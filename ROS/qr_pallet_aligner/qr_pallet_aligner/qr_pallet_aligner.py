#!/usr/bin/env python3

# 1. Standard Python Libraries
import math
import numpy as np

# 2. Third-Party Libraries
import cv2
from cv_bridge import CvBridge

# 3. ROS 2 Core, Parameters & Services
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool

# 4. ROS 2 Standard & Sensor Messages
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import TwistStamped

# 5. ROS 2 QoS Profiles
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data

# 6. Internal vision / control pipeline (pose + transforms handled in-process)
from .pipeline import build_servoing

CONTROL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)


class QrPalletAlignerNode(Node):
    """
    Self-contained PBVS Controller (drop-in for QrPalletAligner).

    Same external contract as the original node — subscribes to the camera feed,
    publishes ``~/cmd_vel_stamped`` / ``~/debug_image`` / ``~/status`` and exposes
    ``~/enable`` — but performs marker detection, pose estimation and the
    camera->robot transform internally.  It therefore does NOT depend on an
    external pose-calculator topic or a published TF tree; the camera extrinsic is
    supplied as the ``camera_x_offset`` / ``camera_z_offset`` parameters instead.
    """

    def __init__(self):
        super().__init__('qr_pallet_aligner')

        # --- CONCISE PARAMETERS ---
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('use_compressed', True)

        # Marker geometry (replaces the external pose calculator).
        self.declare_parameter('qr_size', 0.05)
        self.declare_parameter('qr_content', '')

        # Camera extrinsic relative to the robot centre (replaces TF2 lookups).
        self.declare_parameter('camera_x_offset', -0.065)
        self.declare_parameter('camera_z_offset', 0.075)

        self.declare_parameter('kp_linear', 0.8)
        self.declare_parameter('max_linear_velocity', 0.08)
        self.declare_parameter('nominal_linear_velocity', 0.5)

        self.declare_parameter('kp_angular', 1.5)
        self.declare_parameter('kd_angular', 0.5)
        self.declare_parameter('max_angular_velocity', 0.15)
        self.declare_parameter('nominal_angular_velocity', 3.14159)

        self.declare_parameter('target_forward_distance', 0.6)
        self.declare_parameter('reverse_forward_distance', 1.0)
        self.declare_parameter('position_tolerance', 0.04)

        # --- LOAD PARAMETERS ---
        self.update_rate = self.get_parameter('update_rate').value
        self.image_topic = self.get_parameter('image_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.qr_size = self.get_parameter('qr_size').value
        self.qr_content = self.get_parameter('qr_content').value
        self.camera_x_offset = self.get_parameter('camera_x_offset').value
        self.camera_z_offset = self.get_parameter('camera_z_offset').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.nominal_linear_velocity = self.get_parameter('nominal_linear_velocity').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.kd_angular = self.get_parameter('kd_angular').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.nominal_angular_velocity = self.get_parameter('nominal_angular_velocity').value
        self.target_forward_distance = self.get_parameter('target_forward_distance').value
        self.reverse_forward_distance = self.get_parameter('reverse_forward_distance').value
        self.position_tolerance = self.get_parameter('position_tolerance').value

        self.add_on_set_parameters_callback(self.parameter_callback)

        # State Tracking
        self.K = None
        self.D = None
        self.latest_frame = None        # most recent decoded image (BGR)
        self.servoing = None            # built lazily once intrinsics arrive

        # Tools (no TF2 — transforms are handled inside MarkerServoing)
        self.bridge = CvBridge()

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(TwistStamped, '~/cmd_vel_stamped', CONTROL_QOS)

        if self.use_compressed:
            self.image_pub = self.create_publisher(CompressedImage, '~/debug_image/compressed', qos_profile_sensor_data)
            self.image_sub = self.create_subscription(CompressedImage, f"{self.image_topic}/compressed", self.image_callback, qos_profile_sensor_data)
        else:
            self.image_pub = self.create_publisher(Image, '~/debug_image', qos_profile_sensor_data)
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)

        self.info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, qos_profile_sensor_data)

        # Server State
        self.is_enabled = False
        self.state = "IDLE"
        self.enable_srv = self.create_service(SetBool, '~/enable', self.enable_callback)
        self.status_pub = self.create_publisher(String, '~/status', STATE_QOS)
        self.status_timer = self.create_timer(0.1, self.publish_status)

        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        self.get_logger().info("QrPalletAligner (Standalone PBVS) Start.")

    def publish_status(self):
        self.status_pub.publish(String(data=self.state))

    def enable_callback(self, request, response):
        self.is_enabled = request.data
        self.state = "RUNNING" if self.is_enabled else "IDLE"
        if not self.is_enabled:
            self.stop_robot()
            self.latest_frame = None
            if self.servoing is not None:
                self.servoing.reset()
        response.success = True
        return response

    def stop_robot(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        self.cmd_pub.publish(msg)

    def camera_info_callback(self, msg):
        """Capture intrinsics and build the servoing pipeline once available."""
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64).reshape(-1)
        if self.servoing is None:
            self._build_pipeline()

    def _build_pipeline(self):
        """Construct the detector -> estimator -> tracker -> controller chain."""
        self.servoing = build_servoing(
            self.K, self.D,
            qr_size=self.qr_size,
            qr_content=self.qr_content or None,
            x_offset=self.camera_x_offset,
            z_offset=self.camera_z_offset,
            verbose=False,
        )

    def image_callback(self, msg):
        """Decode and stash the latest frame; control runs on the timer."""
        if not self.is_enabled:
            return
        try:
            self.latest_frame = (self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                                 if self.use_compressed
                                 else self.bridge.imgmsg_to_cv2(msg, 'bgr8'))
        except Exception:
            self.latest_frame = None

    def timer_callback(self):
        if not self.is_enabled or self.servoing is None or self.latest_frame is None:
            return

        frame = self.latest_frame
        self.latest_frame = None
        drawing_frame = frame.copy()

        # Internal pipeline: detect marker, estimate pose, transform to robot
        # frame and compute the command — all behind a single call.
        cmd = self.servoing.update(frame, drawing_frame=drawing_frame)

        # Scale normalized [-1,1] command by nominal velocity, then clamp to max.
        x_cmd = max(-self.max_linear_velocity, min(self.max_linear_velocity, cmd['x'] * self.nominal_linear_velocity))
        w_cmd = max(-self.max_angular_velocity, min(self.max_angular_velocity, cmd['w'] * self.nominal_angular_velocity))

        self.state = "RUNNING"

        # Apply Commands
        tw = TwistStamped()
        tw.header.stamp = self.get_clock().now().to_msg()
        tw.header.frame_id = "base_footprint"
        tw.twist.linear.x, tw.twist.angular.z = float(x_cmd), float(w_cmd)
        self.cmd_pub.publish(tw)

        # Debug overlay (MarkerServoing already annotated drawing_frame)
        self._publish_debug(drawing_frame, msg_header=None)

    def _publish_debug(self, cv_image, msg_header=None):
        try:
            out_msg = (self.bridge.cv2_to_compressed_imgmsg(cv_image)
                       if self.use_compressed
                       else self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
            out_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(out_msg)
        except Exception:
            pass

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'update_rate': self.update_rate = float(param.value)
            elif param.name == 'kp_linear': self.kp_linear = float(param.value)
            elif param.name == 'max_linear_velocity': self.max_linear_velocity = float(param.value)
            elif param.name == 'nominal_linear_velocity': self.nominal_linear_velocity = float(param.value)
            elif param.name == 'kp_angular': self.kp_angular = float(param.value)
            elif param.name == 'kd_angular': self.kd_angular = float(param.value)
            elif param.name == 'max_angular_velocity': self.max_angular_velocity = float(param.value)
            elif param.name == 'nominal_angular_velocity': self.nominal_angular_velocity = float(param.value)
            elif param.name == 'qr_size': self.qr_size = float(param.value)
            elif param.name == 'qr_content': self.qr_content = str(param.value)
            elif param.name == 'camera_x_offset': self.camera_x_offset = float(param.value)
            elif param.name == 'camera_z_offset': self.camera_z_offset = float(param.value)
            elif param.name == 'target_forward_distance': self.target_forward_distance = float(param.value)
            elif param.name == 'reverse_forward_distance': self.reverse_forward_distance = float(param.value)
            elif param.name == 'position_tolerance': self.position_tolerance = float(param.value)
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = QrPalletAlignerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
