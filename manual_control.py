import cv2
import numpy as np
import math
import time
import user_input as inp
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input, PoseFilter, PoseTracker
from marker_det import ArucoDetector, QRCodeDetector, QReaderDetector, HybridQRDetector
from marker_est import PoseEstimator
# from sim_tools import DifferentialCar, sim

# car = DifferentialCar( left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'), right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint') )
# reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id=16, marker_size=0.1)

car = Puzzlebot()
reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), marker_id=0, marker_size=0.034)

# reference = QRCodeDetector(qr_size=0.1, K=car.K, D=car.D)     # 18 Hz
# reference = QReaderDetector(qr_size=0.1)                      # 7 Hz
# reference = HybridQRDetector(qr_size=0.1, K=car.K, D=car.D)

estimator = PoseTracker(
    PoseEstimator(reference=reference, K=car.K, D=car.D),
    PoseFilter(alpha=0.05, max_jump=0.1)
)
plotter = None  # disabled

init_window('Camera', img_size=car.img_size, height=360)

def decompose_pose(pose_T):
    """Extract (x_pos, z_dist, beta) from a 4x4 camera-space pose matrix."""
    x_pos = np.linalg.inv(pose_T)[0, 3]
    z_dist = pose_T[2, 3]
    beta   = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))
    return x_pos, z_dist, beta

def compose_pose(x_pos, z_dist, beta):
    """Reconstruct a 4x4 camera-space pose matrix from (x_pos, z_dist, beta).
    Counterpart to decompose_pose."""
    c, s = math.cos(beta), math.sin(beta)
    tx = (s * z_dist - x_pos) / c if abs(c) > 1e-6 else 0.0
    return np.array([
        [ c, 0,  s,      tx],
        [ 0, 1,  0,       0],
        [-s, 0,  c,  z_dist],
        [ 0, 0,  0,       1],
    ], dtype=np.float64)

def _rot4(axis, deg):
    """4x4 rotation matrix around 'x', 'y', or 'z' by deg degrees."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    if axis == 'x':
        T[1,1], T[1,2], T[2,1], T[2,2] = c, -s, s, c
    elif axis == 'y':
        T[0,0], T[0,2], T[2,0], T[2,2] = c, s, -s, c
    elif axis == 'z':
        T[0,0], T[0,1], T[1,0], T[1,1] = c, -s, s, c
    return T

def car_to_cam_pose(T_car):
    """Convert a car-frame SE(2) matrix (rot-z, trans-xy) to a camera-frame matrix (rot-y, trans-xz).

    Car frame:    x=forward, y=left, z=up   (rotation around z)
    Camera frame: x=left,    y=down, z=fwd  (rotation around y)
    Mapping: car.x → cam.z,  car.y → cam.x,  theta unchanged.
    """
    theta = math.atan2(T_car[1, 0], T_car[0, 0])
    cx, cz = -T_car[1, 3], -T_car[0, 3]  # cam.x = -car.y, cam.z = -car.x
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c,  0, -s,  cx],
        [ 0,  1,  0,   0],
        [ s,  0,  c,  cz],
        [ 0,  0,  0,   1],
    ], dtype=np.float64)

np.set_printoptions(precision=4, suppress=True, sign='+')

last_cam_pose = None
try:
    while True:
        cmd = get_diff_drive_input()
        car.lin_vel  = cmd['x'] * car.nominalLinearVelocity
        car.ang_vel = cmd['w'] * car.nominalAngularVelocity
        car._publish()
        
        # Get camera pose from marker (if available)
        ret, frame = car.get_image()
        cam_pose = None
        if ret:
            drawing_frame = frame.copy()
            res = estimator.get_pose(frame, drawing_frame=drawing_frame)
            cv2.imshow('Camera', drawing_frame)
            if res is not None:
                cam_T, _, _ = res
                cam_pose = decompose_pose(cam_T)
                last_cam_pose = cam_pose

        # Get car pose from odometry (if available)
        odom_pose = car.estimated_pose
        odom_pose = (-odom_pose[1], odom_pose[0], odom_pose[2]) if odom_pose is not None else None  # swap x/y to match cam frame
        
        # Print pose info
        cam_str =  "CAM:  x= ---  z= ---  β=  ---°"
        odom_str =  "ODOM: x= ---  z= ---  β=  ---°"
        if cam_pose is not None:
            cam_x, cam_z, cam_beta = cam_pose
            cam_str = f"CAM:  x={cam_x:+.3f} z={cam_z:.3f} β={math.degrees(cam_beta):+.1f}°"
        if odom_pose is not None:
            odom_x, odom_z, odom_beta = odom_pose
            odom_str = f"ODOM: x={odom_x:+.3f} z={odom_z:.3f} β={math.degrees(odom_beta):+.1f}°"
        print(f"{cam_str} | {odom_str}")
        
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
