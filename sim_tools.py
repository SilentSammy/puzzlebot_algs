from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import math
import cv2

client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')

class DifferentialCar:
    def __init__(self, left_wheel=None, right_wheel=None, cam_handle=None, cam_res_x=2048, cam_res_y=2048, cam_fov_deg=60.0):
        # Get wheel handles from the global sim object
        self.left_wheel = left_wheel or sim.getObject('/DynamicLeftJoint')
        self.right_wheel = right_wheel or sim.getObject('/DynamicRightJoint')
        self._cam = cam_handle or sim.getObject('/visionSensor')

        # Camera intrinsics
        _f = (cam_res_x / 2) / math.tan(math.radians(cam_fov_deg) / 2)
        self.K = np.array([[_f,  0,  cam_res_x / 2],
                           [ 0, _f,  cam_res_y / 2],
                           [ 0,  0,              1]], dtype=np.float64)
        self.D = np.zeros(5, dtype=np.float64)
        self.img_size = (cam_res_x, cam_res_y)

        # Differential car constants
        self.nominalLinearVelocity  = 0.12    # nominal linear speed (m/s)
        self.nominalAngularVelocity = math.pi/6    # nominal angular speed (rad/s)
        self.wheelRadius = 0.027             # wheel radius (m)
        self.interWheelDistance = 0.119      # distance between wheels (m)

        # Internal speeds (m/s and rad/s)
        self._linear_speed = 0.0
        self._angular_speed = 0.0

        # Internal time tracking
        self._last_lin_time = None
        self._last_ang_time = None
        
        # Apply initial wheel speeds
        self._update_wheel_velocities()

    def _update_wheel_velocities(self):
        # Kinematic equations for wheel speeds
        left_speed = (self._linear_speed - (self._angular_speed * self.interWheelDistance / 2)) / self.wheelRadius
        right_speed = (self._linear_speed + (self._angular_speed * self.interWheelDistance / 2)) / self.wheelRadius
        
        # Batch update using stepping
        client.setStepping(True)
        sim.setJointTargetVelocity(self.left_wheel, float(left_speed))
        sim.setJointTargetVelocity(self.right_wheel, float(right_speed))
        client.setStepping(False)

    @property
    def lin_vel(self):
        return self._linear_speed
    
    @lin_vel.setter
    def lin_vel(self, value):
        self._linear_speed = value
        self._update_wheel_velocities()
    
    @property
    def ang_vel(self):
        return self._angular_speed
    
    @ang_vel.setter
    def ang_vel(self, value):
        self._angular_speed = value
        self._update_wheel_velocities()

    def stop(self):
        self.lin_vel = 0.0
        self.ang_vel = 0.0

    def _publish(self):
        pass  # sim applies velocities immediately in the setters

    def get_image(self):
        try:
            frame = get_image(self._cam)
            return True, frame
        except Exception:
            return False, None
    
    def spin_up_to(self, target_ang_vel, acc=math.radians(90)):
        """Smoothly reach a target angular velocity"""
        current_time = time.time()
        dt = current_time - self._last_ang_time if self._last_ang_time and (current_time - self._last_ang_time) <= 0.5 else 0.0
        self._last_ang_time = current_time
        
        # Compute the difference between the target and current angular speed.
        diff = target_ang_vel - self.ang_vel

        # Change in angular speed for this time step.
        dv = abs(acc) * dt
        if abs(diff) < dv:
            dv = abs(diff)
        
        # Update the angular speed.
        self.ang_vel += (1 if diff > 0 else -1) * dv    

    def accelerate_to(self, target_vel, acc=1):
        """Smoothly reach a target linear velocity"""
        # Get dt
        current_time = time.time()
        dt = current_time - self._last_lin_time if self._last_lin_time and (current_time - self._last_lin_time) <= 0.5 else 0.0
        self._last_lin_time = current_time
        
        # Compute the difference.
        diff = target_vel - self.lin_vel

        # Change in velocity for this time step.
        dv = abs(acc) * dt
        if abs(diff) < dv:
            dv = abs(diff)
        
        # Update the speed.
        self.lin_vel += (1 if diff > 0 else -1) * dv

def orient_object(object_handle, alpha=None, beta=None, gamma=None):
    """Sets an object's orientation to specific angles (in radians)."""
    orientation = sim.getObjectOrientation(object_handle, -1)
    orientation[0] = alpha if alpha is not None else orientation[0]
    orientation[1] = beta if beta is not None else orientation[1]
    orientation[2] = gamma if gamma is not None else orientation[2]
    sim.setObjectOrientation(object_handle, -1, orientation)

def move_object(object_handle, x=None, y=None, z=None):
    """Teleports an object to a specific position."""
    position = sim.getObjectPosition(object_handle, -1)
    position[0] = x if x is not None else position[0]
    position[1] = y if y is not None else position[1]
    position[2] = z if z is not None else position[2]
    sim.setObjectPosition(object_handle, -1, position)

def translate_object(object_handle, x=0, y=0, z=0):
    """Adds to an object's position."""
    position = sim.getObjectPosition(object_handle, -1)
    sim.setObjectPosition(object_handle, -1, [position[0] + x, position[1] + y, position[2] + z])

def rotate_object(object_handle, alpha=0, beta=0, gamma=0):
    """Adds to an object's orientation (in radians)."""
    orientation = sim.getObjectOrientation(object_handle, -1)
    sim.setObjectOrientation(object_handle, -1, [orientation[0] + alpha, orientation[1] + beta, orientation[2] + gamma])

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def screenshot_and_exit(cam_handle):
    frame = get_image(cam_handle)
    cv2.imshow('Vision Sensor Image', frame)
    cv2.imwrite('last_frame.png', frame)
    cv2.waitKey(1000)
    raise SystemExit
