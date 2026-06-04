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

def update_pose(base_T, x_pos, z_dist, beta):
    """Return a new 4x4 matrix with updated planar pose (x_pos, z_dist, beta),
    keeping the y-row (pitch, roll, height) from base_T unchanged."""
    T = base_T.copy()
    c, s = math.cos(beta), math.sin(beta)
    # decompose_pose uses hypot (always positive), so beta is in (-90°,+90°).
    # When the true rotation has cos<0, we need the supplementary angle (pi-beta):
    # cos(pi-beta)=-c, sin(pi-beta)=s. Detect this by comparing signs.
    if base_T[0, 0] * c < 0:
        c = -c
    T[0, 0], T[0, 2] =  c,  s
    T[2, 0], T[2, 2] = -s,  c
    T[2, 3] = z_dist
    # Solve tx: x_pos = inv(T)[0,3] = -(c*tx + T[1,0]*T[1,3] - s*z_dist)
    T[0, 3] = (-x_pos - T[1, 0] * T[1, 3] + s * z_dist) / c if abs(c) > 1e-6 else 0.0
    return T

def car_to_cam_pose(odom):
    """Convert (x, y, theta) car-frame odometry to (cam_x, cam_z, beta) camera-frame.
    car.x=fwd→cam.z, car.y=left→cam.x (negated to match camera convention)."""
    ox, oy, theta = odom
    return (-oy, -ox, theta)

np.set_printoptions(precision=4, suppress=True, sign='+')

last_cam_pose = None
last_cam_T = None
last_pnp_result = None
offset_pose = None  # (dx, dz, dbeta): added to odom_pose to match cam_pose
use_update_pose = True  # toggle with 'u'
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
            if res is not None:
                cam_T, pnp_result, _ = res
                cam_pose = decompose_pose(cam_T)
                last_cam_T = cam_T
                last_cam_pose = cam_pose
                last_pnp_result = pnp_result

        # Get car pose from odometry (if available)
        raw_odom = car.estimated_pose
        odom_pose = car_to_cam_pose(raw_odom) if raw_odom is not None else None

        # Update offset whenever both sources are available
        if cam_pose is not None and odom_pose is not None:
            offset_pose = ( cam_pose[0] - odom_pose[0], cam_pose[1] - odom_pose[1], cam_pose[2] - odom_pose[2], )

        # Fused pose: odom + frozen offset
        fused_pose = None
        if offset_pose is not None and odom_pose is not None:
            fused_pose = ( odom_pose[0] + offset_pose[0], odom_pose[1] + offset_pose[1], odom_pose[2] + offset_pose[2], )

        # Print pose info
        cam_str   =  "CAM:   x= ---  z= ---  β=  ---°"
        odom_str  =  "ODOM:  x= ---  z= ---  β=  ---°"
        fused_str =  "FUSED: x= ---  z= ---  β=  ---°"
        if cam_pose is not None:
            cam_x, cam_z, cam_beta = cam_pose
            cam_str = f"CAM:   x={cam_x:+.3f} z={cam_z:.3f} β={math.degrees(cam_beta):+.1f}°"
        if odom_pose is not None:
            odom_x, odom_z, odom_beta = odom_pose
            odom_str = f"ODOM:  x={odom_x:+.3f} z={odom_z:.3f} β={math.degrees(odom_beta):+.1f}°"
        if fused_pose is not None:
            fx, fz, fb = fused_pose
            fused_str = f"FUSED: x={fx:+.3f} z={fz:.3f} β={math.degrees(fb):+.1f}°"
        print(f"{cam_str} | {odom_str} | {fused_str}")

        # Build fused 4x4 matrix from last_cam_T base + fused scalar pose
        if use_update_pose and fused_pose is not None and last_cam_T is not None:
            fused_T = update_pose(last_cam_T, *fused_pose)
        elif last_cam_T is not None:
            fused_T = last_cam_T
        else:
            fused_T = None

        # When marker visible, cam_T and fused_T should be identical — print both to compare
        if cam_pose is not None and fused_T is not None:
            print("cam_T:\n", cam_T)
            print("fused_T:\n", fused_T)

        if ret:
            if fused_T is not None and last_pnp_result is not None:
                pts = estimator._estimator.reproject(fused_T, last_pnp_result, drawing_frame.shape)
            elif cam_T is not None:
                pts = estimator._estimator.reproject(cam_T, pnp_result, drawing_frame.shape)
            else:
                pts = []
            for i, pt in enumerate(pts):
                if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                    continue
                color = (0, 255, 255) if i == 0 else (0, 0, 255)
                cv2.circle(drawing_frame, (int(pt[0]), int(pt[1])), 6, color, -1)
            cv2.imshow('Camera', drawing_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('u'):
            use_update_pose = not use_update_pose
            print(f"update_pose: {'ON' if use_update_pose else 'OFF'}")
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
