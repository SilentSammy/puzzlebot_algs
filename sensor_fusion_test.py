import cv2
import numpy as np
import math
import user_input as inp
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input, PoseFilter, FusedPoseTracker
from marker_det import ArucoDetector
from marker_est import PoseEstimator

car = Puzzlebot()
reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), marker_id=0, marker_size=0.034)

fused_tracker = FusedPoseTracker(
    PoseEstimator(reference=reference, K=car.K, D=car.D),
    PoseFilter(alpha=0.05, max_jump=0.1)
)

init_window('Camera', img_size=car.img_size, height=360)

def car_to_cam_pose(odom):
    """Convert (x, y, theta) car-frame odometry to (cam_x, cam_z, beta) camera-frame.
    car.x=fwd->cam.z, car.y=left->cam.x (negated to match camera convention)."""
    ox, oy, theta = odom
    return (-oy, -ox, theta)

np.set_printoptions(precision=4, suppress=True, sign='+')

try:
    while True:
        cmd = get_diff_drive_input()
        car.lin_vel = cmd['x'] * car.nominalLinearVelocity
        car.ang_vel = cmd['w'] * car.nominalAngularVelocity
        car._publish()

        ret, frame = car.get_image()
        drawing_frame = frame.copy() if ret else None

        raw_odom = car.estimated_pose
        odom_cam = car_to_cam_pose(raw_odom) if raw_odom is not None else None

        result = fused_tracker.update(frame if ret else None, odom_cam, drawing_frame=drawing_frame)

        # Print pose info
        cam_str   = "CAM:   x= ---  z= ---  b=  ---deg"
        odom_str  = "ODOM:  x= ---  z= ---  b=  ---deg"
        fused_str = "FUSED: x= ---  z= ---  b=  ---deg"
        if fused_tracker.cam_pose is not None:
            cx, cz, cb = fused_tracker.cam_pose
            cam_str = f"CAM:   x={cx:+.3f} z={cz:.3f} b={math.degrees(cb):+.1f}deg"
        if odom_cam is not None:
            ox, oz, ob = odom_cam
            odom_str = f"ODOM:  x={ox:+.3f} z={oz:.3f} b={math.degrees(ob):+.1f}deg"
        if fused_tracker.fused_pose is not None:
            fx, fz, fb = fused_tracker.fused_pose
            fused_str = f"FUSED: x={fx:+.3f} z={fz:.3f} b={math.degrees(fb):+.1f}deg"
        print(f"{cam_str} | {odom_str} | {fused_str}")

        if ret:
            if result is not None:
                fused_T, pnp_result, detected = result
                pts = fused_tracker._estimator.reproject(fused_T, pnp_result, drawing_frame.shape)
                for i, pt in enumerate(pts):
                    if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                        continue
                    color = (0, 255, 255) if i == 0 else (0, 0, 255)
                    cv2.circle(drawing_frame, (int(pt[0]), int(pt[1])), 6, color, -1)
            cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    car.lin_vel = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
