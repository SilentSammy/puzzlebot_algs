import cv2
import numpy as np
import math
import user_input as inp
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input, PoseFilter, PoseTracker, cam_to_car
from marker_det import ArucoDetector, QRCodeDetector
from marker_est import PoseEstimator, PosePlotter3D

car = Puzzlebot()
reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id=0, marker_size=0.04)
# reference = QRCodeDetector(qr_size=0.0334, K=car.K, D=car.D)

tracker = PoseTracker(
    PoseEstimator(reference=reference, K=car.K, D=car.D),
    PoseFilter(alpha=0.05, max_jump=0.2),
)

init_window('Camera', img_size=car.img_size, height=360)
plotter = PosePlotter3D(reference, axis_limit=0.3, update_interval=1, camera_at_origin=False)

np.set_printoptions(precision=4, suppress=True, sign='+')

show_car = False  # False: show cam_T, True: show car_T


try:
    while True:
        cmd = get_diff_drive_input()
        car.lin_vel = cmd['x'] * car.nominalLinearVelocity
        car.ang_vel = cmd['w'] * car.nominalAngularVelocity
        car._publish()

        if inp.rising_edge('3'):
            show_car = not show_car
            print(f"Showing: {'car_T' if show_car else 'cam_T'}")

        ret, frame = car.get_image()
        drawing_frame = frame.copy() if ret else None

        result = tracker.get_pose(frame if ret else None, drawing_frame=drawing_frame)

        if result is not None:
            cam_T, pnp_result, detection = result

            car_T = cam_to_car(cam_T, x_off=0.05)

            pose_T = car_T if show_car else cam_T
            plotter.update(pose_T)

            # Decompose pose: position (x, y, z) and orientation (roll, pitch, yaw)
            dec_T = np.linalg.inv(pose_T)
            x = dec_T[0, 3]
            y = dec_T[1, 3]
            z = dec_T[2, 3]
            R = dec_T[:3, :3]
            roll  = math.atan2(R[1, 0], R[0, 0])
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw   = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
            label = 'car_T' if show_car else 'cam_T'
            print(f"[{label}] x={x:+.3f} y={y:+.3f} z={z:.3f} roll={math.degrees(roll):+.1f}deg pitch={math.degrees(pitch):+.1f}deg yaw={math.degrees(yaw):+.1f}deg")

        if ret:
            if result is not None:
                cam_T, pnp_result, detection = result
                pts = tracker._estimator.reproject(cam_T, pnp_result, drawing_frame.shape)
                for i, pt in enumerate(pts):
                    if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                        continue
                    color = (0, 0, 255) if i == 0 else (0, 255, 255)
                    cv2.circle(drawing_frame, (int(float(pt[0])), int(float(pt[1]))), 6, color, -1)
            cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    car.lin_vel = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
    plotter.close()
