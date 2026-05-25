import cv2
import numpy as np
import user_input as inp
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input, PoseFilter
from marker_det import ArucoDetector, QRCodeDetector
from marker_est import PoseEstimator, PosePlotter3D
# from sim_tools import DifferentialCar, sim

# car = DifferentialCar( left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'), right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint') )
# reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id=16, marker_size=0.1)

car = Puzzlebot( K=np.array([[793.9798621618975, 0.0, 628.3131432349588], [0.0, 793.3904503227144, 375.7912014522259], [0.0, 0.0, 1.0]], dtype=np.float32), D=np.array([-0.3515292796708493, 0.158025188818097, -1.861499533667287e-05, -0.00031474130783931936, -0.03843522930855781], dtype=np.float32), img_size=(1280, 720))
reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), marker_id=1, marker_size=0.1)

# reference = QRCodeDetector(qr_size=0.1, K=car.K, D=car.D)
estimator = PoseEstimator(reference=reference, K=car.K, D=car.D)
plotter = PosePlotter3D(reference, axis_limit=1.0, camera_at_origin=False)

init_window('Camera', img_size=car.img_size, height=360)

stream_enabled = True
plotter_enabled = False
filter_enabled = False
pose_filter = PoseFilter(alpha=0.15)

try:
    while True:
        cmd = get_diff_drive_input()
        car.lin_vel  = cmd['x'] * car.nominalLinearVelocity
        car.ang_vel = cmd['w'] * car.nominalAngularVelocity
        car._publish()

        if inp.rising_edge('v'):
            stream_enabled = not stream_enabled
            print(f"Stream: {'ON' if stream_enabled else 'OFF'}")
        if inp.rising_edge('p'):
            plotter_enabled = not plotter_enabled
            print(f"Plotter: {'ON' if plotter_enabled else 'OFF'}")
        if inp.rising_edge('f'):
            filter_enabled = not filter_enabled
            pose_filter.reset()
            print(f"Filter: {'ON' if filter_enabled else 'OFF'}")

        ret, frame = car.get_image()
        if ret:
            drawing_frame = frame.copy() if stream_enabled else None
            res = estimator.get_pose(frame, drawing_frame=drawing_frame)
            res_filtered = pose_filter.update(res) if filter_enabled else None
            active_res = res_filtered if filter_enabled else res
            if res is not None:
                T_raw  = res[0]
                T_inv  = np.linalg.inv(T_raw)
                T_filt = res_filtered[0] if res_filtered is not None else None
                T_filt_inv = np.linalg.inv(T_filt) if T_filt is not None else None
                raw_x      = T_raw[0, 3]
                raw_x_inv  = T_inv[0, 3]
                filt_x     = T_filt[0, 3]     if T_filt     is not None else float('nan')
                filt_x_inv = T_filt_inv[0, 3] if T_filt_inv is not None else float('nan')
                print(f"rx={raw_x:+.3f} ri={raw_x_inv:+.3f} | fx={filt_x:+.3f} fi={filt_x_inv:+.3f} | z={T_raw[2,3]:.3f}")
                if plotter_enabled:
                    plotter.update(active_res[0])
            if stream_enabled:
                cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
