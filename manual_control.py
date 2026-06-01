import cv2
import numpy as np
import math
import time
import user_input as inp
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input, PoseFilter
from marker_det import ArucoDetector, QRCodeDetector, QReaderDetector, HybridQRDetector
from marker_est import PoseEstimator, PosePlotter3D
# from sim_tools import DifferentialCar, sim

# car = DifferentialCar( left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'), right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint') )
# reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id=16, marker_size=0.1)

car = Puzzlebot()
reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), marker_id=0, marker_size=0.1)

# reference = QRCodeDetector(qr_size=0.1, K=car.K, D=car.D) # 18 Hz
# reference = QReaderDetector(qr_size=0.1)                    # 7 Hz
# reference = HybridQRDetector(qr_size=0.1, K=car.K, D=car.D)

estimator = PoseEstimator(reference=reference, K=car.K, D=car.D)
plotter = PosePlotter3D(reference, axis_limit=1.0, camera_at_origin=False)

init_window('Camera', img_size=car.img_size, height=360)

stream_enabled = True
plotter_enabled = False
pose_filter = PoseFilter(alpha=0.15)
_t_last = time.perf_counter()
_loop_hz = 0.0

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
        _t_now = time.perf_counter()
        _dt = _t_now - _t_last
        _t_last = _t_now
        _loop_hz = 0.9 * _loop_hz + 0.1 * (1.0 / _dt) if _dt > 0 else _loop_hz

        ret, frame = car.get_image()
        if ret:
            drawing_frame = frame.copy() if stream_enabled else None
            res = pose_filter.update(estimator.get_pose(frame, drawing_frame=drawing_frame))
            if res is not None:
                pose_T, _, _ = res
                x_pos   = np.linalg.inv(pose_T)[0, 3]
                z_dist  = pose_T[2, 3]
                bearing = math.atan2(pose_T[0, 3], pose_T[2, 3])
                beta    = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))
                print(f"x={x_pos:+.3f} z={z_dist:.3f} b={math.degrees(bearing):+.1f}° β={math.degrees(beta):+.1f}° | {_loop_hz:.1f}Hz")
                if plotter_enabled:
                    plotter.update(pose_T)
            if stream_enabled:
                cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
