import cv2
import numpy as np
import math
import time
import user_input as inp
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input, PoseFilter, PoseTracker
from marker_det import ArucoDetector, QRCodeDetector, QReaderDetector, HybridQRDetector
from marker_est import PoseEstimator, PosePlotter3D
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
    PoseFilter(alpha=0.1)
)
plotter = PosePlotter3D(reference, axis_limit=1.0, camera_at_origin=False)

init_window('Camera', img_size=car.img_size, height=360)

# Reframe: car (x=fwd, y=right, z=up) → camera (x=left, y=down, z=fwd)
_R_car_cam = np.array([
    [ 0., -1.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  1.],
], dtype=np.float64)

stream_enabled = True
plotter_enabled = False
_t_last = time.perf_counter()
_loop_hz = 0.0
_last_cam  = None   # (x_pos, z_dist, beta)
_last_odom = None   # (x, z, beta)
_odom_offset = np.eye(3)  # SE(2): T_corrected = _odom_offset @ T_raw_odom

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
        odom_T = car.estimated_pose
        _raw_odom = None
        if odom_T is not None:
            beta_odom = math.atan2(odom_T[1, 0], odom_T[0, 0])  # yaw around car z
            odom_T = _R_car_cam @ odom_T @ _R_car_cam.T
            _raw_odom = (odom_T[0, 3], -odom_T[2, 3], beta_odom)
            c, s = math.cos(beta_odom), math.sin(beta_odom)
            T_raw = np.array([[c, -s, _raw_odom[0]], [s, c, _raw_odom[1]], [0, 0, 1]])
            T_corr = _odom_offset @ T_raw
            _last_odom = (T_corr[0, 2], T_corr[1, 2], math.atan2(T_corr[1, 0], T_corr[0, 0]))
        if ret:
            drawing_frame = frame.copy() if stream_enabled else None
            res = estimator.get_pose(frame, drawing_frame=drawing_frame)
            if res is not None:
                pose_T, _, _ = res
                x_pos   = np.linalg.inv(pose_T)[0, 3]
                z_dist  = pose_T[2, 3]
                beta    = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))
                _last_cam = (x_pos, z_dist, beta)
                if _raw_odom is not None:
                    c, s = math.cos(beta), math.sin(beta)
                    T_cam = np.array([[c, -s, x_pos], [s, c, z_dist], [0, 0, 1]])
                    rc, rs = math.cos(_raw_odom[2]), math.sin(_raw_odom[2])
                    T_raw = np.array([[rc, -rs, _raw_odom[0]], [rs, rc, _raw_odom[1]], [0, 0, 1]])
                    _odom_offset = T_cam @ np.linalg.inv(T_raw)
                if plotter_enabled:
                    plotter.update(pose_T)
            if stream_enabled:
                cv2.imshow('Camera', drawing_frame)

        cam_str  = (f"cam  x={_last_cam[0]:+.3f} z={_last_cam[1]:.3f} β={math.degrees(_last_cam[2]):+.1f}°"
                    if _last_cam is not None else "cam  --")
        odom_str = (f"  odom x={_last_odom[0]:+.3f} z={_last_odom[1]:.3f} β={math.degrees(_last_odom[2]):+.1f}°"
                    if _last_odom is not None else "  odom --")
        print(f"{cam_str}{odom_str}  | {_loop_hz:.1f}Hz")
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
