import cv2
import numpy as np
import time
from ctrl_helpers import init_window, PoseFilter
from backg_poller import BackgroundPoller
from marker_det import QRCodeDetector, QReaderDetector
from marker_est import PoseEstimator, PosePlotter3D

# car = DifferentialCar( left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'), right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint') )
# reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id=16, marker_size=0.1)

K=np.array([[735.09668766, 0., 308.18011975], [0., 735.62248422, 242.58646203], [0., 0., 1.]], dtype=np.float32)
D=np.array([0.15017654, -1.34648531, 0.00405315, -0.00410719, 2.41472656], dtype=np.float32)

# reference = QRCodeDetector(qr_size=0.1, K=K, D=D)
reference = QReaderDetector(qr_size=0.1)

estimator = PoseEstimator(reference=reference, K=K, D=D)
plotter   = PosePlotter3D(reference, axis_limit=1.0, camera_at_origin=False)

cap = cv2.VideoCapture(0)
init_window('Camera', width=640, height=360)

stream_enabled = True
plotter_enabled = False
filter_enabled = False
pose_filter = PoseFilter(tau=0.25)
_t_last = time.perf_counter()
_loop_hz = 0.0
qr_poller = BackgroundPoller()

try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('v'):
            stream_enabled = not stream_enabled
            print(f"Stream: {'ON' if stream_enabled else 'OFF'}")
        if key == ord('p'):
            plotter_enabled = not plotter_enabled
            print(f"Plotter: {'ON' if plotter_enabled else 'OFF'}")
        if key == ord('f'):
            filter_enabled = not filter_enabled
            pose_filter.reset()
            print(f"Filter: {'ON' if filter_enabled else 'OFF'}")

        _t_now = time.perf_counter()
        _dt = _t_now - _t_last
        _t_last = _t_now
        _loop_hz = 0.9 * _loop_hz + 0.1 * (1.0 / _dt) if _dt > 0 else _loop_hz

        ret, frame = cap.read()
        if ret:
            drawing_frame = frame.copy() if stream_enabled else None
            _frame = frame.copy()
            res = qr_poller.poll_with_annotated(
                _frame, drawing_frame,
                lambda annot: estimator.get_pose(_frame, drawing_frame=annot)
            )
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
                print(f"rx={raw_x:+.3f} ri={raw_x_inv:+.3f} | fx={filt_x:+.3f} fi={filt_x_inv:+.3f} | z={T_raw[2,3]:.3f} | {_loop_hz:.1f}Hz")
                if plotter_enabled:
                    plotter.update(active_res[0])
            if stream_enabled:
                cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    cap.release()
    cv2.destroyAllWindows()
