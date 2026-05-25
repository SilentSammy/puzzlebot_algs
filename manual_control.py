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

init_window('Camera', width=car.img_size[0] * 360 // car.img_size[1], height=360)

stream_enabled = True
plotter_enabled = False
filter_enabled = False
pose_filter = PoseFilter(alpha=0.02)

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
            if plotter_enabled:
                res = estimator.get_pose(frame, drawing_frame=drawing_frame)
                if filter_enabled:
                    res = pose_filter.update(res)
                if res is not None:
                    plotter.update(res[0])
            if stream_enabled:
                cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
