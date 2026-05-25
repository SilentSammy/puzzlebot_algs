import cv2
import numpy as np
import user_input as inp
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input
from marker_det import ArucoDetector
from marker_est import PoseEstimator, PosePlotter3D
from sim_tools import DifferentialCar, sim

# car = DifferentialCar( left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'), right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint') )
car = Puzzlebot( K=np.array([[793.9798621618975, 0.0, 628.3131432349588], [0.0, 793.3904503227144, 375.7912014522259], [0.0, 0.0, 1.0]], dtype=np.float32), D=np.array([-0.3515292796708493, 0.158025188818097, -1.861499533667287e-05, -0.00031474130783931936, -0.03843522930855781], dtype=np.float32), img_size=(1280, 720))

aruco = ArucoDetector(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    marker_id=16,
    marker_size=0.1
)
estimator = PoseEstimator(reference=aruco, K=car.K, D=car.D)
plotter = PosePlotter3D(aruco, axis_limit=1.0, camera_at_origin=False)

init_window('Camera')

stream_enabled = True
plotter_enabled = False

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

    ret, frame = car.get_image()
    if ret:
        drawing_frame = frame.copy() if stream_enabled else None
        if plotter_enabled:
            res = estimator.get_pose(frame, drawing_frame=drawing_frame)
            if res is not None:
                plotter.update(res[0])
        if stream_enabled:
            cv2.imshow('Camera', drawing_frame)
    cv2.waitKey(1)
