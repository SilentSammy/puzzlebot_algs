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
    PoseFilter(alpha=0.05, max_jump=0.1)
)
plotter = PosePlotter3D(reference, axis_limit=0.5, camera_at_origin=False)

init_window('Camera', img_size=car.img_size, height=360)

def decompose_pose(pose_T):
    """Extract (x_pos, z_dist, beta) from a 4x4 camera-space pose matrix."""
    x_pos = np.linalg.inv(pose_T)[0, 3]
    z_dist = pose_T[2, 3]
    beta   = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))
    return x_pos, z_dist, beta

def compose_pose(x_pos, z_dist, beta):
    """Reconstruct a 4x4 camera-space pose matrix from (x_pos, z_dist, beta).
    Counterpart to decompose_pose."""
    c, s = math.cos(beta), math.sin(beta)
    tx = (s * z_dist - x_pos) / c if abs(c) > 1e-6 else 0.0
    return np.array([
        [ c, 0,  s,      tx],
        [ 0, 1,  0,       0],
        [-s, 0,  c,  z_dist],
        [ 0, 0,  0,       1],
    ], dtype=np.float64)

def _rot4(axis, deg):
    """4x4 rotation matrix around 'x', 'y', or 'z' by deg degrees."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    if axis == 'x':
        T[1,1], T[1,2], T[2,1], T[2,2] = c, -s, s, c
    elif axis == 'y':
        T[0,0], T[0,2], T[2,0], T[2,2] = c, s, -s, c
    elif axis == 'z':
        T[0,0], T[0,1], T[1,0], T[1,1] = c, -s, s, c
    return T

def car_to_cam_pose(T_car):
    """Convert a car-frame SE(2) matrix (rot-z, trans-xy) to a camera-frame matrix (rot-y, trans-xz).

    Car frame:    x=forward, y=left, z=up   (rotation around z)
    Camera frame: x=left,    y=down, z=fwd  (rotation around y)
    Mapping: car.x → cam.z,  car.y → cam.x,  theta unchanged.
    """
    theta = math.atan2(T_car[1, 0], T_car[0, 0])
    cx, cz = T_car[1, 3], T_car[0, 3]  # cam.x = car.y, cam.z = car.x
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c,  0, -s,  cx],
        [ 0,  1,  0,   0],
        [ s,  0,  c,  cz],
        [ 0,  0,  0,   1],
    ], dtype=np.float64)

stream_enabled = True
plotter_enabled = False
_t_last = time.perf_counter()
_loop_hz = 0.0
_last_cam = None         # (x_pos, z_dist, beta)
_last_odom = None        # (ox, oz, ob)
_pose_offset = None      # (dx, dz, db) = cam - odom scalars, frozen when camera fails
_last_pnp_result = None  # last successful PnpResult for fallback reproject

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
        odom_cam = None
        if odom_T is not None:
            odom_cam = car_to_cam_pose(odom_T)
            ox = odom_cam[0, 3]
            oz = odom_cam[2, 3]
            ob = math.atan2(-odom_cam[2, 0], math.hypot(odom_cam[0, 0], odom_cam[1, 0]))
            _last_odom = (ox, oz, ob)
        fused_T = None
        if _pose_offset is not None and _last_odom is not None:
            fx = _last_odom[0] + _pose_offset[0]
            fz = _last_odom[1] + _pose_offset[1]
            fb = _last_odom[2] + _pose_offset[2]
            fused_T = compose_pose(fx, fz, fb)
        if ret:
            drawing_frame = frame.copy() if stream_enabled else None
            res = estimator.get_pose(frame, drawing_frame=drawing_frame)
            if res is not None:
                pose_T, pnp_result, _ = res
                _last_cam = decompose_pose(pose_T)
                _last_pnp_result = pnp_result
                if _last_odom is not None:
                    _pose_offset = (
                        _last_cam[0] - _last_odom[0],
                        _last_cam[1] - _last_odom[1],
                        _last_cam[2] - _last_odom[2],
                    )
                    fused_T = compose_pose(_last_cam[0], _last_cam[1], _last_cam[2])
                if plotter_enabled:
                    plotter.update(pose_T)
            if drawing_frame is not None:
                if fused_T is not None and _last_pnp_result is not None:
                    pts = estimator._estimator.reproject(fused_T, _last_pnp_result, drawing_frame.shape)
                    for i, pt in enumerate(pts):
                        if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                            continue
                        color = (0, 255, 255) if i == 0 else (0, 0, 255)
                        cv2.circle(drawing_frame, (int(pt[0]), int(pt[1])), 6, color, -1)
            if stream_enabled:
                cv2.imshow('Camera', drawing_frame)

        np.set_printoptions(precision=4, suppress=True, sign='+')
        if res is not None:
            print(f"cam:\n{pose_T}")
        if fused_T is not None:
            print(f"fused:\n{fused_T}")
        _last_fused = decompose_pose(fused_T) if fused_T is not None else None
        cam_str   = (f"cam   x={_last_cam[0]:+.3f} z={_last_cam[1]:.3f} β={math.degrees(_last_cam[2]):+.1f}°"
                     if _last_cam is not None else "cam   --")
        fused_str = (f"  fused x={_last_fused[0]:+.3f} z={_last_fused[1]:.3f} β={math.degrees(_last_fused[2]):+.1f}°"
                     if _last_fused is not None else "  fused --")
        print(f"{cam_str}{fused_str}  | {_loop_hz:.1f}Hz")
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
