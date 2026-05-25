from sim_tools import sim, get_image, DifferentialCar
import cv2
import numpy as np
import math
import user_input as inp
from ctrl_helpers import init_window, get_diff_drive_input, merge_proportional, get_manual_override
from marker_det import ArucoDetector
from marker_est import PoseEstimator, PosePlotter3D
from pb_bridge import Puzzlebot

def show_frame(name, img, scale=1, wait_for_key=True):
    show_frame.first_time = show_frame.first_time if hasattr(show_frame, 'first_time') else True
    if show_frame.first_time:
        init_window(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        show_frame.first_time = False
    cv2.imshow(name, img)
    if wait_for_key and cv2.waitKey(1) & 0xFF == 27:
        raise KeyboardInterrupt

def follow_2(pose_T, detection):
    # Pose data
    tvec     = pose_T[:3, 3].copy()
    z_distance = tvec[2]
    bearing  = math.atan2(tvec[0], tvec[2])
    beta     = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))

    # ArUco corners in original image coordinates
    img_pts        = detection.img_pts          # shape (4, 2), (u, v) per corner
    xs             = img_pts[:, 0]
    marker_left_u  = xs.min()
    marker_right_u = xs.max()
    marker_mid_u   = (marker_left_u + marker_right_u) / 2

    approach_error = bearing + beta    # rotation-corrected: lateral offset from marker's Z axis

    # Target pixel where marker centre should sit for correct approach geometry
    # (equivalent to gaze_error + approach_error * APPROACH_FACTOR = 0 solved for marker_mid_u)
    APPROACH_FACTOR = 5.0  # 1.0 = full approach correction, 0.0 = pure centering
    scaled_angle   = max(-math.radians(85), min(math.radians(85), approach_error * APPROACH_FACTOR))
    ideal_target_u = CAM_RES_X / 2 + math.tan(scaled_angle) * _f

    # Clamp using the actual marker corners so neither edge can leave the frame
    half_marker    = (marker_right_u - marker_left_u) / 2
    MARGIN         = 0.1 * CAM_RES_X  # px safety margin from image edge (~20% of 2048)
    safe_target_u  = max(MARGIN + half_marker,
                         min(CAM_RES_X - MARGIN - half_marker, ideal_target_u))

    gaze_error = math.atan2(safe_target_u - marker_mid_u, _f)
    lin_auth   = max(0.0, 1.0 - abs(gaze_error) / math.radians(10))  # 0° → 1.0, ≥10° → 0.0

    GAZE_GAIN = 1.0 / math.radians(30)  # normalize: 30° error → full w output
    w = GAZE_GAIN * gaze_error
    w = max(-0.5, min(0.5, w))

    TARGET_DIST  = 0.2   # m
    DIST_GAIN    = 1.0 / 0.5  # normalize: 0.5m error → full x output
    x = lin_auth * DIST_GAIN * (z_distance - TARGET_DIST)
    x = max(-1.0, min(1.0, x))

    GOAL_RADIUS  = 0.05  # m — half-side of goal square in XZ plane
    z_reached    = abs(z_distance - TARGET_DIST) < GOAL_RADIUS
    x_reached    = abs(tvec[0]) < GOAL_RADIUS
    goal_reached = z_reached and x_reached
    at_distance  = z_reached and not x_reached  # right depth, wrong lateral alignment

    status = 'GOAL' if goal_reached else ('AT DISTANCE' if at_distance else '')
    print(f"dist={z_distance:.3f}m  approach={math.degrees(approach_error):.1f}°  target_u={safe_target_u:.0f}  mid_u={marker_mid_u:.0f}  beta={math.degrees(beta):.1f}°  {status}")

    return {'x': x, 'w': w}

reverse_state = False
def follow(frame, drawing_frame=None):
    global reverse_state
    Kp_w = 0.0015
    cmd = {'x': 0.0, 'w': 0.0}
    h, w = frame.shape[:2]
    res = estimator.get_pose(frame, drawing_frame=drawing_frame)

    def get_target_px(aim):
        ref_px = (w / 2) + (aim * (w / 2))
        xs = res[2].img_pts[:, 0]
        marker_left_px  = xs.min()
        marker_right_px = xs.max()
        marker_target_px = (marker_left_px + marker_right_px) / 2 + aim * (marker_right_px - marker_left_px) / 2
        if drawing_frame is not None:
            # Draw vertical lines for reference and marker position
            cv2.line(drawing_frame, (int(ref_px), 0), (int(ref_px), h), (255, 0, 0), 2)  # Blue line for reference
            cv2.line(drawing_frame, (int(marker_target_px), 0), (int(marker_target_px), h), (0, 255, 0), 2)  # Green line for marker
        return ref_px, marker_target_px

    if res is not None:
        pose_T, _, detection = res
        # invert the pose matrix to get the marker's position relative to the camera
        x_pos = np.linalg.inv(pose_T)[0, 3]
        z_dist = pose_T[2, 3]
        
        # Aim shifts left/right based on lateral position (tvec[0]), scaled for sensitivity
        aim = max(-0.8, min(0.8, (-1 if reverse_state else 1) * x_pos * 10.0))

        # Proportional control to determine angular velocity command
        ref_px, marker_target_px = get_target_px(aim)
        error_px = ref_px - marker_target_px
        w_cmd = max(-0.5, min(0.5, Kp_w * error_px))

        # Linear authority: full drive when aligned, zero when off by ≥10°
        gaze_angle = math.atan2(abs(error_px), _f)
        lin_auth   = max(0.0, 1.0 - gaze_angle / math.radians(10))

        TARGET_DIST = 0.2 if not reverse_state else 0.75   # m
        DIST_GAIN   = 1.0 / 0.5  # normalize: 0.5m error → full x output
        x_cmd = lin_auth * DIST_GAIN * (z_dist - TARGET_DIST)
        x_cmd = max(-1.0, min(1.0, x_cmd))

        GOAL_RADIUS  = 0.025  # m — half-side of goal square in XZ plane
        z_reached    = abs(z_dist - TARGET_DIST) < GOAL_RADIUS
        x_reached    = abs(x_pos) < GOAL_RADIUS * (2 if not reverse_state else 1)
        goal_reached = z_reached and x_reached
        at_distance  = z_reached and not x_reached  # right depth, wrong lateral alignment

        if reverse_state and (x_reached or z_reached):
            reverse_state = False
        elif at_distance:
            reverse_state = True

        status = 'GOAL' if goal_reached else ('REVERSING' if reverse_state else ('AT DISTANCE' if at_distance else ''))
        print(f"Marker x: {x_pos:.3f} m, Aim: {aim:.2f}, Error: {error_px:.1f} px, w: {w_cmd:.3f}, z: {z_dist:.3f} m, lin_auth: {lin_auth:.2f}  {status}")

        return {'x': x_cmd, 'w': w_cmd}
    return cmd

# car = DifferentialCar(
#     left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'),
#     right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint'),
#     cam_handle=sim.getObject('/Puzzlebot/visionSensor'),
# )
car = Puzzlebot(
    K=np.array([[793.9798621618975, 0.0, 628.3131432349588], [0.0, 793.3904503227144, 375.7912014522259], [0.0, 0.0, 1.0]], dtype=np.float32),
    D=np.array([-0.3515292796708493, 0.158025188818097, -1.861499533667287e-05, -0.00031474130783931936, -0.03843522930855781], dtype=np.float32),
    img_size=(1280, 720)
)  # use the websocket bridge version of the car

_f = car.K[0, 0]
CAM_RES_X = car.img_size[0]

aruco = ArucoDetector(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    marker_id=16,
    marker_size=0.1
)

estimator = PoseEstimator(reference=aruco, K=car.K, D=car.D)
# plotter = PosePlotter3D(aruco, axis_limit=1.0, camera_at_origin=False)

cmd_enables = {'x': 0.0, 'w': 1.0}

try:
    while True:
        ret, frame = car.get_image()
        if not ret:
            continue
        drawing_frame = frame.copy()

        if inp.rising_edge('1'):
            cmd_enables['x'] = 1.0 - cmd_enables['x']  # toggle between 0.0 and 1.0
            print(f"Auto X: {'ON' if cmd_enables['x'] else 'OFF'}")
        if inp.rising_edge('2'):
            cmd_enables['w'] = 1.0 - cmd_enables['w']  # toggle between 0.0 and 1.0
            print(f"Auto W: {'ON' if cmd_enables['w'] else 'OFF'}")

        auto_cmd = follow(frame, drawing_frame=drawing_frame)  # get automatic command based on vision
        auto_cmd = {axis: auto_cmd[axis] * cmd_enables[axis] for axis in auto_cmd}  # apply enables

        cmd = get_manual_override(auto_cmd)
        car.lin_vel  = cmd['x'] * car.nominalLinearVelocity
        car.ang_vel = cmd['w'] * car.nominalAngularVelocity
        car._publish()

        show_frame('Vision Sensor', drawing_frame, scale=0.3)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
    # plotter.close()
