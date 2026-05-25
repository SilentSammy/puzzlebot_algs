# from sim_tools import sim, get_image, DifferentialCar
import cv2
import numpy as np
import math
import user_input as inp
from ctrl_helpers import init_window, get_diff_drive_input, merge_proportional, get_manual_override, PoseFilter
from marker_det import ArucoDetector
from marker_est import PoseEstimator, PosePlotter3D
from pb_bridge import Puzzlebot

reverse_state = False
def follow(frame, drawing_frame=None):
    global reverse_state
    # --- Tuning constants ---
    Kp_w           = 0.0015            # angular P gain (cmd/px)
    AIM_CLAMP      = 0.75               # max aim offset fraction (0=centre, 1=edge)
    W_CLAMP        = 0.5               # max angular command magnitude
    LIN_AUTH_ANGLE = math.radians(20)  # gaze angle at which forward authority → 0
    TARGET_DIST    = 0.13               # m — normal approach distance
    REVERSE_DIST   = 0.25              # m — back-off distance when reversing
    DIST_GAIN      = 1.0 / 0.25        # full x output at 0.5 m error
    X_CLAMP        = 0.6               # max linear command magnitude
    GOAL_RADIUS    = 0.01             # m — half-side of goal square
    AIM_GAIN       = 10.0              # scales x_pos (m) into aim fraction
    # ------------------------
    cmd = {'x': 0.0, 'w': 0.0}
    h, w = frame.shape[:2]
    res = pose_filter.update(estimator.get_pose(frame, drawing_frame=drawing_frame))

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
        x_pos   = np.linalg.inv(pose_T)[0, 3]
        z_dist  = pose_T[2, 3]
        bearing = math.atan2(pose_T[0, 3], pose_T[2, 3])
        beta    = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))

        # At goal: align to beta=0 in place, skip tracking computations
        if not reverse_state and abs(z_dist - TARGET_DIST) < GOAL_RADIUS and abs(x_pos) < GOAL_RADIUS:
            w_cmd = -max(-W_CLAMP, min(W_CLAMP, Kp_w * math.tan(beta) * _f))
            print(f"x={x_pos:+.3f} z={z_dist:.3f} b={math.degrees(bearing):+.1f}° β={math.degrees(beta):+.1f}° w={w_cmd:+.3f}  GOAL")
            return {'x': 0.0, 'w': w_cmd}

        # Aim shifts left/right based on lateral position (tvec[0]), scaled for sensitivity
        aim = max(-AIM_CLAMP, min(AIM_CLAMP, (-1 if reverse_state else 1) * x_pos * AIM_GAIN))

        # Proportional control to determine angular velocity command
        ref_px, marker_target_px = get_target_px(aim)
        error_px = ref_px - marker_target_px
        w_cmd = max(-W_CLAMP, min(W_CLAMP, Kp_w * error_px))

        # Linear authority: full drive when aligned, fades to 0 at LIN_AUTH_ANGLE
        gaze_angle = math.atan2(abs(error_px), _f)
        lin_auth   = max(0.0, 1.0 - gaze_angle / LIN_AUTH_ANGLE)

        dist_target = TARGET_DIST if not reverse_state else REVERSE_DIST
        x_cmd = lin_auth * DIST_GAIN * (z_dist - dist_target)
        x_cmd = max(-X_CLAMP, min(X_CLAMP, x_cmd))

        z_reached   = abs(z_dist - dist_target) < GOAL_RADIUS
        x_reached   = abs(x_pos) < GOAL_RADIUS
        at_distance = z_reached and not x_reached

        if reverse_state and (x_reached or z_reached):
            reverse_state = False
        elif at_distance:
            reverse_state = True

        status = 'REVERSING' if reverse_state else ('AT DISTANCE' if at_distance else '')
        print(f"x={x_pos:+.3f} z={z_dist:.3f} b={math.degrees(bearing):+.1f}° β={math.degrees(beta):+.1f}° aim={aim:+.2f} err={error_px:+.0f}px w={w_cmd:+.3f} auth={lin_auth:.2f}  {status}")

        return {'x': x_cmd, 'w': w_cmd}
    return cmd

# CAR SETUP
# car = DifferentialCar( left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'), right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint') )
# reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id=16, marker_size=0.1)
car = Puzzlebot( K=np.array([[793.9798621618975, 0.0, 628.3131432349588], [0.0, 793.3904503227144, 375.7912014522259], [0.0, 0.0, 1.0]], dtype=np.float32), D=np.array([-0.3515292796708493, 0.158025188818097, -1.861499533667287e-05, -0.00031474130783931936, -0.03843522930855781], dtype=np.float32), img_size=(1280, 720))
reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), marker_id=0, marker_size=0.1)

# SETUP
init_window('Camera', img_size=car.img_size, height=360)
_f = car.K[0, 0]
estimator = PoseEstimator(reference=reference, K=car.K, D=car.D)
pose_filter = PoseFilter(alpha=0.05)

cmd_enables = {'x': 0.0, 'w': 0.0}
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

        cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
    # plotter.close()
