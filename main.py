# from sim_tools import sim, get_image, DifferentialCar
import cv2
import numpy as np
import math
import user_input as inp
from ctrl_helpers import init_window, get_diff_drive_input, merge_proportional, get_manual_override, PoseFilter, PoseTracker
from marker_det import ArucoDetector, HybridQRDetector, QRCodeDetector
from marker_est import PoseEstimator, PosePlotter3D
from pb_bridge import Puzzlebot

reverse_state = False
goal_state    = False
def follow(frame, drawing_frame=None):
    global reverse_state, goal_state
    # --- Tuning constants ---
    KP_X           = 1.0                # scales distance error (m) into linear command
    X_CLAMP        = 0.15               # max linear command magnitude
    KP_W           = 0.000300           # angular P gain (cmd/px)
    W_CLAMP        = math.radians(60)   # max angular command magnitude

    AIM_CLAMP      = 0.75               # max aim offset fraction (0=centre, 1=edge)
    AIM_GAIN       = 10.0               # scales x_pos (m) into aim fraction
    LIN_AUTH_ANGLE = math.radians(20)   # gaze angle at which forward authority → 0
    TARGET_DIST    = 0.185              # m — normal approach distance
    REVERSE_DIST   = 0.5                # m — back-off distance when reversing

    X_OFFSET       = -0.05              # m — lateral offset of camera from robot center
    GOAL_RADIUS    = 0.01               # m — half-side of goal square (entry)
    GOAL_HYSTERESIS= 0.006              # m — extra margin to stay in goal (exit)

    # ------------------------
    cmd = {'x': 0.0, 'w': 0.0}
    h, w = frame.shape[:2]
    res = estimator.get_pose(frame, drawing_frame=drawing_frame)

    def get_target_px(aim, base_yaw=0.0, aim_clamp=1.0):
        ref_px = yaw_to_pixel(base_yaw, car.K, car.D) + (aim * (w / 2))
        ref_px = max(w/2 - aim_clamp * w/2, min(w/2 + aim_clamp * w/2, ref_px))
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
        x_pos   = np.linalg.inv(pose_T)[0, 3] - X_OFFSET
        z_dist  = pose_T[2, 3]
        bearing = math.atan2(pose_T[0, 3], pose_T[2, 3])
        beta    = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))

        # Base aim yaw: compensates for camera being offset from robot center
        target_yaw = math.atan2(-X_OFFSET, z_dist)
        if drawing_frame is not None:
            aim_px = yaw_to_pixel(target_yaw, car.K, car.D)
            cv2.line(drawing_frame, (int(aim_px), 0), (int(aim_px), h), (255, 255, 0), 2)  # cyan

        # At goal: align to beta=0 in place, skip tracking computations
        goal_threshold = GOAL_RADIUS + (GOAL_HYSTERESIS if goal_state else 0.0)
        if not reverse_state and abs(z_dist - TARGET_DIST) < goal_threshold and abs(x_pos) < goal_threshold:
            goal_state = True
            w_cmd = -max(-W_CLAMP, min(W_CLAMP, KP_W * math.tan(beta) * _f))
            print(f"x={x_pos:+.3f} z={z_dist:.3f} b={math.degrees(bearing):+.1f}° β={math.degrees(beta):+.1f}° w={w_cmd:+.3f}  GOAL")
            return {'x': 0.0, 'w': w_cmd}
        goal_state = False

        # Aim shifts left/right based on lateral position (tvec[0]), scaled for sensitivity
        aim = max(-AIM_CLAMP, min(AIM_CLAMP, (-1 if reverse_state else 1) * x_pos * AIM_GAIN))

        # Proportional control to determine angular velocity command
        ref_px, marker_target_px = get_target_px(aim, base_yaw=target_yaw, aim_clamp=AIM_CLAMP)
        error_px = ref_px - marker_target_px
        w_cmd = max(-W_CLAMP, min(W_CLAMP, KP_W * error_px))

        # Linear authority: full drive when aligned, fades to 0 at LIN_AUTH_ANGLE
        gaze_angle = math.atan2(abs(error_px), _f)
        lin_auth   = max(0.0, 1.0 - gaze_angle / LIN_AUTH_ANGLE)

        dist_target = TARGET_DIST if not reverse_state else REVERSE_DIST
        x_cmd = lin_auth * KP_X * (z_dist - dist_target)
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

def yaw_to_pixel(yaw_rad, K, D):
    """Return the pixel x-coordinate corresponding to a camera yaw angle (rad).
    Uses the full distortion model via cv2.projectPoints."""
    ray = np.array([[[math.sin(yaw_rad), 0.0, math.cos(yaw_rad)]]], dtype=np.float64)
    px, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), K.astype(np.float64), D.astype(np.float64))
    return float(px[0, 0, 0])

def test(frame, drawing_frame=None):
    K = car.K
    D = car.D
    X_OFFSET = -0.05            # m — lateral offset of camera from robot center
    yaw = math.radians(30)  # example yaw angle

    x_px = yaw_to_pixel(yaw, K, D)
    h = frame.shape[0]
    if drawing_frame is not None:
        cv2.line(drawing_frame, (int(x_px), 0), (int(x_px), h), (0, 165, 255), 2)  # orange
        cv2.putText(drawing_frame, f"{math.degrees(yaw):.0f}deg -> {x_px:.0f}px",
                    (int(x_px) + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    h, w = frame.shape[:2]
    res = estimator.get_pose(frame, drawing_frame=drawing_frame)

    if res is not None:
        pose_T, _, detection = res
        x_pos   = np.linalg.inv(pose_T)[0, 3]
        z_dist  = pose_T[2, 3]
        bearing = math.atan2(pose_T[0, 3], pose_T[2, 3])
        beta    = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))

        # Base aim: yaw that points at where the robot center is, not the camera
        target_yaw = math.atan2(-X_OFFSET, z_dist)
        aim_px = yaw_to_pixel(target_yaw, K, D)
        if drawing_frame is not None:
            cv2.line(drawing_frame, (int(aim_px), 0), (int(aim_px), h), (255, 255, 0), 2)  # cyan
            cv2.putText(drawing_frame, f"aim {math.degrees(target_yaw):+.1f}deg",
                        (int(aim_px) + 5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        print(f"x={x_pos:+.3f} z={z_dist:.3f} b={math.degrees(bearing):+.1f}° β={math.degrees(beta):+.1f}° aim_yaw={math.degrees(target_yaw):+.1f}°")
    else:
        print("No detection")

# CAR SETUP
# car = DifferentialCar( left_wheel=sim.getObject('/Puzzlebot/DynamicLeftJoint'), right_wheel=sim.getObject('/Puzzlebot/DynamicRightJoint') )
# reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id=16, marker_size=0.1)
car = Puzzlebot()
reference = ArucoDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), marker_id=0, marker_size=0.034)

# reference = QRCodeDetector(qr_size=0.1, K=car.K, D=car.D)
# reference = HybridQRDetector(qr_size=0.1, K=car.K, D=car.D)
# reference = HybridQRDetector(qr_size=0.0334, K=car.K, D=car.D)

# SETUP
init_window('Camera', img_size=car.img_size, height=360)
_f = car.K[0, 0]
estimator = PoseTracker(
    estimator=PoseEstimator(reference=reference, K=car.K, D=car.D),
    filter=PoseFilter(alpha=0.05)
)

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

        # Send velocity command to car
        auto_cmd = {"x": 0.0, "w": 0.0}
        auto_cmd = follow(frame, drawing_frame=drawing_frame)  # get automatic command based on vision
        # follow(frame, drawing_frame=drawing_frame)  # draw only — output unused
        # test(frame, drawing_frame=drawing_frame)
        auto_cmd = {axis: auto_cmd[axis] * cmd_enables[axis] for axis in auto_cmd}  # apply enables
        man_cmd = get_diff_drive_input()  # get manual input
        cmd = merge_proportional(man_cmd, auto_cmd)  # combine manual and auto commands
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
