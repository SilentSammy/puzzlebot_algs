import math
import cv2
import numpy as np
try:
    from .tracking import cam_to_car
except ImportError:
    from tracking import cam_to_car


class MarkerServoing:
    """Visual-servoing controller that aligns the robot with a reference marker.

    Wraps the full perceive→control pipeline behind a single ``update(frame)``
    call and returns a normalized differential-drive command ``{'x', 'w'}`` (each
    in roughly [-1, 1]; multiply by the robot's nominal linear/angular speed to
    get real units).  All tuning constants are plain instance attributes, so they
    can be changed at runtime without rebuilding the object.

    The caller builds and owns the ``PoseTracker`` (estimator + filter) and passes
    it in, keeping this class agnostic to how poses are produced.

    Parameters
    ----------
    pose_tracker : PoseTracker
        Caller-built tracker exposing ``get_pose(frame, drawing_frame=None)`` ->
        ``(pose_T, pnp_result, detected)`` or None, plus a ``detection`` property.
    K, D : np.ndarray
        Camera intrinsics / distortion coefficients.
    x_offset : float
        Lateral offset of the camera from the robot centre (m).
    z_offset : float
        Forward offset of the camera from the robot centre (m).
    kp_x, x_clamp : float
        Linear P gain (cmd per metre of depth error) and its command clamp.
    kp_w, w_clamp : float
        Angular P gain (cmd per pixel of aim error) and its command clamp (rad).
    aim_clamp, aim_gain : float
        Max aim-offset fraction (0=centre, 1=edge) and the gain mapping lateral
        position (m) into that fraction.
    lin_auth_angle : float
        Gaze angle (rad) at which forward authority fades to zero.
    target_dist, reverse_dist : float
        Normal approach distance and back-off distance (m).
    target_beta : float
        Desired final orientation (rad); 0 = facing the marker.
    goal_radius, goal_hysteresis : float
        Half-side of the goal square (m) and the extra margin to stay in goal.
    use_undistorted : bool
        Do pixel arithmetic in undistorted image space (straight geometry, narrower
        FOV) when True, else in the raw distorted space.
    verbose : bool
        Print per-frame state when True.
    """

    def __init__(self, pose_tracker, K, D, *,
                 x_offset=-0.065,
                 z_offset=0.075,
                 kp_x=2.0, x_clamp=0.15,
                 kp_w=0.000350, w_clamp=math.radians(60),
                 aim_clamp=0.75, aim_gain=10.0,
                 lin_auth_angle=math.radians(20),
                 target_dist=0.30, reverse_dist=0.50,
                 target_beta=math.radians(-2),
                 goal_radius=0.005, goal_hysteresis=0.004,
                 use_undistorted=True, verbose=True):
        self._pose_tracker = pose_tracker
        self.K = K
        self.D = D
        self._f = float(K[0, 0])

        self.x_offset        = x_offset
        self.z_offset        = z_offset
        self.kp_x            = kp_x
        self.x_clamp         = x_clamp
        self.kp_w            = kp_w
        self.w_clamp         = w_clamp
        self.aim_clamp       = aim_clamp
        self.aim_gain        = aim_gain
        self.lin_auth_angle  = lin_auth_angle
        self.target_dist     = target_dist
        self.reverse_dist    = reverse_dist
        self.target_beta     = target_beta
        self.goal_radius     = goal_radius
        self.goal_hysteresis = goal_hysteresis
        self.use_undistorted = use_undistorted
        self.verbose         = verbose

        self._reverse_state = False
        self._goal_state    = False

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    @property
    def pose_tracker(self):
        """The caller-supplied PoseTracker (e.g. for ``.detection`` while drawing)."""
        return self._pose_tracker

    @property
    def reverse_state(self):
        return self._reverse_state

    @property
    def goal_state(self):
        return self._goal_state

    def reset(self):
        """Clear the goal/reverse state machine and the underlying tracker."""
        self._reverse_state = False
        self._goal_state    = False
        self._pose_tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _yaw_to_pixel(self, yaw_rad, D):
        """Pixel x-coordinate corresponding to a camera yaw angle (rad)."""
        ray = np.array([[[math.sin(yaw_rad), 0.0, math.cos(yaw_rad)]]], dtype=np.float64)
        px, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3),
                                  self.K.astype(np.float64), D.astype(np.float64))
        return float(px[0, 0, 0])

    def _to_undist_x(self, pts_xy):
        """Map distorted (N,2) pixel coords into undistorted pixel space."""
        p = pts_xy.reshape(-1, 1, 2).astype(np.float64)
        u = cv2.undistortPoints(p, self.K, self.D, P=self.K)
        return u.reshape(-1, 2)[:, 0]

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, frame, drawing_frame=None):
        """Process one frame and return a normalized ``{'x', 'w'}`` command."""
        result = self._pose_tracker.get_pose(frame, drawing_frame=drawing_frame)
        return self._control(result, frame, drawing_frame=drawing_frame)

    def _control(self, result, frame, drawing_frame=None):
        cmd = {'x': 0.0, 'w': 0.0}
        if result is None:
            return cmd

        h, w = frame.shape[:2]
        fused_T, pnp_result, detected = result

        # Marker x-coords in image: real detection when visible, reprojected when occluded.
        D_aim = np.zeros_like(self.D) if self.use_undistorted else self.D

        detection = self._pose_tracker.detection
        if detected and detection is not None:
            img_pts_x = (self._to_undist_x(detection.img_pts) if self.use_undistorted
                         else detection.img_pts[:, 0])
        else:
            pts = self._pose_tracker._estimator.reproject(fused_T, pnp_result, frame.shape)
            valid = pts[np.isfinite(pts).all(axis=1)]
            if len(valid) > 0:
                img_pts_x = self._to_undist_x(valid) if self.use_undistorted else valid[:, 0]
            else:
                img_pts_x = None

        def get_target_px(aim, base_yaw=0.0, aim_clamp=1.0):
            ref_px = self._yaw_to_pixel(base_yaw, D_aim) + (aim * (w / 2))
            ref_px = max(w / 2 - aim_clamp * w / 2, min(w / 2 + aim_clamp * w / 2, ref_px))
            if img_pts_x is None or len(img_pts_x) == 0:
                return ref_px, ref_px
            marker_left_px  = img_pts_x.min()
            marker_right_px = img_pts_x.max()
            marker_target_px = (marker_left_px + marker_right_px) / 2 \
                + aim * (marker_right_px - marker_left_px) / 2
            if drawing_frame is not None:
                if math.isfinite(ref_px):
                    ref_x = int(float(ref_px))
                    cv2.line(drawing_frame, (ref_x, 0), (ref_x, h), (255, 0, 0), 2)  # blue: reference
                if math.isfinite(marker_target_px):
                    marker_x = int(float(marker_target_px))
                    cv2.line(drawing_frame, (marker_x, 0), (marker_x, h), (0, 255, 0), 2)  # green: marker
            return ref_px, marker_target_px

        # Transform marker pose from the camera frame into the robot/car frame.
        car_T   = cam_to_car(fused_T, x_off=-self.x_offset, z_off=self.z_offset)
        x_pos   = np.linalg.inv(car_T)[0, 3]
        z_dist  = car_T[2, 3]
        bearing = math.atan2(car_T[0, 3], car_T[2, 3])
        beta    = math.atan2(-car_T[2, 0], math.hypot(car_T[0, 0], car_T[1, 0]))

        # Base aim yaw: compensates for camera being offset from robot centre.
        target_yaw = math.atan2(-self.x_offset, z_dist)
        if drawing_frame is not None:
            aim_px = self._yaw_to_pixel(target_yaw, D_aim)
            if math.isfinite(aim_px):
                cv2.line(drawing_frame, (int(aim_px), 0), (int(aim_px), h), (255, 255, 0), 2)  # cyan

        # At goal: align to target_beta in place, skip tracking computations.
        goal_threshold = self.goal_radius + (self.goal_hysteresis if self._goal_state else 0.0)
        if (not self._reverse_state
                and abs(z_dist - self.target_dist) < goal_threshold
                and abs(x_pos) < goal_threshold):
            self._goal_state = True
            beta_error = beta - self.target_beta
            w_cmd = -max(-self.w_clamp, min(self.w_clamp, self.kp_w * math.tan(beta_error) * self._f))
            if self.verbose:
                print(f"x={x_pos:+.3f} z={z_dist:.3f} b={math.degrees(bearing):+.1f}° "
                      f"β={math.degrees(beta):+.1f}° β_err={math.degrees(beta_error):+.1f}° "
                      f"w={w_cmd:+.3f}  GOAL")
            return {'x': 0.0, 'w': w_cmd}
        self._goal_state = False

        # Aim shifts left/right based on lateral position, scaled for sensitivity.
        aim = max(-self.aim_clamp,
                  min(self.aim_clamp, (-1 if self._reverse_state else 1) * x_pos * self.aim_gain))

        # Proportional control for angular velocity.
        ref_px, marker_target_px = get_target_px(aim, base_yaw=target_yaw, aim_clamp=self.aim_clamp)
        error_px = ref_px - marker_target_px
        w_cmd = max(-self.w_clamp, min(self.w_clamp, self.kp_w * error_px))

        # Linear authority: full drive when aligned, fades to 0 at lin_auth_angle.
        gaze_angle = math.atan2(abs(error_px), self._f)
        lin_auth   = max(0.0, 1.0 - gaze_angle / self.lin_auth_angle)

        dist_target = self.target_dist if not self._reverse_state else self.reverse_dist
        x_cmd = lin_auth * self.kp_x * (z_dist - dist_target)
        x_cmd = max(-self.x_clamp, min(self.x_clamp, x_cmd))

        z_reached   = abs(z_dist - dist_target) < self.goal_radius
        x_reached   = abs(x_pos) < self.goal_radius
        at_distance = z_reached and not x_reached

        if self._reverse_state and (x_reached or z_reached):
            self._reverse_state = False
        elif at_distance:
            self._reverse_state = True

        if self.verbose:
            status = 'REVERSING' if self._reverse_state else ('AT DISTANCE' if at_distance else '')
            print(f"x={x_pos:+.3f} z={z_dist:.3f} b={math.degrees(bearing):+.1f}° "
                  f"β={math.degrees(beta):+.1f}° aim={aim:+.2f} err={error_px:+.0f}px "
                  f"w={w_cmd:+.3f} auth={lin_auth:.2f}  {status}")

        return {'x': x_cmd, 'w': w_cmd}
