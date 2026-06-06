import cv2
import math
import numpy as np
import user_input as inp
from marker_est import matrix_to_vecs, vecs_to_matrix


def decompose_pose(pose_T):
    """Extract (x_pos, z_dist, beta) from a 4x4 camera-space pose matrix.
    beta is in (-90°, +90°); pair with update_pose for a lossless round-trip."""
    x_pos = np.linalg.inv(pose_T)[0, 3]
    z_dist = pose_T[2, 3]
    beta   = math.atan2(-pose_T[2, 0], math.hypot(pose_T[0, 0], pose_T[1, 0]))
    return x_pos, z_dist, beta


def update_pose(base_T, x_pos, z_dist, beta):
    """Return a new 4x4 matrix with updated planar pose (x_pos, z_dist, beta),
    keeping the y-row (pitch, roll, height) from base_T unchanged.
    Corrects the cos-sign ambiguity introduced by decompose_pose using hypot."""
    T = base_T.copy()
    c, s = math.cos(beta), math.sin(beta)
    # decompose_pose uses hypot (always positive), so beta is in (-90°, +90°).
    # When the true rotation has cos < 0, flip c to match base_T's sign.
    if base_T[0, 0] * c < 0:
        c = -c
    T[0, 0], T[0, 2] =  c,  s
    T[2, 0], T[2, 2] = -s,  c
    T[2, 3] = z_dist
    # Solve tx: x_pos = inv(T)[0,3] = -(c*tx + T[1,0]*T[1,3] - s*z_dist)
    T[0, 3] = (-x_pos - T[1, 0] * T[1, 3] + s * z_dist) / c if abs(c) > 1e-6 else 0.0
    return T


def init_window(name, width=640, height=360, img_size=None):
    if img_size is not None:
        width = img_size[0] * height // img_size[1]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, width, height)


def get_diff_drive_input(slow=0.25, fast=0.5):
    boost = inp.get_bipolar_ctrl(high_key='c', high_game='RT0')
    scale = slow + (fast - slow) * boost
    x =  inp.get_bipolar_ctrl('w', 's', 'LY0') * scale
    w = -inp.get_bipolar_ctrl('d', 'a', 'RX0') * scale
    return {'x': x, 'w': w}


def merge_proportional(cmd_primary, cmd_secondary):
    cmd_final = {}
    all_axes = set(cmd_primary.keys()) | set(cmd_secondary.keys())
    for axis in all_axes:
        primary_input   = cmd_primary.get(axis, 0.0)
        secondary_input = cmd_secondary.get(axis, 0.0)
        if abs(primary_input) < 0.05:
            cmd_final[axis] = secondary_input
        else:
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[axis] = (1 - override_strength) * secondary_input + override_strength * desired_value
    return cmd_final


def get_manual_override(cmd):
    return merge_proportional(get_diff_drive_input(), cmd)


class PoseFilter:
    """EMA low-pass filter for PoseEstimator.get_pose() output.

    Rotation is stored as a unit quaternion so that sign-continuity correction
    (flipping q to -q when the new sample lands in the opposite hemisphere) is
    always valid — q and -q represent the same rotation, unlike Rodrigues
    vectors where -rvec is the inverse rotation.
    """

    def __init__(self, alpha=0.3, max_jump=0.25, miss_snap=0.15):
        """
        alpha: EMA weight on newest sample (0=frozen, 1=no smoothing).
        max_jump: Max tvec displacement (metres) allowed between samples.
                  Samples exceeding this are discarded. None to disable.
        miss_snap: Extra alpha added per consecutive missed frame on resume.
                   e.g. after 5 misses: alpha_eff = min(1.0, alpha + 5*miss_snap).
        """
        self.alpha = alpha
        self.max_jump = max_jump
        self.miss_snap = miss_snap
        self._quat: np.ndarray | None = None  # unit quaternion [x, y, z, w]
        self._tvec: np.ndarray | None = None
        self._reject_count = 0
        self._miss_count = 0

    @staticmethod
    def _rvec_to_quat(rvec):
        angle = np.linalg.norm(rvec)
        if angle < 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0])
        half = angle * 0.5
        axis = rvec / angle
        return np.array([*(axis * np.sin(half)), np.cos(half)])

    @staticmethod
    def _quat_to_rvec(q):
        q = q / np.linalg.norm(q)
        if q[3] < 0:      # canonical form: w >= 0
            q = -q
        w = float(np.clip(q[3], -1.0, 1.0))
        angle = 2.0 * np.arccos(w)
        if angle < 1e-10:
            return np.zeros(3)
        return (q[:3] / np.sin(angle * 0.5)) * angle

    def update(self, result):
        if result is None:
            self._miss_count += 1
            return None
        alpha_eff = min(1.0, self.alpha + self._miss_count * self.miss_snap)
        self._miss_count = 0
        pose_T, res, detection = result
        rvec, tvec = matrix_to_vecs(pose_T)
        if self._tvec is None:
            self._quat = self._rvec_to_quat(rvec)
            self._tvec = tvec.copy()
        else:
            if self.max_jump is not None and np.linalg.norm(tvec - self._tvec) > self.max_jump:
                self._reject_count += 1
                if self._reject_count >= 5:  # filter too far behind — snap to current measurement
                    self._quat = self._rvec_to_quat(rvec)
                    self._tvec = tvec.copy()
                    self._reject_count = 0
                return vecs_to_matrix(self._quat_to_rvec(self._quat), self._tvec), res, detection
            self._reject_count = 0
            q = self._rvec_to_quat(rvec)
            if np.dot(q, self._quat) < 0:  # shortest-arc: q and -q are the same rotation
                q = -q
            self._quat = alpha_eff * q + (1.0 - alpha_eff) * self._quat
            self._quat /= np.linalg.norm(self._quat)  # renormalize after blend
            self._tvec = alpha_eff * tvec + (1.0 - alpha_eff) * self._tvec
        return vecs_to_matrix(self._quat_to_rvec(self._quat), self._tvec), res, detection

    def reset(self):
        self._quat = None
        self._tvec = None
        self._reject_count = 0
        self._miss_count = 0


class PoseTracker:
    """Stateful wrapper combining a PoseEstimator and a PoseFilter.

    Duck-type compatible with PoseEstimator: call get_pose(frame, drawing_frame)
    and receive a filtered (pose_T, pnp_result, detection) tuple or None.
    """

    def __init__(self, estimator, filter):
        self._estimator = estimator
        self._filter = filter
        self._last_detection = None

    def get_pose(self, frame, drawing_frame=None):
        result = self._filter.update(self._estimator.get_pose(frame, drawing_frame=drawing_frame))
        if result is not None:
            self._last_detection = result[2]
        return result

    @property
    def detection(self):
        """The detection object from the most recent marker detection, or None."""
        return self._last_detection

    def reset(self):
        self._filter.reset()
        self._last_detection = None


class FusedPoseTracker:
    """PoseTracker extended with 2D scalar-offset odometry fusion.

    When the marker is visible, the camera pose drives the estimate and the
    scalar difference (cam − odom) is frozen as an offset.  When the marker
    is occluded, live odometry + frozen offset provides a fallback pose.

    Usage::

        tracker = FusedPoseTracker(PoseEstimator(...), PoseFilter(...))

        # Each tick:
        result = tracker.update(frame_or_None, odom_cam, drawing_frame=drawing_frame)
        if result is not None:
            fused_T, pnp_result, detected = result
            pts = tracker._estimator.reproject(fused_T, pnp_result, img_shape)

    Parameters
    ----------
    frame      : BGR ndarray or None (skip PnP when no image this tick)
    odom_cam   : (cam_x, cam_z, beta) odometry already in camera space, or None
    detected   : True when the marker was visible in this specific frame

    Returns None until the first successful marker detection.
    """

    def __init__(self, estimator, filter, odom_fn=None):
        self._estimator       = estimator
        self._tracker         = PoseTracker(estimator, filter)
        self._odom_fn         = odom_fn   # callable () → (x, y, theta) or None
        self._last_cam_T      = None
        self._last_pnp_result = None
        self._last_detection  = None
        self._cam_pose        = None   # (x_pos, z_dist, beta) from last detection
        self._odom_pose       = None   # current fused prediction (for diagnostics)
        self._marker_world    = None   # (mx, my) estimated marker position in world frame
        self._last_theta_det  = None   # unwrapped theta when marker was last detected
        self._last_beta_det   = None   # beta when marker was last detected
        self._fused_pose      = None   # (x_pos, z_dist, beta) last computed fused pose
        self._odom_theta_last       = None  # last raw theta for unwrapping
        self._odom_theta_unwrapped  = 0.0   # accumulated unwrapped angle

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------

    @property
    def cam_pose(self):
        """(x_pos, z_dist, beta) from the most recent camera detection, or None."""
        return self._cam_pose

    @property
    def fused_pose(self):
        """(x_pos, z_dist, beta) of the most recently computed fused estimate, or None."""
        return self._fused_pose

    @property
    def odom_pose(self):
        """Latest odometry converted to camera space (x_pos, z_dist, beta), or None."""
        return self._odom_pose

    @property
    def detection(self):
        """The detection object from the most recent marker detection, or None."""
        return self._last_detection

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def get_pose(self, frame, drawing_frame=None):
        """Duck-type compatible with PoseTracker. Uses the odom_fn configured at init."""
        if self._odom_fn is None:
            raise RuntimeError("FusedPoseTracker.get_pose() requires odom_fn to be set at init.")
        return self.update(frame, self._odom_fn(), drawing_frame=drawing_frame)

    @staticmethod
    def _car_to_cam_pose(odom):
        """Convert (x, y, theta) car-frame odometry to (cam_x, cam_z, beta) camera-frame.
        Projects world position onto the robot's local axes so cam_z tracks depth
        consistently regardless of heading.
        car forward = (cos θ, sin θ) in world → cam_z (negated: moving forward decreases depth)
        car left    = (-sin θ, cos θ) in world → cam_x"""
        ox, oy, theta = odom
        c, s = math.cos(theta), math.sin(theta)
        cam_z = -(ox * c + oy * s)   # depth: projection onto forward axis, negated
        cam_x =  ox * s - oy * c     # lateral: projection onto left axis
        return (cam_x, cam_z, theta)

    def update(self, frame, odom, drawing_frame=None):
        # --- Unwrap theta ---
        if odom is not None:
            ox, oy, theta = odom
            if self._odom_theta_last is not None:
                delta = (theta - self._odom_theta_last + math.pi) % (2 * math.pi) - math.pi
                self._odom_theta_unwrapped += delta
            else:
                self._odom_theta_unwrapped = theta
            self._odom_theta_last = theta
            theta_u = self._odom_theta_unwrapped
        else:
            ox = oy = None
            theta_u = self._odom_theta_unwrapped

        # --- Camera (PnP) ---
        cam_pose = None
        detected = False
        if frame is not None:
            res = self._tracker.get_pose(frame, drawing_frame=drawing_frame)
            if res is not None:
                cam_T, pnp_result, detection = res
                cam_pose = decompose_pose(cam_T)
                self._last_cam_T      = cam_T
                self._last_pnp_result = pnp_result
                self._last_detection  = detection
                self._cam_pose        = cam_pose
                detected = True

        # --- Store marker in world frame when both sources available ---
        if cam_pose is not None and odom is not None:
            cam_x, cam_z, beta = cam_pose
            c, s = math.cos(theta_u), math.sin(theta_u)
            self._marker_world   = (ox + cam_z * c - cam_x * s,
                                    oy + cam_z * s + cam_x * c)
            self._last_theta_det = theta_u
            self._last_beta_det  = beta

        # --- Reproject marker world pos into current robot frame ---
        if self._marker_world is not None and odom is not None:
            mx, my = self._marker_world
            c, s = math.cos(theta_u), math.sin(theta_u)
            dx, dy = mx - ox, my - oy
            fused_z    =  dx * c + dy * s
            fused_x    = -dx * s + dy * c
            fused_beta = self._last_beta_det + (self._last_theta_det - theta_u)
            self._fused_pose = (fused_x, fused_z, fused_beta)
        else:
            self._fused_pose = None

        self._odom_pose = self._fused_pose

        # --- Build fused 4×4 matrix ---
        if self._last_cam_T is None or self._last_pnp_result is None:
            return None

        if self._fused_pose is not None:
            fused_T = update_pose(self._last_cam_T, *self._fused_pose)
        else:
            fused_T = self._last_cam_T

        return fused_T, self._last_pnp_result, detected

    # ------------------------------------------------------------------

    def reset(self):
        self._tracker.reset()
        self._last_cam_T      = None
        self._last_pnp_result = None
        self._last_detection  = None
        self._cam_pose        = None
        self._odom_pose       = None
        self._marker_world    = None
        self._last_theta_det  = None
        self._last_beta_det   = None
        self._fused_pose      = None
        self._odom_theta_last       = None
        self._odom_theta_unwrapped  = 0.0


