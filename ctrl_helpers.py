import cv2
import math
import time
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


def cam_to_car(cam_T, x_off=0.00, z_off=0.075, return_residual=False):
    """Transform cam_T into car_T via a chain of axis-flattening rotations.

    The transform discards three quantities (pitch tilt, roll tilt, and the
    camera's marker-frame height). Pass return_residual=True to also receive
    these as a (dp, dr, ty) tuple; feed it to car_to_cam for an exact inverse."""
    T = cam_T

    # Rotate about x to flatten pitch to 180
    dp = math.radians(180) + math.atan2(T[2, 1], T[2, 2])
    cp, sp = math.cos(dp), math.sin(dp)
    Rx = np.array([[1, 0, 0, 0],
                   [0, cp, -sp, 0],
                   [0, sp, cp, 0],
                   [0, 0, 0, 1]], dtype=np.float64)
    T = Rx @ T

    # Rotate about z to flatten roll to 180
    dr = math.radians(180) - math.atan2(T[1, 0], T[0, 0])
    cr, sr = math.cos(dr), math.sin(dr)
    Rz = np.array([[cr, -sr, 0, 0],
                   [sr, cr, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float64)
    T = Rz @ T

    # Translate in y by the current inverse-frame y offset, plus x_off and z_off
    ty = np.linalg.inv(T)[1, 3]
    Tr = np.array([[1, 0, 0, x_off],
                   [0, 1, 0, ty],
                   [0, 0, 1, z_off],
                   [0, 0, 0, 1]], dtype=np.float64)
    T = Tr @ T

    if return_residual:
        return T, (dp, dr, ty)
    return T


def car_to_cam(car_T, x_off=0.00, z_off=0.075, residual=None):
    """Inverse of cam_to_car: map a car-frame pose back to the camera frame.

    cam_to_car applies car_T = Tr @ Rz @ Rx @ cam_T, where Rx flattens pitch
    (by dp), Rz flattens roll (by dr), and Tr translates by (x_off, ty, z_off).
    The flatten amounts (dp, dr) and the y-translation (ty) are the only
    information lost by the forward transform.

    Pass residual=(dp, dr, ty) (from cam_to_car(..., return_residual=True)) to
    undo all three steps and exactly recover the original cam_T.

    Without a residual, only the fixed (x_off, z_off) camera offset can be
    undone (ty defaults to 0, pitch/roll stay flat); the result is a
    camera-frame pose lying in the marker plane, which is exact only when the
    original camera already faced the marker."""
    if residual is not None:
        dp, dr, ty = residual
    else:
        dp, dr, ty = 0.0, 0.0, 0.0

    # Undo the translation Tr
    Tr = np.array([[1, 0, 0, x_off],
                   [0, 1, 0, ty],
                   [0, 0, 1, z_off],
                   [0, 0, 0, 1]], dtype=np.float64)
    T = np.linalg.inv(Tr) @ car_T

    # Undo the z rotation Rz (rotate by -dr)
    cr, sr = math.cos(-dr), math.sin(-dr)
    Rz = np.array([[cr, -sr, 0, 0],
                   [sr, cr, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float64)
    T = Rz @ T

    # Undo the x rotation Rx (rotate by -dp)
    cp, sp = math.cos(-dp), math.sin(-dp)
    Rx = np.array([[1, 0, 0, 0],
                   [0, cp, -sp, 0],
                   [0, sp, cp, 0],
                   [0, 0, 0, 1]], dtype=np.float64)
    T = Rx @ T

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
    """Time-constant EMA low-pass filter for PoseEstimator.get_pose() output.

    Smoothing is frame-rate independent: each update blends with
    ``alpha_eff = 1 - exp(-dt/tau)``, where ``dt`` is the measured wall-clock
    interval since the last accepted sample.  This keeps the response identical
    whether the detector runs fast (ArUco) or slow (QR), unlike a fixed
    per-frame alpha.

    Converting from a legacy fixed alpha at rate f:  tau = -dt / ln(1 - alpha).
    e.g. alpha=0.05 at 25 Hz (dt=0.04 s) -> tau = -0.04/ln(0.95) ~= 0.78 s.

    Because ``dt`` grows during missed/occluded frames, the effective alpha
    automatically rises toward 1 on resume, snapping to the fresh measurement
    (this replaces the old per-miss "miss_snap" mechanism).

    Rotation is stored as a unit quaternion so that sign-continuity correction
    (flipping q to -q when the new sample lands in the opposite hemisphere) is
    always valid — q and -q represent the same rotation, unlike Rodrigues
    vectors where -rvec is the inverse rotation.
    """

    def __init__(self, tau=0.78, max_jump=0.25):
        """
        tau: EMA time constant in seconds (0=no smoothing, larger=smoother).
             tau~=0.78 matches the legacy alpha=0.05 @ 25 Hz.
        max_jump: Max tvec displacement (metres) allowed between samples.
                  Samples exceeding this are discarded. None to disable.
        """
        self.tau = tau
        self.max_jump = max_jump
        self._quat: np.ndarray | None = None  # unit quaternion [x, y, z, w]
        self._tvec: np.ndarray | None = None
        self._reject_count = 0
        self._last_time: float | None = None  # perf_counter() of last accepted sample

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
            return None
        now = time.perf_counter()
        pose_T, res, detection = result
        rvec, tvec = matrix_to_vecs(pose_T)
        if self._tvec is None:
            self._quat = self._rvec_to_quat(rvec)
            self._tvec = tvec.copy()
            self._last_time = now
        else:
            if self.max_jump is not None and np.linalg.norm(tvec - self._tvec) > self.max_jump:
                self._reject_count += 1
                if self._reject_count >= 5:  # filter too far behind — snap to current measurement
                    self._quat = self._rvec_to_quat(rvec)
                    self._tvec = tvec.copy()
                    self._reject_count = 0
                    self._last_time = now
                # leave _last_time untouched while rejecting so dt (and alpha_eff)
                # keep growing, giving a fast catch-up once a valid sample lands
                return vecs_to_matrix(self._quat_to_rvec(self._quat), self._tvec), res, detection
            self._reject_count = 0
            dt = (now - self._last_time) if self._last_time is not None else 0.0
            alpha_eff = (1.0 - math.exp(-dt / self.tau)) if self.tau > 0.0 else 1.0
            q = self._rvec_to_quat(rvec)
            if np.dot(q, self._quat) < 0:  # shortest-arc: q and -q are the same rotation
                q = -q
            self._quat = alpha_eff * q + (1.0 - alpha_eff) * self._quat
            self._quat /= np.linalg.norm(self._quat)  # renormalize after blend
            self._tvec = alpha_eff * tvec + (1.0 - alpha_eff) * self._tvec
            self._last_time = now
        return vecs_to_matrix(self._quat_to_rvec(self._quat), self._tvec), res, detection

    def reset(self):
        self._quat = None
        self._tvec = None
        self._reject_count = 0
        self._last_time = None


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
    """PoseTracker extended with odometry fusion at the centre of rotation.

    The camera is mounted a fixed distance ahead of (and possibly beside) the
    robot's centre of rotation, so the camera and the wheel odometry do not move
    1:1 — a pure spin-in-place swings the camera through an arc while odometry
    reports zero translation.  To fuse them consistently this tracker moves the
    *camera* measurement back to the centre of rotation (via ``cam_to_car``)
    before comparing it with odometry, which already lives at the centre.

    When the marker is visible the centred camera pose drives the estimate and
    the scalar difference (cam − odom) is frozen as a registration offset.  When
    the marker is occluded, live odometry + frozen offset provides a fallback
    pose.  The fused centre pose is finally mapped back to the camera frame (via
    ``car_to_cam`` with the stored residual) so the returned matrix matches the
    plain-``PoseTracker`` convention — callers still receive a camera-frame pose
    and convert to car coordinates themselves.

    Usage::

        tracker = FusedPoseTracker(PoseEstimator(...), PoseFilter(...),
                                   odom_fn=lambda: car.estimated_pose)

        # Each tick:
        result = tracker.get_pose(frame_or_None, drawing_frame=drawing_frame)
        if result is not None:
            fused_T, pnp_result, detected = result   # fused_T is camera-frame
            pts = tracker._estimator.reproject(fused_T, pnp_result, img_shape)

    Parameters
    ----------
    estimator : PoseEstimator
    filter    : PoseFilter
    odom_fn   : callable () → (x, y, theta) or None — wheel odometry source
    cam_x_off : lateral camera offset from the centre of rotation (m)
    cam_z_off : forward camera offset from the centre of rotation (m)

    Returns None until the first successful marker detection.
    """

    def __init__(self, estimator, filter, odom_fn=None, cam_x_off=-0.055, cam_z_off=0.065):
        self._estimator       = estimator
        self._tracker         = PoseTracker(estimator, filter)
        self._odom_fn         = odom_fn   # callable () → (x, y, theta) or None
        self._cam_x_off       = cam_x_off  # lateral camera offset from rotation centre (m)
        self._cam_z_off       = cam_z_off  # forward camera offset from rotation centre (m)
        self._last_cam_T      = None
        self._last_car_T      = None   # last detection, centred (lever arm removed)
        self._last_residual   = None   # (dp, dr, ty) mount residual from cam_to_car
        self._last_pnp_result = None
        self._last_detection  = None
        self._cam_pose        = None   # (x_pos, z_dist, beta) centred camera pose
        self._odom_pose       = None   # (x_pos, z_dist, beta) latest odometry at centre
        self._offset_pose     = None   # (dx, dz, dbeta) = cam_pose - odom_pose
        self._fused_pose      = None   # (x_pos, z_dist, beta) last computed fused pose
        self._odom_theta_last       = None  # last raw theta for unwrapping
        self._odom_theta_unwrapped  = 0.0   # accumulated unwrapped angle

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------

    @property
    def cam_pose(self):
        """(x_pos, z_dist, beta) of the most recent camera detection, moved to the
        centre of rotation (lever arm removed), or None."""
        return self._cam_pose

    @property
    def fused_pose(self):
        """(x_pos, z_dist, beta) of the most recently computed fused estimate, or None."""
        return self._fused_pose

    @property
    def odom_pose(self):
        """Latest odometry as a centred (x_pos, z_dist, beta) pose, or None."""
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
    def _odom_to_center_pose(odom):
        """Convert (x, y, theta) car-frame odometry to a centred (x_pos, z_dist, beta)
        pose, in the same convention as decompose_pose(cam_to_car(cam_T)).
        Odometry already reports the robot's centre of rotation, so there is no
        lever arm to remove here — this only projects the world position onto the
        robot's local axes so z_dist tracks depth consistently regardless of heading.
        car forward = (cos θ, sin θ) in world → z_dist (negated: moving forward decreases depth)
        car left    = (-sin θ, cos θ) in world → x_pos"""
        ox, oy, theta = odom
        c, s = math.cos(theta), math.sin(theta)
        z_dist = -(ox * c + oy * s)   # depth: projection onto forward axis, negated
        x_pos  =  ox * s - oy * c     # lateral: projection onto left axis
        return (x_pos, z_dist, theta)

    def update(self, frame, odom, drawing_frame=None):
        if odom is not None:
            ox, oy, theta = odom
            if self._odom_theta_last is not None:
                delta = (theta - self._odom_theta_last + math.pi) % (2 * math.pi) - math.pi
                self._odom_theta_unwrapped += delta
            else:
                self._odom_theta_unwrapped = theta
            self._odom_theta_last = theta
            odom_center = self._odom_to_center_pose((ox, oy, self._odom_theta_unwrapped))
        else:
            odom_center = None
        self._odom_pose = odom_center
        # --- Camera (PnP), moved to the centre of rotation ---
        cam_pose = None
        detected = False
        if frame is not None:
            res = self._tracker.get_pose(frame, drawing_frame=drawing_frame)
            if res is not None:
                cam_T, pnp_result, detection = res
                # Remove the camera's lever arm so the measurement lives at the
                # centre of rotation, matching the odometry's reference point.
                car_T, residual = cam_to_car(cam_T, x_off=self._cam_x_off,
                                             z_off=self._cam_z_off, return_residual=True)
                cam_pose = decompose_pose(car_T)
                self._last_cam_T      = cam_T
                self._last_car_T      = car_T
                self._last_residual   = residual
                self._last_pnp_result = pnp_result
                self._last_detection  = detection
                self._cam_pose        = cam_pose
                detected = True

        # --- Freeze offset when both sources are available ---
        if cam_pose is not None and odom_center is not None:
            self._offset_pose = (
                cam_pose[0] - odom_center[0],
                cam_pose[1] - odom_center[1],
                cam_pose[2] - odom_center[2],
            )

        # --- Fused scalar pose (at the centre of rotation) ---
        if self._offset_pose is not None and odom_center is not None:
            self._fused_pose = (
                odom_center[0] + self._offset_pose[0],
                odom_center[1] + self._offset_pose[1],
                odom_center[2] + self._offset_pose[2],
            )
        else:
            self._fused_pose = None

        # --- Build fused 4×4 matrix, mapped back to the camera frame ---
        if self._last_cam_T is None or self._last_pnp_result is None:
            return None

        if detected:
            # Marker visible this frame: the (filtered) camera pose is ground
            # truth, so return it directly — no planar round-trip, no loss.
            fused_T = self._last_cam_T
        elif self._fused_pose is not None:
            # Occluded: dead-reckon the centred pose with odometry + frozen
            # offset, then map back to the camera frame via the stored residual.
            # The planar (x, z, beta) terms track odometry exactly; only the
            # frozen out-of-plane orientation is approximated during the gap.
            fused_car_T = update_pose(self._last_car_T, *self._fused_pose)
            fused_T = car_to_cam(fused_car_T, x_off=self._cam_x_off,
                                 z_off=self._cam_z_off, residual=self._last_residual)
        else:
            fused_T = self._last_cam_T

        return fused_T, self._last_pnp_result, detected

    # ------------------------------------------------------------------

    def reset(self):
        self._tracker.reset()
        self._last_cam_T      = None
        self._last_car_T      = None
        self._last_residual   = None
        self._last_pnp_result = None
        self._last_detection  = None
        self._cam_pose        = None
        self._odom_pose       = None
        self._offset_pose     = None
        self._fused_pose      = None
        self._odom_theta_last       = None
        self._odom_theta_unwrapped  = 0.0

