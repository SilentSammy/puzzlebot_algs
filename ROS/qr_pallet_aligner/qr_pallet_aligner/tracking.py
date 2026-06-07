import math
import time
import numpy as np
import cv2
try:
    from .marker_det import Detection
    from .marker_est import matrix_to_vecs, vecs_to_matrix
except ImportError:
    from marker_det import Detection
    from marker_est import matrix_to_vecs, vecs_to_matrix


def cam_to_car(cam_T, x_off=0.00, z_off=0.075, return_residual=False):
    """Transform cam_T into car_T via a chain of axis-flattening rotations.

    The transform discards three quantities (pitch tilt, roll tilt, and the
    camera's marker-frame height). Pass return_residual=True to also receive
    these as a (dp, dr, ty) tuple."""
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


class PoseFilter:
    """Time-constant EMA low-pass filter for PoseEstimator.get_pose() output.

    Smoothing is frame-rate independent: each update blends with
    ``alpha_eff = 1 - exp(-dt/tau)``, where ``dt`` is the measured wall-clock
    interval since the last accepted sample.

    Converting from a legacy fixed alpha at rate f:  tau = -dt / ln(1 - alpha).
    e.g. alpha=0.05 at 25 Hz (dt=0.04 s) -> tau = -0.04/ln(0.95) ~= 0.78 s.

    Rotation is stored as a unit quaternion so that sign-continuity correction
    (flipping q to -q when the new sample lands in the opposite hemisphere) is
    always valid.
    """

    def __init__(self, tau=0.78, max_jump=0.25):
        """
        tau:      EMA time constant in seconds (0 = no smoothing).
        max_jump: Max tvec displacement (metres) allowed between samples.
                  Samples exceeding this are discarded. None to disable.
        """
        self.tau = tau
        self.max_jump = max_jump
        self._quat: np.ndarray | None = None  # unit quaternion [x, y, z, w]
        self._tvec: np.ndarray | None = None
        self._reject_count = 0
        self._last_time: float | None = None

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
                if self._reject_count >= 5:  # filter too far behind â€” snap to current measurement
                    self._quat = self._rvec_to_quat(rvec)
                    self._tvec = tvec.copy()
                    self._reject_count = 0
                    self._last_time = now
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


class CornerTracker:
    """Recovers 4 marker corners via Lucas-Kanade optical flow while detection
    is lost.

    On each successful detection, seeds feature points inside the marker quad.
    While lost, tracks those features frame-to-frame with LK optical flow, fits
    a homography from the seed positions to the current positions, and warps the
    last known corners through it.  A suite of sanity checks (convexity, area
    ratio, pixel texture) guards against lock-on to the wrong surface.
    """

    _LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Minimum pixel std-dev inside the quad warped to a 64Ã—64 patch.
    # Marker texture is always â‰¥ 24; drifted-onto-background is â‰¤ 23.
    _PIX_STD_MIN = 23.0

    def __init__(self, min_points=8, fb_threshold=1.0):
        self.min_points   = min_points
        self.fb_threshold = fb_threshold  # max forward-backward flow error (px)
        self.prev_gray      = None
        self.anchor_pts     = None   # (N,1,2) feature positions at detection time
        self.track_pts      = None   # (N,1,2) current feature positions
        self.anchor_corners = None   # (4,2) corners at detection time
        self.anchor_area    = None   # area of anchor quad (pxÂ²)
        self.last_valid     = None   # (4,2) last corners that passed _valid_quad

    def reset(self, gray, corners):
        """Seed features from a fresh detection."""
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                      minDistance=5, mask=mask)
        self.prev_gray      = gray
        self.anchor_corners = corners.astype(np.float32).copy()
        self.anchor_area    = self._quad_area(self.anchor_corners)
        self.last_valid     = corners.astype(np.float32).copy()
        if pts is None or len(pts) < self.min_points:
            self.anchor_pts = None
            self.track_pts  = None
        else:
            self.anchor_pts = pts.astype(np.float32)
            self.track_pts  = pts.astype(np.float32).copy()

    @staticmethod
    def _quad_area(corners):
        return abs(cv2.contourArea(corners.astype(np.float32)))

    @staticmethod
    def _quad_pixel_std(gray, corners):
        """Warp the quad into a 64Ã—64 patch and return its pixel std-dev."""
        side = 64
        dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        patch = cv2.warpPerspective(gray, M, (side, side))
        return float(patch.std())

    def _valid_quad(self, corners, gray):
        """Reject warped corners relative to the previous accepted frame."""
        pts = corners.astype(np.float32)
        if not np.all(np.isfinite(pts)):
            return False
        if not cv2.isContourConvex(pts):
            return False
        for i in range(4):
            a = pts[(i - 1) % 4]; b = pts[i]; c = pts[(i + 1) % 4]
            v1 = a - b; v2 = c - b
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-3 or n2 < 1e-3:
                return False
            cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            if np.degrees(np.arccos(cos_ang)) < 30.0:
                return False
        if self.last_valid is not None:
            area_cur  = self._quad_area(pts)
            area_prev = self._quad_area(self.last_valid)
            if area_prev > 1.0:
                ratio = area_cur / area_prev
                if ratio < 0.78 or ratio > 1.30:
                    return False
        if self.anchor_area and self.anchor_area > 1.0:
            ratio = self._quad_area(pts) / self.anchor_area
            if ratio < 0.20 or ratio > 5.0:
                return False
        if self._quad_pixel_std(gray, pts) < self._PIX_STD_MIN:
            return False
        return True

    def track(self, gray):
        """Propagate features into the current frame. Returns (4,2) corners or None."""
        if self.prev_gray is None or self.track_pts is None:
            self.prev_gray = gray
            return None

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.track_pts, None, **self._LK_PARAMS)
        if new_pts is None:
            self.prev_gray = gray
            self.track_pts = None
            return None

        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, new_pts, None, **self._LK_PARAMS)
        self.prev_gray = gray

        if back_pts is None:
            self.track_pts = None
            return None

        fb_err = np.linalg.norm(
            self.track_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
        good = ((status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
                & (fb_err < self.fb_threshold))

        self.anchor_pts = self.anchor_pts[good]
        self.track_pts  = new_pts[good]

        if len(self.track_pts) < self.min_points:
            self.track_pts = None
            return None

        H, inliers = cv2.findHomography(self.anchor_pts, self.track_pts, cv2.RANSAC, 3.0)
        if H is None:
            return None

        if inliers is not None:
            keep = inliers.reshape(-1) == 1
            if keep.sum() >= self.min_points:
                self.anchor_pts = self.anchor_pts[keep]
                self.track_pts  = self.track_pts[keep]

        warped  = cv2.perspectiveTransform(self.anchor_corners.reshape(-1, 1, 2), H)
        corners = warped.reshape(-1, 2)

        if not self._valid_quad(corners, gray):
            return None

        self.last_valid = corners.copy()
        return corners


class TrackedDetector:
    """Wraps any ReferenceDetector with CornerTracker optical-flow fallback.

    Drop-in replacement for QRCodeDetector / ArucoDetector: exposes the same
    ``detect(frame, drawing_frame=None)`` interface.  When the underlying
    detector finds the marker, the result passes through unchanged and the
    tracker is re-seeded.  When detection fails, the tracker attempts to
    recover the corners via LK optical flow and returns a synthetic Detection
    using the wrapped detector's obj_points.

    Tracked frames are drawn in orange; a red 'LOST' banner is added when
    neither detection nor tracking succeeds.  ``last_state`` is set to
    ``'detected'``, ``'tracked'``, or ``'lost'`` after every call.

    Args:
        detector: Any object with ``detect(frame, drawing_frame=None)`` and
                  ``obj_points`` (all standard ReferenceDetector subclasses qualify).
        tracker:  A CornerTracker instance, or None to use default parameters.
    """

    def __init__(self, detector, tracker=None):
        self._detector  = detector
        self._tracker   = tracker if tracker is not None else CornerTracker()
        self.obj_points = detector.obj_points
        self.last_state = 'lost'   # updated by every detect() call

    @property
    def detector(self):
        """The wrapped underlying detector."""
        return self._detector

    # Forward any attribute the caller might expect from the wrapped detector
    # (e.g. K, D, qr_size, undistorts, get_board_dimensions â€¦).
    def __getattr__(self, name):
        return getattr(self._detector, name)

    def detect(self, frame, drawing_frame=None):
        """Detect or track the marker. Returns Detection or None.
        Sets self.last_state to 'detected', 'tracked', or 'lost'.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        det = self._detector.detect(frame, drawing_frame=drawing_frame)

        if det is not None:
            self._tracker.reset(gray, det.img_pts)
            self.last_state = 'detected'
            return det

        # Detection failed â€” attempt optical-flow recovery.
        corners = self._tracker.track(gray)

        if corners is not None:
            if drawing_frame is not None:
                pts = corners.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(drawing_frame, [pts], True, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(drawing_frame, 'TRACKED', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3, cv2.LINE_AA)
            self.last_state = 'tracked'
            return Detection(obj_pts=self.obj_points.copy(),
                             img_pts=corners.astype(np.float32))

        if drawing_frame is not None:
            cv2.putText(drawing_frame, 'LOST', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        self.last_state = 'lost'
        return None

