import cv2
import numpy as np
import user_input as inp
from marker_est import matrix_to_vecs, vecs_to_matrix


def init_window(name, width=640, height=360, img_size=None):
    if img_size is not None:
        width = img_size[0] * height // img_size[1]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, width, height)


def get_diff_drive_input(slow=0.4, fast=1.0):
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

    def __init__(self, alpha=0.3, max_jump=0.25):
        """
        alpha: EMA weight on newest sample (0=frozen, 1=no smoothing).
        max_jump: Max tvec displacement (metres) allowed between samples.
                  Samples exceeding this are discarded. None to disable.
        """
        self.alpha = alpha
        self.max_jump = max_jump
        self._quat: np.ndarray | None = None  # unit quaternion [x, y, z, w]
        self._tvec: np.ndarray | None = None
        self._reject_count = 0

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
            self.reset()
            return None
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
            self._quat = self.alpha * q + (1.0 - self.alpha) * self._quat
            self._quat /= np.linalg.norm(self._quat)  # renormalize after blend
            self._tvec = self.alpha * tvec + (1.0 - self.alpha) * self._tvec
        return vecs_to_matrix(self._quat_to_rvec(self._quat), self._tvec), res, detection

    def reset(self):
        self._quat = None
        self._tvec = None
        self._reject_count = 0


class PoseTracker:
    """Stateful wrapper combining a PoseEstimator and a PoseFilter.

    Duck-type compatible with PoseEstimator: call get_pose(frame, drawing_frame)
    and receive a filtered (pose_T, pnp_result, detection) tuple or None.
    """

    def __init__(self, estimator, filter):
        self._estimator = estimator
        self._filter = filter

    def get_pose(self, frame, drawing_frame=None):
        return self._filter.update(self._estimator.get_pose(frame, drawing_frame=drawing_frame))

    def reset(self):
        self._filter.reset()


