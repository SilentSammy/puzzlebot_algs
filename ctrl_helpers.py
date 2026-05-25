import cv2
import numpy as np
import user_input as inp
from marker_est import matrix_to_vecs, vecs_to_matrix


def init_window(name, width=640, height=360):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)


def get_diff_drive_input(slow=0.5, fast=1.0):
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

    Smooths rvec and tvec independently to reduce detection jitter while
    preserving the original (pose_T, pnp_result, detection) tuple format.
    Returns None when no detection is available; state is retained across
    None frames so re-acquisition converges smoothly.
    """

    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: EMA weight on the newest sample.  0 = frozen, 1 = no
                   smoothing (pass-through).  Typical range: 0.1 – 0.5.
        """
        self.alpha = alpha
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None

    def update(self, result):
        """Apply EMA smoothing to a get_pose() result.

        Args:
            result: Output of PoseEstimator.get_pose() —
                    (pose_T, pnp_result, detection) or None.

        Returns:
            Filtered (pose_T, pnp_result, detection), or None if result is None.
        """
        if result is None:
            return None
        pose_T, res, detection = result
        rvec, tvec = matrix_to_vecs(pose_T)
        if self._tvec is None:
            self._rvec = rvec.copy()
            self._tvec = tvec.copy()
        else:
            # Rodrigues vectors that represent the same rotation can have opposite
            # signs (axis flipped, angle negated).  Negate before blending so the
            # EMA stays on the correct side and doesn't interpolate through zero.
            if np.dot(rvec, self._rvec) < 0:
                rvec = -rvec
            self._rvec = self.alpha * rvec + (1 - self.alpha) * self._rvec
            self._tvec = self.alpha * tvec + (1 - self.alpha) * self._tvec
        return vecs_to_matrix(self._rvec, self._tvec), res, detection

    def reset(self):
        """Clear filter state (call when re-acquiring after a long marker loss)."""
        self._rvec = None
        self._tvec = None
