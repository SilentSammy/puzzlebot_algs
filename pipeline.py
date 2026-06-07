"""Shared factory for the marker-servoing pipeline.

Builds the full perceive -> control chain (detector -> estimator -> tracker ->
controller) so the construction lives in one place and is reused by both the
desktop loop (``main.py``) and the ROS 2 node (``qr_pallet_aligner_standalone``).
Edit this module to change how the pipeline is assembled; callers stay untouched.
"""

try:
    from .marker_det import QRCodeDetector
    from .marker_est import PoseEstimator
    from .tracking import TrackedDetector, PoseFilter, PoseTracker
    from .servoing import MarkerServoing
except ImportError:
    from marker_det import QRCodeDetector
    from marker_est import PoseEstimator
    from tracking import TrackedDetector, PoseFilter, PoseTracker
    from servoing import MarkerServoing


def build_servoing(K, D, *, qr_size=0.05, qr_content=None, tau=0.78, **servoing_kwargs):
    """Assemble and return a ready-to-use :class:`MarkerServoing` controller.

    Parameters
    ----------
    K, D : np.ndarray
        Camera intrinsics / distortion coefficients.
    qr_size : float
        Side length of the QR marker (m).
    qr_content : str or None
        Expected QR payload; when set, only a marker decoding to this string is
        accepted.  ``None`` accepts any decoded QR.
    tau : float
        Pose filter time constant (s).
    **servoing_kwargs
        Forwarded to :class:`MarkerServoing` (e.g. ``x_offset``, ``z_offset``,
        gains, ``verbose``).  The caller-built ``PoseTracker`` is accessible
        afterwards via the returned controller's ``pose_tracker`` property.
    """
    reference = TrackedDetector(QRCodeDetector(qr_size=qr_size, content=qr_content, K=K, D=D))
    pose_tracker = PoseTracker(
        estimator=PoseEstimator(reference=reference, K=K, D=D),
        filter=PoseFilter(tau=tau),
    )
    return MarkerServoing(pose_tracker, K=K, D=D, **servoing_kwargs)
