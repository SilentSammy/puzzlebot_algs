import cv2
import numpy as np
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window
from marker_det import QRCodeDetector
import user_input as inp

VIDEO_PATH = 'recording_20260607_084736.avi'

# Use the camera intrinsics without connecting to the robot.
K = Puzzlebot._DEFAULT_K
D = Puzzlebot._DEFAULT_D

reference = QRCodeDetector(qr_size=0.0334, K=K, D=D)


def _detect(frame):
    """Try detection on original frame; fall back to unsharpened version."""
    det = reference.detect(frame)
    if det is not None:
        return det
    blurred  = cv2.GaussianBlur(frame, (0, 0), 3.0)
    sharpened = cv2.addWeighted(frame, 5.0, blurred, -4.0, 0)
    return reference.detect(sharpened)


class CornerTracker:
    """Recovers the 4 QR corners while detection is lost.

    On each successful detection, seeds feature points inside the QR quad.
    While lost, tracks those features frame-to-frame with Lucas-Kanade optical
    flow, fits a homography from the seed positions to the current positions,
    and warps the last known corners through it.
    """

    _LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Minimum pixel std-dev inside the quad warped to a 64x64 patch.
    # QR code texture is always >= 24; drifted-onto-background is <= 23.
    _PIX_STD_MIN = 23.0

    def __init__(self, min_points=8, fb_threshold=1.0):
        self.min_points = min_points
        self.fb_threshold = fb_threshold  # max forward-backward flow error (px)
        self.prev_gray = None        # gray frame from the previous iteration
        self.anchor_pts = None       # (N,1,2) feature positions at detection time
        self.track_pts = None        # (N,1,2) current feature positions
        self.anchor_corners = None   # (4,2) corners at detection time
        self.anchor_area = None      # area of anchor quad (px^2)
        self.last_valid = None       # (4,2) last corners that passed _valid_quad

    def reset(self, gray, corners):
        """Seed features from a fresh detection."""
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                      minDistance=5, mask=mask)
        self.prev_gray = gray
        self.anchor_corners = corners.astype(np.float32).copy()
        self.anchor_area = self._quad_area(self.anchor_corners)
        self.last_valid = corners.astype(np.float32).copy()
        if pts is None or len(pts) < self.min_points:
            self.anchor_pts = None
            self.track_pts = None
        else:
            self.anchor_pts = pts.astype(np.float32)
            self.track_pts = pts.astype(np.float32).copy()

    @staticmethod
    def _quad_area(corners):
        return abs(cv2.contourArea(corners.astype(np.float32)))

    @staticmethod
    def _quad_pixel_std(gray, corners):
        """Warp the quad into a 64x64 patch and return its pixel std-dev."""
        side = 64
        dst = np.array([[0, 0], [side, 0], [side, side], [0, side]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        patch = cv2.warpPerspective(gray, M, (side, side))
        return float(patch.std())

    def _valid_quad(self, corners, gray):
        """Reject warped corners relative to the previous accepted frame."""
        pts = corners.astype(np.float32)

        # Must be finite.
        if not np.all(np.isfinite(pts)):
            return False

        # Must be convex.
        if not cv2.isContourConvex(pts):
            return False

        # No degenerate corner angles.
        for i in range(4):
            a = pts[(i - 1) % 4]; b = pts[i]; c = pts[(i + 1) % 4]
            v1 = a - b; v2 = c - b
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-3 or n2 < 1e-3:
                return False
            cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            if np.degrees(np.arccos(cos_ang)) < 30.0:
                return False

        # ---- Relative-to-previous-frame checks ----
        if self.last_valid is not None:
            area_cur  = self._quad_area(pts)
            area_prev = self._quad_area(self.last_valid)
            if area_prev > 1.0:
                ratio = area_cur / area_prev
                # Allow at most ~25 % area change frame-to-frame.
                if ratio < 0.78 or ratio > 1.30:
                    return False

        # Absolute anchor-area backstop (gross failure).
        if self.anchor_area and self.anchor_area > 1.0:
            ratio = self._quad_area(pts) / self.anchor_area
            if ratio < 0.20 or ratio > 5.0:
                return False

        # Pixel texture: QR code std >= 24; plain background <= 23.
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

        # Forward-backward consistency check: track back and keep only points
        # that return close to where they started.
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, new_pts, None, **self._LK_PARAMS)
        self.prev_gray = gray

        if back_pts is None:
            self.track_pts = None
            return None

        fb_err = np.linalg.norm(
            self.track_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
        good = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) \
            & (fb_err < self.fb_threshold)

        self.anchor_pts = self.anchor_pts[good]
        self.track_pts = new_pts[good]

        if len(self.track_pts) < self.min_points:
            self.track_pts = None
            return None

        H, inliers = cv2.findHomography(self.anchor_pts, self.track_pts,
                                        cv2.RANSAC, 3.0)
        if H is None:
            return None

        # Keep only the RANSAC inliers so accumulated drift doesn't poison
        # subsequent frames.
        if inliers is not None:
            keep = inliers.reshape(-1) == 1
            if keep.sum() >= self.min_points:
                self.anchor_pts = self.anchor_pts[keep]
                self.track_pts = self.track_pts[keep]

        warped = cv2.perspectiveTransform(
            self.anchor_corners.reshape(-1, 1, 2), H)
        corners = warped.reshape(-1, 2)

        if not self._valid_quad(corners, gray):
            return None

        self.last_valid = corners.copy()
        return corners

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
init_window('QR Detection', img_size=(w, h), height=360)

# Two capture objects keep navigation honest:
#   proc_cap reads strictly forward and drives the (causal) tracker, so the
#   tracker only ever sees past frames in their original order.
#   disp_cap is free to seek anywhere just to fetch the image for display.
disp_cap = cv2.VideoCapture(VIDEO_PATH)

# ------------------------------------------------------------------
# Phase 1: precompute the full video in one forward pass.
# ------------------------------------------------------------------
# frame_states[i] is one of: 'detected' | 'tracked' | 'lost'
frame_states  = ['lost'] * total
# trk_cache[i] holds the (4,2) corners when state == 'tracked', else None.
trk_cache     = {}

print(f"Precomputing {total} frames...")
tracker = CornerTracker()
for i in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection = _detect(frame)
    if detection is not None:
        tracker.reset(gray, detection.img_pts)
        frame_states[i] = 'detected'
        trk_cache[i]    = None
    else:
        corners = tracker.track(gray)
        if corners is not None:
            frame_states[i] = 'tracked'
            trk_cache[i]    = corners
        else:
            frame_states[i] = 'lost'
            trk_cache[i]    = None

cap.release()

# ------------------------------------------------------------------
# Phase 2: print statistics.
# ------------------------------------------------------------------
n_det     = frame_states.count('detected')
n_tracked = frame_states.count('tracked')
n_lost    = frame_states.count('lost')

print(f"\n{'='*50}")
print(f"  Total  : {total}")
print(f"  Detected (QR found)  : {n_det:>4}  ({100*n_det/total:.1f}%)")
print(f"  Tracked (flow only)  : {n_tracked:>4}  ({100*n_tracked/total:.1f}%)")
print(f"  Lost                 : {n_lost:>4}  ({100*n_lost/total:.1f}%)")

def _ranges(indices):
    """Collapse a sorted list of ints into human-readable ranges."""
    if not indices:
        return "none"
    parts, start, end = [], indices[0], indices[0]
    for v in indices[1:]:
        if v == end + 1:
            end = v
        else:
            parts.append(f"{start}" if start == end else f"{start}-{end}")
            start = end = v
    parts.append(f"{start}" if start == end else f"{start}-{end}")
    return ", ".join(parts)

det_idx  = [i+1 for i, s in enumerate(frame_states) if s == 'detected']
trk_idx  = [i+1 for i, s in enumerate(frame_states) if s == 'tracked']
lost_idx = [i+1 for i, s in enumerate(frame_states) if s == 'lost']

print(f"\n  Detected frames  : {_ranges(det_idx)}")
print(f"  Tracked frames   : {_ranges(trk_idx)}")
print(f"  Lost frames      : {_ranges(lost_idx)}")
print(f"{'='*50}\n")


def render(idx, playing):
    """Fetch frame `idx` and draw detection + cached tracking overlays."""
    disp_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = disp_cap.read()
    if not ret:
        return None

    drawing = frame.copy()
    state   = frame_states[idx]

    if state == 'detected':
        # Re-run detector only for drawing (cheap — already cached in memory).
        reference.detect(frame, drawing_frame=drawing)
    else:
        cv2.putText(drawing, 'LOST', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3, cv2.LINE_AA)
        corners = trk_cache.get(idx)
        if corners is not None:
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(drawing, [pts], True, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(drawing, 'TRACKED', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3,
                        cv2.LINE_AA)

    label = 'PLAY' if playing else 'PAUSE'
    cv2.putText(drawing, f'{idx+1}/{total}  [{label}]  {state.upper()}',
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return drawing


print("Controls: SPACE play/pause | <-/a back | ->/d forward | q quit")

idx = 0
playing = False

try:
    while True:
        if inp.rising_edge('Key.space'):
            playing = not playing

        step = 0
        if playing:
            step = 1
        elif inp.is_pressed('Key.right') or inp.is_pressed('d'):
            step = 1
        elif inp.is_pressed('Key.left') or inp.is_pressed('a'):
            step = -1

        idx += step
        if idx >= total:
            idx = 0
        elif idx < 0:
            idx = 0

        drawing = render(idx, playing)
        if drawing is not None:
            cv2.imshow('QR Detection', drawing)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q') or inp.rising_edge('q'):
            break
finally:
    disp_cap.release()
    cv2.destroyAllWindows()
