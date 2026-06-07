"""
bench_preprocessing.py  –  Compare detection/tracking approaches in one run.

Pass 0: load grays + baseline detections (one video read).
Per approach: one sequential video read, preprocess on-the-fly, track on cached grays.
"""
import cv2
import numpy as np
import time
from pb_bridge import Puzzlebot
from marker_det import QRCodeDetector

VIDEO_PATH = 'recording_20260607_084736.avi'
K = Puzzlebot._DEFAULT_K
D = Puzzlebot._DEFAULT_D
reference = QRCodeDetector(qr_size=0.0334, K=K, D=D)


# ── preprocessing primitives ──────────────────────────────────────────────────

def unsharp(frame, sigma=2.0, amount=1.5):
    b = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.0 + amount, b, -amount, 0)

def clahe_fn(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([cl.apply(l), a, b_]), cv2.COLOR_LAB2BGR)

def identity(frame):
    return frame


# ── CornerTracker (faithful copy from servoing_test.py) ───────────────────────

class CornerTracker:
    _LK = dict(winSize=(21, 21), maxLevel=3,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def __init__(self, pix_std_min=23.0, min_pts=8, fb_thr=1.0):
        self.pix_std_min = pix_std_min
        self.min_pts     = min_pts
        self.fb_thr      = fb_thr
        self.prev_gray   = None
        self.anchor_pts  = None
        self.track_pts   = None
        self.anc_corners = None
        self.anc_area    = None
        self.last_valid  = None

    @staticmethod
    def _area(c):
        return abs(cv2.contourArea(c.astype(np.float32)))

    @staticmethod
    def _pix_std(gray, corners):
        side = 64
        dst  = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
        M    = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        return float(cv2.warpPerspective(gray, M, (side, side)).std())

    def _valid(self, corners, gray):
        pts = corners.astype(np.float32)
        if not np.all(np.isfinite(pts)):            return False
        if not cv2.isContourConvex(pts):             return False
        for i in range(4):
            a = pts[(i-1)%4]; b = pts[i]; c = pts[(i+1)%4]
            v1 = a-b; v2 = c-b
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-3 or n2 < 1e-3:             return False
            cos = np.clip(np.dot(v1, v2) / (n1*n2), -1, 1)
            if np.degrees(np.arccos(cos)) < 30:     return False
        if self.last_valid is not None:
            r = self._area(pts) / max(1.0, self._area(self.last_valid))
            if r < 0.78 or r > 1.30:               return False
        if self.anc_area and self.anc_area > 1.0:
            if not (0.20 < self._area(pts) / self.anc_area < 5.0): return False
        if self._pix_std(gray, pts) < self.pix_std_min:             return False
        return True

    def reset(self, gray, corners):
        mask = np.zeros(gray.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        pts = cv2.goodFeaturesToTrack(gray, 200, 0.01, 5, mask=mask)
        self.prev_gray   = gray
        self.anc_corners = corners.astype(np.float32).copy()
        self.anc_area    = self._area(self.anc_corners)
        self.last_valid  = corners.astype(np.float32).copy()
        if pts is None or len(pts) < self.min_pts:
            self.anchor_pts = self.track_pts = None
        else:
            self.anchor_pts = pts.astype(np.float32)
            self.track_pts  = pts.astype(np.float32).copy()

    def track(self, gray):
        if self.prev_gray is None or self.track_pts is None:
            self.prev_gray = gray; return None
        nw, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.track_pts, None, **self._LK)
        if nw is None:
            self.prev_gray = gray; self.track_pts = None; return None
        bk, bs, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, nw, None, **self._LK)
        self.prev_gray = gray
        if bk is None: self.track_pts = None; return None
        fb  = np.linalg.norm(self.track_pts.reshape(-1, 2) - bk.reshape(-1, 2), axis=1)
        ok  = (st.reshape(-1) == 1) & (bs.reshape(-1) == 1) & (fb < self.fb_thr)
        self.anchor_pts = self.anchor_pts[ok]
        self.track_pts  = nw[ok]
        if len(self.track_pts) < self.min_pts:
            self.track_pts = None; return None
        H, inl = cv2.findHomography(self.anchor_pts, self.track_pts, cv2.RANSAC, 3.0)
        if H is None: return None
        if inl is not None:
            keep = inl.reshape(-1) == 1
            if keep.sum() >= self.min_pts:
                self.anchor_pts = self.anchor_pts[keep]
                self.track_pts  = self.track_pts[keep]
        w       = cv2.perspectiveTransform(self.anc_corners.reshape(-1, 1, 2), H)
        corners = w.reshape(-1, 2)
        if not self._valid(corners, gray): return None
        self.last_valid = corners.copy()
        return corners


# ── helpers ───────────────────────────────────────────────────────────────────

def _ranges(idx):
    if not idx: return "none"
    parts, s, e = [], idx[0], idx[0]
    for v in idx[1:]:
        if v == e + 1: e = v
        else:
            parts.append(f"{s}" if s == e else f"{s}-{e}"); s = e = v
    parts.append(f"{s}" if s == e else f"{s}-{e}")
    return ", ".join(parts)

def print_stats(name, states, total, t):
    nd = states.count('detected')
    nt = states.count('tracked')
    nl = states.count('lost')
    li = [i+1 for i, s in enumerate(states) if s == 'lost']
    marker = '  <<< BEST' if nl == 0 else (f'  (-{min(0, nl-31):+d})' if nl != 31 else '')
    print(f"\n  {name:<42}  {t:5.1f}s")
    print(f"    Det {nd:>3} ({100*nd/total:.1f}%)  "
          f"Trk {nt:>3} ({100*nt/total:.1f}%)  "
          f"Lost {nl:>3} ({100*nl/total:.1f}%)"
          f"{'  BETTER' if nl < 31 else '  WORSE' if nl > 31 else '  SAME':>8}")
    if li:
        print(f"    Lost frames: {_ranges(li)}")


# ── Pass 0: precompute grays + baseline detections (one read) ─────────────────

cap   = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Loading {total} frames (gray only)...", end=' ', flush=True)

grays     = []
base_dets = []   # True where baseline QR detector succeeded
t_load    = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grays.append(gray)
    base_dets.append(reference.detect(frame) is not None)

cap.release()
total = len(grays)
print(f"done  ({time.perf_counter()-t_load:.1f}s, baseline "
      f"{sum(base_dets)}/{total} detected)\n")


# ── run one approach ──────────────────────────────────────────────────────────

def run_approach(prep_fn, trk_kwargs, fallback=False):
    """
    prep_fn     : frame -> preprocessed BGR (used for detection only).
    trk_kwargs  : kwargs forwarded to CornerTracker().
    fallback    : if True, use original frame where baseline succeeded,
                  preprocessed only where baseline failed.
    Tracking always runs on cached original grays.
    """
    cap2 = cv2.VideoCapture(VIDEO_PATH)
    trk  = CornerTracker(**trk_kwargs)
    states = []
    i = 0
    while True:
        ret, frame = cap2.read()
        if not ret: break
        gray = grays[i]
        if fallback:
            proc = frame if base_dets[i] else prep_fn(frame)
        else:
            proc = prep_fn(frame)
        det = reference.detect(proc)
        if det is not None:
            trk.reset(gray, det.img_pts)
            states.append('detected')
        else:
            corners = trk.track(gray)
            states.append('tracked' if corners is not None else 'lost')
        i += 1
    cap2.release()
    return states


# ── approach table ────────────────────────────────────────────────────────────
# (name, prep_fn, trk_kwargs, fallback)

BASE_KW = dict(pix_std_min=23.0, min_pts=8, fb_thr=1.0)

APPROACHES = [
    # ── Detection preprocessing ──────────────────────────────────────────────
    ('baseline',                    identity,                               BASE_KW, False),
    ('unsharp σ=2 a=1.0',          lambda f: unsharp(f, 2.0, 1.0),        BASE_KW, False),
    ('unsharp σ=2 a=1.5',          lambda f: unsharp(f, 2.0, 1.5),        BASE_KW, False),
    ('unsharp σ=2 a=2.5',          lambda f: unsharp(f, 2.0, 2.5),        BASE_KW, False),
    ('unsharp σ=3 a=4.0',          lambda f: unsharp(f, 3.0, 4.0),        BASE_KW, False),
    ('clahe',                       clahe_fn,                               BASE_KW, False),
    ('clahe + unsharp a=1.5',      lambda f: unsharp(clahe_fn(f), 2.0, 1.5), BASE_KW, False),
    # ── Fallback: preprocess only where baseline missed ───────────────────────
    ('fallback → unsharp 1.5',     lambda f: unsharp(f, 2.0, 1.5),        BASE_KW, True),
    ('fallback → unsharp 4.0',     lambda f: unsharp(f, 3.0, 4.0),        BASE_KW, True),
    ('fallback → clahe+unsharp',   lambda f: unsharp(clahe_fn(f), 2.0, 1.5), BASE_KW, True),
    # ── Tracker parameter variations (baseline preprocessing) ─────────────────
    ('relax texture (std≥18)',      identity, dict(pix_std_min=18.0, min_pts=8, fb_thr=1.0), False),
    ('relax texture (std≥20)',      identity, dict(pix_std_min=20.0, min_pts=8, fb_thr=1.0), False),
    ('relax tracking (min_pts=6)',  identity, dict(pix_std_min=23.0, min_pts=6, fb_thr=1.0), False),
    ('relax fb (fb_thr=2.0)',       identity, dict(pix_std_min=23.0, min_pts=8, fb_thr=2.0), False),
    # ── Combined best bets ────────────────────────────────────────────────────
    ('fallback 4.0 + std≥20',      lambda f: unsharp(f, 3.0, 4.0),
                                    dict(pix_std_min=20.0, min_pts=8, fb_thr=1.0), True),
    ('fallback c+u + std≥20',      lambda f: unsharp(clahe_fn(f), 2.0, 1.5),
                                    dict(pix_std_min=20.0, min_pts=8, fb_thr=1.0), True),
]

print(f"{'─'*62}")
print(f"  Baseline stats: Det 163 (62.5%)  Trk 67 (25.7%)  Lost 31 (11.9%)")
print(f"{'─'*62}")

for name, prep_fn, trk_kw, fallback in APPROACHES:
    t0     = time.perf_counter()
    states = run_approach(prep_fn, trk_kw, fallback)
    print_stats(name, states, total, time.perf_counter() - t0)

print(f"\n{'─'*62}")
print("Done.")
