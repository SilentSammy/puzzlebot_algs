# Puzzlebot — Key Findings & Architecture Notes

## Project Overview
CoppeliaSim / real-robot Puzzlebot controller with ArUco-based autonomous following.
- **Sim**: ZMQ Remote API (`sim_tools.py`)
- **Real robot**: rosbridge WebSocket @ `192.168.137.208:9090` → `/cmd_vel_safe`, MJPEG stream @ `:8080` (`pb_bridge.py`)
- **Python 3.14**, OpenCV ArUco, solvePnP pose estimation

---

## PoseFilter — Lessons Learned

### Problem 1: Rodrigues sign-flip fix corrupts rotation

The original filter stored the rotation as a **Rodrigues vector** (`rvec`) and applied a sign-continuity check:
```python
if np.dot(rvec, self._rvec) < 0:
    rvec = -rvec  # WRONG in general
```

**Why it's wrong:** In Rodrigues representation `rvec = θ·axis`, negating gives `-rvec = θ·(−axis)`, which is the *inverse* rotation (rotation by −θ), not the same rotation. The fix is only valid at `|rvec| = π` (180°), where axis and −axis are genuinely equivalent.

**Symptom:** `fx ≈ rx` (translation correct) but `fi ≠ ri` (inverted transform had wrong sign). The corrupted rotation matrix shows up in `inv(T)` because `inv_t = −Rᵀ·t` depends on R. Resetting the filter immediately fixed the inverted value.

**Diagnosis tool added to `manual_control.py`:**
```
rx=+0.046 ri=-0.262 | fx=+0.046 fi=+0.335 | z=0.508   ← fi wrong sign
# after filter reset:
rx=+0.046 ri=-0.264 | fx=+0.046 fi=-0.264 | z=0.506   ← fi now correct
```

### Fix: Store rotation as unit quaternion

For unit quaternions, `q` and `−q` represent the *exact same rotation* (mathematical fact). So the sign-continuity check is unconditionally valid:
```python
if np.dot(q, self._quat) < 0:
    q = -q  # CORRECT: same rotation, opposite hemisphere
```

EMA blend is then renormalised: `self._quat /= np.linalg.norm(self._quat)`.

---

### Problem 2: `max_jump` causes indefinite filter freeze

`max_jump` rejects samples where the raw tvec is more than a threshold from the filtered tvec, intended to catch mirrored solvePnP solutions. With a low `alpha` (slow EMA) the filter lags the true pose significantly during fast movement. Once the lag exceeds `max_jump`, every subsequent sample is rejected and the filter freezes permanently.

**EMA steady-state lag at constant velocity v and frame rate fps:**
$$\text{lag} = \frac{1-\alpha}{\alpha} \cdot \frac{v}{fps}$$

At `alpha=0.04`, 30 fps, and `v=0.3 m/s`: lag ≈ **0.24 m** — right at the 0.25 m threshold.

**Fix 1:** Raise `alpha` (0.04 → 0.15) to reduce lag. Time constant drops from ~0.8 s to ~0.2 s.

**Fix 2:** Add a rejection counter — after 5 consecutive rejections, snap the filter state to the current raw measurement instead of freezing indefinitely:
```python
if self._reject_count >= 5:
    self._quat = self._rvec_to_quat(rvec)
    self._tvec = tvec.copy()
    self._reject_count = 0
```

---

## Coordinate Conventions

| Variable | Meaning |
|---|---|
| `pose_T` | 4×4 camera-to-marker transform from `solvePnP` |
| `pose_T[2, 3]` (`z_dist`) | Depth — distance along camera Z axis to marker |
| `np.linalg.inv(pose_T)[0, 3]` (`x_pos`) | Lateral offset of camera in marker frame |
| `bearing` | `atan2(tvec[0], tvec[2])` — horizontal angle to marker centre |
| `beta` | `atan2(-pose_T[2,0], hypot(pose_T[0,0], pose_T[1,0]))` — marker yaw relative to camera |

`x_pos` requires the matrix inverse because `pose_T[:3, 3]` is the camera origin expressed in the *marker* frame, not the marker in the *camera* frame.

---

## Control Architecture (`follow()`)

```
Detection → PoseFilter (EMA+quaternion) → pose_T
                                              ↓
                              ┌── GOAL? (z near TARGET_DIST, x near 0) ──┐
                              │ yes                                        │ no
                              ▼                                            ▼
                   w = Kp_w · tan(β) · f              aim = f(x_pos)  [bias aim point]
                   x = 0                              error_px = ref_px − marker_px
                   (align β→0 in place)               w = Kp_w · error_px
                                                       x = lin_auth · DIST_GAIN · (z − TARGET_DIST)
```

**Reverse state:** when `z_reached` but not `x_reached` (`AT DISTANCE`), the robot backs off to `REVERSE_DIST` to give itself room to re-approach laterally.

---

## Camera & Calibration

```python
K = [[793.98,   0,    628.31],
     [  0,    793.39, 375.79],
     [  0,       0,     1.0 ]]
D = [-0.352, 0.158, -1.86e-5, -3.15e-4, -0.038]
img_size = (1280, 720)
```

Undistortion is handled inside `ArucoDetector` when `K`/`D` are passed at construction. `PoseEstimator` then uses `D=zeros` to avoid double-applying distortion correction.

---

## File Summary

| File | Purpose |
|---|---|
| `pb_bridge.py` | Real-robot bridge (rosbridge WebSocket + MJPEG stream thread) |
| `ctrl_helpers.py` | Shared helpers: `init_window`, `get_diff_drive_input`, `PoseFilter` |
| `marker_det.py` | `ArucoDetector`, `QRCodeDetector` — detection + optional undistortion |
| `marker_est.py` | `PoseEstimator` (solvePnP wrapper), `PosePlotter3D` |
| `sim_tools.py` | CoppeliaSim `DifferentialCar`, `get_image` |
| `main.py` | Autonomous ArUco following with manual override |
| `manual_control.py` | Manual teleoperation + stream/plotter/filter diagnostics |
