# Visual Marker Following — `follow()` Algorithm

The controller uses a pose estimator to extract the marker's 3D position relative to the camera each frame, giving a lateral offset `x_pos` (metres) and forward distance `z_dist` (metres). Because the camera is mounted off-centre from the robot's physical centre, a **target yaw** is computed — the camera angle that would make the robot centre (not the lens) face the marker — using `target_yaw = atan2(-X_OFFSET, z_dist)`. This yaw is projected into pixel space via the full distortion model (`cv2.projectPoints`) to get an exact reference pixel. An additional **aim offset** proportional to `x_pos` shifts that reference left or right so the robot approaches the marker from the correct lateral angle rather than always driving straight at it. The angular error is the difference between the reference pixel and the detected marker's pixel centre, multiplied by a proportional gain `KP_W` to produce an angular velocity command. Linear authority fades to zero when the gaze angle (how far the camera is looking sideways) exceeds a threshold, preventing the robot from driving forward while still turning to acquire the target.

```
target_yaw  = atan2(-X_OFFSET, z_dist)          # camera angle to align robot centre
ref_px      = project(target_yaw) + aim * (w/2) # reference pixel, shifted by aim
aim         = clamp(x_pos * AIM_GAIN)            # lateral offset → aim shift
error_px    = ref_px - marker_centre_px          # pixel error
w_cmd       = clamp(KP_W * error_px)             # angular command
lin_auth    = max(0, 1 - gaze_angle / AUTH_ANGLE)# forward authority fades when looking sideways
x_cmd       = lin_auth * KP_X * (z_dist - TARGET_DIST)
```

A **hysteresis goal zone** prevents chattering at the target: the robot enters the goal when both `|x_pos|` and `|z_dist - TARGET_DIST|` fall within `GOAL_RADIUS` (1 cm), and only exits once they exceed `GOAL_RADIUS + GOAL_HYSTERESIS` (1.6 cm). Inside the goal, linear motion stops and only a fine orientation correction (`beta`, the marker's facing angle) is applied.
