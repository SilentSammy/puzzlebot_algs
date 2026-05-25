import cv2
import user_input as inp


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
