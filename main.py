# from sim_tools import sim, get_image, DifferentialCar
import cv2
import numpy as np
import math
import time
import user_input as inp
from ctrl_helpers import init_window, get_diff_drive_input, merge_proportional, get_manual_override
from pipeline import build_servoing
from pb_bridge import Puzzlebot


# CAR SETUP
car = Puzzlebot()

# SETUP
init_window('Camera', img_size=car.img_size, height=360)
servoing = build_servoing(car.K, car.D, qr_size=0.05, x_offset=-0.065, z_offset=0.055)

cmd_enables = {'x': 0.0, 'w': 0.0}
last_time = time.perf_counter()
try:
    while True:
        ret, frame = car.get_image()
        if not ret:
            continue
        drawing_frame = frame.copy()
        
        # Calculate FPS
        current_time = time.perf_counter()
        fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0.0
        last_time = current_time
        print(f"FPS: {fps:.1f} Hz", end='  ')

        if inp.rising_edge('1'):
            cmd_enables['x'] = 1.0 - cmd_enables['x']  # toggle between 0.0 and 1.0
            print(f"Auto X: {'ON' if cmd_enables['x'] else 'OFF'}")
        if inp.rising_edge('2'):
            cmd_enables['w'] = 1.0 - cmd_enables['w']  # toggle between 0.0 and 1.0
            print(f"Auto W: {'ON' if cmd_enables['w'] else 'OFF'}")

        # Send velocity command to car
        auto_cmd = servoing.update(frame, drawing_frame=drawing_frame)  # vision-based command
        auto_cmd = {axis: auto_cmd[axis] * cmd_enables[axis] for axis in auto_cmd}  # apply enables
        man_cmd = get_diff_drive_input(0.25, 0.5)  # get manual input
        cmd = merge_proportional(man_cmd, auto_cmd)  # combine manual and auto commands
        car.lin_vel  = cmd['x'] * car.nominalLinearVelocity
        car.ang_vel = cmd['w'] * car.nominalAngularVelocity
        car._publish()

        cv2.imshow('Camera', drawing_frame)
        cv2.waitKey(1)
finally:
    car.lin_vel  = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
    # plotter.close()
