import cv2
import numpy as np
import time
from pb_bridge import Puzzlebot
from ctrl_helpers import init_window, get_diff_drive_input
import user_input as inp

car = Puzzlebot()
init_window('Original',    img_size=car.img_size, height=360)
init_window('Undistorted', img_size=car.img_size, height=360)

writer = None   # cv2.VideoWriter when recording, else None

def start_recording(w, h):
    ts = time.strftime('%Y%m%d_%H%M%S')
    filename = f'recording_{ts}.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 25.0, (w, h))
    print(f"Recording started: {filename}")
    return out

try:
    while True:
        cmd = get_diff_drive_input()
        car.lin_vel = cmd['x'] * car.nominalLinearVelocity
        car.ang_vel = cmd['w'] * car.nominalAngularVelocity
        car._publish()

        ret, frame = car.get_image()
        if not ret:
            continue

        undistorted = cv2.undistort(frame, car.K, car.D)
        orig_drawing = frame.copy()

        # Toggle recording with 'r'
        if inp.rising_edge('r'):
            if writer is None:
                h, w = frame.shape[:2]
                writer = start_recording(w, h)
            else:
                writer.release()
                writer = None
                print("Recording stopped.")

        if writer is not None:
            writer.write(orig_drawing)
            # Red dot in top-right corner to indicate recording
            h, w = frame.shape[:2]
            cv2.circle(orig_drawing, (w - 20, 20), 10, (0, 0, 255), -1)

        cv2.imshow('Original',    orig_drawing)
        cv2.imshow('Undistorted', undistorted)
        cv2.waitKey(1)
finally:
    if writer is not None:
        writer.release()
        print("Recording stopped.")
    car.lin_vel = 0.0
    car.ang_vel = 0.0
    car._publish()
    cv2.destroyAllWindows()
