#!/usr/bin/env python3
"""
Generate 4×4 ArUco markers as 120×120-pixel PNG files.

Dependencies:
    pip install --no-cache-dir --force-reinstall opencv-contrib-python
"""

import os
import sys

import cv2

def main():
    # 1) Verify the aruco module is present
    if not hasattr(cv2, "aruco"):
        sys.exit("ERROR: ArUco module not found. Install opencv-contrib-python, not just opencv-python.")

    # 2) Load dictionary
    dict_id = cv2.aruco.DICT_6X6_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    dict_name = next(
        k.removeprefix("DICT_").lower()
        for k, v in vars(cv2.aruco).items()
        if k.startswith("DICT_") and v == dict_id
    )

    # 3) Draw and save markers for IDs 0 through n-1 in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for marker_id in range(2):
        # generateImageMarker creates the marker image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 120)
        filename = os.path.join(script_dir, f"aruco_{dict_name}_{marker_id:02d}.png")
        cv2.imwrite(filename, marker_img)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
