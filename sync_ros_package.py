#!/usr/bin/env python3
"""Sync shared vision modules into the ROS package.

Run this from the Puzzlebot root whenever any of the source modules change:

    python sync_ros_package.py

Copies the listed modules from the workspace root into the ROS package's
Python directory so they are importable at runtime without a separate
ROS library package.
"""

import shutil
from pathlib import Path

ROOT    = Path(__file__).parent
DEST    = ROOT / 'ROS' / 'qr_pallet_aligner' / 'qr_pallet_aligner'

MODULES = [
    'pipeline.py',
    'marker_det.py',
    'marker_est.py',
    'tracking.py',
    'servoing.py',
]

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ok, missing = [], []
    for name in MODULES:
        src = ROOT / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, DEST / name)
        ok.append(name)

    for name in ok:
        print(f'  copied  {name}')
    for name in missing:
        print(f'  MISSING {name}  (skipped)')

    print(f'\n{len(ok)}/{len(MODULES)} files synced to {DEST}')
    if missing:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
