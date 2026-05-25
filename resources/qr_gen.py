#!/usr/bin/env python3
"""
Generate a QR code PNG from a given string.

Dependencies:
    pip install qrcode[pil]
"""

import os
import sys

import qrcode


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python qr_gen.py <string to encode>")

    data = sys.argv[1]

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=0,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in data)
    filename = os.path.join(script_dir, f"qr_{safe_name}.png")

    img.save(filename)
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
