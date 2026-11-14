"""Write a small photograph-like JPEG (used by CI to try the installed command).

    python .github/scripts/make_sample.py sample.jpg
"""
import sys

import numpy as np
from PIL import Image

rs = np.random.RandomState(7)
yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
base = 110 + 60 * np.sin(xx / 47.0) * np.cos(yy / 31.0) + 0.08 * xx
rgb = np.stack([base, base * 0.92 + 9, base * 0.85 + 18], axis=-1)
rgb = np.clip(rgb + rs.normal(0, 4.0, rgb.shape), 0, 255).astype(np.uint8)
Image.fromarray(rgb).save(sys.argv[1], quality=90)
print("wrote", sys.argv[1])
