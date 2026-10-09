import cv2
import numpy as np
from PIL import Image


def extract_features(img: Image.Image, size: int = 128) -> np.ndarray:
    """Handcrafted features: HSV histogram, dark-pixel share (open grate slots look dark),
    edge density, texture, saturation (garbage is colourful). 80 dims, no GPU needed."""
    rgb = np.array(img.convert("RGB").resize((size, size)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    edges = cv2.Canny(gray, 80, 160)
    extra = [
        (hsv[..., 2] < 50).mean(), (hsv[..., 2] < 90).mean(),
        edges.mean() / 255.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0,
        gray.mean() / 255.0, gray.std() / 255.0,
        hsv[..., 1].mean() / 255.0, (hsv[..., 1] > 90).mean(),
    ]
    return np.concatenate([hist, extra]).astype(np.float32)