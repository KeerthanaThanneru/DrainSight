"""Run: python -m src.train_image"""
import json
import joblib
import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.config import *
from src.image_utils import extract_features

DEBRIS_BGR = [(40, 70, 110), (30, 90, 50), (200, 200, 200), (60, 60, 200),
              (200, 120, 40), (30, 160, 200), (90, 130, 160)]          # leaves, plastic, paper, mud...
COVER = {"clear": (0.0, 0.06), "partial": (0.25, 0.55), "blocked": (0.75, 1.0)}


def synth_image(cls, rng, size=160):
    img = np.full((size, size, 3), int(rng.integers(100, 170)), np.uint8)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = 25, 25, 135, 135
    n = int(rng.integers(6, 9)); step = (x1 - x0) // n
    for i in range(n):                                                  # dark open slots of the grate
        cv2.rectangle(img, (x0 + i * step + 4, y0), (x0 + (i + 1) * step - 4, y1), (20, 20, 20), -1)
    cv2.rectangle(img, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), (70, 70, 70), 4)
    target = rng.uniform(*COVER[cls])
    mask = np.zeros((size, size), np.uint8)
    for _ in range(300):                                                # add debris until coverage reached
        if mask[y0:y1, x0:x1].mean() / 255 >= target:
            break
        c = DEBRIS_BGR[int(rng.integers(len(DEBRIS_BGR)))]
        ctr = (int(rng.integers(x0, x1)), int(rng.integers(y0, y1)))
        axes = (int(rng.integers(6, 22)), int(rng.integers(4, 16)))
        ang = int(rng.integers(0, 180))
        cv2.ellipse(img, ctr, axes, ang, 0, 360, c, -1)
        cv2.ellipse(mask, ctr, axes, ang, 0, 360, 255, -1)
    img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.7, 1.2)), beta=float(rng.uniform(-20, 20)))
    k = int(rng.choice([1, 3, 5]))
    img = cv2.GaussianBlur(img, (k, k), 0)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def load_dataset(n_synth=200):
    real = {}
    for cls in IMAGE_CLASSES:
        folder = IMG_DIR / cls
        folder.mkdir(parents=True, exist_ok=True)
        real[cls] = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    X, y = [], []
    if all(len(v) >= 30 for v in real.values()):
        source = "real"
        for cls, files in real.items():
            for p in files:
                X.append(extract_features(Image.open(p))); y.append(cls)
    else:
        source = "synthetic"
        print("Fewer than 30 real images in some class -> using SYNTHETIC images")
        rng = np.random.default_rng(SEED)
        for cls in IMAGE_CLASSES:
            for _ in range(n_synth):
                X.append(extract_features(synth_image(cls, rng))); y.append(cls)
    return np.array(X), np.array(y), source


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, source = load_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(classification_report(yte, pred))
    rep = classification_report(yte, pred, output_dict=True)
    rep["data_source"] = source
    rep["confusion_matrix"] = confusion_matrix(yte, pred, labels=IMAGE_CLASSES).tolist()
    (MODELS_DIR / "image_metrics.json").write_text(json.dumps(rep, indent=2))
    joblib.dump(clf, MODELS_DIR / "image_clf.joblib", compress=3)


if __name__ == "__main__":
    main()