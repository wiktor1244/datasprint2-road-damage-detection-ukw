"""
inference.py : the contract every team's repo must follow (this file is also the starter baseline).

The organisers' benchmark_speed.py imports this file from the ROOT of your repo and calls, in this order:

    model = load(device)                 # once, not timed. device is "cuda" or "cpu"
    detections = predict(model, image)   # once per image, timed, batch size 1

  image       : numpy array, shape [H, W, 3], dtype uint8, colour order RGB, the original image size
  detections  : a list of tuples (class_id, score, x_min, y_min, x_max, y_max)
                class_id 0 D00, 1 D10, 2 D20, 3 D40; score 0-1; box in pixels of the ORIGINAL image.
                Return low-confidence detections too (score >= 0.001). At most 100 are used.

Rules for this file: everything needed (code and weights) is in your repo; no internet access, no calls to
outside APIs; load weights from a path relative to this file. Test-time augmentation, ensembles and tiling
are allowed inside predict(), and their time is counted.

This baseline fine-tunes an Ultralytics YOLO model; replace the body of load() / predict() with your own.
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(_HERE, "weights", "best.pt")
IMGSZ = 640


class _Model:
    def __init__(self, yolo, device):
        self.yolo, self.device = yolo, device


def load(device="cuda"):
    from ultralytics import YOLO
    return _Model(YOLO(WEIGHTS), device)


def predict(model, image):
    bgr = np.ascontiguousarray(image[:, :, ::-1])  # Ultralytics treats numpy arrays as BGR
    r = model.yolo.predict(bgr, imgsz=IMGSZ, conf=0.001, iou=0.5, max_det=100,
                           device=model.device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return [(int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for c, s, b in zip(cls, conf, xyxy)]
