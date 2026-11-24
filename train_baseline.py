#!/usr/bin/env python3
"""
Starter baseline: fine-tune a small YOLO detector (generic COCO weights) on the hackathon training images.

    python train_baseline.py --kit kit                       # defaults: yolov8n, 50 epochs, 640 px
    python train_baseline.py --kit kit --epochs 5 --device cpu --imgsz 320     # quick check that it runs

Reads  kit/train_images + kit/train_labels.csv  (and the validation images for model selection).
Writes weights/best.pt, which the starter inference.py loads. Images with no damage are kept as background.
"""
import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def build_yolo_split(kit: Path, out: Path, split: str):
    img_dir, lbl_csv = kit / ("%s_images" % split), kit / ("%s_labels.csv" % split)
    boxes = defaultdict(list)
    with open(lbl_csv, newline="") as f:
        for r in csv.DictReader(f):
            boxes[r["image_id"]].append((int(r["class_id"]), *(float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max"))))
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(img_dir.glob("*.jpg")):
        dst = out / "images" / split / p.name
        if not dst.exists():
            try:
                os.symlink(p.resolve(), dst)
            except OSError:
                shutil.copy2(p, dst)
        W, H = Image.open(p).size
        lines = []
        for c, x1, y1, x2, y2 in boxes.get(p.stem, []):
            lines.append("%d %.6f %.6f %.6f %.6f" % (c, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        (out / "labels" / split / (p.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, help="the kit folder with train_images/, val_images/ and the label CSVs")
    ap.add_argument("--model", default="yolov8n.pt", help="generic COCO weights (on the approved list)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--workdir", default="baseline_work")
    a = ap.parse_args()

    from ultralytics import YOLO

    kit, work = Path(a.kit).resolve(), Path(a.workdir).resolve()
    data = work / "yolo_data"
    for s in ("train", "val"):
        print("%s: %d images prepared" % (s, build_yolo_split(kit, data, s)))
    (work / "data.yaml").write_text("path: %s\ntrain: images/train\nval: images/val\nnames:\n%s"
                                    % (data, "".join("  %d: %s\n" % (i, n) for i, n in enumerate(CLASS_NAMES))))

    model = YOLO(a.model)
    model.train(data=str(work / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                workers=a.workers, project=str(work / "runs"), name="baseline", exist_ok=True,
                fliplr=0.5, flipud=0.0, patience=15, plots=False)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy2(model.trainer.best, "weights/best.pt")
    print("\nSaved weights/best.pt (this is what inference.py loads)")


if __name__ == "__main__":
    main()
