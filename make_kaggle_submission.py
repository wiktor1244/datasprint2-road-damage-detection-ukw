#!/usr/bin/env python3
"""
Turn a detailed prediction file (one row per detection) into the file you upload to Kaggle
(one row per image). Every test image gets a row, with PredictionString "none" when nothing was found (Kaggle rejects empty cells).

    python make_kaggle_submission.py --pred val_pred.csv --image-ids test_image_ids.csv --out submission.csv

Input  --pred       image_id,class_id,score,x_min,y_min,x_max,y_max   (what benchmark_speed.py writes)
Output --out        image_id,PredictionString
                    PredictionString = "class score x_min y_min x_max y_max" repeated for each box,
                    separated by spaces, e.g. "3 0.912 412.0 230.5 538.2 301.7 0 0.340 10.0 198.0 622.0 231.0"
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import FormatError, MAX_DETS, load_ids, load_predictions  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--image-ids", required=True, help="test_image_ids.csv (or val_image_ids.csv to test)")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    try:
        ids = load_ids(a.image_ids)
        preds = load_predictions(a.pred, valid_ids=set(ids))
    except FormatError as e:
        print("FAILED:", e)
        return 1
    n_boxes = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "PredictionString"])
        for iid in ids:
            d = preds.get(iid)
            parts = []
            if d is not None:
                d = d[d[:, 1].argsort(kind="stable")[::-1]][:MAX_DETS]
                for c, s, x1, y1, x2, y2 in d:
                    parts.append("%d %.5f %.1f %.1f %.1f %.1f" % (c, s, x1, y1, x2, y2))
                n_boxes += len(d)
            w.writerow([iid, " ".join(parts) or "none"])
    print("wrote %s: %d images, %d boxes" % (a.out, len(ids), n_boxes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
