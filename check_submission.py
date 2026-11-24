#!/usr/bin/env python3
"""
Format check for submission.csv. Run it before every upload; the upload form runs the same checks.

    python check_submission.py --pred submission.csv --image-ids test_image_ids.csv

Rejected (exit code 1): wrong header, a class_id outside 0-3, a score outside 0-1, a box with zero or
negative size, a non-numeric value, an image_id that is not in the test list, a file over the size limit.
Warnings (still accepted): more than 100 detections for one image (the lowest scores are dropped),
or no low-confidence detections (you may have thresholded too early, which lowers mAP).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import FormatError, MAX_DETS, load_ids, load_predictions  # noqa: E402


def check_rows(path, ids):
    """Kaggle upload layout: one row for every test image, and no empty PredictionString cell."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "PredictionString" not in (reader.fieldnames or []):
            return  # the one-row-per-detection layout used by benchmark_speed.py
        rows = list(reader)
    empty = [r["image_id"] for r in rows if not (r["PredictionString"] or "").strip()]
    if empty:
        raise FormatError("%d row(s) have an empty PredictionString, for example %s. Kaggle rejects empty cells: "
                          "write the word none for an image with no detections (make_kaggle_submission.py does this)"
                          % (len(empty), ", ".join(empty[:3])))
    missing = set(ids) - {r["image_id"].strip() for r in rows}
    if missing:
        raise FormatError("%d test image(s) have no row, for example %s. The file needs one row for every test image"
                          % (len(missing), ", ".join(sorted(missing)[:3])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--image-ids", required=True, help="test_image_ids.csv from the starter kit")
    ap.add_argument("--max-mb", type=float, default=50.0)
    a = ap.parse_args()

    try:
        size_mb = os.path.getsize(a.pred) / 1e6
        if size_mb > a.max_mb:
            raise FormatError("file is %.1f MB; the limit is %.0f MB" % (size_mb, a.max_mb))
        ids = load_ids(a.image_ids)
        preds = load_predictions(a.pred, valid_ids=set(ids))
        check_rows(a.pred, ids)
    except (FormatError, OSError) as e:
        print("REJECTED:", e)
        return 1

    n_rows = sum(len(v) for v in preds.values())
    over = [k for k, v in preds.items() if len(v) > MAX_DETS]
    all_scores = [s for v in preds.values() for s in v[:, 1]]
    print("OK: %d detections for %d of %d test images (%.1f MB)"
          % (n_rows, len(preds), len(ids), size_mb))
    if len(preds) < len(ids):
        print("note: %d images have no detections; that is allowed" % (len(ids) - len(preds)))
    if over:
        print("WARNING: %d image(s) have more than %d detections; only the highest %d scores count"
              % (len(over), MAX_DETS, MAX_DETS))
    if all_scores and min(all_scores) > 0.1:
        print("WARNING: your lowest score is %.3f. Keep low-confidence detections (score >= 0.001);"
              " thresholding early lowers mAP." % min(all_scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
