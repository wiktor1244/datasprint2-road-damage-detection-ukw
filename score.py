#!/usr/bin/env python3
"""
Official scorer: Intelligent Road Damage Detection hackathon.

The SAME file is used by teams (on the validation images) and by the organisers (on the hidden test
images), so a score you see locally is computed exactly the way the leaderboard computes it.
Needs only numpy and the Python standard library.

Files (all CSV with a header row; image_id = image file name WITHOUT the extension):
  predictions  image_id,class_id,score,x_min,y_min,x_max,y_max
  ground truth image_id,class_id,x_min,y_min,x_max,y_max
  id list      image_id                      (the images to score; images with no damage have no GT rows)
Boxes are in pixels of the ORIGINAL image. class_id: 0 D00, 1 D10, 2 D20, 3 D40.

Metric definitions (fixed, published in the problem statement):
  * Matching: per image and class, detections are taken in descending score order; each one is matched to
    the unmatched true box of the same class with the highest IoU, if that IoU >= the threshold. A true box
    is matched at most once; every other detection is a false positive.
  * At most 100 detections per image are used (highest scores first).
  * AP: 101-point interpolated average precision (COCO style) per class.
  * mAP@50     = mean over the classes of AP at IoU 0.50.
  * mAP@50:95  = mean over the classes and over IoU 0.50, 0.55, ..., 0.95.
  * Precision / Recall: at IoU 0.50, read at ONE confidence threshold shared by all classes: the threshold
    (grid step 0.001) that maximises the mean F1 over the classes. The scorer finds it; teams do not pick it.
    Precision and recall are the means over the classes. Nothing predicted at a threshold counts as 0.
  * A class with no true box in the scored images is skipped (and a warning is printed).

Usage:
  python score.py --pred submission.csv --gt gt.csv --image-ids ids.csv
  # organisers, full score for one team:
  python score.py --pred submission.csv --gt gt.csv --image-ids private_ids.csv \
                  --stress-ids stress_ids.csv --speed-json speed.json --json-out result.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np

CLASS_NAMES = ["D00_longitudinal_crack", "D10_transverse_crack", "D20_alligator_crack", "D40_pothole"]
NUM_CLASSES = len(CLASS_NAMES)
IOU_THRS = np.round(np.arange(0.50, 0.951, 0.05), 2)  # 10 thresholds: 0.50 ... 0.95
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100
CONF_GRID = np.linspace(0.0, 1.0, 1001)

PRED_COLS = ["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"]
GT_COLS = ["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"]

# Score = 100 * sum(weight * measure), every measure between 0 and 1.
WEIGHTS = {"mAP50": 0.30, "mAP50_95": 0.25, "precision": 0.15, "recall": 0.15, "speed": 0.10, "robustness": 0.05}


class FormatError(Exception):
    """A file does not follow the required format (message says why)."""


# --------------------------------------------------------------------------------------------- reading
def _read_csv(path, required, max_errors=10):
    try:
        f = open(path, newline="", encoding="utf-8-sig")
    except OSError as e:
        raise FormatError("cannot open %s: %s" % (path, e))
    with f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        reader.fieldnames = header
        missing = [c for c in required if c not in header]
        if missing:
            raise FormatError("%s: missing column(s) %s. The header must be: %s"
                              % (path, missing, ",".join(required)))
        rows = []
        for line_no, row in enumerate(reader, start=2):
            rows.append((line_no, row))
    return rows


def _parse_box_row(row, with_score):
    """Return (image_id, class_id, score, x1, y1, x2, y2) or raise ValueError(reason)."""
    iid = (row.get("image_id") or "").strip()
    if not iid:
        raise ValueError("empty image_id")
    try:
        cf = float(row["class_id"])
    except (TypeError, ValueError):
        raise ValueError("class_id is not a number: %r" % row.get("class_id"))
    if not math.isfinite(cf) or cf != int(cf) or not (0 <= int(cf) < NUM_CLASSES):
        raise ValueError("class_id must be 0, 1, 2 or 3 (got %r)" % row.get("class_id"))
    score = 1.0
    if with_score:
        try:
            score = float(row["score"])
        except (TypeError, ValueError):
            raise ValueError("score is not a number: %r" % row.get("score"))
        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
            raise ValueError("score must be between 0 and 1 (got %r)" % row.get("score"))
    try:
        x1, y1, x2, y2 = (float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
    except (TypeError, ValueError):
        raise ValueError("a box coordinate is missing or not a number")
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("a box coordinate is not finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box has zero or negative size (need x_max > x_min and y_max > y_min)")
    return iid, int(cf), score, x1, y1, x2, y2


def _raise_if_errors(path, errors, total_bad):
    if errors:
        more = "" if total_bad <= len(errors) else " (+%d more)" % (total_bad - len(errors))
        raise FormatError("%s: %s%s" % (path, "; ".join(errors), more))


def load_ids(path):
    """Ordered list of image ids from a CSV with an image_id column."""
    rows = _read_csv(path, ["image_id"])
    ids, seen = [], set()
    for line_no, row in rows:
        iid = (row["image_id"] or "").strip()
        if not iid:
            raise FormatError("%s line %d: empty image_id" % (path, line_no))
        if iid in seen:
            raise FormatError("%s line %d: duplicate image_id %r" % (path, line_no, iid))
        seen.add(iid)
        ids.append(iid)
    return ids


def parse_prediction_string(s, with_score=True):
    """'class score x1 y1 x2 y2  class score x1 y1 x2 y2 ...' -> list of (class, score, x1, y1, x2, y2).

    With with_score=False (ground truth) the groups are 'class x1 y1 x2 y2' and the tuples are
    (class, x1, y1, x2, y2). Empty text or the word none means no boxes. Raises ValueError with the reason.
    """
    if s is None:
        return []
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return []
    tok = s.split()
    k = 6 if with_score else 5
    if len(tok) % k:
        raise ValueError("PredictionString must hold groups of %d numbers (class%s x_min y_min x_max y_max); "
                         "it holds %d numbers" % (k, " score" if with_score else "", len(tok)))
    out = []
    for i in range(0, len(tok), k):
        group = tok[i:i + k]
        if with_score:
            row = dict(zip(("class_id", "score", "x_min", "y_min", "x_max", "y_max"), group))
        else:
            row = dict(zip(("class_id", "x_min", "y_min", "x_max", "y_max"), group))
        row["image_id"] = "x"
        _, c, sc, x1, y1, x2, y2 = _parse_box_row(row, with_score)
        out.append((c, sc, x1, y1, x2, y2) if with_score else (c, x1, y1, x2, y2))
    return out


def _load_predictions_by_image(path, valid_ids):
    """The Kaggle upload format: one row per image, columns image_id,PredictionString."""
    rows = _read_csv(path, ["image_id", "PredictionString"])
    out, errors, bad, seen = {}, [], 0, set()
    for line_no, row in rows:
        iid = (row["image_id"] or "").strip()
        problem = None
        if not iid:
            problem = "empty image_id"
        elif iid in seen:
            problem = "image_id %r appears twice (one row per image)" % iid
        elif valid_ids is not None and iid not in valid_ids:
            problem = "image_id %r is not in the list of test images" % iid
        else:
            try:
                recs = parse_prediction_string(row["PredictionString"], True)
            except ValueError as e:
                problem = str(e)
        if problem:
            bad += 1
            if len(errors) < 10:
                errors.append("line %d: %s" % (line_no, problem))
            continue
        seen.add(iid)
        if recs:
            out[iid] = np.asarray(recs, dtype=np.float64).reshape(-1, 6)
    _raise_if_errors(path, errors, bad)
    return out


def load_predictions(path, valid_ids=None):
    """dict image_id -> float array [n, 6] with columns class, score, x1, y1, x2, y2 (file order kept).

    Two layouts are accepted: one row per detection (image_id,class_id,score,x_min,...), and the Kaggle
    upload layout, one row per image (image_id,PredictionString). The layout is read from the header.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = [h.strip() for h in next(csv.reader(f), [])]
    except OSError as e:
        raise FormatError("cannot open %s: %s" % (path, e))
    if "PredictionString" in header:
        return _load_predictions_by_image(path, valid_ids)
    rows = _read_csv(path, PRED_COLS)
    out, errors, bad = defaultdict(list), [], 0
    for line_no, row in rows:
        try:
            iid, c, s, x1, y1, x2, y2 = _parse_box_row(row, True)
        except ValueError as e:
            bad += 1
            if len(errors) < 10:
                errors.append((line_no, "line %d: %s" % (line_no, e)))
            continue
        if valid_ids is not None and iid not in valid_ids:
            bad += 1
            if len(errors) < 10:
                errors.append((line_no, "line %d: image_id %r is not in the list of test images" % (line_no, iid)))
            continue
        out[iid].append((c, s, x1, y1, x2, y2))
    _raise_if_errors(path, [m for _, m in sorted(errors)], bad)
    return {k: np.asarray(v, dtype=np.float64).reshape(-1, 6) for k, v in out.items()}


def load_gt(path):
    """dict image_id -> (boxes [n,4], classes [n])."""
    rows = _read_csv(path, GT_COLS)
    boxes, classes, errors, bad = defaultdict(list), defaultdict(list), [], 0
    for line_no, row in rows:
        try:
            iid, c, _, x1, y1, x2, y2 = _parse_box_row(row, False)
        except ValueError as e:
            bad += 1
            if len(errors) < 10:
                errors.append("line %d: %s" % (line_no, e))
            continue
        boxes[iid].append((x1, y1, x2, y2))
        classes[iid].append(c)
    _raise_if_errors(path, errors, bad)
    return {k: (np.asarray(boxes[k], dtype=np.float64).reshape(-1, 4), np.asarray(classes[k], dtype=np.int64))
            for k in boxes}


# --------------------------------------------------------------------------------------------- metrics
def box_iou(a, b):
    """IoU matrix [len(a), len(b)] for boxes given as x1, y1, x2, y2 (continuous pixel coordinates)."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def _average_precision(tp_sorted, n_gt):
    """101-point interpolated AP for detections already sorted by descending score."""
    if len(tp_sorted) == 0:
        return 0.0
    tps = np.cumsum(tp_sorted)
    fps = np.cumsum(~tp_sorted)
    recall = tps / n_gt
    precision = tps / (tps + fps + np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # make precision non-increasing
    idx = np.searchsorted(recall, REC_THRS, side="left")
    q = np.zeros(len(REC_THRS))
    ok = idx < len(precision)
    q[ok] = precision[idx[ok]]
    return float(q.mean())


def evaluate(gt, preds, image_ids):
    """Score `preds` against `gt` on the images in `image_ids`. Returns a dict of metrics."""
    n_gt = np.zeros(NUM_CLASSES, dtype=np.int64)
    scores = [[] for _ in range(NUM_CLASSES)]
    tps = [[] for _ in range(NUM_CLASSES)]
    empty_boxes, empty_cls = np.zeros((0, 4)), np.zeros(0, dtype=np.int64)

    for iid in image_ids:
        g_boxes, g_cls = gt.get(iid, (empty_boxes, empty_cls))
        d = preds.get(iid)
        if d is None or len(d) == 0:
            d = np.zeros((0, 6))
        elif len(d) > MAX_DETS:  # keep the 100 highest scores of the image
            d = d[np.argsort(-d[:, 1], kind="stable")[:MAX_DETS]]
        for c in range(NUM_CLASSES):
            g = g_boxes[g_cls == c]
            n_gt[c] += len(g)
            dc = d[d[:, 0] == c]
            if len(dc) == 0:
                continue
            dc = dc[np.argsort(-dc[:, 1], kind="stable")]
            tp = np.zeros((len(IOU_THRS), len(dc)), dtype=bool)
            if len(g):
                ious = box_iou(dc[:, 2:6], g)
                for ti, thr in enumerate(IOU_THRS):
                    matched = np.zeros(len(g), dtype=bool)
                    for di in range(len(dc)):
                        cand = np.where(matched, -1.0, ious[di])
                        j = int(np.argmax(cand))
                        if cand[j] >= thr:
                            matched[j] = True
                            tp[ti, di] = True
            scores[c].append(dc[:, 1])
            tps[c].append(tp)

    ap = np.full((NUM_CLASSES, len(IOU_THRS)), np.nan)
    present = [c for c in range(NUM_CLASSES) if n_gt[c] > 0]
    sorted_scores, sorted_tp50 = {}, {}
    for c in range(NUM_CLASSES):
        if n_gt[c] == 0:
            continue
        if scores[c]:
            s = np.concatenate(scores[c])
            tp = np.concatenate(tps[c], axis=1)
            order = np.argsort(-s, kind="stable")
            s, tp = s[order], tp[:, order]
        else:
            s, tp = np.zeros(0), np.zeros((len(IOU_THRS), 0), dtype=bool)
        for ti in range(len(IOU_THRS)):
            ap[c, ti] = _average_precision(tp[ti], n_gt[c])
        sorted_scores[c], sorted_tp50[c] = s, tp[0]

    missing = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if n_gt[c] == 0]
    if missing:
        print("WARNING: no true boxes for %s in the scored images; skipped." % ", ".join(missing), file=sys.stderr)
    if not present:
        raise FormatError("the scored images contain no ground-truth boxes at all")

    map50 = float(np.nanmean(ap[present, 0]))
    map5095 = float(np.nanmean(ap[present, :]))

    # Precision / recall at the single confidence threshold that maximises the mean F1 over classes.
    prec = np.zeros((len(present), len(CONF_GRID)))
    rec = np.zeros_like(prec)
    for i, c in enumerate(present):
        s, tp50 = sorted_scores[c], sorted_tp50[c]
        cum_tp = np.concatenate([[0], np.cumsum(tp50)])
        k = np.searchsorted(-s, -CONF_GRID, side="right")  # detections with score >= threshold
        tp_k = cum_tp[k]
        prec[i] = np.where(k > 0, tp_k / np.maximum(k, 1), 0.0)
        rec[i] = tp_k / n_gt[c]
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.argmax(f1.mean(axis=0)))

    return {
        "n_images": len(image_ids),
        "n_gt_boxes": {CLASS_NAMES[c]: int(n_gt[c]) for c in range(NUM_CLASSES)},
        "mAP50": map50,
        "mAP50_95": map5095,
        "precision": float(prec[:, best].mean()),
        "recall": float(rec[:, best].mean()),
        "f1": float(f1[:, best].mean()),
        "confidence_threshold": float(CONF_GRID[best]),
        "per_class": {CLASS_NAMES[c]: {"AP50": float(ap[c, 0]), "AP50_95": float(np.nanmean(ap[c]))}
                      for c in present},
    }


def combine(main, stress=None, speed_score=None):
    """Combine measures into the leaderboard numbers.

    accuracy_score : 0-100, the available accuracy measures with their weights, scaled to 100
                     (this is what the live leaderboard shows; speed cannot be measured live).
    score          : 0-100, the full formula; only when robustness and speed are both known.
    """
    parts = {"mAP50": main["mAP50"], "mAP50_95": main["mAP50_95"],
             "precision": main["precision"], "recall": main["recall"]}
    if stress is not None:
        parts["robustness"] = stress["mAP50"]
    w = sum(WEIGHTS[k] for k in parts)
    out = {"accuracy_score": 100.0 * sum(WEIGHTS[k] * v for k, v in parts.items()) / w,
           "accuracy_components": sorted(parts), "score": None}
    if stress is not None and speed_score is not None:
        out["score"] = 100.0 * (sum(WEIGHTS[k] * v for k, v in parts.items()) + WEIGHTS["speed"] * speed_score)
    return out


# --------------------------------------------------------------------------------------------- CLI
def _round(o):
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, float):
        return round(o, 6)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions CSV")
    ap.add_argument("--gt", required=True, help="ground-truth CSV")
    ap.add_argument("--image-ids", required=True, help="CSV listing the images to score")
    ap.add_argument("--stress-ids", help="(organisers) CSV listing the degraded images; adds the robustness measure")
    ap.add_argument("--speed-json", help="(organisers) speed.json from benchmark_speed.py; adds the speed measure")
    ap.add_argument("--json-out", help="also write the result to this JSON file")
    a = ap.parse_args(argv)

    try:
        ids = load_ids(a.image_ids)
        preds = load_predictions(a.pred, valid_ids=None)
        gt = load_gt(a.gt)
        result = {"main": evaluate(gt, preds, ids)}
        stress = None
        if a.stress_ids:
            stress = evaluate(gt, preds, load_ids(a.stress_ids))
            result["stress"] = stress
        speed = None
        if a.speed_json:
            with open(a.speed_json) as f:
                speed = json.load(f).get("speed_score")
            result["speed_score"] = speed
        result["combined"] = combine(result["main"], stress, speed)
    except FormatError as e:
        print("SCORING FAILED:", e, file=sys.stderr)
        return 1

    result = _round(result)
    m = result["main"]
    print("images scored      : %d" % m["n_images"])
    print("mAP@50             : %.4f" % m["mAP50"])
    print("mAP@50:95          : %.4f" % m["mAP50_95"])
    print("precision          : %.4f   (confidence threshold %.3f)" % (m["precision"], m["confidence_threshold"]))
    print("recall             : %.4f" % m["recall"])
    for name, v in m["per_class"].items():
        print("  %-24s AP50 %.4f   AP50:95 %.4f" % (name, v["AP50"], v["AP50_95"]))
    if "stress" in result:
        print("robustness (mAP@50 on degraded images): %.4f" % result["stress"]["mAP50"])
    c = result["combined"]
    print("accuracy score     : %.2f / 100  (from %s)" % (c["accuracy_score"], ", ".join(c["accuracy_components"])))
    if c["score"] is not None:
        print("FULL SCORE         : %.2f / 100" % c["score"])
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
