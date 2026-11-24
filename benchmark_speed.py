#!/usr/bin/env python3
"""
Speed benchmark + reproducibility run. The organisers run this for every finalist on the SAME machine
(one reference GPU, nothing else running), so every team is measured the same way.

    python benchmark_speed.py --repo path/to/team_repo --images path/to/test_images \
        --out-csv reproduced_submission.csv --out-json speed.json

What it does
  1. imports <repo>/inference.py and calls load(device) once (not timed);
  2. warms the model up on a few images (not timed);
  3. run 1 : runs predict() on EVERY image, one image at a time (batch size 1), timing each call;
     the detections are written to --out-csv, so the same run gives the reproduced predictions that are
     scored against the uploaded file;
  4. further runs : times the first --repeat-images images again, to smooth out noise;
  5. reports median / mean / p95 milliseconds per image and the speed score.

Timing covers predict() only: pre-processing, the model, test-time augmentation, post-processing and NMS
are all inside it and all counted. Decoding the JPEG from disk is done by this harness, outside the timer,
and is the same for everyone.

Speed score (0 to 1): 1.0 at or below --fast-ms; 0.0 at or above --slow-ms; in between it falls in
a straight line on a log scale (so each doubling of the time costs the same).

Exit status: 0 = finished (status OK); 1 = the repo failed (error / time limit). The JSON says why.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import sys
import time

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png")
PRED_COLS = ["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"]
MAX_DETS = 100

try:  # optional: only used to synchronise the GPU and to report the device
    import torch
except Exception:  # pragma: no cover
    torch = None


def speed_score(ms: float, fast_ms: float = 25.0, slow_ms: float = 250.0) -> float:
    if ms <= fast_ms:
        return 1.0
    if ms >= slow_ms:
        return 0.0
    return (math.log(slow_ms) - math.log(ms)) / (math.log(slow_ms) - math.log(fast_ms))


def _sync(device):
    if torch is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _clean(detections, image_shape):
    """Validate what predict() returned; keep the 100 best. Raises ValueError with the reason."""
    out = []
    for d in detections:
        if len(d) != 6:
            raise ValueError("each detection must be (class_id, score, x_min, y_min, x_max, y_max)")
        c, s, x1, y1, x2, y2 = d
        c = int(c)
        vals = [float(s), float(x1), float(y1), float(x2), float(y2)]
        if not (0 <= c <= 3) or not all(math.isfinite(v) for v in vals):
            raise ValueError("bad detection %r" % (d,))
        if not (0.0 <= vals[0] <= 1.0):
            raise ValueError("score outside 0-1: %r" % (d,))
        if vals[3] <= vals[1] or vals[4] <= vals[2]:
            raise ValueError("box with zero or negative size: %r" % (d,))
        out.append((c, *vals))
    out.sort(key=lambda t: -t[1])
    return out[:MAX_DETS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="team repo (contains inference.py at its root)")
    ap.add_argument("--images", required=True, help="folder with the test images")
    ap.add_argument("--out-csv", default="reproduced_submission.csv")
    ap.add_argument("--out-json", default="speed.json")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=3, help="total passes; passes after the first use --repeat-images")
    ap.add_argument("--repeat-images", type=int, default=300)
    ap.add_argument("--max-minutes", type=float, default=30.0, help="time limit for the first (full) pass")
    ap.add_argument("--fast-ms", type=float, default=25.0)
    ap.add_argument("--slow-ms", type=float, default=250.0)
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    img_dir = os.path.abspath(a.images)
    out_csv, out_json = os.path.abspath(a.out_csv), os.path.abspath(a.out_json)
    device = a.device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")

    info = {"status": "OK", "error": None, "device": device,
            "gpu_name": (torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None),
            "torch": getattr(torch, "__version__", None), "python": platform.python_version(),
            "platform": platform.platform(), "warmup": a.warmup, "runs": a.runs,
            "fast_ms": a.fast_ms, "slow_ms": a.slow_ms, "max_minutes": a.max_minutes}

    def finish(status, error=None, **extra):
        info.update(status=status, error=error, **extra)
        info.setdefault("speed_score", None)
        with open(out_json, "w") as f:
            json.dump(info, f, indent=2)
        print(json.dumps({k: info[k] for k in ("status", "error", "n_images", "median_ms", "mean_ms",
                                               "p95_ms", "speed_score") if k in info}, indent=2))
        return 0 if status == "OK" else 1

    paths = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    if not paths:
        return finish("ERROR", "no images found in %s" % img_dir)
    info["n_images"] = len(paths)

    # --- import the team's code
    module_path = os.path.join(repo, "inference.py")
    if not os.path.isfile(module_path):
        return finish("ERROR", "inference.py not found at the root of the repo")
    sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        spec = importlib.util.spec_from_file_location("team_inference", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn in ("load", "predict"):
            if not callable(getattr(mod, fn, None)):
                return finish("ERROR", "inference.py has no function %s()" % fn)
        t0 = time.perf_counter()
        model = mod.load(device)
        info["load_seconds"] = round(time.perf_counter() - t0, 3)
    except Exception as e:  # noqa: BLE001
        return finish("ERROR", "load() failed: %s: %s" % (type(e).__name__, e))

    def decode(p):
        return np.asarray(Image.open(p).convert("RGB"))

    def timed(img):
        _sync(device)
        t = time.perf_counter()
        out = mod.predict(model, img)
        _sync(device)
        return out, (time.perf_counter() - t) * 1000.0

    # --- warm-up (not timed)
    try:
        for i in range(a.warmup):
            mod.predict(model, decode(paths[i % len(paths)]))
    except Exception as e:  # noqa: BLE001
        return finish("ERROR", "predict() failed during warm-up: %s: %s" % (type(e).__name__, e))
    if torch is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- timed passes
    latencies, rows, first_pass_ms = [], [], 0.0
    for run in range(max(1, a.runs)):
        subset = paths if run == 0 else paths[: a.repeat_images]
        for p in subset:
            try:
                img = decode(p)
                out, ms = timed(img)
                clean = _clean(out, img.shape)
            except Exception as e:  # noqa: BLE001
                return finish("ERROR", "predict() failed on %s: %s: %s" % (os.path.basename(p), type(e).__name__, e))
            latencies.append(ms)
            if run == 0:
                first_pass_ms += ms
                iid = os.path.splitext(os.path.basename(p))[0]
                rows.extend([iid, c, "%.5f" % s, "%.2f" % x1, "%.2f" % y1, "%.2f" % x2, "%.2f" % y2]
                            for c, s, x1, y1, x2, y2 in clean)
                if first_pass_ms > a.max_minutes * 60_000:
                    return finish("TIMEOUT", "the full pass took longer than %.0f minutes" % a.max_minutes,
                                  images_done=len(rows))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PRED_COLS)
        w.writerows(rows)

    lat = np.asarray(latencies)
    med = float(np.median(lat))
    extra = {"median_ms": round(med, 3), "mean_ms": round(float(lat.mean()), 3),
             "p95_ms": round(float(np.percentile(lat, 95)), 3), "fps_median": round(1000.0 / med, 2),
             "timed_calls": int(len(lat)), "first_pass_seconds": round(first_pass_ms / 1000.0, 2),
             "speed_score": round(speed_score(med, a.fast_ms, a.slow_ms), 4),
             "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                                 if torch is not None and str(device).startswith("cuda")
                                 and torch.cuda.is_available() else None)}
    return finish("OK", None, **extra)


if __name__ == "__main__":
    sys.exit(main())
