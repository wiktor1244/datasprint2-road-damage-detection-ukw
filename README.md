# ANVESHAN DataSprint 2.0 - Intelligent Road Damage Detection (Team UKW)

Reproducible computer vision pipeline for automated detection and classification of road surface damage across four classes (`D00` longitudinal crack, `D10` transverse crack, `D20` alligator crack, `D40` pothole) using **YOLOv8m** initialized exclusively from generic COCO weights.

## Repository Contents
- `train_baseline.py` - converts bounding boxes to normalized YOLO format, generates `data.yaml`, and trains the YOLOv8 detector (`imgsz=640`, horizontal flip `fliplr=0.5`, no vertical flip `flipud=0.0`).
- `inference.py` - runs single-image inference (`conf=0.001`, non-TTA) and returns class IDs, confidence scores, and bounding box coordinates.
- `benchmark_speed.py` - measures single-image inference latency and verifies compliance with the T4 GPU speed target (`<= 25 ms/image`, `speed_score = 1.0`).
- `score.py` - computes validation `mAP@50`, `mAP@50:95`, per-class metrics, and composite competition score (`2,150` validation images).
- `make_kaggle_submission.py` - formats test predictions into Kaggle `submission.csv` (`4,200` test images, `none` token for damage-free images).
- `check_submission.py` - validates `submission.csv` schema, row count, `image_id` alignment, and prediction formatting.
- `writeup.pdf` / `presentation.pdf` - technical write-up (2 pages) and final presentation slides.

## Setup
```bash
pip install -r requirements.txt
python train_baseline.py
python benchmark_speed.py
python score.py --pred val_pred.csv --gt val_labels.csv --image-ids val_image_ids.csv --speed-json val_speed.json
python make_kaggle_submission.py
python check_submission.py --pred submission.csv --image-ids test_image_ids.csv


```

## Summary of Results
- YOLOv8n (iou=0.5) - Validation (2,150): 41.86 (mAP@50: 0.4910, mAP@50:95: 0.2082, speed_score: 1.0, med 12.163 ms)
- YOLOv8m (40 ep) - Validation (2,150): 44.53 (mAP@50: 0.5339, mAP@50:95: 0.2239, speed_score: 1.0, med 14.953 ms, P: 0.5837, R: 0.4989)
- YOLOv8n (iou=0.5) - Kaggle Test (4,200): 40.44137 (speed_score: 1.0, med 10.996 ms, 92,329 boxes, 127 none)
- YOLOv8m (40 ep) - Kaggle Test (4,200): 40.32541 (speed_score: 1.0, status: OK, 125,608 boxes, 127 none)

### Key Experimental Findings
- TTA Edge Case: Enabling test-time augmentation (augment=True at conf=0.001) produced a degenerate zero-width box (x_min = x_max = 720.0) on img_005002.jpg; TTA was disabled in favor of deterministic single-pass inference.
- Post-Processing NMS: Applying an extra post-processing NMS (iou=0.50) on test predictions removed 38.66% of boxes (48,559 boxes) and lowered Public Score from 40.32541 to 39.93711, so the unpruned model output was retained.

## Reproducibility & Compliance
- Pretrained Weights: Generic COCO weights (yolov8m.pt / yolov8n.pt) only; no road-damage pretrained weights were used.
- Pinned Commit: <INSERT_PINNED_COMMIT_HASH>
- Compliance: No external road datasets, manual test annotations, output hard-coding, or leaderboard probing were used.
