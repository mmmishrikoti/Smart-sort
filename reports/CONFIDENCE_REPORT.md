# SortSmart AI — Confidence & Calibration Improvement Report

## Executive Summary

We upgraded SortSmart AI from an uncalibrated MobileNetV2 classifier trained with ungrouped splits to a leak-free, temperature-calibrated, TTA-enabled, hybrid YOLO + MobileNetV2 pipeline with validation-derived rejection thresholds, an optional Vision-LLM fallback, and an opt-in user feedback loop.

---

## Before vs. After Comparison

> **Scope Note on Test Sets:**
> - **Before (Task 1):** Evaluated on TrashNet-only held-out test (`379` studio images); TACO crops were previously mixed into training, so no true held-out TACO score existed.
> - **After (Tasks 2–6):** Evaluated on the strict, perceptual-hash-deduplicated, source-image-grouped held-out test set (`1,054` images: `376` TrashNet + `678` unseen TACO crops) with `T = 0.9380` temperature scaling and 4-view TTA (`USE_TTA=True`).

| Metric | Before (Task 1 Baseline) | After (Task 6 Calibrated + TTA) |
|---|---|---|
| **Test accuracy** | 85.49% *(TrashNet-only, 379)* | **66.03%** *(Merged 1,054)* / **79.26%** *(TrashNet)* |
| **Macro F1** | 82.47% *(TrashNet-only)* | **65.68%** *(Merged, single-view)* |
| **ECE (15 bins)** | 0.0566 | **0.0256** *(-54.8% reduction)* |
| **Avg confidence (correct)** | 93.60% | **71.70%** *(label-smoothed + T-scaled)* |
| **Avg confidence (wrong)** | 73.80% | **51.86%** *(19.8-pt gap)* |
| **Coverage @ 95% precision** | Not measured (uncalibrated) | **17.98%** on val (95.24% prec) / **18.60%** on test (92.86% prec) |
| **Real-world (TACO) accuracy** | N/A *(71.15% in-sample)* | **58.70%** *(strictly unseen TACO test, 678 imgs)* |
| **Inference latency (ms)** | 19.43 ms/img *(batched)* | **165.06 ms/img** *(single-image 4-view TTA; 157.98 ms single-view)* |

---

## Acceptance Criteria Check

1. **ECE after calibration is lower than before:** **Passed** (`0.0566` $\rightarrow$ `0.0256` on test; `0.0372` $\rightarrow$ `0.0358` on val).
2. **Average confidence on wrong predictions is clearly lower than on correct ones:** **Passed** (`51.86%` wrong vs `71.70%` correct — nearly a 20-point separation).
3. **No split contains crops from the same source photo as another split:** **Passed** (`group_overlap_across_splits = 0` in `reports/dataset_summary.json`).
4. **Accuracy on the TACO/real-world subset:** Established a true leak-free TACO benchmark (`58.70%` on 678 unseen TACO crops) and verified hybrid YOLO + calibrated CNN rescues YOLO misclassifications on real-world photos (e.g., tea cup rescued from YOLO `plastic` `72.7%` to `glass` `77.9%` with `needs_review: true`).
5. **All existing endpoints still work:** **Passed** (`GET /api/status`, `POST /api/predict`, `GET /api/stats`, plus new `POST /api/detect` and `POST /api/feedback`).

---

## Artifacts Generated

- **Reliability Diagrams:** `reports/reliability_diagram.png`, `reports/reliability_diagram_taco.png`, `reports/reliability_diagram_calibrated.png`
- **Error Analysis:** `reports/errors/` (top 5 confusion pairs, up to 20 images each)
- **Label Issues:** `reports/label_issues.csv` (2,293 Cleanlab-flagged items for human review; none auto-deleted)
- **Configs & Reports:** `model/train_config.json`, `model/calibration.json`, `model/thresholds.json`, `reports/audit_metrics.json`, `reports/dataset_summary.json`, `reports/calibration_report.json`, `reports/thresholds_report.json`, `reports/tta_report.json`
