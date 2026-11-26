"""
Task 5: Derive confidence and margin thresholds from calibrated validation predictions.
- Finds the confidence threshold achieving >=95% precision on accepted validation samples.
- Derives medium confidence threshold (~80% precision) and margin threshold.
- Saves thresholds to model/thresholds.json and reports coverage on val & untouched test.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.special import softmax
from tensorflow.keras.models import Model, load_model


ROOT = Path(r"e:\Hackathons\smart sort")
MODEL_PATH = ROOT / "model" / "waste_classifier.keras"
LABELS_PATH = ROOT / "model" / "labels.json"
CAL_PATH = ROOT / "model" / "calibration.json"
MANIFEST_PATH = ROOT / "dataset-merged" / "manifest.csv"
THRESHOLDS_PATH = ROOT / "model" / "thresholds.json"
REPORTS_DIR = ROOT / "reports"


def load_split_records(manifest_path: Path, split_name: str):
    base = manifest_path.parent
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split_name:
                items.append((base / row["filepath"], row["label"]))
    return items


def preprocess_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img)
    return cv2.resize(arr, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def get_calibrated_probs(logit_model: Model, records, label_to_id: dict, temperature: float, batch_size: int = 32):
    probs_list = []
    y_list = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        imgs = np.stack([preprocess_image(p) for p, _ in chunk], axis=0)
        logits = np.asarray(logit_model.predict(imgs, verbose=0), dtype=np.float64)
        probs = softmax(logits / temperature, axis=1)
        probs_list.append(probs)
        y_list.extend(label_to_id[lbl] for _, lbl in chunk)
    return np.vstack(probs_list), np.array(y_list, dtype=np.int64)


def sweep_thresholds(probs: np.ndarray, y_true: np.ndarray):
    y_pred = np.argmax(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margins = top1 - top2
    correct = (y_pred == y_true)

    curve = []
    for thr in np.round(np.arange(0.30, 0.99, 0.01), 2):
        mask = top1 >= thr
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        prec = float(np.mean(correct[mask]))
        cov = float(cnt / len(y_true))
        curve.append({"threshold": float(thr), "accepted": cnt, "precision": prec, "coverage": cov})

    # Pick lowest threshold achieving >= 0.95 precision with at least 10 accepted samples
    valid_95 = [c for c in curve if c["precision"] >= 0.95 and c["accepted"] >= 10]
    if valid_95:
        best_95 = max(valid_95, key=lambda c: c["coverage"])
    else:
        # Fallback: highest precision threshold with >= 10 samples
        candidates = [c for c in curve if c["accepted"] >= 10]
        best_95 = max(candidates, key=lambda c: (c["precision"], c["coverage"]))

    # Pick medium threshold achieving >= 0.80 precision
    valid_80 = [c for c in curve if c["precision"] >= 0.80 and c["accepted"] >= 20]
    best_80 = max(valid_80, key=lambda c: c["coverage"]) if valid_80 else {"threshold": 0.60, "precision": 0.0, "coverage": 0.0}

    # Margin threshold: 75th percentile of wrong predictions' margin (or 0.15 fallback)
    wrong_margins = margins[~correct]
    margin_thr = float(np.round(np.percentile(wrong_margins, 65), 3)) if len(wrong_margins) > 0 else 0.15
    margin_thr = float(np.clip(margin_thr, 0.10, 0.30))

    return best_95, best_80, margin_thr, curve


def eval_at_thresholds(probs: np.ndarray, y_true: np.ndarray, high_thr: float, med_thr: float, margin_thr: float):
    y_pred = np.argmax(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margins = top1 - top2
    correct = (y_pred == y_true)

    high_mask = (top1 >= high_thr) & (margins >= margin_thr)
    med_mask = (~high_mask) & (top1 >= med_thr) & (margins >= margin_thr * 0.5)
    low_mask = ~(high_mask | med_mask)
    accepted_95_mask = top1 >= high_thr

    return {
        "samples": int(len(y_true)),
        "raw_accuracy": float(np.mean(correct)),
        "precision_at_high_conf_only": float(np.mean(correct[accepted_95_mask])) if np.any(accepted_95_mask) else 0.0,
        "coverage_at_high_conf_only": float(np.mean(accepted_95_mask)),
        "high_band": {
            "count": int(np.sum(high_mask)),
            "coverage": float(np.mean(high_mask)),
            "precision": float(np.mean(correct[high_mask])) if np.any(high_mask) else 0.0,
        },
        "medium_band": {
            "count": int(np.sum(med_mask)),
            "coverage": float(np.mean(med_mask)),
            "precision": float(np.mean(correct[med_mask])) if np.any(med_mask) else 0.0,
        },
        "low_band": {
            "count": int(np.sum(low_mask)),
            "coverage": float(np.mean(low_mask)),
            "precision": float(np.mean(correct[low_mask])) if np.any(low_mask) else 0.0,
        },
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    temperature = float(json.loads(CAL_PATH.read_text(encoding="utf-8")).get("temperature", 1.0))

    model = load_model(MODEL_PATH)
    logit_model = Model(inputs=model.input, outputs=model.get_layer("logits").output)

    val_records = load_split_records(MANIFEST_PATH, "val")
    test_records = load_split_records(MANIFEST_PATH, "test")

    print(f"Evaluating calibrated probabilities (T={temperature:.4f}) on val ({len(val_records)})...")
    val_probs, y_val = get_calibrated_probs(logit_model, val_records, label_to_id, temperature)
    best_95, best_80, margin_thr, curve = sweep_thresholds(val_probs, y_val)

    high_thr = best_95["threshold"]
    med_thr = min(best_80["threshold"], high_thr - 0.05)

    val_eval = eval_at_thresholds(val_probs, y_val, high_thr, med_thr, margin_thr)

    print(f"Evaluating frozen thresholds on untouched test ({len(test_records)})...")
    test_probs, y_test = get_calibrated_probs(logit_model, test_records, label_to_id, temperature)
    test_eval = eval_at_thresholds(test_probs, y_test, high_thr, med_thr, margin_thr)

    thresholds_cfg = {
        "high_confidence_threshold": high_thr,
        "medium_confidence_threshold": round(med_thr, 2),
        "margin_threshold": margin_thr,
        "val_target_precision": 0.95,
        "val_precision_at_high": best_95["precision"],
        "val_coverage_at_high": best_95["coverage"],
    }
    THRESHOLDS_PATH.write_text(json.dumps(thresholds_cfg, indent=2), encoding="utf-8")

    report = {
        "thresholds": thresholds_cfg,
        "val_eval": val_eval,
        "test_eval": test_eval,
        "val_curve": curve,
    }
    (REPORTS_DIR / "thresholds_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== THRESHOLDS DERIVED FROM VAL ===")
    print(json.dumps(thresholds_cfg, indent=2))
    print("\n=== TEST EVAL AT FROZEN THRESHOLDS ===")
    print(json.dumps(test_eval, indent=2))


if __name__ == "__main__":
    main()
