"""
Task 6: Evaluate Test-Time Augmentation (TTA) vs Single-View Inference.
Averages calibrated probabilities over:
1) original, 2) horizontal flip, 3) 88% center crop, 4) 88% center crop flipped.
Measures Accuracy, ECE, Avg Confidence, and single-image inference latency (ms).
"""

import csv
import json
import time
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
REPORTS_DIR = ROOT / "reports"


def load_test_records(manifest_path: Path):
    base = manifest_path.parent
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                items.append((base / row["filepath"], row["label"], row["source"]))
    return items


def make_tta_views(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    orig = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    flip = np.ascontiguousarray(orig[:, ::-1, :])

    ch, cw = max(1, int(h * 0.88)), max(1, int(w * 0.88))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = img_rgb[y0 : y0 + ch, x0 : x0 + cw]
    crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    crop_flip = np.ascontiguousarray(crop_resized[:, ::-1, :])

    return np.stack([orig, flip, crop_resized, crop_flip], axis=0)


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    y_pred = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accs = (y_true == y_pred).astype(float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = boundaries[i], boundaries[i + 1]
        mask = (confs >= low) & (confs <= high) if i == 0 else (confs > low) & (confs <= high)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / len(confs)) * abs(float(np.mean(accs[mask])) - float(np.mean(confs[mask])))
    return float(ece)


def summarize(y_true: np.ndarray, probs: np.ndarray, sources: np.ndarray, latencies_ms: list[float]):
    y_pred = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct = y_true == y_pred
    wrong = ~correct
    per_source = {}
    for src in sorted(set(sources)):
        m = sources == src
        per_source[src] = float(np.mean(correct[m]))
    return {
        "accuracy": float(np.mean(correct)),
        "per_source_accuracy": per_source,
        "ece_15_bins": compute_ece(y_true, probs, n_bins=15),
        "avg_conf_correct": float(np.mean(confs[correct])) if np.any(correct) else 0.0,
        "avg_conf_wrong": float(np.mean(confs[wrong])) if np.any(wrong) else 0.0,
        "single_image_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    l2i = {lbl: i for i, lbl in enumerate(labels)}
    T = float(json.loads(CAL_PATH.read_text(encoding="utf-8")).get("temperature", 1.0))

    model = load_model(MODEL_PATH)
    logit_model = Model(inputs=model.input, outputs=model.get_layer("logits").output)

    records = load_test_records(MANIFEST_PATH)
    y_true = np.array([l2i[lbl] for _, lbl, _ in records], dtype=np.int64)
    sources = np.array([src for _, _, src in records])

    # Warmup
    _ = logit_model.predict(np.zeros((4, 224, 224, 3), dtype=np.float32), verbose=0)

    # Batch evaluate all 1054 images for accuracy/ECE, and measure true single-image latency on first 60 images
    single_probs_all = []
    tta_probs_all = []
    lat_single = []
    lat_tta = []

    print(f"Evaluating Single-View vs 4-View TTA on {len(records)} test images (T={T:.4f})...")
    for idx, (path, _, _) in enumerate(records):
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.asarray(im)
        views = make_tta_views(arr)  # shape (4, 224, 224, 3)

        if idx < 60:
            t0 = time.perf_counter()
            l1 = np.asarray(logit_model.predict(views[:1], verbose=0), dtype=np.float64)
            lat_single.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            l4 = np.asarray(logit_model.predict(views, verbose=0), dtype=np.float64)
            lat_tta.append((time.perf_counter() - t0) * 1000.0)
        else:
            l4 = np.asarray(logit_model.predict(views, verbose=0), dtype=np.float64)
            l1 = l4[:1]

        p1 = softmax(l1 / T, axis=1)[0]
        p4 = np.mean(softmax(l4 / T, axis=1), axis=0)
        single_probs_all.append(p1)
        tta_probs_all.append(p4)

    single_probs = np.vstack(single_probs_all)
    tta_probs = np.vstack(tta_probs_all)

    single_summary = summarize(y_true, single_probs, sources, lat_single)
    tta_summary = summarize(y_true, tta_probs, sources, lat_tta)

    report = {
        "temperature": T,
        "test_samples": len(records),
        "single_view": single_summary,
        "tta_4_views": tta_summary,
        "delta": {
            "accuracy": tta_summary["accuracy"] - single_summary["accuracy"],
            "ece_15_bins": tta_summary["ece_15_bins"] - single_summary["ece_15_bins"],
            "latency_ms": tta_summary["single_image_latency_ms"] - single_summary["single_image_latency_ms"],
        },
    }
    (REPORTS_DIR / "tta_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== TTA COMPARISON REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
