"""
Task 4: Temperature scaling calibration on validation logits.
- Extracts pre-softmax logits from the 'logits' layer on the validation split.
- Optimizes scalar temperature T > 0 by minimizing NLL on validation logits.
- Saves T to model/calibration.json.
- Evaluates ECE (15 bins), NLL, and reliability diagram on the untouched test split.
"""

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize
from scipy.special import softmax
from tensorflow.keras.models import Model, load_model


ROOT = Path(r"e:\Hackathons\smart sort")
MODEL_PATH = ROOT / "model" / "waste_classifier.keras"
LABELS_PATH = ROOT / "model" / "labels.json"
MANIFEST_PATH = ROOT / "dataset-merged" / "manifest.csv"
CALIBRATION_PATH = ROOT / "model" / "calibration.json"
REPORTS_DIR = ROOT / "reports"


def load_labels(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split_records(manifest_path: Path, split_name: str) -> list[tuple[Path, str]]:
    base_dir = manifest_path.parent
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split_name:
                items.append((base_dir / row["filepath"], row["label"]))
    return items


def preprocess_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img)
    return cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)


def extract_logits(logit_model: Model, records: list[tuple[Path, str]], label_to_id: dict, batch_size: int = 32):
    logits_list = []
    y_list = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        imgs = np.stack([preprocess_image(p) for p, _ in chunk], axis=0)
        batch_logits = np.asarray(logit_model.predict(imgs, verbose=0), dtype=np.float64)
        logits_list.append(batch_logits)
        y_list.extend(label_to_id[lbl] for _, lbl in chunk)
    return np.vstack(logits_list), np.array(y_list, dtype=np.int64)


def nll_loss(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    probs = softmax(scaled, axis=1)
    eps = 1e-12
    true_probs = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(true_probs)))


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15):
    y_pred = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accs = (y_true == y_pred).astype(float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        low, high = boundaries[i], boundaries[i + 1]
        mask = (confs >= low) & (confs <= high) if i == 0 else (confs > low) & (confs <= high)
        cnt = int(np.sum(mask))
        if cnt > 0:
            b_acc = float(np.mean(accs[mask]))
            b_conf = float(np.mean(confs[mask]))
            ece += (cnt / len(confs)) * abs(b_acc - b_conf)
        else:
            b_acc = 0.0
            b_conf = float((low + high) / 2.0)
        bins.append({"low": float(low), "high": float(high), "count": cnt, "acc": b_acc, "conf": b_conf})
    return float(ece), bins


def plot_before_after_reliability(bins_before, ece_before, bins_after, ece_after, T: float, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, bins, ece, title in [
        (axes[0], bins_before, ece_before, f"Before Calibration (T=1.000, ECE={ece_before:.4f})"),
        (axes[1], bins_after, ece_after, f"After Calibration (T={T:.3f}, ECE={ece_after:.4f})"),
    ]:
        centers = [(b["low"] + b["high"]) / 2.0 for b in bins]
        accs = [b["acc"] for b in bins]
        width = 1.0 / len(bins)
        ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.75, label="Observed Accuracy")
        ax.plot([0, 1], [0, 1], "--", color="red", label="Perfect Calibration")
        nz_conf = [b["conf"] for b in bins if b["count"] > 0]
        nz_acc = [b["acc"] for b in bins if b["count"] > 0]
        if nz_conf:
            ax.plot(nz_conf, nz_acc, marker="o", color="navy", linewidth=1.5, label="Bin Mean")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS_PATH)
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}

    model = load_model(MODEL_PATH)
    logit_model = Model(inputs=model.input, outputs=model.get_layer("logits").output)

    val_records = load_split_records(MANIFEST_PATH, "val")
    test_records = load_split_records(MANIFEST_PATH, "test")

    print(f"Extracting validation logits ({len(val_records)} images)...")
    val_logits, y_val = extract_logits(logit_model, val_records, label_to_id)

    val_nll_before = nll_loss(val_logits, y_val, 1.0)
    val_probs_before = softmax(val_logits, axis=1)
    val_ece_before, _ = compute_ece(y_val, val_probs_before, n_bins=15)

    # Optimize log_T so T = exp(log_T) > 0
    res = minimize(
        lambda log_t: nll_loss(val_logits, y_val, float(np.exp(log_t[0]))),
        x0=[0.0],
        method="L-BFGS-B",
        bounds=[(-2.5, 2.5)],
    )
    T = float(np.exp(res.x[0]))
    val_nll_after = nll_loss(val_logits, y_val, T)
    val_probs_after = softmax(val_logits / T, axis=1)
    val_ece_after, _ = compute_ece(y_val, val_probs_after, n_bins=15)

    calibration_data = {
        "temperature": T,
        "fitted_on": "val",
        "val_samples": int(len(y_val)),
        "val_nll_before": val_nll_before,
        "val_nll_after": val_nll_after,
        "val_ece_before": val_ece_before,
        "val_ece_after": val_ece_after,
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration_data, indent=2), encoding="utf-8")
    print(f"Saved temperature T={T:.4f} to {CALIBRATION_PATH}")

    print(f"Evaluating on untouched test split ({len(test_records)} images)...")
    test_logits, y_test = extract_logits(logit_model, test_records, label_to_id)

    test_probs_before = softmax(test_logits, axis=1)
    test_probs_after = softmax(test_logits / T, axis=1)

    test_nll_before = nll_loss(test_logits, y_test, 1.0)
    test_nll_after = nll_loss(test_logits, y_test, T)
    test_ece_before, bins_before = compute_ece(y_test, test_probs_before, n_bins=15)
    test_ece_after, bins_after = compute_ece(y_test, test_probs_after, n_bins=15)

    y_pred = np.argmax(test_probs_after, axis=1)
    correct_mask = y_test == y_pred
    wrong_mask = ~correct_mask
    confs_before = np.max(test_probs_before, axis=1)
    confs_after = np.max(test_probs_after, axis=1)

    plot_before_after_reliability(
        bins_before,
        test_ece_before,
        bins_after,
        test_ece_after,
        T,
        REPORTS_DIR / "reliability_diagram_calibrated.png",
    )

    report = {
        "temperature": T,
        "val": {
            "nll_before": val_nll_before,
            "nll_after": val_nll_after,
            "ece_before": val_ece_before,
            "ece_after": val_ece_after,
        },
        "test": {
            "samples": int(len(y_test)),
            "accuracy": float(np.mean(correct_mask)),
            "nll_before": test_nll_before,
            "nll_after": test_nll_after,
            "ece_before": test_ece_before,
            "ece_after": test_ece_after,
            "avg_conf_correct_before": float(np.mean(confs_before[correct_mask])),
            "avg_conf_wrong_before": float(np.mean(confs_before[wrong_mask])),
            "avg_conf_correct_after": float(np.mean(confs_after[correct_mask])),
            "avg_conf_wrong_after": float(np.mean(confs_after[wrong_mask])),
        },
    }
    (REPORTS_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== CALIBRATION REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
