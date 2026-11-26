"""
Task 1: Audit the current SortSmart MobileNetV2 classifier.
- Evaluates on held-out TrashNet test split (seed=42) and TACO crops.
- Computes ECE (15 bins), reliability diagram, avg confidence (correct vs wrong), latency.
- Saves top 5 confusions (up to 20 images each) to reports/errors/.
"""

import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.models import load_model


ROOT = Path(r"e:\Hackathons\smart sort")
MODEL_PATH = ROOT / "model" / "waste_classifier.keras"
LABELS_PATH = ROOT / "model" / "labels.json"
TRASHNET_DIR = ROOT / "dataset" / "trashnet-download" / "dataset-resized"
SUPPLEMENTAL_DIR = ROOT / "dataset-supplemental"
REPORTS_DIR = ROOT / "reports"
ERRORS_DIR = REPORTS_DIR / "errors"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_labels(labels_path: Path):
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        raise ValueError("Invalid labels.json")
    return labels


def get_trashnet_test_split(dataset_dir: Path, labels: list[str], seed: int = 42):
    items = []
    for label in labels:
        class_dir = dataset_dir / label
        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        random.Random(seed).shuffle(images)
        test_end = max(1, round(len(images) * 0.15))
        for p in images[:test_end]:
            items.append((p, label, "trashnet"))
    return items


def get_taco_crops(supp_dir: Path, labels: list[str]):
    items = []
    for label in labels:
        class_dir = supp_dir / label
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and p.name.startswith("taco_"):
                items.append((p, label, "taco_crop"))
    return items


def preprocess_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img)
    resized = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return resized


def predict_batch(model, items: list[tuple[Path, str, str]], labels: list[str], batch_size: int = 32):
    label_to_id = {name: i for i, name in enumerate(labels)}
    y_true = []
    y_pred = []
    confs = []
    probs_all = []
    paths = []
    sources = []
    latencies_ms = []

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        imgs = np.stack([preprocess_image(p) for p, _, _ in chunk], axis=0)
        t0 = time.perf_counter()
        preds = np.asarray(model.predict(imgs, verbose=0))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = dt_ms / len(chunk)

        for (p, lbl, src), prob in zip(chunk, preds):
            pred_idx = int(np.argmax(prob))
            conf = float(prob[pred_idx])
            y_true.append(label_to_id[lbl])
            y_pred.append(pred_idx)
            confs.append(conf)
            probs_all.append(prob)
            paths.append(p)
            sources.append(src)
            latencies_ms.append(per_item_ms)

    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "confs": np.array(confs),
        "probs": np.array(probs_all),
        "paths": paths,
        "sources": sources,
        "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
    }


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, confs: np.ndarray, n_bins: int = 15):
    accuracies = (y_true == y_pred).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs > low) & (confs <= high)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (count / len(confs)) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = float((low + high) / 2.0)
        bin_stats.append({
            "bin": i,
            "low": float(low),
            "high": float(high),
            "count": count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
        })
    return float(ece), bin_stats


def plot_reliability_diagram(bin_stats: list[dict], ece: float, out_path: Path, title_suffix: str = ""):
    confs = [b["confidence"] for b in bin_stats if b["count"] > 0]
    accs = [b["accuracy"] for b in bin_stats if b["count"] > 0]
    centers = [(b["low"] + b["high"]) / 2.0 for b in bin_stats]
    all_accs = [b["accuracy"] for b in bin_stats]
    width = 1.0 / len(bin_stats)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.bar(centers, all_accs, width=width * 0.9, edgecolor="black", alpha=0.75, label="Observed Accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect Calibration")
    if confs:
        ax.plot(confs, accs, marker="o", color="navy", linewidth=1.5, label="Bin Mean (Conf vs Acc)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram {title_suffix} (ECE = {ece:.4f})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_metrics(res: dict, labels: list[str]):
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    confs = res["confs"]
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), zero_division=0
    )
    ece, bin_stats = compute_ece(y_true, y_pred, confs, n_bins=15)

    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "ece_15_bins": ece,
        "avg_conf_correct": float(np.mean(confs[correct_mask])) if np.any(correct_mask) else 0.0,
        "avg_conf_wrong": float(np.mean(confs[wrong_mask])) if np.any(wrong_mask) else 0.0,
        "avg_latency_ms": res["avg_latency_ms"],
        "per_class": {
            lbl: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        "bin_stats": bin_stats,
    }


def save_top_confusions(res: dict, labels: list[str], errors_dir: Path, top_k: int = 5, max_per_pair: int = 20):
    if errors_dir.exists():
        shutil.rmtree(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    y_true = res["y_true"]
    y_pred = res["y_pred"]
    confs = res["confs"]
    paths = res["paths"]

    pair_items = {}
    for idx, (t, p, c, path) in enumerate(zip(y_true, y_pred, confs, paths)):
        if t != p:
            pair = (labels[t], labels[p])
            pair_items.setdefault(pair, []).append((c, path))

    sorted_pairs = sorted(pair_items.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_k]
    top_confusions_report = []

    for (true_lbl, pred_lbl), items in sorted_pairs:
        # Sort highest-confidence mistakes first so worst overconfident errors are visible
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        pair_dir = errors_dir / f"{true_lbl}_as_{pred_lbl}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for conf, src_path in items_sorted[:max_per_pair]:
            dest_name = f"conf{int(round(conf * 100)):03d}_{src_path.name}"
            shutil.copy2(src_path, pair_dir / dest_name)
            saved += 1
        top_confusions_report.append({
            "true": true_lbl,
            "predicted": pred_lbl,
            "count": len(items),
            "saved_samples": saved,
            "avg_confidence": float(np.mean([c for c, _ in items])),
        })
    return top_confusions_report


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS_PATH)
    model = load_model(MODEL_PATH)

    trashnet_test = get_trashnet_test_split(TRASHNET_DIR, labels, seed=42)
    taco_crops = get_taco_crops(SUPPLEMENTAL_DIR, labels)

    print(f"Evaluating TrashNet held-out test set ({len(trashnet_test)} images)...")
    res_trashnet = predict_batch(model, trashnet_test, labels)
    metrics_trashnet = summarize_metrics(res_trashnet, labels)

    print(f"Evaluating TACO crops ({len(taco_crops)} images)...")
    res_taco = predict_batch(model, taco_crops, labels)
    metrics_taco = summarize_metrics(res_taco, labels)

    plot_reliability_diagram(
        metrics_trashnet["bin_stats"],
        metrics_trashnet["ece_15_bins"],
        REPORTS_DIR / "reliability_diagram.png",
        "(TrashNet Held-Out Test)",
    )
    plot_reliability_diagram(
        metrics_taco["bin_stats"],
        metrics_taco["ece_15_bins"],
        REPORTS_DIR / "reliability_diagram_taco.png",
        "(TACO Crops)",
    )

    # Save top 5 confusions from held-out TrashNet test set, and also from TACO crops
    top_conf_trashnet = save_top_confusions(res_trashnet, labels, ERRORS_DIR / "trashnet_test", top_k=5, max_per_pair=20)
    top_conf_taco = save_top_confusions(res_taco, labels, ERRORS_DIR / "taco_crops", top_k=5, max_per_pair=20)

    report = {
        "labels": labels,
        "trashnet_held_out_test": metrics_trashnet,
        "taco_crops_note": "TACO crops were part of dataset-supplemental during the previous training run; true held-out TACO evaluation will be established in Task 2.",
        "taco_crops": metrics_taco,
        "top_5_confusions_trashnet": top_conf_trashnet,
        "top_5_confusions_taco": top_conf_taco,
    }

    (REPORTS_DIR / "audit_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== AUDIT SUMMARY (TrashNet Held-Out Test) ===")
    print(f"Accuracy      : {metrics_trashnet['accuracy']:.4f}")
    print(f"Macro F1      : {metrics_trashnet['macro_f1']:.4f}")
    print(f"ECE (15 bins) : {metrics_trashnet['ece_15_bins']:.4f}")
    print(f"Avg Conf (OK) : {metrics_trashnet['avg_conf_correct']:.4f}")
    print(f"Avg Conf (Err): {metrics_trashnet['avg_conf_wrong']:.4f}")
    print(f"Latency (ms)  : {metrics_trashnet['avg_latency_ms']:.2f} ms/img")
    print("Top 5 Confusions (TrashNet):", json.dumps(top_conf_trashnet, indent=2))
    print("\n=== AUDIT SUMMARY (TACO Crops) ===")
    print(f"Accuracy      : {metrics_taco['accuracy']:.4f}")
    print(f"Macro F1      : {metrics_taco['macro_f1']:.4f}")
    print(f"ECE (15 bins) : {metrics_taco['ece_15_bins']:.4f}")
    print(f"Avg Conf (OK) : {metrics_taco['avg_conf_correct']:.4f}")
    print(f"Avg Conf (Err): {metrics_taco['avg_conf_wrong']:.4f}")
    print("Top 5 Confusions (TACO):", json.dumps(top_conf_taco, indent=2))


if __name__ == "__main__":
    main()
