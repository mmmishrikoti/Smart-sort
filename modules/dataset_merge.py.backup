"""
Task 2: Source-aware dataset merging, perceptual-hash deduplication,
stratified group 70/15/15 splitting, and Cleanlab label-issue detection.

Generates:
- dataset-merged/manifest.csv (with source, group_id, phash, split)
- dataset-merged/{train,val,test}/{class}/
- reports/label_issues.csv (suspected mislabeled items; never auto-deleted)
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import imagehash
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from cleanlab.filter import find_label_issues


ROOT = Path(r"e:\Hackathons\smart sort")
LABELS_PATH = ROOT / "model" / "labels.json"
TRASHNET_DIR = ROOT / "dataset" / "trashnet-download" / "dataset-resized"
SUPPLEMENTAL_DIR = ROOT / "dataset-supplemental"
MERGED_DIR = ROOT / "dataset-merged"
REPORTS_DIR = ROOT / "reports"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Optional label mapping for extra datasets (RealWaste, Garbage Classification 12-class,
# Drinking Waste, e-waste, etc.) -> canonical SortSmart classes.
EXTRA_LABEL_MAPPINGS = {
    # TrashNet / standard aliases
    "cardboard": "cardboard",
    "corrugated": "cardboard",
    "carton": "cardboard",
    "glass": "glass",
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "metal": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "tin": "metal",
    "cans": "metal",
    "paper": "paper",
    "newspaper": "paper",
    "plastic": "plastic",
    "pet": "plastic",
    "hdpe": "plastic",
    "trash": "trash",
    "biological": "trash",
    "organic": "trash",
    "food organics": "trash",
    "miscellaneous trash": "trash",
    "textile trash": "trash",
    "clothes": "trash",
    "shoes": "trash",
    "battery": "trash",
    "e-waste": "trash",
    # TODO: If an 'other' (reject) dataset is provided, map out-of-scope folders to 'other'
    # and add 'other' to model/labels.json.
}

TACO_IMG_RE = re.compile(r"^taco_img(\d+)_ann(\d+)\.")


def load_labels(labels_path: Path = LABELS_PATH) -> list[str]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Invalid labels file: {labels_path}")
    return labels


def compute_phash(path: Path) -> tuple[str, int]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        area = img.width * img.height
        h = str(imagehash.phash(img))
    return h, area


def collect_records(
    labels: list[str],
    trashnet_dir: Path = TRASHNET_DIR,
    supplemental_dir: Path = SUPPLEMENTAL_DIR,
    extra_dirs: list[Path] | None = None,
    other_dir: Path | None = None,
) -> list[dict]:
    records = []
    valid_set = set(labels)

    # 1. TrashNet
    if trashnet_dir.is_dir():
        for lbl in labels:
            cdir = trashnet_dir / lbl
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                    records.append({
                        "src_path": p,
                        "label": lbl,
                        "source": "trashnet",
                        "group_id": f"trashnet_{p.stem}",
                        "force_train": False,
                    })

    # 2. TACO crops + custom cup variations in dataset-supplemental
    if supplemental_dir.is_dir():
        for lbl in labels:
            cdir = supplemental_dir / lbl
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.rglob("*")):
                if not (p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES):
                    continue
                m = TACO_IMG_RE.match(p.name)
                if m:
                    taco_img_id = m.group(1)
                    records.append({
                        "src_path": p,
                        "label": lbl,
                        "source": "taco",
                        "group_id": f"taco_img_{taco_img_id}",
                        "force_train": False,
                    })
                elif p.name.startswith("glass_cup_"):
                    # Synthetic variations of one user-provided cup photo: keep grouped in train only
                    records.append({
                        "src_path": p,
                        "label": lbl,
                        "source": "custom_cup",
                        "group_id": "custom_cup_001",
                        "force_train": True,
                    })
                else:
                    records.append({
                        "src_path": p,
                        "label": lbl,
                        "source": "supplemental",
                        "group_id": f"supp_{p.stem}",
                        "force_train": False,
                    })

    # 3. Optional extra datasets (RealWaste, Garbage-12, Drinking Waste, etc.)
    for ed in extra_dirs or []:
        if not ed.is_dir():
            print(f"[!] Extra dataset dir not found, skipping: {ed}")
            continue
        source_name = ed.name.lower().replace(" ", "_")
        for sub in sorted(ed.iterdir()):
            if not sub.is_dir():
                continue
            mapped = EXTRA_LABEL_MAPPINGS.get(sub.name.strip().lower())
            if not mapped or mapped not in valid_set:
                continue
            for p in sorted(sub.rglob("*")):
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                    records.append({
                        "src_path": p,
                        "label": mapped,
                        "source": source_name,
                        "group_id": f"{source_name}_{p.stem}",
                        "force_train": False,
                    })

    # 4. Optional 'other' (reject) class
    # TODO: Provide out-of-scope images (hands, tables, floors, people, empty backgrounds)
    # via --other-dir and add 'other' to model/labels.json when ready.
    if other_dir and other_dir.is_dir() and "other" in valid_set:
        for p in sorted(other_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                records.append({
                    "src_path": p,
                    "label": "other",
                    "source": "other_reject",
                    "group_id": f"other_{p.stem}",
                    "force_train": False,
                })

    return records


def deduplicate_by_phash(records: list[dict], hamming_threshold: int = 2) -> tuple[list[dict], int]:
    """
    Computes perceptual hashes and removes exact/near-duplicate real images
    (keeping the largest-resolution copy). Synthetic 'force_train' augmentations
    are hashed for manifest logging but not deduplicated against each other.
    """
    print(f"Computing perceptual hashes for {len(records)} images...")
    for r in records:
        ph, area = compute_phash(r["src_path"])
        r["phash"] = ph
        r["phash_obj"] = imagehash.hex_to_hash(ph)
        r["area"] = area

    kept = []
    removed = 0
    seen_by_exact = {}
    # Sort descending by resolution so we keep the sharpest image on collision
    real_records = sorted([r for r in records if not r["force_train"]], key=lambda x: x["area"], reverse=True)
    aug_records = [r for r in records if r["force_train"]]

    kept_hashes_by_source = defaultdict(list)
    for r in real_records:
        ph = r["phash"]
        ph_obj = r["phash_obj"]
        src = r["source"]

        # Exact hash duplicate across any source
        if ph in seen_by_exact:
            removed += 1
            continue

        # Near-duplicate (Hamming <= threshold) within same source & label
        is_near_dup = False
        for existing_obj, existing_lbl in kept_hashes_by_source[src]:
            if existing_lbl == r["label"] and (ph_obj - existing_obj) <= hamming_threshold:
                is_near_dup = True
                break
        if is_near_dup:
            removed += 1
            continue

        seen_by_exact[ph] = r
        kept_hashes_by_source[src].append((ph_obj, r["label"]))
        kept.append(r)

    kept.extend(aug_records)
    print(f"Deduplication complete: kept {len(kept)}, removed {removed} exact/near-duplicates.")
    return kept, removed


def assign_source_aware_splits(records: list[dict], seed: int = 42) -> list[dict]:
    """
    Assigns train (~70%), val (~15%), test (~15%) splits grouped by group_id
    and stratified by (source, label) so crops from the same TACO source image
    never appear in more than one split.
    """
    rng = np.random.default_rng(seed)
    group_to_records = defaultdict(list)
    for r in records:
        group_to_records[r["group_id"]].append(r)

    # Separate force_train groups (e.g. synthetic cup variations)
    normal_groups = []
    for gid, grecs in group_to_records.items():
        if any(r["force_train"] for r in grecs):
            for r in grecs:
                r["split"] = "train"
        else:
            # Stratify key: primary (source, label) of the group
            lbl_counts = Counter((r["source"], r["label"]) for r in grecs)
            primary_key = lbl_counts.most_common(1)[0][0]
            normal_groups.append((gid, grecs, primary_key))

    # Group by (source, label) and split groups 70 / 15 / 15
    by_stratum = defaultdict(list)
    for gid, grecs, key in normal_groups:
        by_stratum[key].append((gid, grecs))

    for key, g_list in sorted(by_stratum.items()):
        idx = rng.permutation(len(g_list))
        shuffled = [g_list[i] for i in idx]
        n = len(shuffled)
        n_test = max(1, round(n * 0.15)) if n >= 3 else 0
        n_val = max(1, round(n * 0.15)) if n >= 5 else 0
        # Ensure at least 1 group stays in train
        while n_test + n_val >= n and (n_test > 0 or n_val > 0):
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        for i, (gid, grecs) in enumerate(shuffled):
            if i < n_test:
                sp = "test"
            elif i < n_test + n_val:
                sp = "val"
            else:
                sp = "train"
            for r in grecs:
                r["split"] = sp

    return records


def materialize_merged_dataset(records: list[dict], labels: list[str], merged_dir: Path):
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    for sp in ("train", "val", "test"):
        for lbl in labels:
            (merged_dir / sp / lbl).mkdir(parents=True, exist_ok=True)

    manifest_path = merged_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "source", "group_id", "phash", "split"])
        for idx, r in enumerate(records):
            dest_name = f"{r['source']}_{idx:05d}{r['src_path'].suffix.lower()}"
            dest_path = merged_dir / r["split"] / r["label"] / dest_name
            shutil.copy2(r["src_path"], dest_path)
            r["merged_path"] = dest_path
            rel = dest_path.relative_to(merged_dir).as_posix()
            writer.writerow([rel, r["label"], r["source"], r["group_id"], r["phash"], r["split"]])

    print(f"Materialized merged dataset & manifest at {manifest_path}")


def run_cleanlab_audit(records: list[dict], labels: list[str], out_csv: Path, seed: int = 42):
    """
    Extracts frozen MobileNetV2 embeddings, runs 5-fold StratifiedGroupKFold
    LogisticRegression for out-of-fold probabilities, and invokes cleanlab
    to flag suspected mislabeled samples without deleting anything.
    """
    import cv2
    import tensorflow as tf

    print("Extracting MobileNetV2 embeddings for Cleanlab out-of-fold audit...")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}

    # Only audit real images (skip synthetic force_train augmentations)
    audit_recs = [r for r in records if not r["force_train"]]
    X_list = []
    y_list = []
    groups = []

    batch_size = 64
    for start in range(0, len(audit_recs), batch_size):
        chunk = audit_recs[start : start + batch_size]
        imgs = []
        for r in chunk:
            with Image.open(r["merged_path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                arr = cv2.resize(np.asarray(im), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
            imgs.append(arr)
        batch_arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.stack(imgs, axis=0))
        feats = base.predict(batch_arr, verbose=0)
        X_list.append(feats)
        for r in chunk:
            y_list.append(label_to_id[r["label"]])
            groups.append(r["group_id"])

    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(groups)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros((len(y), len(labels)), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X[tr_idx], y[tr_idx])
        oof_probs[va_idx] = clf.predict_proba(X[va_idx])

    issue_indices = find_label_issues(
        labels=y,
        pred_probs=oof_probs,
        return_indices_ranked_by="self_confidence",
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "filepath", "source", "split", "given_label",
            "predicted_label", "given_prob", "predicted_prob"
        ])
        for rank, idx in enumerate(issue_indices, 1):
            r = audit_recs[idx]
            given_id = y[idx]
            pred_id = int(np.argmax(oof_probs[idx]))
            writer.writerow([
                rank,
                r["merged_path"].relative_to(ROOT).as_posix(),
                r["source"],
                r["split"],
                labels[given_id],
                labels[pred_id],
                f"{oof_probs[idx, given_id]:.4f}",
                f"{oof_probs[idx, pred_id]:.4f}",
            ])

    print(f"Cleanlab flagged {len(issue_indices)} suspected label issues -> {out_csv}")
    return len(issue_indices)


def print_and_save_summary(records: list[dict], labels: list[str], dedup_removed: int, label_issues_count: int):
    by_split_class = defaultdict(Counter)
    by_source_class = defaultdict(Counter)
    groups_by_split = defaultdict(set)

    for r in records:
        by_split_class[r["split"]][r["label"]] += 1
        by_source_class[r["source"]][r["label"]] += 1
        groups_by_split[r["split"]].add(r["group_id"])

    # Verify zero group overlap across splits
    tr_g = groups_by_split["train"]
    va_g = groups_by_split["val"]
    te_g = groups_by_split["test"]
    overlap = len(tr_g & va_g) + len(tr_g & te_g) + len(va_g & te_g)

    summary = {
        "total_images": len(records),
        "dedup_removed": dedup_removed,
        "group_overlap_across_splits": overlap,
        "cleanlab_flagged_issues": label_issues_count,
        "counts_by_split": {sp: dict(by_split_class[sp]) for sp in ("train", "val", "test")},
        "counts_by_source": {src: dict(cnt) for src, cnt in sorted(by_source_class.items())},
    }
    (REPORTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== MERGED DATASET SUMMARY ===")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Merge, deduplicate, split, and audit SortSmart datasets.")
    parser.add_argument("--extra-dirs", nargs="*", default=[], help="Optional extra dataset directories.")
    parser.add_argument("--other-dir", default=None, help="Optional out-of-scope 'other' reject class directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = load_labels(LABELS_PATH)
    extra_paths = [Path(d) for d in args.extra_dirs]
    other_path = Path(args.other_dir) if args.other_dir else None

    records = collect_records(labels, TRASHNET_DIR, SUPPLEMENTAL_DIR, extra_paths, other_path)
    records, removed = deduplicate_by_phash(records, hamming_threshold=2)
    records = assign_source_aware_splits(records, seed=args.seed)
    materialize_merged_dataset(records, labels, MERGED_DIR)
    issues_count = run_cleanlab_audit(records, labels, REPORTS_DIR / "label_issues.csv", seed=args.seed)
    print_and_save_summary(records, labels, removed, issues_count)


if __name__ == "__main__":
    main()
