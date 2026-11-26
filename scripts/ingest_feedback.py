"""
Task 9: Ingest reviewed user feedback from feedback/feedback_log.csv into
dataset-merged/train/ and manifest.csv for the next training run.
Checks perceptual hashes against existing manifest entries so no duplicates
or val/test leaks are introduced.
"""

import csv
import json
import shutil
from pathlib import Path

import imagehash
from PIL import Image, ImageOps


ROOT = Path(r"e:\Hackathons\smart sort")
FEEDBACK_DIR = ROOT / "feedback"
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_log.csv"
MERGED_DIR = ROOT / "dataset-merged"
MANIFEST_PATH = MERGED_DIR / "manifest.csv"
LABELS_PATH = ROOT / "model" / "labels.json"


def main():
    if not FEEDBACK_LOG.is_file():
        print(f"No feedback log found at {FEEDBACK_LOG}. Nothing to ingest.")
        return
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run modules/dataset_merge.py first.")

    valid_labels = set(json.loads(LABELS_PATH.read_text(encoding="utf-8")))

    existing_hashes = set()
    existing_paths = set()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing_hashes.add(row["phash"])
            existing_paths.add(row["filepath"])

    added = 0
    skipped_unreviewed = 0
    skipped_dup = 0

    new_rows = []
    with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("consent") != "1" or row.get("reviewed") != "1":
                skipped_unreviewed += 1
                continue
            label = row["corrected_category"].strip().lower()
            if label not in valid_labels:
                continue
            src = FEEDBACK_DIR / row["rel_path"]
            if not src.is_file():
                continue

            with Image.open(src) as im:
                ph = str(imagehash.phash(ImageOps.exif_transpose(im).convert("RGB")))
            if ph in existing_hashes:
                skipped_dup += 1
                continue

            dest_name = f"feedback_{src.name}"
            rel_dest = f"train/{label}/{dest_name}"
            if rel_dest in existing_paths:
                skipped_dup += 1
                continue

            dest = MERGED_DIR / rel_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

            existing_hashes.add(ph)
            existing_paths.add(rel_dest)
            new_rows.append([rel_dest, label, "user_feedback", f"fb_{ph}", ph, "train"])
            added += 1

    if new_rows:
        with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)

    print(f"Feedback ingestion complete: added={added}, skipped_dup={skipped_dup}, skipped_unreviewed={skipped_unreviewed}")


if __name__ == "__main__":
    main()
