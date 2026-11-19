"""
Module A: Crop TACO items using bounding box annotations and organize them
into category folders for SortSmart's image classifier (MobileNet / EfficientNet).
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter
from PIL import Image

# Import shared mapping
try:
    from .taco_mapping import TACO_ID_TO_SORTSMART, SORTSMART_CLASSES
except ImportError:
    from taco_mapping import TACO_ID_TO_SORTSMART, SORTSMART_CLASSES


def crop_taco_dataset(
    taco_root: str = "TACO",
    output_dir: str = "dataset-supplemental",
    min_size: int = 25,
    padding_pct: float = 0.08,
):
    taco_path = Path(taco_root)
    anno_file = taco_path / "data" / "annotations.json"
    if not anno_file.is_file():
        raise FileNotFoundError(f"Annotations file not found: {anno_file}")

    out_path = Path(output_dir)
    for cls_name in SORTSMART_CLASSES:
        (out_path / cls_name).mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {anno_file}...")
    with open(anno_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build image lookup: id -> image info
    images_by_id = {img["id"]: img for img in data["images"]}

    print(f"Found {len(images_by_id)} images and {len(data['annotations'])} annotations.")

    stats = Counter()
    skipped_too_small = 0
    skipped_missing_img = 0

    # Group annotations by image for efficient file opening
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    total_images = len(anns_by_img)
    for idx, (img_id, anns) in enumerate(anns_by_img.items(), 1):
        img_info = images_by_id.get(img_id)
        if not img_info:
            continue

        rel_path = img_info["file_name"]
        full_img_path = taco_path / "data" / rel_path

        if not full_img_path.is_file():
            skipped_missing_img += len(anns)
            continue

        try:
            with Image.open(full_img_path) as full_img:
                img_w, img_h = full_img.size

                for ann in anns:
                    cat_id = ann["category_id"]
                    target_class = TACO_ID_TO_SORTSMART.get(cat_id)
                    if not target_class:
                        continue

                    bbox = ann.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue

                    bx, by, bw, bh = bbox
                    if bw < min_size or bh < min_size:
                        skipped_too_small += 1
                        continue

                    # Add padding around the object
                    pad_x = bw * padding_pct
                    pad_y = bh * padding_pct

                    x1 = max(0, int(bx - pad_x))
                    y1 = max(0, int(by - pad_y))
                    x2 = min(img_w, int(bx + bw + pad_x))
                    y2 = min(img_h, int(by + bh + pad_y))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = full_img.crop((x1, y1, x2, y2))
                    if crop.mode != "RGB":
                        crop = crop.convert("RGB")

                    # Generate unique filename
                    ann_id = ann["id"]
                    out_filename = f"taco_img{img_id}_ann{ann_id}.jpg"
                    dest_file = out_path / target_class / out_filename

                    crop.save(dest_file, format="JPEG", quality=90)
                    stats[target_class] += 1

        except Exception as e:
            print(f"Warning: Error processing {full_img_path}: {e}")

        if idx % 100 == 0 or idx == total_images:
            print(f"Processed [{idx}/{total_images}] images... Total crops: {sum(stats.values())}")

    print("\n" + "=" * 50)
    print("Crop Extraction Summary:")
    print("=" * 50)
    for cls_name in SORTSMART_CLASSES:
        print(f"  - {cls_name:<12}: {stats.get(cls_name, 0)} items")
    print("-" * 50)
    print(f"Total crops saved: {sum(stats.values())}")
    print(f"Skipped (too small < {min_size}px): {skipped_too_small}")
    if skipped_missing_img > 0:
        print(f"Skipped (missing image file): {skipped_missing_img}")
    print(f"Saved into: {out_path.resolve()}")
    print("=" * 50)
    return stats


if __name__ == "__main__":
    taco_dir = sys.argv[1] if len(sys.argv) > 1 else r"e:\Hackathons\smart sort\TACO"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"e:\Hackathons\smart sort\dataset-supplemental"
    crop_taco_dataset(taco_dir, out_dir)
