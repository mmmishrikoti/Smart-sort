"""
Module B1: Convert TACO dataset (COCO format) into Ultralytics YOLO format
with normalized bounding boxes and train/val/test splits.
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path
from PIL import Image

try:
    from .taco_mapping import TACO_ID_TO_SORTSMART, SORTSMART_CLASSES, CLASS_TO_ID
except ImportError:
    from taco_mapping import TACO_ID_TO_SORTSMART, SORTSMART_CLASSES, CLASS_TO_ID


def convert_taco_to_yolo(
    taco_root: str = "TACO",
    output_dir: str = "dataset-yolo",
    split_ratios: tuple = (0.75, 0.15, 0.10),
    seed: int = 42,
):
    random.seed(seed)
    taco_path = Path(taco_root)
    anno_file = taco_path / "data" / "annotations.json"
    if not anno_file.is_file():
        raise FileNotFoundError(f"Annotations file not found: {anno_file}")

    out_path = Path(output_dir)

    # Prepare directories
    for split in ["train", "val", "test"]:
        (out_path / split / "images").mkdir(parents=True, exist_ok=True)
        (out_path / split / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Reading annotations from {anno_file}...")
    with open(anno_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}

    # Group annotations by image_id
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Filter images that actually exist on disk and have valid annotations
    valid_image_ids = []
    for img_id, anns in anns_by_img.items():
        img_info = images_by_id.get(img_id)
        if not img_info:
            continue
        full_img_path = taco_path / "data" / img_info["file_name"]
        if full_img_path.is_file():
            valid_image_ids.append(img_id)

    print(f"Found {len(valid_image_ids)} valid annotated images on disk.")
    random.shuffle(valid_image_ids)

    n_total = len(valid_image_ids)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    train_ids = set(valid_image_ids[:n_train])
    val_ids = set(valid_image_ids[n_train : n_train + n_val])
    test_ids = set(valid_image_ids[n_train + n_val :])

    splits_map = {}
    for i in train_ids:
        splits_map[i] = "train"
    for i in val_ids:
        splits_map[i] = "val"
    for i in test_ids:
        splits_map[i] = "test"

    print(f"Split distribution: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test.")

    total_bboxes_written = 0
    class_bbox_counts = {c: 0 for c in SORTSMART_CLASSES}

    for idx, img_id in enumerate(valid_image_ids, 1):
        img_info = images_by_id[img_id]
        split = splits_map[img_id]
        src_path = taco_path / "data" / img_info["file_name"]

        # Read actual image dimensions (fallback to json metadata)
        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            with Image.open(src_path) as im:
                img_w, img_h = im.size

        # Create unique filename base
        safe_name = f"taco_{img_id:05d}"
        dest_img_path = out_path / split / "images" / f"{safe_name}.jpg"
        dest_lbl_path = out_path / split / "labels" / f"{safe_name}.txt"

        # Copy image file
        shutil.copy2(src_path, dest_img_path)

        # Write YOLO formatted labels
        yolo_lines = []
        for ann in anns_by_img.get(img_id, []):
            cat_id = ann["category_id"]
            sortsmart_class = TACO_ID_TO_SORTSMART.get(cat_id)
            if not sortsmart_class:
                continue

            yolo_class_id = CLASS_TO_ID[sortsmart_class]
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            bx, by, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                continue

            # Convert to YOLO normalized format: center_x, center_y, width, height
            x_center = (bx + bw / 2.0) / img_w
            y_center = (by + bh / 2.0) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h

            # Clamp between 0.0 and 1.0
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            class_bbox_counts[sortsmart_class] += 1
            total_bboxes_written += 1

        with open(dest_lbl_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(yolo_lines) + "\n")

        if idx % 200 == 0 or idx == n_total:
            print(f"Exported [{idx}/{n_total}] images to YOLO format...")

    # Write data.yaml configuration for Ultralytics YOLO
    yaml_content = f"""# SortSmart AI - TACO Dataset YAML for Ultralytics YOLO
path: {out_path.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

nc: {len(SORTSMART_CLASSES)}
names: {SORTSMART_CLASSES}
"""
    yaml_path = out_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print("\n" + "=" * 55)
    print("YOLO Conversion Complete!")
    print("=" * 55)
    print(f"Output Directory : {out_path.resolve()}")
    print(f"YAML Config File : {yaml_path.resolve()}")
    print(f"Total Images     : {n_total} (train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)})")
    print(f"Total Bounding Boxes: {total_bboxes_written}")
    print("\nBounding Boxes by Class:")
    for c in SORTSMART_CLASSES:
        print(f"  - {c:<12}: {class_bbox_counts[c]} bboxes")
    print("=" * 55)


if __name__ == "__main__":
    taco_dir = sys.argv[1] if len(sys.argv) > 1 else r"e:\Hackathons\smart sort\TACO"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"e:\Hackathons\smart sort\dataset-yolo"
    convert_taco_to_yolo(taco_dir, out_dir)
