import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance

def generate_cup_variations():
    src_path = Path(r"e:\Hackathons\smart sort\uploads\8fd08e223f9846839e7db14b3b1b5c2a.jpg")
    if not src_path.is_file():
        raise FileNotFoundError(f"Source cup image not found at {src_path}")

    dest_dir = Path(r"e:\Hackathons\smart sort\dataset-supplemental\glass")
    dest_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    # The cup bounding box in the original image
    # (x1, y1, x2, y2)
    cup_bbox = (120, 250, 820, 1020)
    tight_bbox = (150, 280, 780, 950)
    wide_bbox = (50, 150, 880, 1200)

    crops = [
        ("full", img),
        ("cup_standard", img.crop(cup_bbox)),
        ("cup_tight", img.crop(tight_bbox)),
        ("cup_wide", img.crop(wide_bbox)),
    ]

    saved_count = 0

    for name, base_crop in crops:
        for flip in [False, True]:
            current = base_crop.transpose(Image.FLIP_LEFT_RIGHT) if flip else base_crop

            for rot in [-8, 0, 8]:
                if rot != 0:
                    rotated = current.rotate(rot, resample=Image.BICUBIC, expand=False)
                else:
                    rotated = current

                for bright in [0.9, 1.0, 1.1]:
                    enhancer = ImageEnhance.Brightness(rotated)
                    adjusted = enhancer.enhance(bright)

                    enhancer_c = ImageEnhance.Contrast(adjusted)
                    final_img = enhancer_c.enhance(random.uniform(0.95, 1.1))

                    flip_str = "flip" if flip else "orig"
                    fname = f"glass_cup_{name}_{flip_str}_r{rot}_b{int(bright*100)}_{saved_count:03d}.jpg"
                    out_path = dest_dir / fname
                    final_img.save(out_path, format="JPEG", quality=92)
                    saved_count += 1

    print(f"Successfully generated and saved {saved_count} glass cup training variations into {dest_dir}!")

if __name__ == "__main__":
    generate_cup_variations()
