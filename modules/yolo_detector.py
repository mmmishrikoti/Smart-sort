"""
Module B3: Real-Time / Image Object Detector using trained YOLO model.
Detects multiple waste items in an image and draws color-coded bounding boxes.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Color palette for SortSmart classes (RGB)
CLASS_COLORS = {
    "cardboard": (180, 115, 60),   # Brown / Amber
    "glass":     (0, 168, 150),    # Teal / Cyan
    "metal":     (120, 140, 160),  # Steel Blue / Gray
    "paper":     (52, 120, 246),   # Royal Blue
    "plastic":   (245, 140, 44),   # Orange
    "trash":     (220, 53, 69),    # Crimson Red
}


class SortSmartYOLODetector:
    def __init__(self, model_path: Union[str, Path] = "runs/detect/sortsmart_yolo/weights/best.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.is_file():
            print(f"[!] Warning: YOLO weights not found at {self.model_path}")
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            print(f"[+] Loaded YOLO detector from {self.model_path}")
        except Exception as e:
            print(f"[!] Failed to load YOLO model: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def detect(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Runs object detection on the input image.
        Returns a list of detected objects:
        [
            {
                "category": "plastic",
                "confidence": 0.88,
                "bbox": [x1, y1, x2, y2],  # pixel coordinates
                "class_id": 4
            },
            ...
        ]
        """
        if not self.is_ready():
            raise RuntimeError("YOLO model is not loaded. Train a model or provide valid weights.")

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = first_res.names

        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            category_name = names.get(cls_id, str(cls_id))

            detections.append({
                "category": category_name,
                "confidence": round(conf, 4),
                "bbox": [round(coord, 1) for coord in xyxy],
                "class_id": cls_id,
            })

        return detections

    def draw_detections(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        box_thickness: int = 3,
    ) -> Image.Image:
        """
        Draws stylish bounding boxes and labels on a PIL Image.
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            category = det["category"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(category, (40, 167, 69))

            # Draw outer box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

            # Label text
            label_text = f"{category.upper()} {int(conf * 100)}%"

            # Calculate text background banner
            text_bbox = draw.textbbox((x1, y1), label_text)
            text_w = text_bbox[2] - text_bbox[0] + 8
            text_h = text_bbox[3] - text_bbox[1] + 6

            banner_y1 = max(0, y1 - text_h)
            banner_y2 = y1 if y1 >= text_h else y1 + text_h

            draw.rectangle([x1, banner_y1, x1 + text_w, banner_y2], fill=color)
            draw.text((x1 + 4, banner_y1 + 2), label_text, fill=(255, 255, 255))

        return annotated


if __name__ == "__main__":
    import sys
    print("SortSmart YOLO Detector module ready.")
    if len(sys.argv) > 2:
        weights = sys.argv[1]
        img_file = sys.argv[2]
        detector = SortSmartYOLODetector(weights)
        if detector.is_ready():
            img = Image.open(img_file)
            res = detector.detect(img)
            print(f"Detections: {res}")
            annotated = detector.draw_detections(img, res)
            annotated.save("detected_output.jpg")
            print("Saved output to detected_output.jpg")
