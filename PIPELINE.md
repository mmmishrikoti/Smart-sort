# SortSmart AI - Modular Waste Detection Pipeline

This directory contains the modular architecture to upgrade SortSmart from single-image TrashNet classification to real-world multi-item detection using the TACO dataset.

---

## Architecture Overview

```text
TACO Dataset (1,500 images, 4,784 annotations)
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[ Module A: Cropping ]    [ Module B: YOLO Converter ]
       │                           │
  dataset-supplemental/       dataset-yolo/
  (4,308 cropped items)       (1,500 split scenes + labels)
       │                           │
       ▼                           ▼
[ Existing Classifier ]   [ YOLOv8 Object Detector ]
MobileNet / EfficientNet   Real-Time Bounding Boxes
```

---

## Modules Directory (`modules/`)

### 1. `modules/taco_mapping.py`
The single source of truth mapping all 60 fine-grained TACO categories into SortSmart's 6 target classes:
* `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

### 2. `modules/crop_classifier.py` (Option A)
* Extracts every bounding-box annotated item with 8% context padding.
* Saves crops directly into `dataset-supplemental/<category>/`.
* **Extracted dataset:** **4,308 real-world crops** (including **918 trash/litter items** to fix the original class imbalance).

**Run extraction again (if needed):**
```powershell
python modules/crop_classifier.py
```

**Train existing classifier with supplemental data:**
```powershell
python train.py --dataset dataset\trashnet-download\dataset-resized --supplemental-dataset dataset-supplemental --epochs 15 --batch-size 32
```

---

### 3. `modules/yolo_converter.py` (Option B)
* Converts TACO COCO bounding boxes into normalized YOLO format: `<class_id> <x_center> <y_center> <width> <height>`.
* Generates `dataset-yolo/` partitioned into:
  * `train/`: 1,125 images
  * `val/`: 225 images
  * `test/`: 150 images
* Generates `dataset-yolo/data.yaml`.

**Run conversion again (if needed):**
```powershell
python modules/yolo_converter.py
```

---

### 4. `modules/yolo_trainer.py`
Trains an Ultralytics YOLOv8/YOLOv11 model on `dataset-yolo/data.yaml`.

**Prerequisites:**
```powershell
pip install ultralytics
```

**Run YOLO Training:**
```powershell
python modules/yolo_trainer.py --model yolov8n.pt --epochs 30 --imgsz 640 --batch 16
```
Best model weights will be saved in `runs/detect/sortsmart_yolo/weights/best.pt`.

---

### 5. `modules/yolo_detector.py`
Production-ready inference class to detect multiple items simultaneously on images or live video, drawing color-coded bounding boxes matching SortSmart's branding:
* **Cardboard:** Amber / Brown
* **Glass:** Teal
* **Metal:** Steel Blue
* **Paper:** Royal Blue
* **Plastic:** Orange
* **Trash:** Crimson Red

**Python Usage:**
```python
from PIL import Image
from modules.yolo_detector import SortSmartYOLODetector

detector = SortSmartYOLODetector("runs/detect/sortsmart_yolo/weights/best.pt")
image = Image.open("sample.jpg")

# 1. Get structured detections
detections = detector.detect(image, conf_threshold=0.25)
print(detections)
# Output: [{'category': 'plastic', 'confidence': 0.89, 'bbox': [102.5, 45.0, 310.2, 540.8], 'class_id': 4}]

# 2. Draw styled boxes
annotated_img = detector.draw_detections(image, detections)
annotated_img.save("result.jpg")
```
