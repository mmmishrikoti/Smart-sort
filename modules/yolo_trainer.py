"""
Module B2: Train YOLOv8 / YOLOv11 on the converted TACO dataset.
"""

import argparse
import sys
from pathlib import Path


def train_yolo(
    data_yaml: str = "dataset-yolo/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "sortsmart_yolo",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[!] Ultralytics is not installed. Run:")
        print("    pip install ultralytics\n")
        sys.exit(1)

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.is_file():
        print(f"[!] data.yaml not found at {yaml_path}")
        print("    Run: python modules/yolo_converter.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"Starting SortSmart YOLO Training")
    print(f"Base Model : {model_name}")
    print(f"Data Config: {yaml_path}")
    print(f"Epochs     : {epochs}")
    print(f"Image Size : {imgsz}")
    print(f"Batch Size : {batch}")
    print("=" * 60)

    # Load pretrained model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        pretrained=True,
        verbose=True,
    )

    print("\nTraining completed successfully!")
    print(f"Best model weights saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on SortSmart TACO dataset.")
    parser.add_argument("--data", default="dataset-yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_yolo(args.data, args.model, args.epochs, args.imgsz, args.batch)
