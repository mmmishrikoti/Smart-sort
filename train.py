import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Invalid labels file: {labels_path}")
    return labels


def load_manifest_splits(manifest_path: Path, class_names: list[str]):
    """Load pre-computed source-aware splits from dataset-merged/manifest.csv."""
    base_dir = manifest_path.parent
    valid = set(class_names)
    splits = {lbl: {"train": [], "val": [], "test": [], "sources": {}} for lbl in class_names}
    source_records = {"train": [], "val": [], "test": []}

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["label"]
            sp = row["split"]
            src = row["source"]
            if lbl not in valid or sp not in ("train", "val", "test"):
                continue
            full_path = base_dir / row["filepath"]
            splits[lbl][sp].append(full_path)
            source_records[sp].append((full_path, lbl, src))
    return splits, source_records


def _gaussian_blur_3x3(image: tf.Tensor) -> tf.Tensor:
    kernel = tf.constant(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=tf.float32
    ) / 16.0
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(
        tf.expand_dims(image, 0), kernel, strides=[1, 1, 1, 1], padding="SAME"
    )
    return tf.squeeze(blurred, 0)


def _random_erasing(image: tf.Tensor, max_frac: float = 0.2) -> tf.Tensor:
    h, w = IMAGE_SIZE
    eh = tf.random.uniform([], minval=int(h * 0.08), maxval=int(h * max_frac), dtype=tf.int32)
    ew = tf.random.uniform([], minval=int(w * 0.08), maxval=int(w * max_frac), dtype=tf.int32)
    y0 = tf.random.uniform([], minval=0, maxval=h - eh, dtype=tf.int32)
    x0 = tf.random.uniform([], minval=0, maxval=w - ew, dtype=tf.int32)

    yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
    mask = tf.logical_and(
        tf.logical_and(yy >= y0, yy < y0 + eh),
        tf.logical_and(xx >= x0, xx < x0 + ew),
    )
    mask = tf.expand_dims(mask, -1)
    fill_val = tf.random.uniform([1, 1, 3], minval=0.0, maxval=255.0, dtype=tf.float32)
    return tf.where(mask, tf.broadcast_to(fill_val, [h, w, 3]), image)


def augment_image(image: tf.Tensor) -> tf.Tensor:
    # 1. Random resized crop
    scale = tf.random.uniform([], minval=0.75, maxval=1.0)
    crop_h = tf.cast(tf.cast(IMAGE_SIZE[0], tf.float32) * scale, tf.int32)
    crop_w = tf.cast(tf.cast(IMAGE_SIZE[1], tf.float32) * scale, tf.int32)
    # Resize slightly larger before random crop
    image = tf.image.resize(image, [256, 256])
    crop_h = tf.cast(256.0 * scale, tf.int32)
    crop_w = tf.cast(256.0 * scale, tf.int32)
    image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    image = tf.image.resize(image, IMAGE_SIZE)

    # 2. Horizontal flip
    image = tf.image.random_flip_left_right(image)

    # 3. Color jitter (brightness, contrast, saturation, hue)
    image = tf.image.random_brightness(image, max_delta=25.0)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.clip_by_value(image, 0.0, 255.0)

    # 4. Probabilistic Gaussian blur (p=0.2)
    if tf.random.uniform([]) < 0.2:
        image = _gaussian_blur_3x3(image)

    # 5. Probabilistic Random Erasing (p=0.25)
    if tf.random.uniform([]) < 0.25:
        image = _random_erasing(image)

    return tf.clip_by_value(image, 0.0, 255.0)


def apply_mixup(images: tf.Tensor, labels: tf.Tensor, alpha: float = 0.2):
    """Apply MixUp regularization probabilistically on a training batch."""
    if tf.random.uniform([]) >= 0.35:
        return images, labels
    batch_size = tf.shape(images)[0]
    g1 = tf.random.gamma([batch_size], alpha)
    g2 = tf.random.gamma([batch_size], alpha)
    lam = g1 / (g1 + g2 + 1e-8)
    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam_x * images + (1.0 - lam_x) * tf.gather(images, indices)
    mixed_labels = lam_y * labels + (1.0 - lam_y) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def make_dataset(records: list[tuple[Path, str, str]], class_names: list[str], batch_size: int, training: bool, use_mixup: bool = True):
    label_ids = {name: index for index, name in enumerate(class_names)}
    files = [str(p) for p, _, _ in records]
    labels = [label_ids[lbl] for _, lbl, _ in records]

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))

    def load_image(path, label):
        contents = tf.io.read_file(path)
        image = tf.io.decode_image(contents, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        if training:
            image = augment_image(image)
        return image, tf.one_hot(label, depth=len(class_names))

    if training:
        dataset = dataset.shuffle(min(len(files), 4096), seed=42, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    if training and use_mixup:
        dataset = dataset.map(lambda x, y: apply_mixup(x, y, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(class_count: int, backbone: str = "mobilenetv2", label_smoothing: float = 0.08):
    # Mild geometric rotation inside model graph; main augmentations run in tf.data
    geo_aug = tf.keras.Sequential(
        [tf.keras.layers.RandomRotation(0.08)],
        name="geo_augmentation",
    )

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    x = geo_aug(inputs)

    if backbone == "mobilenetv2":
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_rescale")(x)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
        )
    elif backbone == "efficientnetv2s":
        # EfficientNetV2S expects [0, 255] when include_preprocessing=True (default)
        base = tf.keras.applications.EfficientNetV2S(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    logits = tf.keras.layers.Dense(class_count, name="logits")(x)
    outputs = tf.keras.layers.Activation("softmax", name="predictions")(logits)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model, base


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, confs: np.ndarray, n_bins: int = 15) -> float:
    accuracies = (y_true == y_pred).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs >= low) & (confs <= high) if i == 0 else (confs > low) & (confs <= high)
        count = int(np.sum(mask))
        if count > 0:
            ece += (count / len(confs)) * abs(float(np.mean(accuracies[mask])) - float(np.mean(confs[mask])))
    return float(ece)


def evaluate_records(model, records: list[tuple[Path, str, str]], class_names: list[str], batch_size: int):
    ds = make_dataset(records, class_names, batch_size, training=False, use_mixup=False)
    truth = []
    probs = []
    for images, labels in ds:
        truth.extend(np.argmax(labels.numpy(), axis=1))
        probs.extend(model.predict(images, verbose=0))

    y_true = np.array(truth)
    probs = np.array(probs)
    y_pred = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask

    # Per-source accuracy
    per_source = {}
    sources_arr = np.array([src for _, _, src in records])
    for src in sorted(set(sources_arr)):
        mask = sources_arr == src
        per_source[src] = {
            "samples": int(np.sum(mask)),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
        }

    return {
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1_score": float(np.mean(f1)),
        "ece_15_bins": compute_ece(y_true, y_pred, confs, n_bins=15),
        "avg_conf_correct": float(np.mean(confs[correct_mask])) if np.any(correct_mask) else 0.0,
        "avg_conf_wrong": float(np.mean(confs[wrong_mask])) if np.any(wrong_mask) else 0.0,
        "per_source": per_source,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
            }
            for index, label in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist(),
        "class_order": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the SortSmart waste classifier.")
    parser.add_argument("--manifest", default="dataset-merged/manifest.csv", help="Path to source-aware split manifest.")
    parser.add_argument("--labels", default="model/labels.json", help="Path to canonical labels.json.")
    parser.add_argument("--output", default="model", help="Folder for the trained model and evaluation files.")
    parser.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "efficientnetv2s"])
    parser.add_argument("--phase1-epochs", type=int, default=8)
    parser.add_argument("--phase2-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mixup", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    labels_path = Path(args.labels)
    class_names = load_labels(labels_path)
    manifest_path = Path(args.manifest)
    splits, source_records = load_manifest_splits(manifest_path, class_names)

    train_records = source_records["train"]
    val_records = source_records["val"]
    test_records = source_records["test"]
    print(f"Loaded manifest: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    train_data = make_dataset(train_records, class_names, args.batch_size, training=True, use_mixup=not args.no_mixup)
    val_data = make_dataset(val_records, class_names, args.batch_size, training=False, use_mixup=False)

    train_counts = [len(splits[name]["train"]) for name in class_names]
    total_samples = sum(train_counts)
    class_weights = {
        i: total_samples / (len(class_names) * count)
        for i, count in enumerate(train_counts)
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ckpt_path = output / f"best_{args.backbone}.keras"

    model, base = build_model(len(class_names), backbone=args.backbone, label_smoothing=args.label_smoothing)

    # Phase 1: Train classification head
    print(f"\n[Phase 1] Backbone={args.backbone}, Epochs={args.phase1_epochs}, LabelSmoothing={args.label_smoothing}")
    p1_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.phase1_epochs,
        callbacks=p1_callbacks,
        class_weight=class_weights,
    )

    # Phase 2: Fine-tune top N layers (keep BatchNorm frozen for stability)
    print(f"\n[Phase 2] Unfreezing top {args.unfreeze_layers} layers (excluding BatchNorm) with lr=1e-5...")
    base.trainable = True
    for layer in base.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"],
    )
    p2_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.phase2_epochs,
        callbacks=p2_callbacks,
        class_weight=class_weights,
    )

    # Load best checkpoint
    best_model = tf.keras.models.load_model(ckpt_path)

    print("\nEvaluating on held-out validation and test splits...")
    val_metrics = evaluate_records(best_model, val_records, class_names, args.batch_size)
    test_metrics = evaluate_records(best_model, test_records, class_names, args.batch_size)

    model_file = output / "waste_classifier.keras"
    best_model.save(model_file)

    metrics_file = output / f"metrics_{args.backbone}.json"
    metrics_file.write_text(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    train_config = {
        "backbone": args.backbone,
        "manifest": str(manifest_path),
        "class_order": class_names,
        "image_size": list(IMAGE_SIZE),
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "batch_size": args.batch_size,
        "label_smoothing": args.label_smoothing,
        "unfreeze_layers": args.unfreeze_layers,
        "mixup": not args.no_mixup,
        "augmentations": [
            "random_resized_crop(0.75-1.0)",
            "random_horizontal_flip",
            "random_rotation(0.08)",
            "color_jitter(brightness=25, contrast=0.8-1.2, saturation=0.8-1.2, hue=0.04)",
            "gaussian_blur_3x3(p=0.2)",
            "random_erasing(p=0.25)",
            "mixup(alpha=0.2, p=0.35)",
        ],
        "class_weights": {class_names[i]: round(w, 4) for i, w in class_weights.items()},
        "seed": args.seed,
    }
    (output / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    print(f"\n=== TEST METRICS ({args.backbone}) ===")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
