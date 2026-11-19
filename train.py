import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


IMAGE_SIZE = (224, 224)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
STARTER_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}


def split_dataset(dataset_dir, seed):
    root = Path(dataset_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    missing = STARTER_CLASSES.difference(classes)
    if missing:
        raise ValueError(f"Missing starter category folders in {root}: {', '.join(sorted(missing))}")
    paths_by_class = {}
    for label in classes:
        images = sorted(
            path
            for path in (root / label).rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if len(images) < 10:
            raise ValueError(f"Category '{label}' needs at least 10 supported images; found {len(images)}.")
        random.Random(seed).shuffle(images)
        test_end = max(1, round(len(images) * 0.15))
        validation_end = test_end + max(1, round(len(images) * 0.15))
        if validation_end >= len(images):
            raise ValueError(f"Category '{label}' needs more images for three separate splits.")
        paths_by_class[label] = {
            "train": images[validation_end:],
            "validation": images[test_end:validation_end],
            "test": images[:test_end],
        }
    return classes, paths_by_class


def add_supplemental_training_images(split, supplemental_dir, class_names):
    root = Path(supplemental_dir)
    if not root.is_dir():
        raise ValueError(f"Supplemental dataset folder does not exist: {root}")
    class_dirs = [path for path in root.iterdir() if path.is_dir()]
    unknown = {path.name for path in class_dirs}.difference(class_names)
    if unknown:
        raise ValueError(
            f"Supplemental dataset has unknown category folders: {', '.join(sorted(unknown))}"
        )
    added = {}
    for label in class_names:
        class_dir = root / label
        images = (
            sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
            if class_dir.is_dir()
            else []
        )
        split[label]["train"].extend(images)
        if images:
            added[label] = len(images)
    if not added:
        raise ValueError(f"No supported supplemental images found in {root}.")
    return added


def make_dataset(paths, class_names, batch_size, training):
    label_ids = {name: index for index, name in enumerate(class_names)}
    files = []
    labels = []
    for category, category_paths in paths.items():
        for path in category_paths:
            files.append(str(path))
            labels.append(label_ids[category])
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))

    def load_image(path, label):
        contents = tf.io.read_file(path)
        image = tf.io.decode_image(contents, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, tf.one_hot(label, depth=len(class_names))

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(files), 2048), seed=42, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(class_count):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the SortSmart MobileNetV2 classifier.")
    parser.add_argument("--dataset", default="dataset", help="Folder containing one subfolder per category.")
    parser.add_argument("--output", default="model", help="Folder for the trained model and evaluation files.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--supplemental-dataset",
        help="Optional labeled image folders added to training only, never validation or test.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers.")

    tf.keras.utils.set_random_seed(args.seed)
    classes, split = split_dataset(args.dataset, args.seed)
    if args.supplemental_dataset:
        added = add_supplemental_training_images(
            split, args.supplemental_dataset, classes
        )
        print(f"Added supplemental training images: {json.dumps(added, sort_keys=True)}")
    train_data = make_dataset(
        {name: split[name]["train"] for name in classes}, classes, args.batch_size, True
    )
    validation_data = make_dataset(
        {name: split[name]["validation"] for name in classes}, classes, args.batch_size, False
    )
    test_data = make_dataset(
        {name: split[name]["test"] for name in classes}, classes, args.batch_size, False
    )

    model, _ = build_model(len(classes))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    ]
    train_counts = [len(split[name]["train"]) for name in classes]
    total_samples = sum(train_counts)
    class_weights = {
        i: total_samples / (len(classes) * count)
        for i, count in enumerate(train_counts)
    }
    print(f"Calculated class weights: {json.dumps({classes[i]: round(w, 3) for i, w in class_weights.items()}, indent=2)}")
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    truth = []
    predictions = []
    for images, labels in test_data:
        truth.extend(np.argmax(labels.numpy(), axis=1))
        predictions.extend(np.argmax(model.predict(images, verbose=0), axis=1))
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=range(len(classes)), zero_division=0
    )
    metrics = {
        "test_samples": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1_score": float(np.mean(f1)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
            }
            for index, label in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=range(len(classes))
        ).tolist(),
        "class_order": classes,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "waste_classifier.keras")
    (output / "labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
