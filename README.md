---
title: SortSmart AI
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SortSmart AI

SortSmart AI is a software-only waste sorting assistant built with Flask, SQLite, TensorFlow/Keras, MobileNetV2, and OpenCV. Its online Hugging Face Space loads the trained model and offers real image predictions. The Vercel deployment remains a lightweight site without prediction support; neither deployment fabricates AI results.

## Requirements

- Python 3.11 recommended for local model training (TensorFlow wheel availability depends on your operating system and Python version).
- VS Code with the Python extension.
- A webcam is optional; image upload works without one.
- A labeled image dataset is required to train the model. This project does not bundle or claim rights to a third-party dataset.

## Install and run on Windows

Open the project folder in VS Code, then run these commands in the integrated PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
python app.py
```

Open <http://127.0.0.1:5000>. If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. TensorFlow and OpenCV downloads can be large.

The site and guide run before a model is trained. On the scanner, model status will say **Model not installed** and an analysis request will explain that a trained model is required. This is expected, not a mock prediction.

## Train a model

Put images in one folder per class. For the six starter classes, use:

```text
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Use appropriately licensed images, and keep each image in exactly one class. The trainer needs at least 10 supported images per class, with more strongly recommended. It reproducibly shuffles each class and creates separate 70/15/15 train, validation, and test splits. Training uses 224 × 224 images, augmentation, and ImageNet-pretrained MobileNetV2 transfer learning with a frozen feature extractor. The pretrained weights may be downloaded the first time.

The starter six-class dataset used during local development is TrashNet, downloaded from [garythung/trashnet on Hugging Face](https://huggingface.co/datasets/garythung/trashnet), based on the [original TrashNet repository](https://github.com/garythung/trashnet). The dataset card declares the MIT license. The downloaded archive is in `dataset/trashnet-resized.zip`, and extracted class images are in `dataset/trashnet-download/dataset-resized/`. Please retain the dataset attribution and verify license terms before redistribution.

```powershell
python train.py --dataset dataset --output model --epochs 10 --batch-size 32
```

For the included dataset, install the training dependencies in the environment first, then use:

```powershell
python train.py --dataset dataset\trashnet-download\dataset-resized --output model --epochs 10 --batch-size 16
```

You can add new, correctly labeled examples without putting them into the held-out validation or test sets. Make a separate folder with one subfolder per known category, then pass it using `--supplemental-dataset`:

```text
dataset-supplemental/
  plastic/
    colored-bottle.jpg
```

```powershell
python train.py --dataset dataset\trashnet-download\dataset-resized --supplemental-dataset dataset-supplemental --output model --epochs 10 --batch-size 16
```

Supplemental images are used for training only. For a meaningful independent evaluation, keep separate new examples out of this folder and compare the reported metrics with the same held-out split.

The trainer reports the measured test-set accuracy, per-class precision/recall/F1, confusion matrix, and test-set size to the terminal. It writes:

- `model/waste_classifier.keras` — trained Keras model.
- `model/labels.json` — output class order used by the model.
- `model/metrics.json` — measured test-set evaluation.

The dataset must be representative of the images you expect to scan. A test score is not a guarantee of real-world accuracy. Do not compare results across datasets or splits as if they were equivalent.

## Features and API

- `/` home, `/scanner`, `/guide`, `/dashboard`, `/history`.
- `GET /api/status` — reports whether the model and labels are available.
- `GET /api/recommendations/<category>` — returns disposal guidance for a known category.
- `GET /api/model-metrics` — returns the measured held-out test report after training.
- `POST /api/predict` — accepts an `image` multipart field; validates actual image contents, supports JPEG/PNG/WebP, and limits uploads to 10 MB.
- `GET /api/history?q=&category=` and `DELETE /api/history/<id>` — view/filter or delete saved scans and their stored image.
- `GET /api/categories` — lists guide and scanned categories for dynamic filters.
- `GET /api/stats?period=daily|weekly` — category, activity, and scan totals.

Model predictions below the 55% confidence threshold are presented as uncertain. Change the threshold with the `CONFIDENCE_THRESHOLD` Flask config if there is a measured reason to do so; do not lower it just to make the UI show more confident results.

SQLite is stored at `database/sortsmart.sqlite3`; validated scan images are stored under `uploads/`. Configure `SORTSMART_DATABASE`, `SORTSMART_UPLOAD_FOLDER`, `SORTSMART_MODEL_PATH`, `SORTSMART_LABELS_PATH`, and `SORTSMART_SECRET_KEY` with environment variables when needed. The default secret key is for local development only.

## Security and production notes

This starter application has no accounts or authentication and is intended for local development or a trusted single-user demo. Before deploying publicly, add authentication and authorization, CSRF protection for state-changing requests, HTTPS, rate limiting, managed secret configuration, retention controls, and an operational backup/cleanup policy. Uploaded images and scan records are retained on the server until deleted. Camera access generally requires localhost or HTTPS.

## Online image predictions

The full scanner and trained model can be hosted together on Hugging Face Spaces using the included `Dockerfile`. Create a Docker Space, upload the project files, and the app listens on port 7860. The trained `.keras` model is included in the Docker image; uploaded scans and SQLite history are kept in `/tmp` and are therefore temporary and may be cleared when a Space restarts. Deploy only the source and model you have permission to publish.

Vercel serves the lightweight website and measured evaluation report, but omits TensorFlow and the model because of serverless bundle limits; online predictions are not available on the Vercel deployment. Use the Hugging Face Space URL for online image scanning.

## Vercel deployment

The Vercel deployment serves the Flask pages and measured model evaluation report. To keep the serverless function small, `.vercelignore` excludes the TensorFlow model and training dataset; image prediction remains unavailable there. Vercel's `/tmp` filesystem is temporary, so SQLite scan history is not durable across serverless instances or restarts.

With the Vercel CLI installed and authenticated, deploy from the project directory:

```powershell
vercel deploy --prod
```

`requirements.txt` contains lightweight Vercel web runtime dependencies. Install `requirements-training.txt` for local training, and use `requirements-space.txt` for the Docker inference service.

## Troubleshooting

- **Model not installed:** complete the dataset preparation and training steps, then restart Flask.
- **TensorFlow/OpenCV installation fails:** use a supported 64-bit Python version and check the package's platform-specific wheel support.
- **Camera cannot open:** allow browser camera permission and use localhost/HTTPS, or upload an image.
- **No history or charts:** scans are saved only after a trained model successfully analyzes an image.
