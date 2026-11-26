import io
import json
import math
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, current_app, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
GUIDE = {
    "cardboard": {
        "label": "Cardboard",
        "examples": "Shipping boxes, clean packaging, and corrugated board.",
        "disposal": "Flatten boxes and keep them dry. Remove tape and liners when practical.",
        "recycling": "Place clean, dry cardboard in your local paper and cardboard collection.",
        "impact": "Recycling cardboard reduces demand for virgin fibers and landfill volume.",
        "reuse": "Reuse sturdy boxes for storage or shipping before recycling.",
        "recyclable": True,
    },
    "glass": {
        "label": "Glass",
        "examples": "Bottles and food jars. Ceramics, mirrors, and window glass are different materials.",
        "disposal": "Empty and rinse containers. Separate colors if your local program asks.",
        "recycling": "Use a glass collection point or curbside bin that accepts container glass.",
        "impact": "Glass can be recycled repeatedly, but correct sorting prevents contamination.",
        "reuse": "Reuse jars for storage after cleaning.",
        "recyclable": True,
    },
    "metal": {
        "label": "Metal",
        "examples": "Aluminum drink cans, steel food cans, and clean metal lids.",
        "disposal": "Empty containers and check local guidance for aerosols and scrap metal.",
        "recycling": "Put accepted cans in your local metal or mixed-recycling collection.",
        "impact": "Recovering metal can save energy and conserve mined resources.",
        "reuse": "Reuse clean tins for organizing small items.",
        "recyclable": True,
    },
    "paper": {
        "label": "Paper",
        "examples": "Office paper, newspapers, magazines, and clean paper packaging.",
        "disposal": "Keep paper clean and dry. Do not include heavily soiled or wax-coated paper.",
        "recycling": "Use a paper collection accepted by your local recycling service.",
        "impact": "Paper recovery can reduce landfill waste and demand for new pulp.",
        "reuse": "Use blank sides for notes or craft projects.",
        "recyclable": True,
    },
    "plastic": {
        "label": "Plastic",
        "examples": "Bottles, tubs, and containers; resin and accepted shapes vary by location.",
        "disposal": "Empty and rinse packaging. Check local rules rather than relying on the resin number alone.",
        "recycling": "Recycle only the plastic formats your local program accepts.",
        "impact": "Correct sorting reduces litter and helps keep reusable material in circulation.",
        "reuse": "Choose durable reusable containers where appropriate.",
        "recyclable": True,
    },
    "trash": {
        "label": "Trash / Other",
        "examples": "Mixed, contaminated, or currently unclassified everyday waste.",
        "disposal": "Check local waste rules. Keep batteries, electronics, and hazardous items out of general trash.",
        "recycling": "Separate any identifiable recyclable parts before disposal.",
        "impact": "Landfilled waste uses space and may release greenhouse gases as it breaks down.",
        "reuse": "Consider repair, donation, or reuse before discarding an item.",
        "recyclable": False,
    },
}


@contextmanager
def _database_connection():
    connection = sqlite3.connect(current_app.config["DATABASE"])
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _init_database():
    database = current_app.config["DATABASE"]
    Path(database).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                uncertain INTEGER NOT NULL DEFAULT 0,
                image_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def _model_and_labels():
    model_path = Path(current_app.config["MODEL_PATH"])
    labels_path = Path(current_app.config["LABELS_PATH"])
    if not model_path.is_file() or not labels_path.is_file():
        return None, None
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("The model labels file is unreadable or invalid.") from error
    if not isinstance(labels, list) or not labels or not all(
        isinstance(label, str) and label for label in labels
    ):
        raise RuntimeError("The model labels file must contain a non-empty list of labels.")

    cached = current_app.extensions.get("sortsmart_model")
    if cached and cached[0] == str(model_path.resolve()):
        return cached[1], cached[2], cached[3], labels

    try:
        from tensorflow.keras.models import Model, load_model
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is not installed. Install requirements.txt to enable predictions."
        ) from error
    model = load_model(model_path)
    try:
        logit_layer = model.get_layer("logits")
        logit_model = Model(inputs=model.input, outputs=logit_layer.output)
    except ValueError:
        logit_model = None

    temperature = 1.0
    cal_path = model_path.parent / "calibration.json"
    if cal_path.is_file():
        try:
            cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
            t_val = float(cal_data.get("temperature", 1.0))
            if math.isfinite(t_val) and t_val > 0:
                temperature = t_val
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            temperature = 1.0

    current_app.extensions["sortsmart_model"] = (
        str(model_path.resolve()),
        model,
        logit_model,
        temperature,
    )
    return model, logit_model, temperature, labels


def _decode_image(image_bytes):
    if not image_bytes:
        raise ValueError("Choose an image before analyzing.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Use a JPEG, PNG, or WebP image.")
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.width < 1 or image.height < 1 or image.width * image.height > 25_000_000:
                raise ValueError("The image dimensions are not supported.")
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as error:
        raise ValueError("The uploaded file is not a valid supported image.") from error


def _canonical_category(label):
    normalized = label.strip().lower().replace("_", " ")
    if normalized in {"trash / other", "trash / other waste", "other"}:
        return "trash"
    return normalized


def _load_thresholds(model_path: Path):
    thr_path = model_path.parent / "thresholds.json"
    defaults = {
        "high_confidence_threshold": 0.90,
        "medium_confidence_threshold": float(current_app.config.get("CONFIDENCE_THRESHOLD", 0.55)),
        "margin_threshold": 0.30,
    }
    if thr_path.is_file():
        try:
            data = json.loads(thr_path.read_text(encoding="utf-8"))
            for key in defaults:
                if key in data and math.isfinite(float(data[key])):
                    defaults[key] = float(data[key])
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
    return defaults


def _make_tta_views(img_rgb):
    import cv2
    import numpy as np

    h, w = img_rgb.shape[:2]
    orig = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    flip = np.ascontiguousarray(orig[:, ::-1, :])

    ch, cw = max(1, int(h * 0.88)), max(1, int(w * 0.88))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = img_rgb[y0 : y0 + ch, x0 : x0 + cw]
    crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    crop_flip = np.ascontiguousarray(crop_resized[:, ::-1, :])
    return np.stack([orig, flip, crop_resized, crop_flip], axis=0)


def _predict_probs(image):
    loaded = _model_and_labels()
    if loaded[0] is None:
        return None, None
    model, logit_model, temperature, labels = loaded
    import cv2
    import numpy as np

    img_rgb = np.asarray(image)
    use_tta = bool(current_app.config.get("USE_TTA", True))
    if use_tta:
        batch = _make_tta_views(img_rgb)
    else:
        resized = cv2.resize(
            img_rgb,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        batch = np.expand_dims(resized, axis=0)
    from tensorflow.errors import OpError

    try:
        if logit_model is not None:
            logits = np.asarray(logit_model.predict(batch, verbose=0), dtype=np.float64)
            scaled = logits / temperature
            shifted = scaled - np.max(scaled, axis=1, keepdims=True)
            exp_vals = np.exp(shifted)
            probs_batch = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
            predictions = np.mean(probs_batch, axis=0)
        else:
            raw = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] != len(labels):
                raise RuntimeError("The model output does not match the saved class labels.")
            predictions = np.mean(raw, axis=0)
    except OpError as error:
        raise RuntimeError("The trained model could not process this image.") from error
    if predictions.shape != (len(labels),):
        raise RuntimeError("The model output does not match the saved class labels.")
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("The model returned an invalid prediction.")
    if np.any(predictions < 0) or not np.isclose(np.sum(predictions), 1.0, atol=1e-3):
        shifted = predictions - np.max(predictions)
        probabilities = np.exp(shifted)
        predictions = probabilities / np.sum(probabilities)
    return predictions, labels


def _classify_probs(predictions, labels):
    import numpy as np

    sorted_indices = np.argsort(predictions)[::-1]
    index = int(sorted_indices[0])
    confidence = float(predictions[index])
    second_conf = float(predictions[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0
    margin = float(confidence - second_conf)
    if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise RuntimeError("The model returned an invalid confidence score.")

    top_3 = [
        {
            "category": _canonical_category(labels[int(idx)]),
            "confidence": float(predictions[int(idx)]),
        }
        for idx in sorted_indices[:3]
    ]

    thresholds = _load_thresholds(Path(current_app.config["MODEL_PATH"]))
    high_thr = thresholds["high_confidence_threshold"]
    med_thr = thresholds["medium_confidence_threshold"]
    margin_thr = thresholds["margin_threshold"]

    if confidence >= high_thr and margin >= margin_thr:
        confidence_level = "high"
    elif confidence >= med_thr and margin >= margin_thr * 0.5:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    needs_review = (confidence < high_thr) or (margin < margin_thr)
    uncertain = confidence_level == "low"
    category = _canonical_category(labels[index])
    message = (
        "Not sure, try a closer photo with a plain background."
        if confidence_level == "low"
        else None
    )

    return {
        "category": "uncertain" if uncertain else category,
        "predicted_category": category,
        "confidence": confidence,
        "margin": margin,
        "confidence_level": confidence_level,
        "needs_review": needs_review,
        "top_3": top_3,
        "message": message,
        "uncertain": uncertain,
        "recommendation": GUIDE.get(
            category,
            {
                "label": labels[index].replace("_", " ").title(),
                "disposal": "Check your local waste authority for safe disposal guidance.",
                "recycling": "Confirm whether this material is accepted in your area.",
                "reuse": "Consider reuse or repair before disposal.",
                "recyclable": False,
            },
        ),
    }


def _predict(image):
    predictions, labels = _predict_probs(image)
    if predictions is None:
        return None
    return _classify_probs(predictions, labels)


def _get_yolo_detector():
    yolo_path = Path(current_app.config["YOLO_MODEL_PATH"])
    if not yolo_path.is_file():
        return None
    cached = current_app.extensions.get("sortsmart_yolo")
    if cached and cached[0] == str(yolo_path.resolve()):
        return cached[1]
    from modules.yolo_detector import SortSmartYOLODetector

    detector = SortSmartYOLODetector(yolo_path)
    if not detector.is_ready():
        return None
    current_app.extensions["sortsmart_yolo"] = (str(yolo_path.resolve()), detector)
    return detector


def _hybrid_detect(image, conf_threshold: float = 0.15):
    import numpy as np

    detector = _get_yolo_detector()
    if detector is None:
        return None
    raw_dets = detector.detect(image, conf_threshold=conf_threshold)
    img_w, img_h = image.size
    thresholds = _load_thresholds(Path(current_app.config["MODEL_PATH"]))
    high_thr = thresholds["high_confidence_threshold"]
    med_thr = thresholds["medium_confidence_threshold"]
    margin_thr = thresholds["margin_threshold"]

    results = []
    for det in raw_dets:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad_x, pad_y = bw * 0.10, bh * 0.10
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x))
        cy2 = min(img_h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = image.crop((cx1, cy1, cx2, cy2))
        cnn_probs, labels = _predict_probs(crop)
        yolo_cat = _canonical_category(det["category"])
        yolo_conf = float(det["confidence"])

        if cnn_probs is not None and labels is not None:
            cnn_idx = int(np.argmax(cnn_probs))
            cnn_cat = _canonical_category(labels[cnn_idx])
            cnn_conf = float(cnn_probs[cnn_idx])
            sources_agree = yolo_cat == cnn_cat

            # Weighted geometric mean combining calibrated CNN (0.65) and YOLO (0.35)
            n_cls = len(labels)
            other_yolo = max(1e-6, (1.0 - yolo_conf) / max(1, n_cls - 1))
            combined = np.zeros(n_cls, dtype=np.float64)
            for i, lbl in enumerate(labels):
                cname = _canonical_category(lbl)
                p_cnn = max(1e-6, float(cnn_probs[i]))
                p_yolo = max(1e-6, yolo_conf if cname == yolo_cat else other_yolo)
                combined[i] = (p_cnn ** 0.65) * (p_yolo ** 0.35)
            combined /= np.sum(combined)

            sorted_idx = np.argsort(combined)[::-1]
            top_idx = int(sorted_idx[0])
            final_conf = float(combined[top_idx])
            second_conf = float(combined[int(sorted_idx[1])]) if n_cls > 1 else 0.0
            margin = float(final_conf - second_conf)
            final_cat = _canonical_category(labels[top_idx])
            top_3 = [
                {"category": _canonical_category(labels[int(k)]), "confidence": float(combined[int(k)])}
                for k in sorted_idx[:3]
            ]
        else:
            cnn_cat = None
            cnn_conf = None
            sources_agree = True
            final_cat = yolo_cat
            final_conf = yolo_conf
            margin = yolo_conf
            top_3 = [{"category": yolo_cat, "confidence": yolo_conf}]

        if final_conf >= high_thr and margin >= margin_thr and sources_agree:
            confidence_level = "high"
        elif final_conf >= med_thr and margin >= margin_thr * 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        disagreement_uncertain = (not sources_agree) and (yolo_conf < high_thr) and (cnn_conf is None or cnn_conf < high_thr)
        needs_review = disagreement_uncertain or (final_conf < high_thr) or (margin < margin_thr)

        results.append({
            "category": final_cat,
            "confidence": final_conf,
            "margin": margin,
            "confidence_level": confidence_level,
            "sources_agree": sources_agree,
            "needs_review": needs_review,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "yolo": {"category": yolo_cat, "confidence": yolo_conf},
            "classifier": {"category": cnn_cat, "confidence": cnn_conf} if cnn_cat is not None else None,
            "top_3": top_3,
            "recommendation": GUIDE.get(final_cat, GUIDE["trash"]),
        })

    return {
        "count": len(results),
        "image_size": [img_w, img_h],
        "detections": results,
    }


def _json_error(message, status):
    return jsonify({"error": message}), status


def create_app(test_config=None):
    app = Flask(__name__)
    is_vercel = os.environ.get("VERCEL") == "1"
    runtime_data = Path(tempfile.gettempdir()) / "sortsmart"
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SORTSMART_SECRET_KEY", "local-development-key"),
        DATABASE=os.environ.get(
            "SORTSMART_DATABASE",
            str(runtime_data / "sortsmart.sqlite3" if is_vercel else ROOT / "database" / "sortsmart.sqlite3"),
        ),
        UPLOAD_FOLDER=os.environ.get(
            "SORTSMART_UPLOAD_FOLDER",
            str(runtime_data / "uploads" if is_vercel else ROOT / "uploads"),
        ),
        FEEDBACK_FOLDER=os.environ.get(
            "SORTSMART_FEEDBACK_FOLDER",
            str(runtime_data / "feedback" if is_vercel else ROOT / "feedback"),
        ),
        MODEL_PATH=os.environ.get("SORTSMART_MODEL_PATH", str(ROOT / "model" / "waste_classifier.keras")),
        LABELS_PATH=os.environ.get("SORTSMART_LABELS_PATH", str(ROOT / "model" / "labels.json")),
        YOLO_MODEL_PATH=os.environ.get("SORTSMART_YOLO_MODEL_PATH", str(ROOT / "model" / "yolov8_best.pt")),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        CONFIDENCE_THRESHOLD=0.55,
        USE_TTA=os.environ.get("SORTSMART_USE_TTA", "1") == "1",
    )
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    with app.app_context():
        _init_database()

    @app.errorhandler(413)
    def request_too_large(_error):
        if request.path.startswith("/api/"):
            return _json_error("Image exceeds the 10 MB upload limit.", 413)
        return "Upload exceeds the 10 MB limit.", 413

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/scanner")
    def scanner():
        return render_template("scanner.html")

    @app.get("/guide")
    def guide():
        return render_template("guide.html", categories=GUIDE)

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/history")
    def history():
        return render_template("history.html")

    @app.get("/uploads/<path:image_name>")
    def uploaded_image(image_name):
        return send_from_directory(app.config["UPLOAD_FOLDER"], image_name)

    @app.get("/api/status")
    def status():
        available = Path(app.config["MODEL_PATH"]).is_file() and Path(
            app.config["LABELS_PATH"]
        ).is_file()
        return jsonify(
            {
                "model_available": available,
                "message": (
                    "A trained model is ready."
                    if available
                    else "No trained model is installed yet. Train one with train.py to enable AI predictions."
                ),
            }
        )

    @app.get("/api/recommendations/<category>")
    def recommendations(category):
        normalized = _canonical_category(category)
        recommendation = GUIDE.get(normalized)
        if recommendation is None:
            return _json_error("No disposal guide is available for that category.", 404)
        return jsonify(recommendation)

    @app.get("/api/model-metrics")
    def model_metrics():
        metrics_path = Path(app.config["MODEL_PATH"]).with_name("metrics.json")
        if not metrics_path.is_file():
            return jsonify({"available": False})
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            app.logger.exception("The saved model evaluation could not be read.")
            return _json_error("The saved model evaluation is unreadable.", 503)
        if not isinstance(metrics, dict) or not {
            "test_samples",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1_score",
            "per_class",
            "confusion_matrix",
            "class_order",
        }.issubset(metrics):
            return _json_error("The saved model evaluation has an invalid format.", 503)
        return jsonify({"available": True, **metrics})

    @app.post("/api/detect")
    def detect_items():
        upload = request.files.get("image")
        if upload is None:
            return _json_error("Choose an image to analyze.", 400)
        image_bytes = upload.stream.read(app.config["MAX_CONTENT_LENGTH"] + 1)
        if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
            return _json_error("Image exceeds the 10 MB upload limit.", 413)
        try:
            pixels = _decode_image(image_bytes)
        except ValueError as error:
            return _json_error(str(error), 400)
        try:
            detection_result = _hybrid_detect(pixels)
        except (RuntimeError, ValueError, OSError) as error:
            app.logger.exception("Hybrid detection could not run.")
            return _json_error(str(error), 503)
        if detection_result is None:
            return _json_error("YOLO detector weights are not installed.", 503)
        return jsonify(detection_result)

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None:
            return _json_error("Choose an image to analyze.", 400)
        image_bytes = upload.stream.read(app.config["MAX_CONTENT_LENGTH"] + 1)
        if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
            return _json_error("Image exceeds the 10 MB upload limit.", 413)
        try:
            pixels = _decode_image(image_bytes)
        except ValueError as error:
            return _json_error(str(error), 400)
        try:
            result = _predict(pixels)
        except (RuntimeError, ValueError, OSError) as error:
            app.logger.exception("Prediction could not run.")
            return _json_error(str(error), 503)
        if result is None:
            return _json_error(
                "No trained model is installed. Add model/waste_classifier.keras and model/labels.json after training.",
                503,
            )

        if result.get("needs_review") and result.get("confidence_level") != "high":
            from modules.llm_fallback import run_llm_fallback

            cache_path = Path(app.config["DATABASE"]).with_name("llm_cache.json")
            result["llm_fallback"] = run_llm_fallback(pixels, result, cache_path=cache_path)
        else:
            result["llm_fallback"] = {"used": False, "reason": "high_confidence_cnn"}

        image_name = f"{uuid.uuid4().hex}.jpg"
        image_path = Path(app.config["UPLOAD_FOLDER"]) / image_name
        pixels.save(image_path, format="JPEG", quality=88)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _database_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scans (predicted_category, confidence, uncertain, image_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result["predicted_category"],
                    result["confidence"],
                    int(result["uncertain"]),
                    image_name,
                    created_at,
                ),
            )
            result.update(
                {
                    "id": cursor.lastrowid,
                    "image_url": f"/uploads/{image_name}",
                    "created_at": created_at,
                }
            )
        return jsonify(result)

    @app.post("/api/feedback")
    def submit_feedback():
        import csv
        import shutil

        payload = request.get_json(silent=True) or {}
        if payload.get("consent") is not True:
            return _json_error("Explicit consent is required before saving feedback.", 400)

        scan_id = payload.get("scan_id")
        is_correct = payload.get("is_correct")
        if not isinstance(scan_id, int) or not isinstance(is_correct, bool):
            return _json_error("scan_id and is_correct are required.", 400)

        with _database_connection() as connection:
            row = connection.execute(
                "SELECT id, predicted_category, image_name FROM scans WHERE id = ?",
                (scan_id,),
            ).fetchone()
        if row is None:
            return _json_error("Scan record not found.", 404)

        labels_path = Path(app.config["LABELS_PATH"])
        valid_labels = set(json.loads(labels_path.read_text(encoding="utf-8"))) if labels_path.is_file() else set(CLASSES)
        predicted = _canonical_category(row["predicted_category"])
        corrected = _canonical_category(str(payload.get("corrected_category") or predicted))
        if is_correct:
            corrected = predicted
        if corrected not in valid_labels:
            return _json_error(f"Unsupported label '{corrected}'.", 400)

        src_img = Path(app.config["UPLOAD_FOLDER"]) / row["image_name"]
        if not src_img.is_file():
            return _json_error("Original scan image is no longer on disk.", 404)

        fb_root = Path(app.config["FEEDBACK_FOLDER"])
        dest_dir = fb_root / corrected
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"fb_{scan_id}_{uuid.uuid4().hex[:8]}.jpg"
        dest_img = dest_dir / dest_name
        shutil.copy2(src_img, dest_img)

        log_path = fb_root / "feedback_log.csv"
        write_header = not log_path.is_file()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "scan_id", "rel_path", "predicted_category", "corrected_category", "is_correct", "consent", "reviewed"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                scan_id,
                f"{corrected}/{dest_name}",
                predicted,
                corrected,
                int(is_correct),
                1,
                1 if is_correct or bool(payload.get("corrected_category")) else 0,
            ])

        return jsonify({"saved": True, "label": corrected, "path": f"{corrected}/{dest_name}"})

    @app.get("/api/history")
    def scan_history():
        search = request.args.get("q", "").strip()[:100]
        category = request.args.get("category", "").strip()
        query = "SELECT * FROM scans WHERE 1 = 1"
        parameters = []
        if search:
            query += " AND (predicted_category LIKE ? OR created_at LIKE ?)"
            parameters.extend((f"%{search}%", f"%{search}%"))
        if category == "uncertain":
            query += " AND uncertain = 1"
        elif category and category != "all":
            query += " AND predicted_category = ?"
            parameters.append(category)
        query += " ORDER BY id DESC LIMIT 500"
        with _database_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return jsonify(
            [
                {
                    "id": row["id"],
                    "category": "uncertain" if row["uncertain"] else row["predicted_category"],
                    "predicted_category": row["predicted_category"],
                    "confidence": row["confidence"],
                    "uncertain": bool(row["uncertain"]),
                    "image_url": f"/uploads/{row['image_name']}",
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        )

    @app.get("/api/categories")
    def categories():
        with _database_connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT predicted_category, uncertain FROM scans"
            ).fetchall()
        names = set(GUIDE)
        names.update(row["predicted_category"] for row in rows)
        if any(row["uncertain"] for row in rows):
            names.add("uncertain")
        return jsonify({"categories": sorted(names)})

    @app.delete("/api/history/<int:scan_id>")
    def delete_scan(scan_id):
        with _database_connection() as connection:
            row = connection.execute(
                "SELECT image_name FROM scans WHERE id = ?", (scan_id,)
            ).fetchone()
            if row is None:
                return _json_error("Scan record was not found.", 404)
            connection.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        image_path = Path(app.config["UPLOAD_FOLDER"]) / row["image_name"]
        if image_path.parent.resolve() == Path(app.config["UPLOAD_FOLDER"]).resolve():
            image_path.unlink(missing_ok=True)
        return jsonify({"deleted": True})

    @app.get("/api/stats")
    def stats():
        period = request.args.get("period", "weekly")
        if period not in {"daily", "weekly"}:
            return _json_error("Period must be daily or weekly.", 400)
        today = datetime.now(timezone.utc).date()
        start_day = today if period == "daily" else today - timedelta(days=6)
        cutoff = start_day.isoformat()
        with _database_connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            recyclable_categories = tuple(
                category for category, details in GUIDE.items() if details["recyclable"]
            )
            placeholders = ",".join("?" for _ in recyclable_categories)
            recyclable = connection.execute(
                f"SELECT COUNT(*) FROM scans WHERE uncertain = 0 "
                f"AND predicted_category IN ({placeholders})",
                recyclable_categories,
            ).fetchone()[0]
            categories = connection.execute(
                "SELECT predicted_category, COUNT(*) AS count FROM scans GROUP BY predicted_category"
            ).fetchall()
            recent = connection.execute(
                "SELECT id, predicted_category, confidence, uncertain, created_at "
                "FROM scans ORDER BY id DESC LIMIT 6"
            ).fetchall()
            daily = connection.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count
                FROM scans WHERE created_at >= ?
                GROUP BY day ORDER BY day
                """,
                (cutoff,),
            ).fetchall()
        return jsonify(
            {
                "total_scans": total,
                "recyclable_scans": recyclable,
                "categories": {row["predicted_category"]: row["count"] for row in categories},
                "recent": [
                    {
                        "id": row["id"],
                        "category": "uncertain" if row["uncertain"] else row["predicted_category"],
                        "predicted_category": row["predicted_category"],
                        "confidence": row["confidence"],
                        "created_at": row["created_at"],
                    }
                    for row in recent
                ],
                "daily": [{"day": row["day"], "count": row["count"]} for row in daily],
                "period": period,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
