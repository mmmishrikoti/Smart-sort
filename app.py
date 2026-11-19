import io
import json
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
        return cached[1], labels

    try:
        from tensorflow.keras.models import load_model
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is not installed. Install requirements.txt to enable predictions."
        ) from error
    model = load_model(model_path)
    current_app.extensions["sortsmart_model"] = (str(model_path.resolve()), model)
    return model, labels


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


def _predict(image):
    model, labels = _model_and_labels()
    if model is None:
        return None
    import cv2
    import numpy as np

    image = cv2.resize(
        np.asarray(image),
        (224, 224),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    from tensorflow.errors import OpError

    try:
        predictions = np.asarray(model.predict(np.expand_dims(image, axis=0), verbose=0))
    except OpError as error:
        raise RuntimeError("The trained model could not process this image.") from error
    if predictions.ndim != 2 or predictions.shape != (1, len(labels)):
        raise RuntimeError("The model output does not match the saved class labels.")
    predictions = predictions[0]
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("The model returned an invalid prediction.")
    if np.any(predictions < 0) or not np.isclose(np.sum(predictions), 1.0, atol=1e-3):
        shifted = predictions - np.max(predictions)
        probabilities = np.exp(shifted)
        predictions = probabilities / np.sum(probabilities)
    index = int(np.argmax(predictions))
    confidence = float(predictions[index])
    if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise RuntimeError("The model returned an invalid confidence score.")
    category = _canonical_category(labels[index])
    uncertain = confidence < current_app.config["CONFIDENCE_THRESHOLD"]
    return {
        "category": "uncertain" if uncertain else category,
        "predicted_category": category,
        "confidence": confidence,
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
        MODEL_PATH=os.environ.get("SORTSMART_MODEL_PATH", str(ROOT / "model" / "waste_classifier.keras")),
        LABELS_PATH=os.environ.get("SORTSMART_LABELS_PATH", str(ROOT / "model" / "labels.json")),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        CONFIDENCE_THRESHOLD=0.55,
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
