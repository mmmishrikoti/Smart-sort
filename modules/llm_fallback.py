"""
Task 8: Optional Vision-LLM fallback (Google Gemini) for low/medium-confidence items.
- Only invoked when needs_review is True and confidence_level != "high".
- Server-side GEMINI_API_KEY only, cached by image SHA-256 hash.
- Requests structured JSON (temperature=0.1) and validates against the Part 2 schema.
"""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image


ALLOWED_CATEGORIES = {
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
    "organic",
    "e-waste",
    "hazardous",
    "unknown",
}

SYSTEM_PROMPT = (
    "You are a waste-sorting expert. You will see one photo of a waste item (or a few items). "
    "Identify the primary item and its material, and say how it should be disposed of.\n\n"
    "Rules:\n"
    "- Choose \"category\" ONLY from: cardboard, glass, metal, paper, plastic, trash, organic, e-waste, hazardous, unknown.\n"
    "- If the image is blurry, too dark, shows no waste item, or you are not sure, use \"unknown\" and explain what photo would help. Do not guess.\n"
    "- Recycling rules vary by city. State the general rule, and set \"check_local_rules\" to true when the answer depends on local programs "
    "(for example: plastic film, greasy pizza boxes, ceramics, mixed materials).\n"
    "- Mention contamination (food residue, liquids) and whether the cap, lid or label should be removed.\n"
    "- Batteries, bulbs, electronics, and chemicals are never regular trash: category = \"hazardous\" or \"e-waste\" with drop-off advice.\n"
    "- Output valid JSON only. No markdown, no extra text."
)

PRIVACY_NOTICE = "Images sent to the optional fallback go to a third-party AI service (Google Gemini)."


def validate_llm_response(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be a JSON object.")

    required_str = ("item", "material", "category", "reasoning")
    for k in required_str:
        if not isinstance(payload.get(k), str) or not payload[k].strip():
            raise ValueError(f"Missing or invalid string field: {k}")

    cat = payload["category"].strip().lower()
    if cat not in ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid category '{cat}'.")

    for k in ("recyclable", "agrees_with_classifier", "check_local_rules"):
        if not isinstance(payload.get(k), bool):
            raise ValueError(f"Missing or invalid boolean field: {k}")

    conf = payload.get("confidence")
    if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
        raise ValueError("Invalid confidence value (must be 0.0-1.0).")

    steps = payload.get("disposal_steps")
    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise ValueError("Invalid disposal_steps (must be list of strings).")

    for opt_k in ("contamination_warning", "reuse_idea", "retake_tip"):
        v = payload.get(opt_k)
        if v is not None and not isinstance(v, str):
            raise ValueError(f"Invalid optional string field: {opt_k}")

    return {
        "item": payload["item"].strip(),
        "material": payload["material"].strip(),
        "category": cat,
        "recyclable": payload["recyclable"],
        "confidence": float(conf),
        "agrees_with_classifier": payload["agrees_with_classifier"],
        "disposal_steps": [s.strip() for s in steps if s.strip()],
        "contamination_warning": payload.get("contamination_warning"),
        "reuse_idea": payload.get("reuse_idea"),
        "check_local_rules": payload["check_local_rules"],
        "reasoning": payload["reasoning"].strip(),
        "retake_tip": payload.get("retake_tip"),
    }


def _image_sha256_and_jpeg(image: Image.Image) -> tuple[str, bytes]:
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.thumbnail((768, 768))
    rgb.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    return hashlib.sha256(raw).hexdigest(), raw


def _load_cache(cache_path: Path) -> dict:
    if cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def run_llm_fallback(
    image: Image.Image,
    cnn_result: dict,
    cache_path: Path,
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict:
    """
    Calls Gemini Vision API when needs_review is True and confidence_level != 'high'.
    Never overrides a high-confidence CNN result.
    """
    if not cnn_result.get("needs_review") or cnn_result.get("confidence_level") == "high":
        return {"used": False, "reason": "high_confidence_cnn"}

    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        return {"used": False, "reason": "api_key_not_configured", "privacy_notice": PRIVACY_NOTICE}

    img_hash, jpeg_bytes = _image_sha256_and_jpeg(image)
    cache = _load_cache(cache_path)
    if img_hash in cache:
        return {"used": True, "cached": True, "privacy_notice": PRIVACY_NOTICE, **cache[img_hash]}

    top_3 = cnn_result.get("top_3") or []
    cnn_label = top_3[0]["category"] if len(top_3) > 0 else cnn_result.get("predicted_category", "unknown")
    cnn_conf = round((top_3[0]["confidence"] if len(top_3) > 0 else cnn_result.get("confidence", 0.0)) * 100, 1)
    cnn_label_2 = top_3[1]["category"] if len(top_3) > 1 else "none"
    cnn_conf_2 = round((top_3[1]["confidence"] if len(top_3) > 1 else 0.0) * 100, 1)

    user_message = (
        f"Classifier suggestion (may be wrong): {cnn_label} at {cnn_conf}%.\n"
        f"Second guess: {cnn_label_2} at {cnn_conf_2}%.\n"
        "Identify the item in the image and return JSON in this schema."
    )

    target_model = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "parts": [
                    {"text": user_message},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(jpeg_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = requests.post(url, params={"key": key}, json=body, timeout=20)
        resp.raise_for_status()
        resp_json = resp.json()
        raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)
        validated = validate_llm_response(parsed)
    except Exception as exc:
        return {"used": False, "reason": f"fallback_error: {type(exc).__name__}", "privacy_notice": PRIVACY_NOTICE}

    cache[img_hash] = validated
    _save_cache(cache_path, cache)
    return {"used": True, "cached": False, "privacy_notice": PRIVACY_NOTICE, **validated}
