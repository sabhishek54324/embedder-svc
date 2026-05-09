import os

from decouple import config
from flask import Flask, jsonify, request

from embeddings import (
    DEVICE,
    EMBEDDING_MODEL,
    SUPPORTS_IMAGES,
    embed_image,
    embed_images_batch,
    embed_text,
    embed_texts_batch,
)
from logger import get_logger

log = get_logger("main")
app = Flask(__name__)

API_KEY = config("VECTOR_EMBEDDER_API_KEY", default="abc123")
MAX_TEXTS_PER_BATCH = config("MAX_TEXTS_PER_BATCH", default=100, cast=int)
MAX_IMAGES_PER_BATCH = config("MAX_IMAGES_PER_BATCH", default=20, cast=int)


# ─── Auth helper ─────────────────────────────────────────────────────────────
def _auth(req) -> bool:
    return req.headers.get("X-API-Key") == API_KEY


def _unauthorized():
    return jsonify({"error": "Invalid or missing API key"}), 401


# ─── Health ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/health/ready", methods=["GET"])
def health_ready():
    import embeddings as emb
    is_ready = getattr(emb, "model", None) is not None
    if not is_ready:
        return jsonify({"status": "not_ready", "error": "Model not loaded"}), 503
    return (
        jsonify(
            {
                "status": "ready",
                "model": EMBEDDING_MODEL,
                "device": DEVICE,
                "supports_images": SUPPORTS_IMAGES,
            }
        ),
        200,
    )


# ─── Single text ─────────────────────────────────────────────────────────────
@app.route("/embeddings", methods=["POST"])
def generate_embeddings():
    if not _auth(request):
        return _unauthorized()
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "text is required"}), 400
    log.info("POST /embeddings  len=%d  device=%s", len(text), DEVICE)
    result = embed_text(text)
    return jsonify({"embeddings": result.tolist(), "device": DEVICE}), 200


# ─── Batch text ──────────────────────────────────────────────────────────────
@app.route("/embeddings/batch", methods=["POST"])
def generate_embeddings_batch():
    if not _auth(request):
        return _unauthorized()
    texts = request.json.get("texts")
    if not texts or not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "texts must be a non-empty array"}), 400
    if len(texts) > MAX_TEXTS_PER_BATCH:
        return jsonify({"error": f"Batch size exceeds max {MAX_TEXTS_PER_BATCH}"}), 400
    valid = [t for t in texts if t and isinstance(t, str)]
    if len(valid) != len(texts):
        return jsonify({"error": "All texts must be non-empty strings"}), 400
    log.info("POST /embeddings/batch  n=%d  device=%s", len(valid), DEVICE)
    results = embed_texts_batch(valid)
    return jsonify({"embeddings": [e.tolist() for e in results], "device": DEVICE}), 200


# ─── Single image ────────────────────────────────────────────────────────────
@app.route("/embeddings/image", methods=["POST"])
def generate_image_embedding():
    if not _auth(request):
        return _unauthorized()
    if not SUPPORTS_IMAGES:
        return jsonify({"error": "Image embeddings not supported by current model"}), 501
    img_b64 = request.json.get("image")
    if not img_b64:
        return jsonify({"error": "image (base64) is required"}), 400
    try:
        result = embed_image(img_b64)
        return jsonify({"embeddings": result.tolist(), "device": DEVICE}), 200
    except (ValueError, Exception) as e:
        return jsonify({"error": str(e)}), 400


# ─── Batch image ─────────────────────────────────────────────────────────────
@app.route("/embeddings/image/batch", methods=["POST"])
def generate_image_embeddings_batch():
    if not _auth(request):
        return _unauthorized()
    if not SUPPORTS_IMAGES:
        return jsonify({"error": "Image embeddings not supported by current model"}), 501
    images = request.json.get("images")
    if not images or not isinstance(images, list) or len(images) == 0:
        return jsonify({"error": "images must be a non-empty array"}), 400
    if len(images) > MAX_IMAGES_PER_BATCH:
        return jsonify({"error": f"Batch size exceeds max {MAX_IMAGES_PER_BATCH}"}), 400
    valid = [i for i in images if i and isinstance(i, str)]
    if len(valid) != len(images):
        return jsonify({"error": "All images must be non-empty base64 strings"}), 400
    try:
        results = embed_images_batch(valid)
        return jsonify({"embeddings": [e.tolist() for e in results], "device": DEVICE}), 200
    except (ValueError, Exception) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    log.info("Starting embedder-svc on port 5001 | device=%s", DEVICE)
    app.run(host="0.0.0.0", port=5001)