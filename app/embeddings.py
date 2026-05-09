import base64
import io
import os

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from logger import get_logger

log = get_logger("embeddings")

# ─── Config ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multi-qa-MiniLM-L6-cos-v1")
TOKENIZER_MODEL = os.getenv(
    "TOKENIZER_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10 MB

# ─── Device detection ────────────────────────────────────────────────────────
def _detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info("🟢 Accelerator device : CUDA")
        log.info("   GPU name           : %s", name)
        log.info("   VRAM               : %.2f GB", vram)
        return "cuda"
    if torch.backends.mps.is_available():
        log.info("🟢 Accelerator device : MPS (Apple Silicon)")
        return "mps"
    log.warning("🟡 Accelerator device : CPU  (no GPU detected)")
    return "cpu"


DEVICE = _detect_device()

# ─── Model load ──────────────────────────────────────────────────────────────
log.info("Loading embedding model : %s", EMBEDDING_MODEL)
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
log.info("Model loaded on device  : %s", DEVICE)


# ─── Image support probe ─────────────────────────────────────────────────────
def _check_image_support() -> bool:
    try:
        dummy = Image.new("RGB", (224, 224))
        model.encode(dummy)
        log.info("Image embedding support : ✅ YES")
        return True
    except Exception:
        log.info("Image embedding support : ❌ NO (text-only model)")
        return False


SUPPORTS_IMAGES = _check_image_support()

# ─── Tokenizer (text-only models only) ───────────────────────────────────────
tokenizer = None
if TOKENIZER_MODEL and not SUPPORTS_IMAGES:
    log.info("Loading tokenizer : %s", TOKENIZER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    log.info("Tokenizer loaded.")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def decode_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(
            f"Image size {len(image_bytes)} bytes exceeds limit of {MAX_IMAGE_SIZE} bytes"
        )
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def chunk_by_tokens(content: str, max_token_length: int = 512) -> list[str]:
    tok = tokenizer
    if tok is None:
        return [content]

    tokens = tok.tokenize(content)
    token_chunks = [
        tokens[i : i + max_token_length]
        for i in range(0, len(tokens), max_token_length)
    ]
    chunk_lengths = [len(" ".join(c)) for c in token_chunks]

    chunks, next_start = [], 0
    for idx, length in enumerate(chunk_lengths):
        chunk = (
            content[next_start:]
            if idx == len(chunk_lengths) - 1
            else content[next_start : next_start + length]
        )
        next_start += length
        if chunk:
            chunks.append(chunk)

    return chunks


# ─── Public API ──────────────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    if SUPPORTS_IMAGES:
        return model.encode(text, device=DEVICE).reshape(1, -1)
    return _embed_text_chunked(text)


def _embed_text_chunked(text: str) -> np.ndarray:
    chunks = chunk_by_tokens(text, max_token_length=512)[:5]
    log.debug("Encoding %d chunk(s) on %s", len(chunks), DEVICE)
    embeddings = model.encode(chunks, device=DEVICE)
    return np.mean(embeddings, axis=0, keepdims=True)


def embed_texts_batch(texts: list[str], max_batch_size: int = None) -> list[np.ndarray]:
    if max_batch_size is None:
        max_batch_size = MAX_BATCH_SIZE

    if SUPPORTS_IMAGES:
        embeddings = model.encode(texts, device=DEVICE)
        return [e.reshape(1, -1) for e in embeddings]

    all_chunks, boundaries = [], []
    for text in texts:
        chunks = chunk_by_tokens(text, max_token_length=512)[:5]
        boundaries.append((len(all_chunks), len(all_chunks) + len(chunks)))
        all_chunks.extend(chunks)

    all_embeddings = []
    for i in range(0, len(all_chunks), max_batch_size):
        batch = all_chunks[i : i + max_batch_size]
        log.debug(
            "Batch encode %d/%d chunks on %s", i + len(batch), len(all_chunks), DEVICE
        )
        all_embeddings.extend(model.encode(batch, device=DEVICE))

    results = []
    for start, end in boundaries:
        avg = np.mean(np.array(all_embeddings[start:end]), axis=0, keepdims=True)
        results.append(avg)
    return results


def embed_image(image_base64: str) -> np.ndarray:
    if not SUPPORTS_IMAGES:
        raise NotImplementedError("Current model does not support image embeddings")
    img = decode_image(image_base64)
    return model.encode(img, device=DEVICE).reshape(1, -1)


def embed_images_batch(images_base64: list[str]) -> list[np.ndarray]:
    if not SUPPORTS_IMAGES:
        raise NotImplementedError("Current model does not support image embeddings")
    images = [decode_image(b) for b in images_base64]
    embeddings = model.encode(images, device=DEVICE)
    return [e.reshape(1, -1) for e in embeddings]