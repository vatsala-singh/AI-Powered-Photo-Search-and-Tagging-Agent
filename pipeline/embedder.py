#this class loads the CLIP model once and exposes two functions:
#1. embed_image
#2. embed_text
#both return 512-d vector that Qdrant can directly store and compare

import numpy as np
from pathlib import Path
from fastembed import TextEmbedding, ImageEmbedding
from config import TEXT_MODEL_NAME, VISION_MODEL_NAME, MODELS_DIR, EMBED_DIM

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Singleton model state
_text_model = None
_image_model = None

def _load_text_model():
    global _text_model
    if _text_model is not None:
        return
    print(f"[embedder] Loading text model '{TEXT_MODEL_NAME}' from cache_dir '{MODELS_DIR}'...")
    _text_model = TextEmbedding(
        model_name=TEXT_MODEL_NAME,
        cache_dir=str(MODELS_DIR)
    )
    print("[embedder] Text model loaded successfully")

def _load_image_model():
    global _image_model
    if _image_model is not None:
        return
    print(f"[embedder] Loading image model '{VISION_MODEL_NAME}' from cache_dir '{MODELS_DIR}'...")
    _image_model = ImageEmbedding(
        model_name=VISION_MODEL_NAME,
        cache_dir=str(MODELS_DIR)
    )
    print("[embedder] Image model loaded successfully")

def embed_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk and return a normalized 512-d embedding.
    Returns None if the file can't be opened.
    """
    _load_image_model()
    try:
        path = Path(path)
        # ImageEmbedding.embed accepts an iterable of paths and returns embeddings
        embeddings = list(_image_model.embed([str(path)]))
        if not embeddings:
            print(f"[embedder] Warning: Failed to embed image at '{path}'")
            return None
        return embeddings[0].astype(np.float32)
    except Exception as e:
        print(f"[embedder] Warning: Failed to open image at '{path}': {e}")
        return None

def embed_text(query: str) -> np.ndarray:
    """
    Convert a natural language string into a normalized 512-d embedding.
    FastEmbed returns already normalized embeddings.
    """
    _load_text_model()
    try:
        # TextEmbedding.embed accepts an iterable of strings
        embeddings = list(_text_model.embed([query]))
        if not embeddings:
            print(f"[embedder] Warning: Failed to embed text '{query}'")
            return None
        return embeddings[0].astype(np.float32)
    except Exception as e:
        print(f"[embedder] Warning: Failed to embed text '{query}': {e}")
        return None
        
