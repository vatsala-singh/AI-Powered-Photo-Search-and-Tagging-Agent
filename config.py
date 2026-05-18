from pathlib import Path

# --- Embedding ---
TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
VISION_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"
MODELS_DIR = Path("./qdrant-edge-data/models")
EMBED_DIM = 512

# --- Qdrant ---
SHARD_DIR   = Path("./qdrant-edge-data/photos")  # where the shard lives on disk
VECTOR_NAME = "clip"                         # named vector key inside the shard
# --- Search ---
TOP_K = 10                         # default number of results to return
DUPLICATE_THRESHOLD = 0.97         # cosine similarity threshold for duplicates

# --- Paths ---
PHOTO_DIR = Path.home() / "Pictures"