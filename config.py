from pathlib import Path

# --- Embedding ---
EMBED_MODEL = "ViT-B-32"          # open-clip model name
EMBED_PRETRAINED = "openai"        # pretrained weights tag
EMBED_DIM = 512                    # vector dimension for ViT-B/32

# --- Qdrant ---
SHARD_DIR   = Path("./qdrant-edge-data/photos")  # where the shard lives on disk
VECTOR_NAME = "clip"                         # named vector key inside the shard
# --- Search ---
TOP_K = 10                         # default number of results to return
DUPLICATE_THRESHOLD = 0.97         # cosine similarity threshold for duplicates

# --- Paths ---
PHOTO_DIR = Path.home() / "Pictures" 