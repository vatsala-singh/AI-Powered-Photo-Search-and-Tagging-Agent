from pathlib import Path

# --- Embedding ---
EMBED_MODEL = "ViT-B-32"          # open-clip model name
EMBED_PRETRAINED = "openai"        # pretrained weights tag
EMBED_DIM = 512                    # vector dimension for ViT-B/32

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "photos"

# --- Search ---
TOP_K = 10                         # default number of results to return
DUPLICATE_THRESHOLD = 0.97         # cosine similarity threshold for duplicates

# --- Paths ---
PHOTO_DIR = Path.home() / "Pictures" 