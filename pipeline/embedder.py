#this class loads the CLIP model once and exposes two functions:
#1. embed_image
#2. embed_text
#both return 512-d vector that Qdrant can directly store and compare

import open_clip
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from config import EMBED_MODEL, EMBED_PRETRAINED, EMBED_DIM

#select device
DEVICE = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

#singleton model state
_model = None
_preprocess = None
_tokenizer = None

def _load_model():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return
    print(f"[embedder] Loading model '{EMBED_MODEL}' with pretrained weights '{EMBED_PRETRAINED}' on device '{DEVICE}'...")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        EMBED_MODEL,
        pretrained=EMBED_PRETRAINED,
        device=DEVICE
    )    
    _tokenizer = open_clip.get_tokenizer(EMBED_MODEL)
    _model.eval()  # set to evaluation mode
    print("[embedder] Model loaded successfully")
    
def _normalize(vec: np.ndarray) -> np.ndarray:
    #L2 normalize the vector for cosine similarity, essentially dot product becomes cosine similarity
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def embed_image(path:str | Path) -> np.ndarray:
#    Load an image from disk and return a normalized 512-d embedding.
#     Returns None if the file can't be opened.
    _load_model()
    try:
        img = Image.open(path).convert("RGB")
    except:
        print(f"[embedder] Warning: Failed to open image at '{path}'")
        return None
    
    tensor = _preprocess(img).unsqueeze(0).to(DEVICE)  # add batch dimension and move to device
    with torch.no_grad():
        features = _model.encode_image(tensor)
    vec = features.cpu().numpy()[0].astype(np.float32)  # convert to numpy and remove batch dimension
    return _normalize(vec)

def embed_text(query: str) ->np.ndarray:
    # Convert a natural language string into a normalized 512-d embedding.
    _load_model()
    tokens = _tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
    vec = features.cpu().numpy()[0].astype(np.float32)
    return _normalize(vec)
        
