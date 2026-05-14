from pipeline.embedder import embed_image, embed_text
import numpy as np

# Test text embedding
vec = embed_text("a dog on a beach")
print(f"Text vector shape : {vec.shape}")       # (512,)
print(f"Text vector norm  : {np.linalg.norm(vec):.4f}")  # ~1.0

# Test image embedding with a sample image (any jpg on your Mac)
from pathlib import Path
samples = list(Path.home().glob("Pictures/**/*.jpg"))
if samples:
    img_vec = embed_image(samples[0])
    print(f"Image vector shape: {img_vec.shape}")
    print(f"Image vector norm : {np.linalg.norm(img_vec):.4f}")
else:
    print("No jpg found in ~/Pictures — drop any .jpg in there to test")