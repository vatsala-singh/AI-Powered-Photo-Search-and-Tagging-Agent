import numpy as np
from pipeline.embedder import embed_text, embed_image
from pathlib import Path

# Text embedding
vec = embed_text("a dog running on a beach")
assert vec.shape == (512,), f"Wrong shape: {vec.shape}"
assert abs(np.linalg.norm(vec) - 1.0) < 1e-5, "Not normalized"
print(f"Text vector OK — shape: {vec.shape}, norm: {np.linalg.norm(vec):.5f}")

# Image embedding
samples = list(Path.home().glob("Pictures/test/*.jpg"))
if not samples:
    print("WARNING: no test images found in ~/Pictures/test — skipping image test")
else:
    img_vec = embed_image(samples[0])
    assert img_vec is not None
    assert img_vec.shape == (512,)
    assert abs(np.linalg.norm(img_vec) - 1.0) < 1e-5
    print(f"Image vector OK — shape: {img_vec.shape}, norm: {np.linalg.norm(img_vec):.5f}")

# Similarity between two text embeddings
vec_a = embed_text("sunset at the beach")
vec_b = embed_text("sunrise over the ocean")
vec_c = embed_text("quarterly earnings report")
sim_ab = float(np.dot(vec_a, vec_b))
sim_ac = float(np.dot(vec_a, vec_c))
print(f"Similarity (beach/ocean): {sim_ab:.3f}  ← should be high")
print(f"Similarity (beach/earnings): {sim_ac:.3f}  ← should be low")
assert sim_ab > sim_ac, "Semantic similarity is wrong — model may not have loaded"
print("Similarity ordering OK")