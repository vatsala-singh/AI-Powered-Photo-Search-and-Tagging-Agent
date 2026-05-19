from qdrant_edge import Query, QueryRequest, ScrollRequest, UpdateOperation

from config import DUPLICATE_THRESHOLD, VECTOR_NAME
from pipeline.embedder import embed_image, embed_text
from store.qdrant_client import get_shard
import numpy as np

TAG_VOCABULARY = [
    "sunset", "sunrise", "beach", "ocean", "mountain", "forest", "city",
    "night", "snow", "rain", "fog", "sunny", "cloudy",
    "dog", "cat", "bird", "people", "crowd", "portrait", "selfie",
    "food", "coffee", "restaurant", "travel", "architecture",
    "car", "road", "nature", "flowers", "trees",
    "indoor", "outdoor", "party", "celebration", "sport",
    "screenshot", "document", "text", "map",
]

_tag_vectors: dict[str, np.ndarray] = {}

def _get_tag_vectors() -> dict[str, np.ndarray]:
    global _tag_vectors
    if not _tag_vectors:
        print("[tagger] Pre-computing tag embeddings ...")
        _tag_vectors = {tag: embed_text(tag) for tag in TAG_VOCABULARY}
    return _tag_vectors

def generate_tags_from_vector(img_vec: np.ndarray, threshold: float = 0.20, max_tags: int = 6) -> list[str]:
    """
    Generate tags for an image vector using zero-shot CLIP classification.
    Tags with cosine similarity above threshold are included (up to max_tags).
    
    This is a utility function used for generating tags during indexing
    or when you already have an image vector.
    """
    tag_vecs = _get_tag_vectors()
    
    scores = {
        tag: float(np.dot(img_vec, vec))   # both normalized → cosine similarity
        for tag, vec in tag_vecs.items()
    }
    
    tags = sorted(
        [t for t, s in scores.items() if s >= threshold],
        key=lambda t: scores[t],
        reverse=True,
    )[:max_tags]
    
    return tags

def _load_all_points() -> list:
    """
    Page through the entire shard and return all points with vectors + payloads.
    EdgeShard has scroll method — we use ScrollRequest with offset pagination.
    """
    shard = get_shard()
    all_points = []
    offset = 0
    batch_size = 256

    while True:
        batch, _ = shard.scroll(
            ScrollRequest(
                offset=offset,
                limit=batch_size,
                with_vector=True,
                with_payload=True,
            )
        )
        all_points.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size

    return all_points


def find_duplicates(threshold: float = DUPLICATE_THRESHOLD) -> dict:
    """
    Scan all vectors and find clusters of near-identical images
    using cosine similarity (vectors are pre-normalized by embedder).

    Returns a dict with a 'clusters' list — each cluster is a list of paths.
    O(n²) — fine up to ~5k images for a personal library.
    """
    all_points = _load_all_points()
    print(f"[duplicates] Scanning {len(all_points)} images at threshold={threshold}")

    visited = set()
    clusters = []

    for i, point_a in enumerate(all_points):
        if point_a.id in visited:
            continue

        # Extract named vector for point_a
        vec_a = point_a.vector.get(VECTOR_NAME) if isinstance(point_a.vector, dict) else point_a.vector
        if vec_a is None:
            continue

        cluster = [point_a]
        visited.add(point_a.id)

        for point_b in all_points[i + 1:]:
            if point_b.id in visited:
                continue

            vec_b = point_b.vector.get(VECTOR_NAME) if isinstance(point_b.vector, dict) else point_b.vector
            if vec_b is None:
                continue

            # Cosine similarity — both vectors are L2-normalized by embedder
            sim = sum(a * b for a, b in zip(vec_a, vec_b))
            if sim >= threshold:
                cluster.append(point_b)
                visited.add(point_b.id)

        if len(cluster) > 1:
            clusters.append([
                (p.payload or {}).get("path", str(p.id))
                for p in cluster
            ])

    return {
        "threshold":          threshold,
        "total_images":       len(all_points),
        "duplicate_clusters": len(clusters),
        "clusters":           clusters,
    }
    
def tag_image(path: str, threshold: float = 0.20, max_tags: int = 6) -> dict:
    """
    Generate tags for an image using zero-shot CLIP classification.
    Tags with cosine similarity above `threshold` are included.
    Tags are persisted back into the shard's payload for that point.
    """
    img_vec = embed_image(path)
    if img_vec is None:
        return {"path": path, "tags": [], "error": "Could not load image"}
 
    tags = generate_tags_from_vector(img_vec, threshold=threshold, max_tags=max_tags)
    _store_tags(path, tags)
 
    return {"path": path, "tags": tags}
 
 
def _store_tags(path: str, tags: list[str]) -> None:
    """
    Find the point with this file path and update its 'tags' payload.
    Uses Qdrant Edge's UpdateOperation.set_payload.
    """
    shard = get_shard()
 
    # Find the point ID for this path via a scroll scan
    results,_ = shard.scroll(
        ScrollRequest(
            offset=0,
            limit=10_000,
            with_vector=False,
            with_payload=True,
        )
    )
 
    point_id = None
    for point in results:
        if (point.payload or {}).get("path") == path:
            point_id = point.id
            break
 
    if point_id is None:
        print(f"[tagger] Warning: could not find point for path {path}")
        return
 
    shard.update(
        UpdateOperation.set_payload(
            [point_id],
            {"tags": tags},
        )
    )
 