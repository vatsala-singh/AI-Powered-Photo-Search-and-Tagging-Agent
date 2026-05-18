# tools/duplicates.py
from qdrant_edge import Query, QueryRequest

from config import DUPLICATE_THRESHOLD, VECTOR_NAME
from store.qdrant_client import get_shard


def _load_all_points() -> list:
    """
    Page through the entire shard and return all points with vectors + payloads.
    EdgeShard has no scroll method — we use Query.Scroll with offset pagination.
    """
    shard = get_shard()
    all_points = []
    offset = 0
    batch_size = 256

    while True:
        batch = shard.query(
            QueryRequest(
                query=Query.Scroll(offset=offset),
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