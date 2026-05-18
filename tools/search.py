from qdrant_edge import Query, QueryRequest
from config import TOP_K, VECTOR_NAME
from store.qdrant_client import get_shard
from pipeline.embedder import embed_image, embed_text


def search_photos(query: str, top_k: int = TOP_K, tags: list[str] = None) -> list[dict]:
    #search photo library with a Natural language query
    #takes in query, no of results to be displayed, and a list of tags
    # returns list of dicts with photo metadata and relevance score
    print(f"[search] Received query='{query}' with tags={tags} and top_k={top_k}")
    shard = get_shard()
    query_vector = embed_text(query)
    
    search_filter = None
    results = shard.query(
        QueryRequest(
            query=Query.Nearest(query_vector.tolist(), using=VECTOR_NAME),
            limit=top_k if not tags else top_k * 3,  # over-fetch when tag filtering
            with_vector=False,
            with_payload=True,
        )
    )
    print(f"[search] Found {len(results)} initial hits for query='{query}' with tags={tags}")
    hits = []
    for point in results:
        payload = point.payload or {}
        point_tags = payload.get("tags", [])
        
        # Post-filter by tags if specified
        # (EdgeShard doesn't support complex filters, so we filter in Python
        # after over-fetching more results than needed)
        if tags and not any(t in point_tags for t in tags):
            continue
        
        hits.append(
            {
                "path": payload.get("path"),
                "filename": payload.get("filename"),
                "tags": point_tags,
                "timestamp": payload.get("timestamp"),
                "score": round(point.score, 4)
            }
        )
        if len(hits) >= top_k:
            break

    return hits
    
    