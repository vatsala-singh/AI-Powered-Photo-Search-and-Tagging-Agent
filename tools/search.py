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
    
    # Over-fetch when tag filtering is requested to have enough candidates
    # after post-filtering by tags
    over_fetch_multiplier = 5 if tags else 1
    fetch_limit = top_k * over_fetch_multiplier
    
    results = shard.query(
        QueryRequest(
            query=Query.Nearest(query_vector.tolist(), using=VECTOR_NAME),
            limit=fetch_limit,
            with_vector=False,
            with_payload=True,
        )
    )
    print(f"[search] Found {len(results)} initial hits for query='{query}' with tags={tags}")
    
    hits = []
    untagged_hits = []  # Fallback results for images without tags
    
    for point in results:
        payload = point.payload or {}
        point_tags = payload.get("tags", [])
        
        result_dict = {
            "path": payload.get("path"),
            "filename": payload.get("filename"),
            "tags": point_tags,
            "timestamp": payload.get("timestamp"),
            "score": round(point.score, 4)
        }
        
        # Post-filter by tags if specified
        # (EdgeShard doesn't support complex filters, so we filter in Python
        # after over-fetching more results than needed)
        if tags:
            if point_tags and any(t in point_tags for t in tags):
                # Has tags and matches the filter
                hits.append(result_dict)
            elif not point_tags:
                # No tags yet (images not auto-tagged), save as fallback
                untagged_hits.append(result_dict)
        else:
            # No tag filter specified, include all results
            hits.append(result_dict)
        
        # Stop if we have enough tagged results
        if len(hits) >= top_k:
            break
    
    # If we don't have enough tagged results, include untagged ones that match the query
    if tags and len(hits) < top_k:
        hits.extend(untagged_hits[:top_k - len(hits)])
    
    return hits[:top_k]
    
    