from qdrant_client.models import Filter, FieldCondition, MatchAny
from config import COLLECTION_NAME, TOP_K
from store.qdrant_client import get_client
from pipeline.embedder import embed_image, embed_text


def search_photos(query: str, top_k: int = TOP_K, tags: list[str] = None) -> list[dict]:
    #search photo library with a Natural language query
    #takes in query, no of results to be displayed, and a list of tags
    # returns list of dicts with photo metadata and relevance score
    
    client = get_client()
    query_vector = embed_text(query)
    
    search_filter = None
    if tags:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="tags",
                    match=MatchAny(any=tags)
                )
            ]
        )
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "path": hit.payload.get("path"),
            "filename": hit.payload.get("filename"),
            "tags": hit.payload.get("tags", []),
            "timestamp": hit.payload.get("timestamp"),
            "score": round(hit.score, 4)
        }
        for hit in hits.points
    ]
    
    