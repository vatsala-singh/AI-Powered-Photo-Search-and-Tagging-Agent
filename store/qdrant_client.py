from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType
)
from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBED_DIM

_client: QdrantClient | None = None

def get_client() -> QdrantClient:
    #singleton Qdrant client
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client

def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, 
                                        distance=Distance.COSINE),
        )
        #index 'path' for faster lookups and duplicate detection
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="path",
            schema=PayloadSchemaType.KEYWORD
        )
        print(f"[store] Created Collection '{COLLECTION_NAME}' with vector size {EMBED_DIM}")
    else:
        print(f"[store] Collection '{COLLECTION_NAME}' already exists")
        
    
