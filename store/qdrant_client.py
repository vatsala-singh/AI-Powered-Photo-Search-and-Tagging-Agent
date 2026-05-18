from pathlib import Path
from qdrant_edge import(
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
)
from config import SHARD_DIR, VECTOR_NAME, EMBED_DIM

_shard: EdgeShard | None = None

def get_shard() -> EdgeShard:
    """
    Return the singleton EdgeShard, creating it on first call.
 
    - If SHARD_DIR does not exist → create a brand-new shard.
    - If SHARD_DIR already contains data → reopen it (no config needed).
 
    EdgeShard runs entirely in-process. No binary, no port, no network.
    """
    global _shard
    if _shard is not None:
        return _shard
    
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    # Detect whether this is a fresh shard or an existing one.
    # EdgeShard.create() fails if data already exists on disk.
    shard_has_data = any(SHARD_DIR.iterdir())
    
    if shard_has_data:
        print(f"[qdrant_client] Reopening existing shard at '{SHARD_DIR}'")
        _shard = EdgeShard.load(path=SHARD_DIR)
    else:
        print(f"[qdrant_client] Creating new shard at '{SHARD_DIR}'")
        config = EdgeConfig(
            vectors={
                VECTOR_NAME: EdgeVectorParams(
                    size=EMBED_DIM,
                    distance=Distance.Cosine
                )
            }
        )
        _shard = EdgeShard.create(path=str(SHARD_DIR), config=config)
        print(f"[store] Shard ready — vector: '{VECTOR_NAME}', dim: {EMBED_DIM}")
    return _shard

def close_shard() -> None:
    """
    Flush and close the shard. Must be called on application shutdown
    to guarantee all in-flight writes are persisted to disk.
    """
    global _shard
    if _shard is not None:
        _shard.close()
        _shard = None
        print("[store] Shard closed and flushed to disk.")