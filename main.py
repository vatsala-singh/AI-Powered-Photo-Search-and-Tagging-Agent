from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path

from qdrant_edge import QueryRequest, CountRequest

from pipeline.indexer import index_folder
from tools.search import search_photos

from store.qdrant_client import close_shard, get_shard


# ── Lifespan: ensure EdgeShard is flushed on shutdown ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup — shard is created lazily on first use, nothing to do here
    yield
    # shutdown — MUST flush and close to persist in-flight writes
    close_shard()

app = FastAPI(title="Photo Library Search Agent", version="1.0")

#Request and response samples for strong typing and documentation
class IndexRequest(BaseModel):
    folder: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    tags: list[str] = []

class TagRequest(BaseModel):
    path: str
 
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    
#app routes
@app.post("/index")
def index(req: IndexRequest):
    folder = Path(req.folder)
    if not folder.exists():
        raise HTTPException(status_code=400, detail="Folder does not exist")
    summary = index_folder(folder)
    return {"status": "success", "summary": summary}

@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = search_photos(req.query, top_k=req.top_k, tags=req.tags or None)
    return {"query": req.query, "results": results}

@app.get("/tags/{image_id}")
def get_tags(image_id: str):
    shard = get_shard()
    results = shard.retrieve(
        QueryRequest(
            points_ids=[image_id],
            with_payload=True,
            with_vector=False
        )
    )
    if not results or not results[0].payload:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"id": image_id, "tags": results[0].payload.get("tags", [])}
    
    
    
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/shard-stats")
def shard_stats():
    """Check shard statistics - useful for debugging indexing issues."""
    shard = get_shard()
    point_count = shard.count(CountRequest(exact=True))
    return {
        "shard_path": str(shard.path) if hasattr(shard, 'path') else "unknown",
        "total_points": point_count,
        "status": "has_data" if point_count > 0 else "empty"
    }

@app.post("/chat")
def chat(req: ChatRequest):
    from agent import run_agent
    reply = run_agent(req.message, history=req.history)
    return {"reply": reply}


@app.post("/tag")
def tag(req: TagRequest):
    from tools.tag import tag_image
    result = tag_image(req.path)
    return result