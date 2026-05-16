from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from pipeline.indexer import index_folder
from tools.search import search_photos

from store.qdrant_client import get_client
from config import COLLECTION_NAME

app = FastAPI(title="Photo Library Search Agent", version="1.0")

#Request and response samples for strong typing and documentation
class IndexRequest(BaseModel):
    folder: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    tags: list[str] = []
    
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
    client = get_client()
    results = client.retrieve(collection_name=COLLECTION_NAME, 
                             ids=[image_id],
                             with_payload=True)
    if not results or not results[0].payload:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"id": image_id, "tags": results[0].payload.get("tags", [])}
    
    
    
    
@app.get("/health")
def health():
    return {"status": "ok"}
