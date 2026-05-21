from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from openclaw import Agent

from qdrant_edge import QueryRequest, CountRequest

from pipeline.indexer import index_folder
from tools.search import search_photos
from tools.duplicates import find_duplicates
from tools.tag import generate_tags_from_vector

from store.qdrant_client import close_shard, get_shard



# ── Lifespan: ensure EdgeShard is flushed on shutdown ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup — shard is created lazily on first use, nothing to do here
    yield
    # shutdown — MUST flush and close to persist in-flight writes
    close_shard()

app = FastAPI(title="Photo Library Search Agent", version="1.0")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

#Request and response samples for strong typing and documentation
class IndexRequest(BaseModel):
    folder: str
    auto_tag: bool = True  # Automatically generate tags during indexing

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
@app.get("/")
def root():
    """Redirect to chat interface"""
    from fastapi.responses import FileResponse
    return FileResponse(static_dir / "chat.html")

@app.get("/image")
def get_image(path: str):
    """Serve image files"""
    from fastapi.responses import FileResponse
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.post("/index")
def index(req: IndexRequest):
    folder = Path(req.folder)
    if not folder.exists():
        raise HTTPException(status_code=400, detail="Folder does not exist")
    summary = index_folder(folder, auto_tag=req.auto_tag)
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
    """
    Conversational endpoint. Accepts user message and conversation history,
    returns agent's reply after processing with tools.
    """
    # Define tool functions that the agent can call
    def search_tool(query: str, top_k: int = 10, tag_filter: list = None):
        """Search photos by natural language query"""
        return search_photos(query=query, top_k=top_k, tags=tag_filter)
    
    def duplicates_tool(threshold: float = 0.97):
        """Find duplicate or near-duplicate photos"""
        return find_duplicates(threshold=threshold)
    
    def tag_tool(image_path: str):
        """Generate and update tags for a specific photo"""
        return generate_tags_from_vector(image_path=image_path)
    
    # Create the agent with tools
    agent = Agent(
        tools=[search_tool, duplicates_tool, tag_tool],
        system_prompt="""
        You are a personal photo assistant. You help users search, organize,
        and understand their local photo library. You have access to tools
        for semantic search, duplicate detection, and tagging.
        
        When helping users:
        - Use the search tool to find photos by describing their content
        - Use duplicates tool to find and clean up duplicate shots
        - Use tag tool to inspect or update tags for specific photos
        
        Always be concise and helpful. When returning photo results, 
        format them clearly with filenames, similarity scores, and tags. 
        Use emojis sparingly but helpfully.
        """
    )
    
    # Run the agent conversation
    response = agent.chat(
        message=req.message,
        history=req.history
    )
    return {"response": response}


@app.post("/tag")
def tag(req: TagRequest):
    from tools.tag import tag_image
    result = tag_image(req.path)
    return result