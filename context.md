# Photo Agent — Project Scaffold & Design Plan

AI-powered local photo search and tagging agent using CLIP embeddings, Qdrant Edge, and OpenClaw.

---

## What We're Building

A **local-first, AI-powered photo assistant** that:

1. Ingests images from a local folder
2. Embeds them with CLIP or SigLIP
3. Stores vectors and metadata in Qdrant Edge (on-device)
4. Exposes a FastAPI interface for natural language search
5. Wraps everything in an OpenClaw agent with tool calling

---

## System Architecture

```
Photos → Embedding Model → Qdrant Edge → OpenClaw Agent → Search Results
```

### Indexing sub-pipeline

```
Load images → Generate vectors → Attach metadata → Upsert collection
```

### Agent tools (OpenClaw)

| Tool | Purpose |
|------|---------|
| `search` | Natural language → ranked image results |
| `tag` | Zero-shot auto-tagging from image content |
| `find_duplicates` | Cluster near-identical images by vector distance |
| `smart_albums` | Group images by semantic query |
| `ocr_faces` | OCR on screenshots, face clustering |

---

## Semantic Search Workflow

```
Natural language query
        ↓
  CLIP text encoder (text embedding)
        ↓
   Query vector (512-d float array)
        ↓
  Qdrant Edge — cosine similarity search
   ↑
  Stored image vectors (+ tags, filename, timestamp)
        ↓
   Ranked matches (image paths + similarity scores)
        ↓
  OpenClaw agent response (formatted reply + image paths)
```

---

## Project Structure

```
photo-agent/
├── main.py                  # FastAPI app entrypoint
├── agent.py                 # OpenClaw agent definition + tool registration
│
├── pipeline/
│   ├── __init__.py
│   ├── embedder.py          # CLIP/SigLIP model loader + embed()
│   ├── indexer.py           # Batch & incremental image indexer
│   └── tagger.py            # Zero-shot multimodal auto-tagging
│
├── store/
│   ├── __init__.py
│   ├── qdrant_client.py     # Qdrant Edge connection + collection setup
│   └── schema.py            # Payload schema: filename, tags, timestamp, path
│
├── tools/
│   ├── __init__.py
│   ├── search.py            # search_photos(query: str) → ranked results
│   ├── tag.py               # tag_image(path: str) → list[str]
│   ├── duplicates.py        # find_duplicates(threshold: float) → clusters
│   └── albums.py            # create_smart_album(query: str) → album
│
├── config.py                # Paths, model name, Qdrant host/port, top-k
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

### Embedding model

Use `open-clip-torch` with `ViT-B/32` by default — small, fast, CPU-friendly. SigLIP is better for zero-shot tagging. Make the model swappable via `config.py`:

```python
# config.py
EMBED_MODEL = "ViT-B/32"          # swap to "SigLIP" for better tagging
EMBED_DIM = 512
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "photos"
TOP_K = 10
PHOTO_DIR = "/Users/you/Photos"
```

### Qdrant Edge

Runs as a local process (no Docker). One collection named `photos`, vector size `512`, distance `Cosine`.

**Payload schema per point:**

```python
{
  "filename": "IMG_2045.jpg",
  "path": "/Users/you/Photos/2024/IMG_2045.jpg",
  "tags": ["sunset", "beach", "golden hour"],
  "timestamp": "2024-07-12T18:43:00",
  "width": 4032,
  "height": 3024
}
```

### FastAPI

Three endpoints to start:

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/index` | Kick off indexing a folder |
| `POST` | `/search` | Natural language query → ranked results |
| `GET` | `/tags/{image_id}` | Return tags for a single image |

The OpenClaw agent calls these internally via its registered tools.

### OpenClaw agent

Registered tools map directly to modules in `tools/`. Each tool returns a structured dict the agent reasons over and formats into a natural language reply.

```python
# agent.py (sketch)
agent = OpenClawAgent(
    tools=[search_photos, tag_image, find_duplicates, create_smart_album],
    model="...",
    system="You are a photo assistant. Use tools to answer queries about the user's photo library."
)
```

### Incremental indexing

On `POST /index`, skip images whose path already exists as a payload match in Qdrant (cheap scroll query before embedding). This keeps re-indexing fast when new photos are added.

```python
# indexer.py (sketch)
existing = qdrant.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter=Filter(must=[FieldCondition(key="path", match=MatchValue(value=img_path))])
)
if existing[0]:
    continue  # already indexed
```

---

## Scaffolding Sequence

Build in this order so each layer is independently testable before the next depends on it:

1. **`config.py` + `requirements.txt`** — lock the model name and Qdrant settings first
2. **`store/qdrant_client.py`** — get the collection created and a point insertable
3. **`pipeline/embedder.py`** — load model, write `embed_image()` and `embed_text()`
4. **`pipeline/indexer.py`** — walk a folder, call embedder, upsert to Qdrant
5. **`tools/search.py`** — `embed_text(query)` → Qdrant search → return ranked paths
6. **`main.py`** — wire the three FastAPI routes
7. **`agent.py`** — register tools with OpenClaw, test a round-trip query
8. **`pipeline/tagger.py` + `tools/tag.py`** — add zero-shot tagging last (depends on embedder already working)

---

## Required Stack

```
# requirements.txt
open-clip-torch
qdrant-client
fastapi
uvicorn
pillow
torch
transformers        # for SigLIP / zero-shot tagging
openclaw            # agent orchestration layer
```

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant Edge (local binary)
./qdrant --config-path config/local.yaml

# 3. Index your photo library
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"folder": "/Users/you/Photos"}'

# 4. Start the API
uvicorn main:app --reload

# 5. Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "beach sunsets with dogs"}'
```

---

## Advanced Enhancements (Post-MVP)

- **Face clustering** — group photos by detected face embeddings
- **OCR for screenshots** — extract and index text from screenshots and documents
- **Video frame indexing** — sample keyframes, embed, and index alongside photos
- **Hybrid search** — combine keyword filters (date, filename, tag) with vector similarity
- **Time-aware retrieval** — weight recent photos higher for contextual queries

---

## References

- [Qdrant Edge Documentation](https://qdrant.tech/documentation/edge/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenClaw GitHub](https://github.com/openclaw)
- [OpenAI CLIP Research](https://openai.com/research/clip)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [open-clip-torch](https://github.com/mlfoundations/open_clip)