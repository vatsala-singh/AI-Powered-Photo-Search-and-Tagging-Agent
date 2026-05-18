# AI-Powered Photo Search and Tagging Agent

A **local-first, AI-powered photo assistant** that enables natural language search, automatic tagging, and intelligent organization of your photo library using CLIP embeddings, Qdrant Edge vector database, and OpenClaw agent framework.

## Features

- **Natural Language Search** – Find photos using plain English queries ("sunset over mountains", "group photo of friends")
- **Auto-Tagging** – Zero-shot automatic tagging of images based on visual content
- **Duplicate Detection** – Identify and cluster near-identical images by vector similarity
- **Smart Albums** – Group images semantically by topic or theme
- **OCR & Face Detection** – Extract text from screenshots and cluster faces
- **Local-First** – All processing happens on-device; no cloud services required
- **Fast** – CPU-friendly embedding model (ViT-B/32) with instant search results

## System Architecture

```
Photos → CLIP Embeddings → Qdrant Edge → OpenClaw Agent → Search Results
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Embedder** | CLIP/SigLIP model for generating vector embeddings from images |
| **Indexer** | Batch and incremental image indexing pipeline |
| **Qdrant Edge** | In-process vector database (no server needed) for fast local similarity search |
| **Search Tool** | Natural language photo search via semantic similarity |
| **Tagging Tool** | Zero-shot multi-label image classification |
| **Duplicates Tool** | Cluster detection for finding duplicate/similar images |
| **FastAPI Server** | REST API for indexing, searching, and retrieval |

### About Qdrant Edge

**Qdrant Edge** is an embedded, in-process vector database that:
- ✅ Runs entirely on your machine (no network, no external services)
- ✅ Automatically persists data to disk at `./qdrant-edge-data/photos`
- ✅ Provides instant startup with zero configuration
- ✅ Supports the same powerful vector search operations as server-mode Qdrant
- ✅ Flushes writes on application shutdown to ensure data integrity

## Project Structure

```
ai-photo-agent/
├── main.py                    # FastAPI app entry point
├── agent.py                   # OpenClaw agent setup & tool registration
├── config.py                  # Configuration (model, paths, Qdrant settings)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── pipeline/                  # Data processing pipeline
│   ├── embedder.py            # CLIP/SigLIP embedding model
│   ├── indexer.py             # Image indexing & ingestion
│   └── tagger.py              # Zero-shot auto-tagging
│
├── store/                     # Vector database & storage
│   ├── qdrant_client.py       # Qdrant connection & collection setup
│   └── schema.py              # Payload schema (tags, filename, timestamp, path)
│
├── tools/                     # Agent tools
│   ├── search.py              # Natural language search tool
│   ├── tag.py                 # Image tagging tool
│   ├── duplicates.py          # Duplicate detection tool
│   └── albums.py              # Smart album creation tool
│
└── test/                      # Test suite
    └── embedder_test.py       # Embedding model tests
```

## Prerequisites

- Python 3.9+
- pip or conda
- ~2GB free disk space (for models and vector database)
- **No external services required** – Qdrant Edge runs entirely in-process, locally on your machine

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-Powered-Photo-Search-and-Tagging-Agent.git
cd AI-Powered-Photo-Search-and-Tagging-Agent
```

### 2. Create a virtual environment

```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n photo-agent python=3.9
conda activate photo-agent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `open-clip-torch` – CLIP embedding model
- `qdrant-edge` – Local in-process vector database (no server needed)
- `fastapi` & `uvicorn` – Web server
- `pillow` – Image processing
- `torch` & `torchvision` – PyTorch deep learning framework
- Additional utilities

### 4. Verify installation

```bash
python -c "import torch; import clip; from qdrant_edge import EdgeShard; print('✓ All dependencies installed')"
```

## Configuration

Edit [config.py](config.py) to customize:

```python
# Embedding model
EMBED_MODEL = "ViT-B-32"              # CLIP model variant
EMBED_PRETRAINED = "openai"            # Pretrained weights source
EMBED_DIM = 512                        # Vector dimension

# Qdrant Edge (local, in-process)
SHARD_DIR = Path("./qdrant-edge-data/photos")  # Local shard directory
VECTOR_NAME = "clip"                   # Named vector key

# Search settings
TOP_K = 10                             # Default number of results
DUPLICATE_THRESHOLD = 0.97             # Similarity threshold for duplicates

# Paths
PHOTO_DIR = Path.home() / "Pictures"   # Default photo directory
```

## Quick Start

### Option 1: Using the FastAPI Server

#### Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Qdrant Edge automatically initializes its local shard on first use and reuses it on subsequent runs.

#### API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### API Usage with cURL

**Index photos**
```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"folder": "/path/to/photos"}'
```

**Search photos**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sunset over the ocean",
    "top_k": 10,
    "tags": []
  }'
```

**Get tags for an image**
```bash
curl "http://localhost:8000/tags/img_001"
```

**Chat with the agent**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me photos of my dog playing in the park",
    "history": []
  }'
```

**Debug: Get shard statistics**
```bash
curl "http://localhost:8000/debug/shard-stats"
```

**Health check**
```bash
curl "http://localhost:8000/health"
```

### Option 2: Python Script

Index a folder and search:

```python
from pipeline.indexer import index_folder
from tools.search import search_photos
from pathlib import Path

# Index photos
result = index_folder(Path("/path/to/your/photos"))
print(f"Indexed {result['total']} photos")

# Search
results = search_photos("beautiful landscape", top_k=5)
for match in results:
    print(f"  {match['path']} (score: {match['score']:.3f})")
```

### Option 3: Using the Agent

```python
from agent import create_agent

agent = create_agent()

# Use agent to answer questions about photos
response = agent.run("Show me photos of my dog playing in the park")
print(response)
```

## Usage Examples

### 1. Index Your Photo Library

```python
from pipeline.indexer import index_folder
from pathlib import Path

result = index_folder(Path("/Users/you/Pictures"))
print(result)
# Output: {
#   "total": 1543,
#   "successful": 1540,
#   "failed": 3,
#   "duration_seconds": 45.2
# }
```

### 2. Search with Natural Language

```python
from tools.search import search_photos

results = search_photos("sunset at beach", top_k=5)
for match in results:
    print(f"{match['path']}: {match['score']:.2%} match")
```

### 3. Auto-Tag an Image

```python
from tools.tag import tag_image

tags = tag_image("/path/to/photo.jpg")
print(tags)  # ['sunset', 'beach', 'ocean', 'sky', 'landscape']
```

### 4. Find Duplicate Images

```python
from tools.duplicates import find_duplicates

clusters = find_duplicates(threshold=0.95)
for cluster in clusters:
    print(f"Found {len(cluster)} duplicate images")
    for image in cluster:
        print(f"  - {image}")
```

### 5. Create Smart Albums

```python
from tools.albums import create_smart_album

album = create_smart_album("photos from 2024 vacation")
print(f"Created album with {len(album)} images")
```

## API Endpoints

### POST `/index`
Index a folder of photos.

**Request:**
```json
{
  "folder": "/path/to/photos"
}
```

**Response:**
```json
{
  "status": "success",
  "summary": {
    "total": 1543,
    "successful": 1540,
    "failed": 3,
    "duration_seconds": 45.2
  }
}
```

### POST `/search`
Search photos with natural language.

**Request:**
```json
{
  "query": "sunset over mountains",
  "top_k": 10,
  "tags": []
}
```

**Response:**
```json
{
  "query": "sunset over mountains",
  "results": [
    {
      "id": "img_001",
      "path": "/photos/sunset_001.jpg",
      "score": 0.87,
      "tags": ["sunset", "mountains", "landscape"]
    }
  ]
}
```

### GET `/tags/{image_id}`
Get tags for a specific image.

**Response:**
```json
{
  "id": "img_001",
  "tags": ["sunset", "mountains", "landscape", "sky"]
}
```

### GET `/health`
Health check.

**Response:**
```json
{
  "status": "ok"
}
```

### POST `/chat`
Chat with the agent to perform complex photo queries and tasks.

**Request:**
```json
{
  "message": "Show me photos of my dog playing in the park",
  "history": []
}
```

**Response:**
```json
{
  "reply": "I found 5 photos of your dog playing. The most recent ones show..."
}
```

**Example with history:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you also find sunset photos?",
    "history": [
      {"role": "user", "content": "Show me photos of my dog"},
      {"role": "assistant", "content": "I found 5 photos..."}
    ]
  }'
```

### GET `/debug/shard-stats`
Get vector database statistics for debugging and monitoring.

**Response:**
```json
{
  "shard_path": "/Users/vatsalasingh/Documents/GitHub/AI-Powered-Photo-Search-and-Tagging-Agent/storage/collections/photos/0",
  "total_points": 1540,
  "status": "has_data"
}
```

## Performance Tips

- **First Index**: The initial indexing of a large photo library takes time (embedding generation is CPU-intensive). For 1000 photos, expect 5-10 minutes on a modern CPU.
- **Batch Size**: Adjust batch sizes in [pipeline/indexer.py](pipeline/indexer.py) for your available RAM.
- **Model**: ViT-B/32 is fast and memory-efficient. For higher quality, use ViT-L/14 (slower, ~2x larger).
- **GPU**: If you have a CUDA-capable GPU, PyTorch will automatically use it for faster embeddings.

## Troubleshooting

### Issue: "Shard directory error" or "Failed to create shard"
**Solution**: Ensure the shard directory path exists and is writable. Check your `SHARD_DIR` setting in [config.py](config.py):
```python
SHARD_DIR = Path("./qdrant-edge-data/photos")
```
The directory will be created automatically if it doesn't exist.

### Issue: Out of memory errors
**Solution**: Reduce batch size in config or process fewer images at once.

### Issue: Slow search results
**Solution**: 
- Ensure you have GPU available (CUDA/Metal)
- Check the number of indexed images
- Verify shard statistics: `curl http://localhost:8000/debug/shard-stats`
- Reduce batch size in [pipeline/indexer.py](pipeline/indexer.py) if experiencing memory issues

### Issue: Poor search quality
**Solution**:
- Make sure you've indexed enough photos (at least 50-100)
- Try more specific queries
- Check that auto-tagging is enabled for better metadata

## Development

### Running Tests

```bash
pytest test/ -v
```

### Running Tests for Embedder

```bash
python test/embedder_test.py
```

### Agent

#### Install Ollama + pull a model
```bash
# Install Ollama
brew install ollama

# Pull the model (one-time download, ~4.7 GB)
ollama pull qwen2.5:7b

# Verify it's working
ollama run qwen2.5:7b "reply with: pong"
```
#### leave Ollama running
```bash
ollama serve &
```

#### Install OpenClaw CLI
```bash
npm install -g openclaw@latest

# Verify
openclaw --version
```
#### Register Ollama as the provider

```bash
# Tell OpenClaw Ollama is the provider
export OLLAMA_API_KEY="ollama-local"

openclaw models list --provider ollama    # shows installed models
openclaw models set ollama/qwen2.5:7b    # set as default
```

Verify the model can receive tool calls:
```bash
OLLAMA_API_KEY=ollama-local \
openclaw infer model run \
  --local \
  --model ollama/qwen2.5:7b \
  --prompt "Reply with exactly: pong" \
  --json
```
#### Onboard the photo agent
```bash
# Run in your photo-agent project directory
openclaw onboard --install-daemon
```


### Code Structure Guidelines

- **Pipeline modules** handle data processing (embeddings, indexing)
- **Store modules** manage vector database operations
- **Tools** are stateless functions called by the agent
- **Main/Agent** orchestrate everything

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License – see LICENSE file for details.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) – Vision-language model
- [Qdrant](https://qdrant.tech/) – Vector database
- [FastAPI](https://fastapi.tiangolo.com/) – Web framework
- [OpenClaw](https://github.com/ucb-oarc/openclaw) – Agent framework

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
---

**Happy searching! 📸**


