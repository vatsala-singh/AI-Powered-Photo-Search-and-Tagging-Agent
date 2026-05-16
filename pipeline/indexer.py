#This class is reponsible for walking through a folder, 
# embed images
# skip already indexed images,
# upsert to Qdrant

import uuid 
from pathlib import Path
from datetime import datetime

from PIL import Image
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from config import COLLECTION_NAME
from store.qdrant_client import get_client, ensure_collection
from store.schema import PhotoPayload
from pipeline.embedder import embed_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff"}

def _already_indexed(path: str) -> bool:
    #check if a photo with the same path already exists in Qdrant
    client = get_client()
    result,_ = client.scroll(
        collection_name = COLLECTION_NAME,
        scroll_filter = Filter(
            must =[
                FieldCondition(
                    key="path",
                    match=MatchValue(value=path)
                )]),
                limit=1,
    )
    return len(result) > 0

def _get_image_meta(path: Path) -> dict:
    #extract basic metadata from the image file
    try:
        with Image.open(path) as img:
            width, height = img.size
    except:
        width, height = None, None
    mtime = path.stat().st_mtime
    timestamp = datetime.fromtimestamp(mtime).isoformat()
    return {"width": width, "height": height, "timestamp": timestamp}

def index_folder(folder: str | Path, batch_size: int = 32) -> dict:
    # walktrough the folder recursively
    # check if an image was already indexed
    # if not, embed and upsert to Qdrant in batches
    # already indexed images are skipped
    
    ensure_collection()
    client = get_client()
    folder = Path(folder)
    
    all_images = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    print(f"[indexer] Found {len(all_images)} images in '{folder}'")
    indexed = skipped = failed = 0
    batch: list[PointStruct] = []
    
    def flush(batch):
        if batch:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
    for img_path in all_images:
        
        path_str = str(img_path.resolve())
        
        if _already_indexed(path_str):
            skipped += 1
            continue
        vec = embed_image(img_path)
        if vec is None:
            failed += 1
            continue
        meta = _get_image_meta(img_path)
        payload =PhotoPayload(
            filename=img_path.name,
            path=path_str,
            tags=[],
            timestamp=meta["timestamp"],
            width=meta["width"],
            height=meta["height"]
        )
        
        batch.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload=payload.to_dict()
            )
        )
        if len(batch) >= batch_size:
            flush(batch)
            indexed += len(batch)
            print(f"[indexer] Indexed {indexed} images so far... Skipped: {skipped}, Failed: {failed}")
            batch = []
            
    #flush remaining
    flush(batch)
    indexed += len(batch)
    
    summary = {
        "total_found": len(all_images),
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed
    }
    print(f"[indexer] Done. Summary: {summary}")
    return summary