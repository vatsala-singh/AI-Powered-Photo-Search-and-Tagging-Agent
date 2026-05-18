# pipeline/indexer.py
import uuid
from pathlib import Path
from datetime import datetime

from PIL import Image
from qdrant_edge import Point, UpdateOperation

from config import VECTOR_NAME
from store.qdrant_client import get_shard
from store.schema import PhotoPayload
from pipeline.embedder import embed_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff"}


def _load_indexed_paths() -> set[str]:
    """
    Scan the shard for all stored 'path' payload values and return them
    as a set. Used to skip already-indexed images without re-embedding.

    EdgeShard has no payload index, so we do a full in-memory scan once
    at indexing time. For a personal library (<100k photos) this is fast.
    """
    shard = get_shard()
    indexed = set()
    offset = 0
    batch_size = 256

    while True:
        from qdrant_edge import Query, QueryRequest
        # Use a dummy zero-vector broad scan to page through all points.
        # We only need payloads, not vectors.
        results = shard.query(
            QueryRequest(
                query=Query.Nearest([0.0] * 512, using=VECTOR_NAME),
                limit=batch_size,
                with_vector=False,
                with_payload=True,
            )
        )
        for point in results:
            path = (point.payload or {}).get("path")
            if path:
                indexed.add(path)

        if len(results) < batch_size:
            break
        offset += batch_size

    return indexed


def _get_image_meta(path: Path) -> dict:
    """Extract width, height, and file modification timestamp."""
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception:
        width, height = None, None

    mtime = path.stat().st_mtime
    timestamp = datetime.fromtimestamp(mtime).isoformat()
    return {"width": width, "height": height, "timestamp": timestamp}


def index_folder(folder: str | Path, batch_size: int = 32) -> dict:
    """
    Walk `folder` recursively, embed every supported image, and upsert
    into the EdgeShard. Already-indexed images are skipped.

    Returns a summary dict: {indexed, skipped, failed}.
    """
    shard = get_shard()
    folder = Path(folder)

    all_images = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"[indexer] Found {len(all_images)} images in {folder}")

    # Load already-indexed paths once upfront
    print("[indexer] Loading indexed paths ...")
    already_indexed = _load_indexed_paths()
    print(f"[indexer] {len(already_indexed)} images already in shard")

    indexed = skipped = failed = 0
    batch: list[Point] = []

    def flush(batch: list[Point]) -> None:
        if batch:
            shard.update(UpdateOperation.upsert_points(batch))
            shard.flush()  # Persist to disk immediately

    for img_path in all_images:
        path_str = str(img_path.resolve())

        if path_str in already_indexed:
            skipped += 1
            continue

        vec = embed_image(img_path)
        if vec is None:
            failed += 1
            continue

        meta = _get_image_meta(img_path)
        payload = PhotoPayload(
            filename=img_path.name,
            path=path_str,
            tags=[],
            timestamp=meta["timestamp"],
            width=meta["width"],
            height=meta["height"],
        )

        batch.append(
            Point(
                id=str(uuid.uuid4()),
                vector={VECTOR_NAME: vec.tolist()},
                payload=payload.to_dict(),
            )
        )

        if len(batch) >= batch_size:
            batch_size_actual = len(batch)
            flush(batch)
            indexed += batch_size_actual
            print(f"[indexer] Flushed {batch_size_actual} — total indexed: {indexed}")
            batch = []

    # Flush any remaining points
    if batch:
        final_batch_size = len(batch)
        flush(batch)
        indexed += final_batch_size
        print(f"[indexer] Flushed final {final_batch_size} — total indexed: {indexed}")

    summary = {"indexed": indexed, "skipped": skipped, "failed": failed}
    print(f"[indexer] Done — {summary}")
    return summary