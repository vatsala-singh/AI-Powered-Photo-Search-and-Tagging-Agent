import shutil
import pytest
from pathlib import Path
from pipeline.indexer import index_folder


def test_indexer_smoke():
    """Smoke test: verify indexer module loads and runs without error."""
    # Clean shard for a fresh test
    shutil.rmtree("./qdrant-edge-data", ignore_errors=True)

    folder = Path.home() / "Users" / "vatsalasingh" / "Documents" / "Datasets" / "sample"
    if not folder.exists():
        pytest.skip("Test data not available at ~/Pictures/test")

    # First run — should index everything
    result1 = index_folder(folder)
    assert isinstance(result1, dict), "index_folder should return a dict"
    assert "indexed" in result1, "Result should contain 'indexed' key"
    assert "skipped" in result1, "Result should contain 'skipped' key"

    # Second run — everything should be skipped (dedup check)
    result2 = index_folder(folder)
    assert result2["indexed"] == 0, "Should not re-index already-indexed images"