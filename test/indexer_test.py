from pipeline.indexer import index_folder
from pathlib import Path

# Point this at any folder with a few images
result = index_folder(Path.home() / "Pictures")
print(result)