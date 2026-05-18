import shutil
from pathlib import Path

# Start clean
test_dir = Path("./qdrant-edge-data")
if test_dir.exists():
    shutil.rmtree(test_dir)

from store.qdrant_client import get_shard, close_shard

# First call — should CREATE a new shard
shard = get_shard()
print("Shard created:", shard)

info = shard.info()
print("Shard info:", info)

close_shard()
print("Shard closed OK")

# Second call — should REOPEN the existing shard
shard2 = get_shard()
print("Shard reopened:", shard2)
close_shard()