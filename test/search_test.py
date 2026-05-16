from tools.search import search_photos
import json

results = search_photos("sunset over water", top_k=5)
for r in results:
    print(f"{r['score']:.3f}  {r['filename']}")