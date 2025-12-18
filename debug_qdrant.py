from qdrant_client import QdrantClient
import inspect

print("Checking QdrantClient API...")
client = QdrantClient(location="./qdrant_db")
print(f"Client type: {type(client)}")
print("Methods:")
for name in dir(client):
    if not name.startswith("_"):
        print(f" - {name}")

if hasattr(client, 'search'):
    print("\nHas 'search' method.")
else:
    print("\nDoes NOT have 'search' method.")
