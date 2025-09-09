import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ----------------------------
# Environment / configuration
# ----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "crawled_docs_pages")

# Choose how to talk to Chroma:
#   - "persistent" (recommended): in-process client with local on-disk storage (fast)
#   - "http": use Chroma server over HTTP (only if you truly need a remote service)
CHROMA_MODE = os.getenv("CHROMA_MODE", "persistent")  # "persistent" | "http"

# Only used if CHROMA_MODE="http"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

# Only used if CHROMA_MODE="persistent"
# Put this on a fast SSD/NVMe if possible
CHROMA_PATH = os.getenv("CHROMA_PATH", "/var/lib/chroma")

print("CHROMA PATH:", CHROMA_PATH)

# ----------------------------
# Singletons (loaded at startup)
# ----------------------------
_client = None
_collection = None
_embedder = None


def init_vector():
    """
    Initialize the Chroma client/collection and embedding model as singletons.
    Returns (client, collection, embedder).
    """
    global _client, _collection, _embedder

    # 1) Chroma client
    if _client is None:
        if CHROMA_MODE.lower() == "http":
            _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        else:
            # In-process, persistent storage (recommended for lower latency)
            # Ensure the path exists and is writable by the app user.
            os.makedirs(CHROMA_PATH, exist_ok=True)
            _client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False),
            )

    # 2) Embedder (load once)
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)

    # 3) Collection (create or get)
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"embedding_model": EMBED_MODEL},
            embedding_function=None,
        )

    return _client, _collection, _embedder


# Helpers (optional but handy)
# ----------------------------
def get_query_embedding(query: str):
    """
    Compute a single query embedding using the loaded SentenceTransformer.
    Assumes init_vector() was called earlier (e.g., at app startup).
    """
    if _embedder is None:
        init_vector()
    # normalize embeddings for better cosine similarity behavior
    return _embedder.encode([query], normalize_embeddings=True)


def ann_search(
    query: str,
    n_results: int = 10,
    include=("ids", "distances"),
    where: dict | None = None,
    where_document: dict | None = None,
    use_precomputed_embedding: bool = True,
):
    """
    Thin wrapper around collection.query with sane defaults for latency:
      - Return only ids/distances (hydrate details from DB later)
      - Keep n_results tight (10–20)
      - Optionally filter via `where` / `where_document` to shrink the candidate set
    """
    if _collection is None or _embedder is None:
        init_vector()

    if use_precomputed_embedding:
        q_emb = _embedder.encode([query], normalize_embeddings=True)
        return _collection.query(
            query_embeddings=q_emb,
            n_results=max(1, min(n_results, 50)),
            include=list(include),
            where=where,
            where_document=where_document,
        )
    else:
        # Lets Chroma embed the query internally (slightly higher latency)
        return _collection.query(
            query_texts=[query],
            n_results=max(1, min(n_results, 50)),
            include=list(include),
            where=where,
            where_document=where_document,
        )