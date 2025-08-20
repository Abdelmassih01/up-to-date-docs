import os
import chromadb
from sentence_transformers import SentenceTransformer

# Singletons (loaded at startup)
client = None
collection = None
embedder = None
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "crawled_docs_pages")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

def init_vector():
    global client, collection, embedder
    if client is None:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)
    if collection is None:
        # cosine is default in recent versions; metadata purely informational
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"embedding_model": EMBED_MODEL}
        )
    return client, collection, embedder
