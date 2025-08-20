from datetime import datetime
from typing import Dict
from app.core.vector import init_vector

def build_page_text(sections) -> str:
    """Concatenate sections' text and code (preserving code context)."""
    parts = []
    for s in sections or []:
        if s.heading:
            parts.append(f"# {s.heading}\n")
        if s.text:
            parts.append(s.text.strip() + "\n")
        for c in (s.codes or []):
            lang = c.language or "text"
            parts.append(f"```{lang}\n{c.code}\n```\n")
    return "\n".join(parts).strip()

def embed_text(text: str):
    _, _, embedder = init_vector()
    return embedder.encode([text], normalize_embeddings=True)[0].tolist()

def chroma_upsert(doc_id: str, text: str, meta: Dict):
    _, collection, _ = init_vector()
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[meta]
    )

def chroma_delete(doc_id: str):
    _, collection, _ = init_vector()
    try:
        collection.delete(ids=[doc_id])
    except Exception:
        # safe to ignore; delete is idempotent
        pass
