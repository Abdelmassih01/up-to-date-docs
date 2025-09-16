from fastapi import APIRouter, Query, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import HTMLResponse
import os
from app.core.vector import init_vector
from app.api.public_pages import get_page_documents_by_ids, render_combined_pages_html

# Resolve Chroma connection from env (with safe defaults)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

# Connect to Chroma
_client, _collection, _embedder = init_vector()

router = APIRouter(tags=["Query"])

class QueryResult(BaseModel):
    id: str
    document: str
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    results: List[QueryResult]


@router.get("", response_model=QueryResponse)
def query_collection(query: str = Query(..., description="Search query")) -> QueryResponse:
    """
    Query via query string: GET /?query=...
    Always returns up to 10 results.
    """
    # 🔹 change: embed query text using HuggingFace
    q_emb = _embedder.encode([query], normalize_embeddings=True)

    # 🔹 change: pass query_embeddings instead of query_texts
    results = _collection.query(
        query_embeddings=q_emb,
        n_results=10,  # fixed default
    )

    results_array: List[QueryResult] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])
    metas = metas[0] if metas else []

    for i in range(len(ids)):
        results_array.append(
            QueryResult(
                id=ids[i],
                document=docs[i],
                metadata=metas[i] if i < len(metas) else None,
            )
        )

    return QueryResponse(results=results_array)


@router.get("/direct", response_class=HTMLResponse)
async def direct_query_to_html(query: str = Query(..., description="Search query")) -> HTMLResponse:
    """
    HTML-rendered results via query string: GET /direct?query=...
    Always uses up to 10 results.
    """
    # 🔹 change: embed query text using HuggingFace
    q_emb = _embedder.encode([query], normalize_embeddings=True)

    # 🔹 use embeddings in query
    results = _collection.query(
        query_embeddings=q_emb,
        n_results=10,  # fixed default
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])
    metas = metas[0] if metas else []

    results_array: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        results_array.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i] if i < len(metas) else None,
            }
        )

    doc_ids = [r["id"] for r in results_array]

    if not doc_ids:
        return HTMLResponse(content=_render_no_results_html(query), status_code=200)

    # Fetch corresponding PageDocuments
    pages = await get_page_documents_by_ids(doc_ids)

    if not pages:
        return HTMLResponse(content=_render_no_documents_html(query), status_code=200)

    html = render_combined_pages_html(pages, query)
    return HTMLResponse(
        content=html,
        status_code=200,
        headers={"Cache-Control": "public, max-age=300"},
    )

@router.get("/instructions.txt")
async def get_instructions():
    content = """# Instructions for querying this documentation API

    BASE_URL = "https://abdelmassih.vps.webdock.cloud/query/direct"

    ## How to build a query URL

    1. Always start with the BASE_URL.
    2. Append "?query=" followed by your search terms.
    3. Encode spaces as "%20".
    4. Do NOT include quotation marks in the query.
    5. Do NOT add extra parameters unless specified below.
    6. Keep the query terms in plain English (lowercase recommended, but not required).
    7. Each request must return a full URL that starts with BASE_URL and includes ?query=.

    ## Examples

    Search for MongoDB aggregations:
        BASE_URL?query=mongodb%20aggregations

    Search for Python aggregation examples:
        BASE_URL?query=python%20mongodb%20aggregation

    Search for $lookup usage in Python:
        BASE_URL?query=$lookup%20python

    Search for aggregation memory limit details:
        BASE_URL?query=aggregation%20memory%20limit

    ## Rules for LLMs

    - If the user provides just keywords (e.g., "python aggregation stages"), 
    you MUST build the correct URL by inserting those keywords into the query parameter.
    - Never return only the keywords. Always return the FULL URL.
    - Ensure proper URL encoding (spaces → %20, special characters like $ kept as-is).
    - Always prepend the BASE_URL exactly as given above.
    - Do not assume additional endpoints exist — only use BASE_URL + ?query=.

    """
    return Response(content=content, media_type="text/plain")

def _render_no_results_html(query: str) -> str:
    return f"""<!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>No Results Found</title>
        <style>
            body{{font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:2rem;color:#111;max-width:900px}}
            h1{{margin:0 0 1rem 0;color:#666}}
        </style>
    </head>
    <body>
        <h1>No Results Found</h1>
        <p>Your query "{query}" didn't match any documents.</p>
    </body>
    </html>"""


def _render_no_documents_html(query: str) -> str:
    return f"""<!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Documents Not Found</title>
        <style>
            body{{font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:2rem;color:#111;max-width:900px}}
            h1{{margin:0 0 1rem 0;color:#666}}
        </style>
    </head>
    <body>
        <h1>Documents Not Found</h1>
        <p>The documents matching "{query}" could not be retrieved.</p>
    </body>
    </html>"""
