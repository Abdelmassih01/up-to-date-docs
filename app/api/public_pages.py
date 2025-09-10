from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from app.models.page import PageDocument
from html import escape
from bson import ObjectId
from datetime import datetime
from typing import List, Optional

router = APIRouter()

def _slugify(title: str) -> str:
    return "-".join(
        "".join(ch.lower() if ch.isalnum() else "-" for ch in title).split("-")
    ) or "page"

def _clamp_heading_level(level: int | None) -> int:
    # Use h2..h6 for sections (default h2)
    if not isinstance(level, int):
        return 2
    return min(6, max(2, level))

def _render_html(doc: PageDocument, public_url: str) -> str:
    # Build simple static HTML (no JS). All user content is escaped.
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(f"<title>{escape(doc.title or 'Document')}</title>")
    parts.append('<meta name="robots" content="index,follow">')
    parts.append(f'<link rel="canonical" href="{escape(public_url)}">')
    parts.append("""<style>
      body{font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:2rem;color:#111;max-width:900px}
      h1{margin:0 0 .25rem 0} .meta{color:#555;margin:0 0 1.25rem 0;font-size:.95rem}
      pre{padding:.75rem;border:1px solid #ddd;border-radius:6px;overflow:auto;background:#fafafa}
      code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
      section{margin:1.25rem 0}
      a{color:#0b5fff;text-decoration:none} a:hover{text-decoration:underline}
    </style>""")
    parts.append("</head><body>")

    parts.append(f"<h1>{escape(doc.title or 'Document')}</h1>")
    last = doc.last_crawled if isinstance(doc.last_crawled, datetime) else None
    last_str = last.isoformat(timespec="seconds") if last else ""
    parts.append('<div class="meta">')
    parts.append(f'Source: <a href="{escape(doc.url)}" rel="noopener nofollow">{escape(doc.url)}</a><br>')
    if last_str:
        parts.append(f"Last crawled: {escape(last_str)}")
    parts.append("</div>")

    # Sections
    for s in (doc.sections or []):
        parts.append("<section>")
        if s.heading:
            lvl = _clamp_heading_level(s.heading_level)
            parts.append(f"<h{lvl}>{escape(s.heading)}</h{lvl}>")
        if s.text:
            # Split paragraphs by double newline for readability
            for para in (s.text.split("\n\n")):
                if para.strip():
                    parts.append(f"<p>{escape(para)}</p>")
        for c in (s.codes or []):
            lang = escape(c.language or "text")
            code_text = escape(c.code or "")
            parts.append(f'<pre><code class="language-{lang}">{code_text}</code></pre>')
        parts.append("</section>")

    # Optional footer linking back to source
    parts.append("<hr>")
    parts.append(f'<p class="meta">Canonical source: <a href="{escape(doc.url)}" rel="noopener nofollow">{escape(doc.url)}</a></p>')

    parts.append("</body></html>")
    return "".join(parts)

# NEW: Extracted reusable functions
async def get_page_document_by_id(doc_id: str) -> Optional[PageDocument]:
    """Get a PageDocument by ID, returns None if not found or invalid ID"""
    try:
        oid = ObjectId(doc_id)
        return await PageDocument.get(oid)
    except Exception:
        return None
    
async def get_page_documents_by_ids(doc_ids: List[str]) -> List[PageDocument]:
    """Fetch multiple PageDocuments by IDs in a single query."""
    object_ids = []
    for doc_id in doc_ids:
        try:
            object_ids.append(ObjectId(doc_id))
        except Exception:
            continue
    if not object_ids:
        return []

    # ✅ Correct Beanie query
    return await PageDocument.find({"_id": {"$in": object_ids}}).to_list()

def render_single_page_html(doc: PageDocument) -> str:
    """Render a single PageDocument to HTML"""
    slug = _slugify(doc.title or "page")
    public_url = f"/public/pages/{doc.id}/{slug}"
    return _render_html(doc, public_url=public_url)

def render_combined_pages_html(pages: List[PageDocument], query: str) -> str:
    """Render multiple PageDocuments into a single search results HTML page"""
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(f"<title>Search Results: {escape(query)}</title>")
    parts.append('<meta name="robots" content="noindex,nofollow">')
    parts.append("""<style>
      body{font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:2rem;color:#111;max-width:900px}
      h1{margin:0 0 1rem 0} .query-info{color:#555;margin:0 0 2rem 0;font-size:.95rem;padding:.75rem;background:#f8f9fa;border-radius:6px}
      .document{margin:2rem 0;padding:1.5rem;border:1px solid #e1e5e9;border-radius:8px;background:#fefefe}
      .document h2{margin:0 0 .5rem 0;color:#0b5fff} 
      .document .meta{color:#555;margin:0 0 1.25rem 0;font-size:.9rem}
      pre{padding:.75rem;border:1px solid #ddd;border-radius:6px;overflow:auto;background:#fafafa}
      code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
      section{margin:1.25rem 0}
      a{color:#0b5fff;text-decoration:none} a:hover{text-decoration:underline}
      .document-separator{margin:2rem 0;border:none;height:2px;background:#e1e5e9}
    </style>""")
    parts.append("</head><body>")

    # Header with search info
    parts.append(f"<h1>Search Results</h1>")
    parts.append(f'<div class="query-info">Query: <strong>{escape(query)}</strong><br>Found {len(pages)} document(s)</div>')

    # Render each document
    for i, doc in enumerate(pages):
        if i > 0:
            parts.append('<hr class="document-separator">')
        
        parts.append('<div class="document">')
        parts.append(f"<h2>{escape(doc.title or 'Document')}</h2>")
        
        last = doc.last_crawled if isinstance(doc.last_crawled, datetime) else None
        last_str = last.isoformat(timespec="seconds") if last else ""
        parts.append('<div class="meta">')
        parts.append(f'Source: <a href="{escape(doc.url)}" rel="noopener nofollow" target="_blank">{escape(doc.url)}</a><br>')
        if last_str:
            parts.append(f"Last crawled: {escape(last_str)}")
        parts.append("</div>")

        # Sections (reusing existing logic)
        for s in (doc.sections or []):
            parts.append("<section>")
            if s.heading:
                lvl = _clamp_heading_level(s.heading_level)
                parts.append(f"<h{lvl}>{escape(s.heading)}</h{lvl}>")
            if s.text:
                for para in (s.text.split("\n\n")):
                    if para.strip():
                        parts.append(f"<p>{escape(para)}</p>")
            for c in (s.codes or []):
                lang = escape(c.language or "text")
                code_text = escape(c.code or "")
                parts.append(f'<pre><code class="language-{lang}">{code_text}</code></pre>')
            parts.append("</section>")
        
        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)

# Updated endpoints using the extracted functions
@router.get("/public/pages/{doc_id}", response_class=HTMLResponse, tags=["public"])
async def get_public_page(doc_id: str):
    doc = await get_page_document_by_id(doc_id)
    if not doc:
        print("Page not found")
        raise HTTPException(status_code=404, detail="Page not found")

    html = render_single_page_html(doc)
    return HTMLResponse(
        content=html,
        status_code=200,
        headers={"Cache-Control": "public, max-age=300"}
    )

@router.get("/public/pages/{doc_id}/{slug}", response_class=HTMLResponse, tags=["public"])
async def get_public_page_with_slug(doc_id: str, slug: str):
    # Delegate to the primary handler; slug is ignored for lookup (SEO only)
    return await get_public_page(doc_id)