from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from app.models.page import PageDocument
from html import escape
from bson import ObjectId
from datetime import datetime

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

@router.get("/public/pages/{doc_id}", response_class=HTMLResponse, tags=["public"])
async def get_public_page(doc_id: str):
    # Accept both raw hex and standard string ObjectIds
    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Page not found")

    doc = await PageDocument.get(oid)
    if not doc:
        raise HTTPException(status_code=404, detail="Page not found")

    # Pretty slug is optional; canonical points back to this path
    slug = _slugify(doc.title or "page")
    public_url = f"/public/pages/{doc_id}/{slug}"
    html = _render_html(doc, public_url=public_url)
    return HTMLResponse(
        content=html,
        status_code=200,
        headers={"Cache-Control": "public, max-age=300"}
    )

@router.get("/public/pages/{doc_id}/{slug}", response_class=HTMLResponse, tags=["public"])
async def get_public_page_with_slug(doc_id: str, slug: str):
    # Delegate to the primary handler; slug is ignored for lookup (SEO only)
    return await get_public_page(doc_id)
