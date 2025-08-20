import re
import time
import hashlib
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from beanie import PydanticObjectId

from app.models.page import PageDocument, Heading, Section, CodeBlock
import asyncio
from app.services.embedding_service import build_page_text, embed_text, chroma_upsert, chroma_delete
from app.core.vector import EMBED_MODEL
from app.models.page import PageDocument


# ---------- Config ----------
REQ_TIMEOUT = 25
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# ---------- Helpers ----------
def normalize_url(url: str) -> str:
    clean, _ = urldefrag(url.strip())
    return clean

def same_scope(url: str, base_url: str) -> bool:
    pu, pb = urlparse(url), urlparse(base_url)
    if pu.netloc != pb.netloc:
        return False
    base_path = pb.path if pb.path.endswith("/") else pb.path + "/"
    return pu.path.startswith(base_path) or pu.path == pb.path

def fetch(url: str) -> Optional[str]:
    try:
        res = requests.get(url, timeout=REQ_TIMEOUT, headers=HEADERS)
        if res.ok:
            return res.text
        print(f"[ERROR] {url} -> HTTP {res.status_code}")
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
    return None

def pick_content_root(soup: BeautifulSoup):
    candidates = [
        ".devsite-article-body",
        ".devsite-main-content",
        "article.devsite-article",
        "main.devsite-main-content",
        "main",
    ]
    for sel in candidates:
        el = soup.select_one(sel)
        if el:
            return el
    return soup.body or soup

def heading_level(tag_name: str) -> Optional[int]:
    m = re.fullmatch(r"h([1-6])", tag_name.lower())
    return int(m.group(1)) if m else None

def detect_lang_from_classes(el) -> Optional[str]:
    classes = el.get("class") or []
    for prefix in ("language-", "lang-", "code-", "highlight-"):
        for cls in classes:
            cls_l = cls.lower()
            if cls_l.startswith(prefix):
                return cls_l.replace(prefix, "")
    for attr in ("data-language", "data-lang"):
        if el.has_attr(attr) and el[attr]:
            return str(el[attr]).lower()
    if el.name == "pre":
        code = el.find("code")
        if code:
            c2 = code.get("class") or []
            for cls in c2:
                cls_l = cls.lower()
                for prefix in ("language-", "lang-", "highlight-"):
                    if cls_l.startswith(prefix):
                        return cls_l.replace(prefix, "")
    return None

def extract_rich_sections(soup: BeautifulSoup) -> Tuple[List[Section], List[str]]:
    root = pick_content_root(soup)
    rich: List[Section] = []

    current = {
        "heading": None,
        "heading_level": None,
        "text_chunks": [],
        "codes": [],  # list[CodeBlock]
    }

    def flush():
        text = "\n\n".join([t for t in current["text_chunks"] if t.strip()]).strip()
        has_code = len(current["codes"]) > 0
        if (current["heading"] or text or has_code):
            rich.append(Section(
                heading=current["heading"],
                heading_level=current["heading_level"],
                text=(text if text else None),
                codes=list(current["codes"]),
            ))

    block_selectors = (
        "h1,h2,h3,h4,h5,h6,"
        "p,ul,ol,li,pre,code,table,blockquote,"
        "div.devsite-article-body,div.devsite-code,devsite-code,"
        "div,section,article"
    )
    nodes = root.select(block_selectors)

    def is_code_under_pre(el):
        return el.name == "code" and el.find_parent("pre")

    for el in nodes:
        if is_code_under_pre(el):
            continue

        lvl = heading_level(el.name) if el.name else None
        if lvl:
            flush()
            current = {
                "heading": el.get_text(" ", strip=True),
                "heading_level": lvl,
                "text_chunks": [],
                "codes": [],
            }
            continue

        if el.name in ("div", "devsite-code") and (
            "devsite-code" in (el.get("class") or []) or el.name == "devsite-code"
        ):
            pres = el.find_all("pre")
            if not pres:
                code_text = el.get_text("\n", strip=True)
                if code_text:
                    current["codes"].append(CodeBlock(language=detect_lang_from_classes(el), code=code_text))
            else:
                for pre in pres:
                    code_el = pre.find("code") or pre
                    code_text = code_el.get_text("\n", strip=True)
                    if code_text:
                        current["codes"].append(
                            CodeBlock(language=detect_lang_from_classes(pre) or detect_lang_from_classes(code_el),
                                      code=code_text)
                        )
            continue

        if el.name == "pre":
            code_el = el.find("code") or el
            code_text = code_el.get_text("\n", strip=True)
            if code_text:
                current["codes"].append(
                    CodeBlock(language=detect_lang_from_classes(el) or detect_lang_from_classes(code_el),
                              code=code_text)
                )
            continue

        if el.name == "table":
            text = el.get_text(" ", strip=True)
            if text:
                current["text_chunks"].append(text)
            continue

        if el.name in ("p", "li", "blockquote"):
            txt = el.get_text(" ", strip=True)
            if txt:
                current["text_chunks"].append(txt)
            continue

        if el.name in ("div", "section", "article"):
            if el.find(["h1", "h2", "h3", "h4", "h5", "h6"]) is None:
                if el.find("pre"):
                    continue
                t = el.get_text(" ", strip=True)
                if t and len(t) < 600:
                    current["text_chunks"].append(t)
            continue

    flush()

    flat_codes: List[str] = []
    for sec in rich:
        for c in sec.codes:
            flat_codes.append(c.code)

    return rich, flat_codes

def extract_page(doc_url: str, html: str):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else doc_url

    # headings (filter obvious UI clutter later if desired)
    headings: List[Heading] = []
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        headings.append(Heading(level=int(tag.name[1]), text=tag.get_text(" ", strip=True)))

    sections, flat_code_blocks = extract_rich_sections(soup)

    # Build a hash (for future change detection)
    concat = title + "\n".join(h.text or "" for h in headings) \
             + "\n".join((s.text or "") + "".join(cb.code for cb in s.codes) for s in sections)
    content_hash = hashlib.sha256(concat.encode("utf-8")).hexdigest()

    doc = PageDocument(
        url=doc_url,
        title=title,
        headings=headings,
        sections=sections,
        last_crawled=datetime.utcnow(),
        code_blocks_flat=flat_code_blocks,
        summary=None,
        metadata={},
        hash=content_hash,
    )
    return doc

EXCLUDE_PATH_RE = re.compile(
    r"(?:/blob/|/tree/|/raw/|/pulls?|/issues?|/actions?|/commits?/|/compare/"
    r"|/releases?/|/tags?/|/forks?/|/pulse/?$|/security/?$|/contributors?/?$"
    r"|/\.git/|/\.github/|/\.devcontainer/|/node_modules/|/vendor/|/dist/|/build/)"
)

EXCLUDE_EXTS = {
    "lock","yml","yaml","toml","ini","cfg","env","properties","json","ndjson","map",
    "zip","tar","gz","tgz","rar","7z",
    "jpg","jpeg","png","gif","bmp","webp","svg","ico","mp4","mp3","mov","avi","webm","wav",
    "pdf","ppt","pptx","doc","docx","xls","xlsx","csv",
    "tsconfig","eslintrc","prettierrc","npmrc","babelrc",
}
EXCLUDE_BASENAMES = {"LICENSE","CHANGELOG","CONTRIBUTING","SECURITY","README","NOTICE","CODE_OF_CONDUCT"}
EXCLUDE_QUERY_KEYS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","ref","source"}

def should_skip_url(url: str) -> bool:
    u = urlparse(url)
    path = u.path or "/"
    if EXCLUDE_PATH_RE.search(path):
        return True
    base = path.rsplit("/", 1)[-1]
    if (base.split(".", 1)[0].upper() in EXCLUDE_BASENAMES):
        return True
    if "." in base:
        parts = base.lower().split(".")[1:]
        if any(p in EXCLUDE_EXTS for p in parts):
            return True
    if u.query:
        for qp in u.query.split("&"):
            k = qp.split("=", 1)[0].lower()
            if k in EXCLUDE_QUERY_KEYS:
                return True
    return False

async def upsert_page(doc: PageDocument):
    # upsert by URL
    existing = await PageDocument.find_one(PageDocument.url == doc.url)
    if existing:
        # optional: skip if hash unchanged
        if existing.hash == doc.hash:
            print(f"[SKIP] {doc.url} (unchanged)")
            return existing.id
        existing.title = doc.title
        existing.headings = doc.headings
        existing.sections = doc.sections
        existing.last_crawled = doc.last_crawled
        existing.code_blocks_flat = doc.code_blocks_flat
        existing.summary = doc.summary
        existing.metadata = doc.metadata
        existing.hash = doc.hash
        await existing.save()
        print(f"[UPDATE] {doc.url}")
        return existing.id
    else:
        saved = await doc.insert()
        print(f"[INSERT] {doc.url}")
        return saved.id

async def crawl_site(base_url: str):
    """
    Minimal change from your script:
    - Still uses requests/BeautifulSoup (sync) for fetching/parsing
    - BUT saves directly to Mongo via Beanie (async) using upsert_page()
    """
    base_url = normalize_url(base_url)

    visited = set()
    crawled = set()  # if you later want to persist a set, move it to Mongo
    q = deque([base_url])

    while q:
        url = normalize_url(q.popleft())

        if url in visited or url in crawled:
            continue
        if not same_scope(url, base_url):
            continue

        html = fetch(url)
        if not html:
            continue

        # extract page and SAVE TO MONGO
        doc = extract_page(url, html)
        
        # 1) Save/update Mongo (returns id + whether it existed)
        mongo_id, pre_existed = await upsert_page(doc)

        # If the page is unchanged (we SKIP), we won't re-embed here.
        # But you can mark it "ok" to reflect previously embedded status.
        page = await PageDocument.get(mongo_id)
        if page and page.hash and pre_existed:
            # If you prefer to skip re-embedding unchanged:
            # (comment out the next block to always re-embed)
            print(f"[EMBED] skip unchanged {url}")
        else:
            # 2) Build embedding text (sections + code)
            text = build_page_text(doc.sections)

            try:
                # 3) Compute embedding off the event loop
                vector = await asyncio.to_thread(embed_text, text)

                # 4) Upsert to Chroma
                meta = {
                    "url": doc.url,
                    "title": doc.title,
                    "hash": doc.hash,
                    "site": doc.metadata.get("site") if doc.metadata else None,
                    "last_crawled": doc.last_crawled.isoformat()
                }
                await asyncio.to_thread(chroma_upsert, mongo_id, text, meta)

                # 5) Mark page as embedded
                saved = await PageDocument.get(mongo_id)
                if saved:
                    saved.embedding_status = "ok"
                    saved.last_embedded_at = datetime.utcnow()
                    saved.embedding_model = EMBED_MODEL
                    saved.vector_store_id = mongo_id
                    await saved.save()
                print(f"[EMBED] ok {url}")

            except Exception as e:
                print(f"[EMBED][ERROR] {url} -> {e}")

                # Rollback: if brand-new Mongo doc and Chroma failed, delete the Mongo doc
                if not pre_existed:
                    doomed = await PageDocument.get(mongo_id)
                    if doomed:
                        await doomed.delete()
                        print(f"[ROLLBACK] deleted Mongo for {url} (brand-new insert)")
                else:
                    # Existing doc — keep it, mark failed so a later retry can fix
                    existing = await PageDocument.get(mongo_id)
                    if existing:
                        existing.embedding_status = "failed"
                        existing.last_embedded_at = datetime.utcnow()
                        await existing.save()

        visited.add(url)
        crawled.add(url)

        # enqueue new links
        try:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                nxt = normalize_url(urljoin(url, a["href"]))
                if not nxt.lower().startswith(("http://", "https://")):
                    continue
                if (same_scope(nxt, base_url)
                    and nxt not in visited
                    and nxt not in crawled
                    and not should_skip_url(nxt)):
                    q.append(nxt)
        except Exception as e:
            print(f"[WARN] link parse failed on {url}: {e}")
            
async def upsert_page(doc: PageDocument) -> tuple[str, bool]:
    existing = await PageDocument.find_one(PageDocument.url == doc.url)
    if existing:
        if existing.hash == doc.hash:
            print(f"[SKIP] {doc.url} (unchanged)")
            return str(existing.id), True
        existing.title = doc.title
        existing.headings = doc.headings
        existing.sections = doc.sections
        existing.last_crawled = doc.last_crawled
        existing.code_blocks_flat = doc.code_blocks_flat
        existing.summary = doc.summary
        existing.metadata = doc.metadata
        existing.hash = doc.hash
        await existing.save()
        print(f"[UPDATE] {doc.url}")
        return str(existing.id), True
    saved = await doc.insert()
    print(f"[INSERT] {doc.url}")
    return str(saved.id), False
