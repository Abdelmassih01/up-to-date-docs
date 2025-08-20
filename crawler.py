import os
import re
import json
import time
import argparse
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

# ---------- Config ----------
OUTPUT_DIR = Path("outputs")
CRAWLED_LIST = Path("crawled_urls.txt")
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
    """Remove URL fragment and normalize."""
    clean, _ = urldefrag(url.strip())
    return clean

def same_scope(url: str, base_url: str) -> bool:
    """Stay within the same domain and under the base path."""
    pu, pb = urlparse(url), urlparse(base_url)
    if pu.netloc != pb.netloc:
        return False
    # Keep within base path prefix (if any)
    base_path = pb.path if pb.path.endswith("/") else pb.path + "/"
    return pu.path.startswith(base_path) or pu.path == pb.path

def sanitize_filename(text: str) -> str:
    text = text.strip() or "index"
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    return text[:180]

def load_crawled_urls(output_dir: Path, list_file: Path) -> set:
    """Collect URLs from existing outputs and the persisted list file."""
    crawled = set()
    if output_dir.exists():
        for p in output_dir.glob("*.json"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                u = data.get("url")
                if u:
                    crawled.add(normalize_url(u))
            except Exception:
                pass
    if list_file.exists():
        with list_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    crawled.add(normalize_url(line))
    return crawled

def fetch(url: str) -> str | None:
    try:
        res = requests.get(url, timeout=REQ_TIMEOUT, headers=HEADERS)
        if res.ok:
            return res.text
        print(f"[ERROR] {url} -> HTTP {res.status_code}")
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
    return None

def pick_content_root(soup: BeautifulSoup):
    """
    Try to pick the main content container used by Google DevSite/Firebase.
    Falls back to <main> or <body>.
    """
    candidates = [
        # common devsite containers
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

def heading_level(tag_name: str) -> int | None:
    m = re.fullmatch(r"h([1-6])", tag_name.lower())
    return int(m.group(1)) if m else None

def detect_lang_from_classes(el) -> str | None:
    classes = el.get("class") or []
    joined = " ".join(classes).lower()
    # common class hints
    for prefix in ("language-", "lang-", "code-", "highlight-"):
        for cls in classes:
            cls_l = cls.lower()
            if cls_l.startswith(prefix):
                return cls_l.replace(prefix, "")
    # devsite tabs may include language in data attributes
    for attr in ("data-language", "data-lang"):
        if el.has_attr(attr) and el[attr]:
            return str(el[attr]).lower()
    # check code tag's class if this is a <pre>
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

def extract_rich_sections(soup: BeautifulSoup, page_url: str):
    """
    Walk the content in order, creating rich sections that keep:
      - nearest heading text & level
      - nearby prose
      - code blocks (pre>code or devsite code containers)
    """
    root = pick_content_root(soup)
    rich = []

    current = {
        "heading": None,
        "heading_level": None,
        "text_chunks": [],
        "codes": [],  # list[{language, code}]
    }

    def flush():
        # finalize current into rich list if it has content
        text = "\n\n".join([t for t in current["text_chunks"] if t.strip()]).strip()
        has_code = len(current["codes"]) > 0
        if (current["heading"] or text or has_code):
            rich.append({
                "heading": current["heading"],
                "heading_level": current["heading_level"],
                "text": text if text else None,
                "codes": current["codes"] if has_code else [],
            })

    # Devsite often nests content; iterate visible block-level elements
    block_selectors = (
        "h1,h2,h3,h4,h5,h6,"
        "p,ul,ol,li,pre,code,table,blockquote,"
        "div.devsite-article-body,div.devsite-code,devsite-code,"
        "div,section,article"
    )
    nodes = root.select(block_selectors)

    # Avoid double-processing: we'll skip nodes fully contained in a <pre> we handle
    def is_within_pre(el):
        return el.find_parent("pre") is not None

    for el in nodes:
        # Skip elements that belong to another block already captured
        if el.name == "code" and el.find_parent("pre"):
            continue  # will be handled by the parent <pre>

        # New heading -> flush previous section and start a new one
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

        # Devsite code containers (tabbed or not)
        if el.name in ("div", "devsite-code") and (
            "devsite-code" in (el.get("class") or []) or el.name == "devsite-code"
        ):
            # collect all pre/code inside
            pres = el.find_all("pre")
            if not pres:
                # maybe inline code container
                code_text = el.get_text("\n", strip=True)
                if code_text:
                    current["codes"].append({"language": detect_lang_from_classes(el), "code": code_text})
            else:
                for pre in pres:
                    code_el = pre.find("code") or pre
                    code_text = code_el.get_text("\n", strip=True)
                    if code_text:
                        current["codes"].append({
                            "language": detect_lang_from_classes(pre) or detect_lang_from_classes(code_el),
                            "code": code_text,
                        })
            continue

        # Regular <pre> blocks (often contain code)
        if el.name == "pre":
            code_el = el.find("code") or el
            code_text = code_el.get_text("\n", strip=True)
            if code_text:
                current["codes"].append({
                    "language": detect_lang_from_classes(el) or detect_lang_from_classes(code_el),
                    "code": code_text,
                })
            continue

        # Tables sometimes contain important params docs; capture as text
        if el.name == "table":
            text = el.get_text(" ", strip=True)
            if text:
                current["text_chunks"].append(text)
            continue

        # Lists and paragraphs -> add as prose
        if el.name in ("p", "li", "blockquote"):
            txt = el.get_text(" ", strip=True)
            if txt:
                current["text_chunks"].append(txt)
            continue

        # Generic div/section/article: capture simple standalone code/paragraphs inside (light touch)
        if el.name in ("div", "section", "article"):
            # avoid massive duplication: only pick direct text nodes in simple containers
            if el.find(["h1", "h2", "h3", "h4", "h5", "h6"]) is None:
                # if it contains a lone pre, we already handled via <pre> path
                if el.find("pre"):
                    continue
                # capture short text-only blobs
                t = el.get_text(" ", strip=True)
                if t and len(t) < 600:  # heuristic to avoid grabbing whole page chrome
                    current["text_chunks"].append(t)
            continue

    # flush the last section
    flush()

    # Also provide a flat list of code blocks (for convenience/back-compat)
    flat_codes = []
    for sec in rich:
        for c in sec.get("codes", []):
            flat_codes.append(c["code"])

    return rich, flat_codes

def extract_page(doc_url: str, html: str):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else doc_url

    # headings list (quick overview)
    headings = []
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        headings.append({"level": int(tag.name[1]), "text": tag.get_text(" ", strip=True)})

    rich_sections, flat_code_blocks = extract_rich_sections(soup, doc_url)

    return {
        "url": doc_url,
        "title": title,
        "headings": headings,
        "rich_sections": rich_sections,   # <-- text + code grouped by nearest heading
        "code_blocks": flat_code_blocks,  # <-- convenience/legacy
        "summary": None,
        "last_crawled": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

# ---------- Minimal Exclusion Helper ----------
EXCLUDE_PATH_RE = re.compile(
    r"(?:/blob/|/tree/|/raw/|/pulls?|/issues?|/actions?|/commits?/|/compare/"
    r"|/releases?/|/tags?/|/forks?/|/pulse/?$|/security/?$|/contributors?/?$"
    r"|/\.git/|/\.github/|/\.devcontainer/|/node_modules/|/vendor/|/dist/|/build/)"
)

EXCLUDE_EXTS = {
    # code/config/data blobs & archives
    "lock","yml","yaml","toml","ini","cfg","env","properties","json","ndjson","map",
    "zip","tar","gz","tgz","rar","7z",
    # media
    "jpg","jpeg","png","gif","bmp","webp","svg","ico","mp4","mp3","mov","avi","webm","wav",
    # office/PDF (enable later if you parse them)
    "pdf","ppt","pptx","doc","docx","xls","xlsx","csv",
    # project configs
    "tsconfig","eslintrc","prettierrc","npmrc","babelrc",
}

EXCLUDE_BASENAMES = {"LICENSE","CHANGELOG","CONTRIBUTING","SECURITY","README","NOTICE","CODE_OF_CONDUCT"}
EXCLUDE_QUERY_KEYS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","ref","source"}

def should_skip_url(url: str) -> bool:
    u = urlparse(url)
    path = u.path or "/"

    # repo/UI routes
    if EXCLUDE_PATH_RE.search(path):
        return True

    # filenames to skip regardless of extension
    base = path.rsplit("/", 1)[-1]
    if (base.split(".", 1)[0].upper() in EXCLUDE_BASENAMES):
        return True

    # multi-dot extension check (e.g., .prettierrc.json -> {prettierrc, json})
    if "." in base:
        parts = base.lower().split(".")[1:]
        if any(p in EXCLUDE_EXTS for p in parts):
            return True

    # junky tracking queries
    if u.query:
        for qp in u.query.split("&"):
            k = qp.split("=", 1)[0].lower()
            if k in EXCLUDE_QUERY_KEYS:
                return True

    return False

# ---------- Crawler ----------
def crawl(base_url: str, output_dir: Path):
    base_url = normalize_url(base_url)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    crawled = load_crawled_urls(output_dir, CRAWLED_LIST)
    visited = set()
    q = deque([base_url])

    while q:
        url = normalize_url(q.popleft())

        if url in visited or url in crawled:
            # already seen/written in a previous run
            continue
        if not same_scope(url, base_url):
            continue

        html = fetch(url)
        if not html:
            continue

        doc = extract_page(url, html)

        # save
        fname = sanitize_filename(doc["title"]) + ".json"
        out_path = output_dir / fname
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        with CRAWLED_LIST.open("a", encoding="utf-8") as f:
            f.write(url + "\n")

        visited.add(url)
        crawled.add(url)
        print(f"[OK] {url} -> {out_path.name}")

        # enqueue new links
        try:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                nxt = normalize_url(urljoin(url, a["href"]))
                # ignore mailto/js and non-http(s)
                if not nxt.lower().startswith(("http://", "https://")):
                    continue
                if (same_scope(nxt, base_url)
                    and nxt not in visited
                    and nxt not in crawled
                    and not should_skip_url(nxt)):   # <-- minimal addition
                    q.append(nxt)
        except Exception as e:
            print(f"[WARN] link parse failed on {url}: {e}")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-aware docs crawler (groups code with nearby text).")
    parser.add_argument("--base-url", required=True, help="Starting URL (scope limited to its domain and path prefix).")
    parser.add_argument("--output", default="outputs", help="Directory to write JSON files.")
    args = parser.parse_args()

    outdir = Path(args.output)
    crawl(args.base_url, outdir)
