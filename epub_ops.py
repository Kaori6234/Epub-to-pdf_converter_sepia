import io
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import unquote, urldefrag, urljoin
from xml.sax.saxutils import escape

import ebooklib
import requests
from bs4 import BeautifulSoup, Tag
from ebooklib import epub
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer


@dataclass(frozen=True)
class EpubPdfOptions:
    page_format: str = "A4"  # e.g. A4, Letter
    margin_mm: int = 12
    use_merriweather: bool = True
    base_font_size_pt: int = 13


class _EpubDocTemplate(BaseDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._outline_added: set[str] = set()

    def afterFlowable(self, flowable):
        meta = getattr(flowable, "_outline", None)
        if not meta:
            return

        title, key, level = meta
        if not key or key in self._outline_added:
            return

        try:
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title=title, key=key, level=int(level), closed=False)
            self._outline_added.add(key)
        except Exception:
            # Outline/bookmark creation is best-effort; don't fail conversion.
            return


def _extract_epub_html(epub_path: str, progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    book = epub.read_epub(epub_path)

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    total = max(1, len(items))

    parts: list[str] = []
    for i, item in enumerate(items, start=1):
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            body = soup.body
            if body:
                parts.append(str(body))
            else:
                parts.append(str(soup))
        except Exception:
            # Fallback: raw bytes decoded
            parts.append(item.get_content().decode("utf-8", errors="ignore"))

        if progress_cb is not None:
            progress_cb(i, total)

    return "\n<hr/>\n".join(parts)


def _get_book_title(book: epub.EpubBook) -> str:
    try:
        md = book.get_metadata("DC", "title")
        if md and md[0] and md[0][0]:
            return str(md[0][0])
    except Exception:
        pass
    return "Untitled"


def _flatten_toc(toc) -> list[tuple[int, str, Optional[str]]]:
    """Flatten ebooklib's toc structure into (level, title, href) entries."""

    def node_title(obj) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        t = getattr(obj, "title", None)
        if callable(t):
            t = None
        if t:
            return str(t)
        get_fn = getattr(obj, "get", None)
        if callable(get_fn):
            try:
                t2 = get_fn("title")
                if t2:
                    return str(t2)
            except Exception:
                pass
        return None

    def node_href(obj) -> Optional[str]:
        if obj is None:
            return None
        href = getattr(obj, "href", None)
        if callable(href):
            href = None
        if href:
            return str(href)
        get_fn = getattr(obj, "get", None)
        if callable(get_fn):
            try:
                h2 = get_fn("href")
                if h2:
                    return str(h2)
            except Exception:
                pass
        return None

    def add(entries: list[tuple[int, str, Optional[str]]], level: int, title: str, href: Optional[str]) -> None:
        t = (title or "").strip()
        if t:
            entries.append((level, t, href))

    def walk(node, level: int, entries: list[tuple[int, str, Optional[str]]]) -> None:
        if node is None:
            return

        if isinstance(node, str):
            add(entries, level, node, None)
            return

        # ebooklib TOC can be nested lists/tuples like: (Section/Link, [children])
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)):
                parent, children = node
                # parent may be a Section or Link
                title = node_title(parent)
                href = node_href(parent)
                if title:
                    add(entries, level, title, href)
                for child in children:
                    walk(child, level + 1, entries)
                return

            for child in node:
                walk(child, level, entries)
            return

        title = node_title(node)
        if title:
            add(entries, level, title, node_href(node))
            # Some nodes may have nested items
            for child in getattr(node, "subitems", []) or []:
                walk(child, level + 1, entries)

    out: list[tuple[int, str, Optional[str]]] = []
    walk(toc, 0, out)
    return out


def _spine_document_items(book: epub.EpubBook):
    """Return document items in spine order (best match for reading order)."""
    idrefs = []
    try:
        for entry in book.spine:
            # entry is usually (idref, linear)
            if isinstance(entry, (list, tuple)) and entry:
                idrefs.append(entry[0])
            elif isinstance(entry, str):
                idrefs.append(entry)
    except Exception:
        idrefs = []

    seen = set()
    items = []
    for idref in idrefs:
        if not idref or idref in seen:
            continue
        seen.add(idref)
        item = book.get_item_with_id(idref)
        if item is None:
            continue
        try:
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                items.append(item)
        except Exception:
            pass

    # Fallback: all documents
    if not items:
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    return items


def _slugify(value: str) -> str:
    value = (value or "").strip().lower()
    out = []
    prev_dash = False
    for ch in value:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "section"


def _normalize_epub_href(href: Optional[str]) -> tuple[str, str]:
    """Return (doc_path, fragment) from an EPUB href like 'Text/ch1.xhtml#sec1'."""
    href = (href or "").strip()
    if not href:
        return ("", "")

    href = href.replace("\\", "/").lstrip("/")
    try:
        doc, frag = urldefrag(href)
    except Exception:
        doc, frag = href, ""

    doc = unquote((doc or "").split("?", 1)[0]).replace("\\", "/").lstrip("/")
    while doc.startswith("./"):
        doc = doc[2:]
    frag = unquote(frag or "")
    return (doc, frag)


def _iter_epub_elements(book: epub.EpubBook, progress_cb: Optional[Callable[[int, int], None]] = None):
    """Yield (kind, data, base_href) blocks from the EPUB in reading order.

    - For headings/paragraphs: (kind, text, None)
    - For images: ("img", src, base_href)
    """
    items = _spine_document_items(book)
    total = max(1, len(items))

    def _tag_text(tag: Tag) -> str:
        try:
            return (tag.get_text(" ", strip=True) or "").strip()
        except Exception:
            return ""

    def _tag_has_text(tag: Tag) -> bool:
        return bool(_tag_text(tag))

    def _captionish(tag: Tag) -> bool:
        try:
            cls = " ".join(tag.get("class") or [])
            ident = (tag.get("id") or "")
            blob = f"{cls} {ident}".lower()
        except Exception:
            blob = ""
        return any(k in blob for k in ("caption", "credit", "figcaption", "image-caption", "photo-caption"))

    def _image_has_context(el: Tag) -> bool:
        # Alt/title text implies a caption or context.
        try:
            if (el.get("alt") or "").strip() or (el.get("title") or "").strip():
                return True
        except Exception:
            pass

        # Figure/figcaption pattern.
        try:
            figure = el.find_parent("figure")
            if figure is not None:
                figcap = figure.find("figcaption")
                if figcap is not None and _tag_has_text(figcap):
                    return True
        except Exception:
            pass

        # Siblings with text (common caption layout).
        try:
            parent = el.parent
            if isinstance(parent, Tag):
                for sib in list(el.previous_siblings) + list(el.next_siblings):
                    if isinstance(sib, Tag) and _tag_has_text(sib):
                        return True
                # Any direct children of the parent with text (but not the img itself).
                for child in parent.find_all(["p", "div", "span", "figcaption", "caption", "h1", "h2", "h3", "li"], recursive=False):
                    if child is el:
                        continue
                    if _tag_has_text(child):
                        return True
                    if _captionish(child):
                        return True
        except Exception:
            pass

        # Caption-ish ancestor (e.g., div.caption containing image and text)
        try:
            for anc in el.parents:
                if not isinstance(anc, Tag):
                    continue
                if _captionish(anc) and _tag_has_text(anc):
                    return True
                # Stop if we hit body/html to avoid scanning too wide.
                if (anc.name or "").lower() in ("body", "html"):
                    break
        except Exception:
            pass

        return False

    for i, item in enumerate(items, start=1):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        body = soup.body or soup

        base_href = None
        try:
            base_href = item.get_name()
        except Exception:
            base_href = None

        # Document start marker (used for TOC/outline anchoring).
        try:
            doc_path = (base_href or "").replace("\\", "/").lstrip("/")
        except Exception:
            doc_path = ""
        yield ("docstart", doc_path, None)

        for el in body.find_all(["h1", "h2", "h3", "p", "li", "img", "br", "hr"], recursive=True):
            tag = (el.name or "p").lower()

            if tag == "br":
                yield ("softbreak", "", None)
                continue

            if tag == "hr":
                yield ("softbreak", "", None)
                continue

            text = el.get_text("\n", strip=True)
            tag = (el.name or "p").lower()

            if tag == "img":
                src = (el.get("src") or "").strip()
                if not src:
                    continue
                # If an image is surrounded by text/caption in its local context,
                # treat it as an inline image (do not force it onto its own page).
                contextual = False
                try:
                    contextual = _image_has_context(el)
                except Exception:
                    contextual = False

                yield ("img_inline" if contextual else "img", src, base_href)
                continue

            if tag in ("h1", "h2", "h3"):
                raw_fragment = (el.get("id") or el.get("name") or "").strip()
                if not text and not raw_fragment:
                    continue
                yield (tag, text, None)
            else:
                if not text:
                    continue
                # Preserve <br/> within paragraphs as separate lines.
                lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
                if not lines:
                    continue
                for j, ln in enumerate(lines):
                    yield ("p", ln, None)
                    if j != len(lines) - 1:
                        yield ("softbreak", "", None)

        # Emit a document boundary marker. The PDF builder decides whether this
        # becomes a hard page break (chapter boundary) or just spacing.
        yield ("docbreak", "", None)

        if progress_cb is not None:
            progress_cb(i, total)


def _scan_headings(book: epub.EpubBook):
    """Return heading specs in reading order.

    Each entry is: (kind, text, anchor, level, doc_path, raw_fragment)
    - doc_path comes from the spine document item's name
    - raw_fragment is the HTML id/name if present (used for TOC href mapping)
    """
    counters: dict[str, int] = {}
    result: list[tuple[str, str, str, int, str, str]] = []
    level_map = {"h1": 0, "h2": 1, "h3": 2}

    for item in _spine_document_items(book):
        try:
            doc_path = (item.get_name() or "").replace("\\", "/").lstrip("/")
        except Exception:
            doc_path = ""

        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            body = soup.body or soup
        except Exception:
            continue

        for el in body.find_all(["h1", "h2", "h3"], recursive=True):
            kind = (el.name or "").lower()
            if kind not in level_map:
                continue
            text = el.get_text("\n", strip=True) or ""
            raw_fragment = (el.get("id") or el.get("name") or "").strip()
            if not text and not raw_fragment:
                continue

            base = _slugify(raw_fragment) if raw_fragment else _slugify(text)
            n = counters.get(base, 0) + 1
            counters[base] = n
            anchor = f"{base}-{n}" if n > 1 else base

            result.append((kind, text, anchor, level_map[kind], doc_path, raw_fragment))

    return result


def _page_size(name: str):
    name = (name or "A4").strip().lower()
    if name == "letter":
        return LETTER
    return A4


def _ensure_merriweather_font(cache_dir: str) -> tuple[str, str]:
    """Download Merriweather into cache_dir and register with ReportLab.

    Uses the variable font file from the official google/fonts repository.
    If download fails for any reason, the caller should fall back to built-in fonts.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # The static TTFs for Merriweather appear to be no longer present at the old paths.
    # This variable font file is present and works.
    vf_path = os.path.join(cache_dir, "Merriweather[opsz,wdth,wght].ttf")
    vf_url = "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather%5Bopsz%2Cwdth%2Cwght%5D.ttf"

    if not (os.path.exists(vf_path) and os.path.getsize(vf_path) > 0):
        r = requests.get(vf_url, timeout=60)
        r.raise_for_status()
        with open(vf_path, "wb") as f:
            f.write(r.content)

    # Register font (idempotent)
    if "Merriweather" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Merriweather", vf_path))

    # ReportLab needs a bold font name for headings; if we only have the variable font,
    # use the same face for bold as a pragmatic fallback.
    return ("Merriweather", "Merriweather")


def _try_register_windows_font(font_name: str, filename: str) -> Optional[str]:
    """Register a Windows system font (best-effort)."""
    try:
        fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        path = os.path.join(fonts_dir, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name
    except Exception:
        return None
    return None


def _register_cjk_cid_fonts() -> dict[str, str]:
    """Register built-in CID fonts for CJK (best-effort, no external files)."""
    out: dict[str, str] = {}
    for key, cid_name in (
        ("ja", "HeiseiMin-W3"),
        ("zh", "STSong-Light"),
        ("ko", "HYSMyeongJo-Medium"),
    ):
        try:
            if cid_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(UnicodeCIDFont(cid_name))
            out[key] = cid_name
        except Exception:
            continue
    return out


def _text_script_hint(text: str) -> str:
    """Return 'arabic', 'cjk', or 'default' based on Unicode ranges."""
    for ch in text or "":
        cp = ord(ch)
        if (
            0x0600 <= cp <= 0x06FF
            or 0x0750 <= cp <= 0x077F
            or 0x08A0 <= cp <= 0x08FF
            or 0xFB50 <= cp <= 0xFDFF
            or 0xFE70 <= cp <= 0xFEFF
        ):
            return "arabic"
    for ch in text or "":
        cp = ord(ch)
        if (
            0x3000 <= cp <= 0x30FF
            or 0x3400 <= cp <= 0x4DBF
            or 0x4E00 <= cp <= 0x9FFF
            or 0xAC00 <= cp <= 0xD7AF
        ):
            return "cjk"
    return "default"


def _maybe_shape_arabic(text: str) -> str:
    """Best-effort Arabic shaping + bidi. No-op if deps are missing."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _get_cover_image_bytes(book: epub.EpubBook) -> Optional[bytes]:
    """Best-effort cover image extraction from an EPUB."""
    cover_const = getattr(ebooklib, "ITEM_COVER", None)
    if cover_const is not None:
        try:
            items = list(book.get_items_of_type(cover_const))
            if items:
                return items[0].get_content()
        except Exception:
            pass

    # OPF <meta name="cover" content="id"/>
    try:
        metas = book.get_metadata("OPF", "meta")
        for _value, attrs in metas or []:
            try:
                name = (attrs or {}).get("name")
                content = (attrs or {}).get("content")
                if (name or "").lower() == "cover" and content:
                    item = book.get_item_with_id(content)
                    if item is not None:
                        return item.get_content()
            except Exception:
                continue
    except Exception:
        pass

    # Heuristic: first image with "cover" in filename
    try:
        for img_item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            try:
                name = (img_item.get_name() or "").lower()
            except Exception:
                name = ""
            if "cover" in name:
                try:
                    return img_item.get_content()
                except Exception:
                    continue
    except Exception:
        pass

    return None


def _iter_epub_blocks(book: epub.EpubBook, progress_cb: Optional[Callable[[int, int], None]] = None):
    # Backward-compatible wrapper
    yield from _iter_epub_elements(book, progress_cb=progress_cb)


def _try_decode_data_image_uri(src: str) -> Optional[bytes]:
    src = (src or "").strip()
    if not src.startswith("data:"):
        return None
    # Only handle base64 data URIs
    if ";base64," not in src:
        return None
    try:
        header, b64 = src.split(",", 1)
        if not header.lower().startswith("data:image/"):
            return None
        import base64

        return base64.b64decode(b64)
    except Exception:
        return None


def _resolve_epub_image_bytes(book: epub.EpubBook, base_href: Optional[str], src: str) -> Optional[bytes]:
    """Resolve an <img src> to raw image bytes from inside the EPUB."""
    if not src:
        return None

    data_bytes = _try_decode_data_image_uri(src)
    if data_bytes is not None:
        return data_bytes

    # Ignore external images
    s = src.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return None

    # Drop URL fragments and decode %xx
    s, _frag = urldefrag(s)
    s = unquote(s)

    # Normalize path style
    s = s.replace("\\", "/")
    s = s.lstrip("/")

    href = s
    if base_href:
        try:
            href = urljoin(base_href, s)
        except Exception:
            href = s

    # ebooklib uses POSIX-style hrefs internally
    href = (href or "").replace("\\", "/").lstrip("/")

    item = None
    try:
        item = book.get_item_with_href(href)
    except Exception:
        item = None

    if item is None:
        # Try common variations
        candidates = [href, href.lstrip("./"), s, s.lstrip("./")]
        for cand in candidates:
            if not cand:
                continue
            try:
                item = book.get_item_with_href(cand)
            except Exception:
                item = None
            if item is not None:
                break

    if item is None:
        # Fallback: best-effort suffix match across image items
        try:
            for img_item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
                try:
                    name = (img_item.get_name() or "").replace("\\", "/").lstrip("/")
                except Exception:
                    continue
                if name and (name.endswith(href) or href.endswith(name) or name.endswith(s) or s.endswith(name)):
                    item = img_item
                    break
        except Exception:
            item = None

    if item is None:
        return None

    try:
        return item.get_content()
    except Exception:
        return None


def _build_image_flowable(
    img_bytes: bytes,
    max_width_pt: float,
    max_height_pt: float,
) -> Optional[RLImage]:
    """Create a ReportLab Image flowable without altering pixel data.

    The image is only *scaled down* to fit the page (no upscaling).
    """
    if not img_bytes:
        return None

    # Skip SVG (ReportLab doesn't handle it without extra deps)
    head = img_bytes[:256].lstrip()
    if head.startswith(b"<svg") or head.startswith(b"<?xml") and b"<svg" in head:
        return None

    try:
        reader = ImageReader(io.BytesIO(img_bytes))
        w_px, h_px = reader.getSize()
        w_pt, h_pt = float(w_px), float(h_px)  # ReportLab treats pixels as points by default
    except Exception:
        return None

    # Be slightly conservative to avoid borderline overflow due to frame padding
    # and floating-point rounding.
    eps = 0.5  # points
    safe_max_w = max(1.0, float(max_width_pt) - eps)
    safe_max_h = max(1.0, float(max_height_pt) - eps)

    scale = 1.0
    if w_pt > 0 and h_pt > 0:
        # Only scale down (no upscaling) and never crop.
        scale = min(1.0, safe_max_w / w_pt, safe_max_h / h_pt)

    try:
        img = RLImage(io.BytesIO(img_bytes), width=w_pt * scale, height=h_pt * scale)
        img.hAlign = "CENTER"
        return img
    except Exception:
        return None


class _PageFitImageFlowable(Flowable):
    """Draw an image scaled to fit within the available box (no cropping).

    If consume_full_height=True, the flowable will take the full available height
    and center the image vertically (useful for image-only pages).
    """

    def __init__(self, img_bytes: bytes, consume_full_height: bool = True):
        super().__init__()
        self._reader = ImageReader(io.BytesIO(img_bytes))
        self._consume_full_height = bool(consume_full_height)
        self._wrap_w: float = 0.0
        self._wrap_h: float = 0.0

    def wrap(self, availWidth, availHeight):
        aw = max(1.0, float(availWidth))
        ah = max(1.0, float(availHeight))

        self._wrap_w = aw
        if self._consume_full_height:
            self._wrap_h = ah
        else:
            try:
                iw, ih = self._reader.getSize()
                if iw and ih:
                    scale = min(1.0, aw / float(iw), ah / float(ih))
                    self._wrap_h = max(1.0, float(ih) * scale)
                else:
                    self._wrap_h = 1.0
            except Exception:
                self._wrap_h = 1.0

        return (self._wrap_w, self._wrap_h)

    def draw(self):
        try:
            iw, ih = self._reader.getSize()
            if not iw or not ih:
                return

            aw = max(1.0, float(getattr(self, "_wrap_w", 0.0) or 0.0))
            ah = max(1.0, float(getattr(self, "_wrap_h", 0.0) or 0.0))

            # Contain / fit (never crop) and do not upscale.
            scale = min(1.0, aw / float(iw), ah / float(ih))
            dw = float(iw) * scale
            dh = float(ih) * scale
            x = (aw - dw) / 2.0
            y = (ah - dh) / 2.0
            self.canv.drawImage(self._reader, x, y, width=dw, height=dh, mask="auto")
        except Exception:
            return


def _build_full_bleed_image_page(img_bytes: bytes, page_w: float, page_h: float) -> Optional[_PageFitImageFlowable]:
    if not img_bytes:
        return None
    head = img_bytes[:256].lstrip()
    if head.startswith(b"<svg") or head.startswith(b"<?xml") and b"<svg" in head:
        return None
    try:
        # Historical name, but do NOT crop-to-fill; page-fit avoids cutting wide images.
        return _PageFitImageFlowable(img_bytes, consume_full_height=True)
    except Exception:
        return None


def epub_bytes_to_styled_pdf(
    epub_bytes: bytes,
    options: EpubPdfOptions,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """Convert EPUB bytes to a styled PDF (sepia background, black text, Merriweather).

    This implementation is pure-Python (ReportLab) to avoid Playwright/asyncio subprocess issues.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as f:
        epub_path = f.name
        f.write(epub_bytes)

    try:
        page_w, page_h = _page_size(options.page_format)
        margin = int(options.margin_mm) * mm

        book = epub.read_epub(epub_path)
        book_title = _get_book_title(book)
        toc_entries = _flatten_toc(getattr(book, "toc", None))
        heading_specs = _scan_headings(book)
        heading_anchor_set = {h[2] for h in heading_specs if h and len(h) >= 3}
        title_to_first_anchor: dict[str, str] = {}
        doc_first_anchor: dict[str, str] = {}
        docfrag_to_anchor: dict[tuple[str, str], str] = {}

        for _kind, title, anchor, _level, doc_path, raw_fragment in heading_specs:
            if title and title not in title_to_first_anchor:
                title_to_first_anchor[title] = anchor
            if doc_path and doc_path not in doc_first_anchor:
                doc_first_anchor[doc_path] = anchor
            if doc_path and raw_fragment:
                docfrag_to_anchor[(doc_path, raw_fragment)] = anchor

        spine_doc_paths: list[str] = []
        for item in _spine_document_items(book):
            try:
                name = (item.get_name() or "").replace("\\", "/").lstrip("/")
            except Exception:
                name = ""
            if name:
                spine_doc_paths.append(name)

        doc_paths = sorted(set(spine_doc_paths))

        def _resolve_doc_path(doc: str) -> str:
            d = (doc or "").replace("\\", "/").lstrip("/")
            while d.startswith("./"):
                d = d[2:]
            if not d:
                return ""
            if d in doc_first_anchor:
                return d
            # Try suffix match (TOC often uses paths relative to the package root)
            for full in doc_paths:
                if full.endswith(d):
                    return full
            return d

        # Normalize TOC entries and pre-resolve their targets
        toc_items: list[tuple[int, str, str, str]] = []  # (level, title, doc_path, frag)
        for level, title, href in toc_entries:
            dpath, frag = _normalize_epub_href(href)
            dpath = _resolve_doc_path(dpath)
            toc_items.append((int(level), title or "", dpath, frag))

        # Build TOC anchors, preferring existing heading anchors.
        toc_anchor_for: dict[int, str] = {}
        toc_doc_entries: dict[str, list[tuple[int, str, str]]] = {}
        anchor_counts: dict[str, int] = {}

        def _unique_anchor(base: str) -> str:
            key = _slugify(base)
            n = anchor_counts.get(key, 0) + 1
            anchor_counts[key] = n
            return f"{key}-{n}" if n > 1 else key

        valid_doc_paths = set(doc_paths)

        for idx, (level, title, dpath, frag) in enumerate(toc_items):
            anchor = None
            if dpath and frag:
                anchor = docfrag_to_anchor.get((dpath, frag))
            if anchor is None and dpath:
                anchor = doc_first_anchor.get(dpath)
            if anchor is None and title:
                anchor = title_to_first_anchor.get(title)
            if anchor is None:
                anchor = _unique_anchor(f"toc-{title or dpath or frag or 'section'}")

            toc_anchor_for[idx] = anchor
            if dpath and dpath in valid_doc_paths:
                toc_doc_entries.setdefault(dpath, []).append((int(level), title or "", anchor))

        # Only link TOC items to anchors that will exist in the document.
        valid_anchor_targets: set[str] = set(heading_anchor_set)
        for idx, (level, title, dpath, _frag) in enumerate(toc_items):
            anchor = toc_anchor_for.get(idx)
            if not anchor:
                continue
            if anchor in heading_anchor_set:
                valid_anchor_targets.add(anchor)
                continue
            if dpath and dpath in valid_doc_paths and (title or "").strip():
                valid_anchor_targets.add(anchor)

        font_name = "Times-Roman"
        font_bold = "Times-Bold"
        if options.use_merriweather:
            try:
                cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "fonts")
                font_name, font_bold = _ensure_merriweather_font(cache_dir)
            except Exception:
                # No hard-fail on font download; keep conversion working.
                font_name = "Times-Roman"
                font_bold = "Times-Bold"

        # Font fallbacks (best-effort)
        fallback_sans = _try_register_windows_font("FallbackSans", "arial.ttf") or "Helvetica"
        cjk_fonts = _register_cjk_cid_fonts()
        cjk_font = cjk_fonts.get("ja") or cjk_fonts.get("zh") or cjk_fonts.get("ko") or font_name

        sepia_bg = Color(244 / 255.0, 236 / 255.0, 216 / 255.0)

        def draw_bg(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(sepia_bg)
            canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)
            canvas.restoreState()

        cover_reader = None
        try:
            cover_bytes = _get_cover_image_bytes(book)
            if cover_bytes:
                cover_reader = ImageReader(io.BytesIO(cover_bytes))
        except Exception:
            cover_reader = None

        def draw_cover(canvas, doc):
            # Full-bleed cover page (no sepia background).
            if cover_reader is None:
                return
            try:
                iw, ih = cover_reader.getSize()
                if not iw or not ih:
                    return
                scale = max(page_w / float(iw), page_h / float(ih))
                dw = float(iw) * scale
                dh = float(ih) * scale
                x = (page_w - dw) / 2.0
                y = (page_h - dh) / 2.0
                canvas.drawImage(cover_reader, x, y, width=dw, height=dh, mask="auto")
            except Exception:
                return

        styles = getSampleStyleSheet()
        normal_default = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=int(options.base_font_size_pt),
            leading=int(options.base_font_size_pt * 1.55),
            textColor=Color(0, 0, 0),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
        normal_fallback = ParagraphStyle("BodyFallback", parent=normal_default, fontName=fallback_sans)
        normal_cjk = ParagraphStyle("BodyCJK", parent=normal_default, fontName=cjk_font)

        h1_default = ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName=font_bold,
            fontSize=int(options.base_font_size_pt * 1.6) + 2,
            leading=int(options.base_font_size_pt * 1.9) + 2,
            textColor=Color(0, 0, 0),
            spaceAfter=10,
        )
        h1_fallback = ParagraphStyle("H1Fallback", parent=h1_default, fontName=fallback_sans)
        h1_cjk = ParagraphStyle("H1CJK", parent=h1_default, fontName=cjk_font)

        h2_default = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName=font_bold,
            fontSize=int(options.base_font_size_pt * 1.35) + 2,
            leading=int(options.base_font_size_pt * 1.6) + 2,
            textColor=Color(0, 0, 0),
            spaceAfter=8,
        )
        h2_fallback = ParagraphStyle("H2Fallback", parent=h2_default, fontName=fallback_sans)
        h2_cjk = ParagraphStyle("H2CJK", parent=h2_default, fontName=cjk_font)

        h3_default = ParagraphStyle(
            "H3",
            parent=styles["Heading3"],
            fontName=font_bold,
            fontSize=int(options.base_font_size_pt * 1.15) + 2,
            leading=int(options.base_font_size_pt * 1.4) + 2,
            textColor=Color(0, 0, 0),
            spaceAfter=6,
        )
        h3_fallback = ParagraphStyle("H3Fallback", parent=h3_default, fontName=fallback_sans)
        h3_cjk = ParagraphStyle("H3CJK", parent=h3_default, fontName=cjk_font)

        toc_title = ParagraphStyle(
            "TOCTitle",
            parent=styles["Heading2"],
            fontName=font_bold,
            fontSize=int(options.base_font_size_pt * 1.4) + 2,
            leading=int(options.base_font_size_pt * 1.7) + 2,
            textColor=Color(0, 0, 0),
            spaceAfter=10,
        )
        toc_item = ParagraphStyle(
            "TOCItem",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=int(options.base_font_size_pt),
            leading=int(options.base_font_size_pt * 1.4),
            textColor=Color(0, 0, 0),
            leftIndent=0,
            firstLineIndent=0,
            spaceAfter=2,
        )
        anchor_only = ParagraphStyle(
            "AnchorOnly",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=1,
            leading=1,
            textColor=Color(0, 0, 0),
            spaceBefore=0,
            spaceAfter=0,
        )

        buf = io.BytesIO()
        doc = _EpubDocTemplate(buf, pagesize=(page_w, page_h), title="EPUB export")

        # Use zero frame padding so images can sit flush within margins ("borderless")
        # and to reduce surprise overflows.
        frame_pad = 0.0
        main_frame = Frame(
            margin,
            margin,
            page_w - 2 * margin,
            page_h - 2 * margin,
            leftPadding=frame_pad,
            rightPadding=frame_pad,
            topPadding=frame_pad,
            bottomPadding=frame_pad,
            id="main",
        )

        # Full-bleed template for large images.
        full_bleed_frame = Frame(
            0,
            0,
            page_w,
            page_h,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="fullbleed",
        )

        if cover_reader is not None:
            cover_frame = Frame(
                0,
                0,
                page_w,
                page_h,
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
                id="cover",
            )
            doc.addPageTemplates(
                [
                    PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_cover),
                    PageTemplate(id="Main", frames=[main_frame], onPage=draw_bg),
                    PageTemplate(id="FullBleed", frames=[full_bleed_frame], onPage=draw_bg),
                ]
            )
        else:
            doc.addPageTemplates(
                [
                    PageTemplate(id="Main", frames=[main_frame], onPage=draw_bg),
                    PageTemplate(id="FullBleed", frames=[full_bleed_frame], onPage=draw_bg),
                ]
            )

        story = []
        toc_anchor_emitted: set[str] = set()

        def pick_style(text: str, default_style: ParagraphStyle, fallback_style: ParagraphStyle, cjk_style: ParagraphStyle):
            hint = _text_script_hint(text)
            if hint == "arabic":
                return fallback_style
            if hint == "cjk":
                return cjk_style
            return default_style

        def shape_text(text: str) -> str:
            if _text_script_hint(text) == "arabic":
                return _maybe_shape_arabic(text)
            return text

        # If cover exists: first page is cover-only, then TOC/content starts on next page.
        if cover_reader is not None:
            story.append(NextPageTemplate("Main"))
            story.append(PageBreak())

        # Title (first content page)
        title_text = shape_text(book_title)
        title_style = pick_style(book_title, h1_default, h1_fallback, h1_cjk)
        title_para = Paragraph(escape(title_text), title_style)
        title_para._outline = (book_title, "title", 0)
        story.append(title_para)
        story.append(Spacer(1, 12))

        # Table of contents (best-effort)
        if toc_items:
            toc_para = Paragraph("Table of Contents", toc_title)
            toc_para._outline = ("Table of Contents", "toc", 0)
            story.append(toc_para)
            for n, (level, title, _dpath, _frag) in enumerate(toc_items, start=1):
                indent = min(36, int(level) * 12)
                anchor = toc_anchor_for.get(n - 1)

                safe_title = escape(shape_text(title))
                if anchor and anchor in valid_anchor_targets:
                    line = f"<a href=\"#{escape(anchor)}\">{n}. {safe_title}</a>"
                else:
                    line = f"{n}. {safe_title}"
                style = ParagraphStyle(
                    f"TOCItem{n}",
                    parent=toc_item,
                    leftIndent=indent,
                )
                story.append(Paragraph(line, style))
            story.append(PageBreak())

        heading_idx = 0
        max_w = float(page_w - 2 * margin - 2 * frame_pad)
        max_h = float(page_h - 2 * margin - 2 * frame_pad)

        last_was_large_image = False

        def _ensure_break() -> None:
            if story and not isinstance(story[-1], PageBreak):
                story.append(PageBreak())

        for kind, text, base_href in _iter_epub_blocks(book, progress_cb=progress_cb):
            if kind == "docstart":
                doc_path = (text or "").replace("\\", "/").lstrip("/")
                for level, title, anchor in toc_doc_entries.get(doc_path, []) or []:
                    if not anchor or anchor in toc_anchor_emitted:
                        continue
                    if not (title or "").strip():
                        continue
                    if anchor in heading_anchor_set:
                        continue
                    # Emit an invisible anchor at the start of the document and
                    # add an outline entry for the PDF sidebar.
                    safe_anchor = escape(anchor)
                    para = Paragraph(f"<a name=\"{safe_anchor}\"/>", anchor_only)
                    para._outline = (title or "", anchor, max(0, int(level)))
                    story.append(para)
                    toc_anchor_emitted.add(anchor)
                continue

            if kind in ("pagebreak", "docbreak"):
                # Prefer a hard break at document boundaries unless the last thing
                # was a very large image (those often already consumed the page).
                if kind == "docbreak" and last_was_large_image:
                    last_was_large_image = False
                    continue
                if story and not isinstance(story[-1], PageBreak):
                    story.append(PageBreak())
                last_was_large_image = False
                continue

            if kind == "softbreak":
                # Light spacing for <br> and <hr>
                story.append(Spacer(1, 6))
                last_was_large_image = False
                continue

            if kind in ("img", "img_inline"):
                img_bytes = _resolve_epub_image_bytes(book, base_href, text)
                if img_bytes:
                    try:
                        r = ImageReader(io.BytesIO(img_bytes))
                        iw, ih = r.getSize()
                        # Heuristic: images that are very large or have extreme aspect ratios
                        # often represent full-page illustrations or spreads. These should
                        # get their own page, but MUST NOT be cropped.
                        img_aspect = (float(iw) / float(ih)) if iw and ih else 0.0
                        is_large = bool(
                            iw
                            and ih
                            and (
                                float(iw) >= 0.95 * float(max_w)
                                or float(ih) >= 0.95 * float(max_h)
                                or img_aspect >= 1.60
                                or (img_aspect > 0 and img_aspect <= 0.60)
                            )
                        )
                    except Exception:
                        is_large = False

                    # Only standalone images are eligible for an image-only page.
                    if kind == "img" and is_large:
                        # Render on its own page, but fit/contain within the normal margins
                        # (no cropping, no forced borderless/bleed).
                        _ensure_break()
                        img_page = _PageFitImageFlowable(img_bytes, consume_full_height=True)
                        story.append(img_page)
                        story.append(PageBreak())
                        last_was_large_image = True
                        continue

                img_flow = _build_image_flowable(img_bytes or b"", max_width_pt=max_w, max_height_pt=max_h)
                if img_flow is not None:
                    story.append(img_flow)
                    last_was_large_image = False
                continue
            if kind == "h1":
                anchor = heading_specs[heading_idx][2] if heading_idx < len(heading_specs) else _slugify(text)
                style = pick_style(text, h1_default, h1_fallback, h1_cjk)
                if not (text or "").strip():
                    story.append(Paragraph(f"<a name=\"{escape(anchor)}\"/>", style))
                    heading_idx += 1
                    last_was_large_image = False
                else:
                    styled = shape_text(text)
                    title_txt = escape(styled)
                    para = Paragraph(f"<a name=\"{escape(anchor)}\"/>{title_txt}", style)
                    para._outline = (text, anchor, 0)
                    story.append(para)
                    heading_idx += 1
                    story.append(Spacer(1, 6))
                    last_was_large_image = False
            elif kind == "h2":
                anchor = heading_specs[heading_idx][2] if heading_idx < len(heading_specs) else _slugify(text)
                style = pick_style(text, h2_default, h2_fallback, h2_cjk)
                if not (text or "").strip():
                    story.append(Paragraph(f"<a name=\"{escape(anchor)}\"/>", style))
                    heading_idx += 1
                    last_was_large_image = False
                else:
                    styled = shape_text(text)
                    title_txt = escape(styled)
                    para = Paragraph(f"<a name=\"{escape(anchor)}\"/>{title_txt}", style)
                    para._outline = (text, anchor, 1)
                    story.append(para)
                    heading_idx += 1
                    story.append(Spacer(1, 4))
                    last_was_large_image = False
            elif kind == "h3":
                anchor = heading_specs[heading_idx][2] if heading_idx < len(heading_specs) else _slugify(text)
                style = pick_style(text, h3_default, h3_fallback, h3_cjk)
                if not (text or "").strip():
                    story.append(Paragraph(f"<a name=\"{escape(anchor)}\"/>", style))
                    heading_idx += 1
                    last_was_large_image = False
                else:
                    styled = shape_text(text)
                    title_txt = escape(styled)
                    para = Paragraph(f"<a name=\"{escape(anchor)}\"/>{title_txt}", style)
                    para._outline = (text, anchor, 2)
                    story.append(para)
                    heading_idx += 1
                    story.append(Spacer(1, 3))
                    last_was_large_image = False
            else:
                styled = shape_text(text)
                style = pick_style(text, normal_default, normal_fallback, normal_cjk)
                story.append(Paragraph(escape(styled), style))
                last_was_large_image = False

        doc.build(story)
        return buf.getvalue()
    finally:
        try:
            os.remove(epub_path)
        except OSError:
            pass
