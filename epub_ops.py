import io
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional
from xml.sax.saxutils import escape

import ebooklib
import requests
from bs4 import BeautifulSoup
from ebooklib import epub
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


@dataclass(frozen=True)
class EpubPdfOptions:
    page_format: str = "A4"  # e.g. A4, Letter
    margin_mm: int = 12
    use_merriweather: bool = True
    base_font_size_pt: int = 13


class _EpubDocTemplate(SimpleDocTemplate):
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


def _flatten_toc(toc) -> list[tuple[int, str]]:
    """Flatten ebooklib's toc structure into (level, title) entries."""

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

    def add(entries: list[tuple[int, str]], level: int, title: str) -> None:
        t = (title or "").strip()
        if t:
            entries.append((level, t))

    def walk(node, level: int, entries: list[tuple[int, str]]) -> None:
        if node is None:
            return

        if isinstance(node, str):
            add(entries, level, node)
            return

        # ebooklib TOC can be nested lists/tuples like: (Section/Link, [children])
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)):
                parent, children = node
                # parent may be a Section or Link
                title = node_title(parent)
                if title:
                    add(entries, level, title)
                for child in children:
                    walk(child, level + 1, entries)
                return

            for child in node:
                walk(child, level, entries)
            return

        title = node_title(node)
        if title:
            add(entries, level, title)
            # Some nodes may have nested items
            for child in getattr(node, "subitems", []) or []:
                walk(child, level + 1, entries)

    out: list[tuple[int, str]] = []
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


def _iter_epub_elements(book: epub.EpubBook, progress_cb: Optional[Callable[[int, int], None]] = None):
    """Yield (kind, text) blocks from the EPUB in reading order."""
    items = _spine_document_items(book)
    total = max(1, len(items))

    for i, item in enumerate(items, start=1):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        body = soup.body or soup

        for el in body.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            tag = (el.name or "p").lower()
            if tag in ("h1", "h2", "h3"):
                yield (tag, text)
            else:
                yield ("p", text)

        # Page break between documents
        yield ("pagebreak", "")

        if progress_cb is not None:
            progress_cb(i, total)


def _scan_headings(book: epub.EpubBook):
    """Return a list of heading specs in reading order: (kind, text, anchor, level)."""
    counters: dict[str, int] = {}
    result: list[tuple[str, str, str, int]] = []
    level_map = {"h1": 0, "h2": 1, "h3": 2}

    for kind, text in _iter_epub_elements(book, progress_cb=None):
        if kind not in level_map:
            continue
        base = _slugify(text)
        n = counters.get(base, 0) + 1
        counters[base] = n
        anchor = f"{base}-{n}" if n > 1 else base
        result.append((kind, text, anchor, level_map[kind]))

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


def _iter_epub_blocks(book: epub.EpubBook, progress_cb: Optional[Callable[[int, int], None]] = None):
    # Backward-compatible wrapper
    yield from _iter_epub_elements(book, progress_cb=progress_cb)


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
        title_to_first_anchor: dict[str, str] = {}
        for _kind, title, anchor, _level in heading_specs:
            if title not in title_to_first_anchor:
                title_to_first_anchor[title] = anchor

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

        sepia_bg = Color(244 / 255.0, 236 / 255.0, 216 / 255.0)

        def draw_bg(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(sepia_bg)
            canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)
            canvas.restoreState()

        styles = getSampleStyleSheet()
        normal = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=int(options.base_font_size_pt),
            leading=int(options.base_font_size_pt * 1.55),
            textColor=Color(0, 0, 0),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
        h1 = ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName=font_bold,
            fontSize=int(options.base_font_size_pt * 1.6) + 2,
            leading=int(options.base_font_size_pt * 1.9) + 2,
            textColor=Color(0, 0, 0),
            spaceAfter=10,
        )
        h2 = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName=font_bold,
            fontSize=int(options.base_font_size_pt * 1.35) + 2,
            leading=int(options.base_font_size_pt * 1.6) + 2,
            textColor=Color(0, 0, 0),
            spaceAfter=8,
        )
        h3 = ParagraphStyle(
            "H3",
            parent=styles["Heading3"],
            fontName=font_bold,
            fontSize=int(options.base_font_size_pt * 1.15) + 2,
            leading=int(options.base_font_size_pt * 1.4) + 2,
            textColor=Color(0, 0, 0),
            spaceAfter=6,
        )

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

        buf = io.BytesIO()
        doc = _EpubDocTemplate(
            buf,
            pagesize=(page_w, page_h),
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
            title="EPUB export",
        )

        story = []

        # Title page
        title_para = Paragraph(book_title, h1)
        title_para._outline = (book_title, "title", 0)
        story.append(title_para)
        story.append(Spacer(1, 12))

        # Table of contents (best-effort)
        if toc_entries:
            toc_para = Paragraph("Table of Contents", toc_title)
            toc_para._outline = ("Table of Contents", "toc", 0)
            story.append(toc_para)
            for n, (level, title) in enumerate(toc_entries, start=1):
                indent = min(36, int(level) * 12)
                anchor = title_to_first_anchor.get(title)
                safe_title = escape(title)
                if anchor:
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
        for kind, text in _iter_epub_blocks(book, progress_cb=progress_cb):
            if kind == "pagebreak":
                if story and not isinstance(story[-1], PageBreak):
                    story.append(PageBreak())
                continue
            if kind == "h1":
                anchor = heading_specs[heading_idx][2] if heading_idx < len(heading_specs) else _slugify(text)
                title_txt = escape(text)
                para = Paragraph(f"<a name=\"{escape(anchor)}\"/>{title_txt}", h1)
                para._outline = (text, anchor, 0)
                story.append(para)
                heading_idx += 1
                story.append(Spacer(1, 6))
            elif kind == "h2":
                anchor = heading_specs[heading_idx][2] if heading_idx < len(heading_specs) else _slugify(text)
                title_txt = escape(text)
                para = Paragraph(f"<a name=\"{escape(anchor)}\"/>{title_txt}", h2)
                para._outline = (text, anchor, 1)
                story.append(para)
                heading_idx += 1
                story.append(Spacer(1, 4))
            elif kind == "h3":
                anchor = heading_specs[heading_idx][2] if heading_idx < len(heading_specs) else _slugify(text)
                title_txt = escape(text)
                para = Paragraph(f"<a name=\"{escape(anchor)}\"/>{title_txt}", h3)
                para._outline = (text, anchor, 2)
                story.append(para)
                heading_idx += 1
                story.append(Spacer(1, 3))
            else:
                story.append(Paragraph(escape(text), normal))

        doc.build(story, onFirstPage=draw_bg, onLaterPages=draw_bg)
        return buf.getvalue()
    finally:
        try:
            os.remove(epub_path)
        except OSError:
            pass
