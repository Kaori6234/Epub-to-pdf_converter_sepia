import time
import io
import zipfile
import os

import streamlit as st

from epub_ops import EpubPdfOptions, epub_bytes_to_styled_pdf
from pdf_ops import SepiaOptions, pdf_to_sepia_pdf


st.set_page_config(page_title="PDF → Sepia", layout="centered")

st.title("Document to sepia")
st.write("Upload a PDF or EPUB, convert to a sepia-styled PDF, then download the result.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Input type", options=["PDF", "EPUB"], horizontal=True)

    batch_pdf = False
    if mode == "PDF":
        batch_pdf = st.toggle(
            "Batch convert PDFs",
            value=False,
            help="Upload multiple PDFs and download a ZIP of sepia PDFs.",
        )

    batch_epub = False
    if mode == "EPUB":
        batch_epub = st.toggle(
            "Batch convert EPUBs",
            value=False,
            help="Upload multiple EPUBs and download a ZIP of styled PDFs.",
        )

    intensity = st.slider(
        "Sepia intensity",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="For PDF inputs, this is applied after rendering each page. EPUB conversion is styled directly.",
    )
    dpi = st.selectbox("PDF render quality (DPI)", options=[100, 150, 200, 300], index=1)
    background_only = st.toggle(
        "Keep text black (background-only)",
        value=True,
        help="Recommended: tints light background pixels to sepia while keeping dark text unchanged.",
    )
    background_threshold = st.slider(
        "Background threshold",
        min_value=0.60,
        max_value=0.95,
        value=0.80,
        step=0.01,
        help="Higher preserves more (keeps more pixels unchanged).",
    )
    image_format = st.selectbox("Output image format", options=["JPEG", "PNG"], index=0)
    jpeg_quality = st.slider("JPEG quality", min_value=50, max_value=95, value=85, step=5)

    st.divider()
    st.subheader("EPUB to PDF styling")
    page_format = st.selectbox("PDF page size", options=["A4", "Letter"], index=0)
    margin_mm = st.slider("Margins (mm)", min_value=6, max_value=25, value=12, step=1)
    use_merriweather = st.toggle(
        "Use Merriweather font",
        value=True,
        help="Downloads and embeds Merriweather (internet required). If off, falls back to built-in serif fonts.",
    )


def _files_from_zip(zip_bytes: bytes, allowed_exts: set[str], max_files: int = 200, max_total_uncompressed: int = 250_000_000):
    """Return [(filename, bytes)] from a ZIP, with basic safety limits."""
    out: list[tuple[str, bytes]] = []
    total_uncompressed = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            name = (info.filename or "").replace("\\", "/")
            base = os.path.basename(name)
            ext = base.lower().rsplit(".", 1)[-1] if "." in base else ""
            if ext not in allowed_exts:
                continue

            total_uncompressed += int(info.file_size or 0)
            if total_uncompressed > max_total_uncompressed:
                raise ValueError("ZIP is too large (uncompressed).")

            data = zf.read(info)
            out.append((base, data))
            if len(out) >= max_files:
                break

    return out

is_pdf_batch = mode == "PDF" and batch_pdf
is_epub_batch = mode == "EPUB" and batch_epub

if is_pdf_batch:
    batch_source = st.radio("Batch input", options=["Select multiple files", "Upload a ZIP"], horizontal=True)
    pdf_inputs: list[tuple[str, bytes]] = []

    if batch_source == "Select multiple files":
        uploaded_many = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Tip: in the file picker you can multi-select (Ctrl/Shift) to add many PDFs in one go.",
        )
        if not uploaded_many:
            st.info("Choose one or more PDF files to get started.")
            st.stop()
        pdf_inputs = [(u.name, u.getvalue()) for u in uploaded_many]
    else:
        up_zip = st.file_uploader(
            "Upload a ZIP containing PDFs",
            type=["zip"],
            accept_multiple_files=False,
            help="ZIP may contain folders; only .pdf files are processed.",
        )
        if up_zip is None:
            st.info("Upload a ZIP file to get started.")
            st.stop()
        pdf_inputs = _files_from_zip(up_zip.getvalue(), allowed_exts={"pdf"})
        if not pdf_inputs:
            st.warning("No .pdf files found inside the ZIP.")
            st.stop()

elif is_epub_batch:
    batch_source = st.radio("Batch input", options=["Select multiple files", "Upload a ZIP"], horizontal=True)
    epub_inputs: list[tuple[str, bytes]] = []

    if batch_source == "Select multiple files":
        uploaded_many = st.file_uploader(
            "Upload EPUB files",
            type=["epub"],
            accept_multiple_files=True,
            help="Tip: in the file picker you can multi-select (Ctrl/Shift) to add many EPUBs in one go.",
        )
        if not uploaded_many:
            st.info("Choose one or more EPUB files to get started.")
            st.stop()
        epub_inputs = [(u.name, u.getvalue()) for u in uploaded_many]
    else:
        up_zip = st.file_uploader(
            "Upload a ZIP containing EPUBs",
            type=["zip"],
            accept_multiple_files=False,
            help="ZIP may contain folders; only .epub files are processed.",
        )
        if up_zip is None:
            st.info("Upload a ZIP file to get started.")
            st.stop()
        epub_inputs = _files_from_zip(up_zip.getvalue(), allowed_exts={"epub"})
        if not epub_inputs:
            st.warning("No .epub files found inside the ZIP.")
            st.stop()

else:
    uploaded = st.file_uploader(
        "Upload a file",
        type=["pdf", "epub"],
        help="PDF: rendered per page then sepia-applied. EPUB: converted to a styled PDF. Batch modes are available in the sidebar.",
    )

    if uploaded is None:
        st.info("Choose a PDF or EPUB file to get started.")
        st.stop()

    ext = uploaded.name.lower().rsplit(".", 1)[-1] if "." in uploaded.name else ""

    if mode == "PDF" and ext != "pdf":
        st.warning("Sidebar is set to PDF, but you uploaded a non-PDF. Switch to EPUB or upload a PDF.")
    if mode == "EPUB" and ext != "epub":
        st.warning("Sidebar is set to EPUB, but you uploaded a non-EPUB. Switch to PDF or upload an EPUB.")

if mode == "PDF" and batch_pdf:
    options = SepiaOptions(
        intensity=float(intensity),
        dpi=int(dpi),
        image_format=str(image_format),
        jpeg_quality=int(jpeg_quality),
        background_only=bool(background_only),
        background_threshold=float(background_threshold),
    )

    st.caption(
        f"Files: {len(pdf_inputs)} | dpi={options.dpi} | intensity={options.intensity:.2f} | format={options.image_format}"
    )

    if st.button("Convert PDFs to sepia (ZIP)", type="primary"):
        overall = st.progress(0)
        status = st.empty()

        out_zip = io.BytesIO()
        ts = time.strftime("%Y%m%d-%H%M%S")

        with st.spinner("Converting batch…"):
            try:
                with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    total_files = len(pdf_inputs)
                    for file_idx, (name, input_bytes) in enumerate(pdf_inputs, start=1):
                        status.write(f"Converting {file_idx}/{total_files}: {name}…")
                        output_bytes = pdf_to_sepia_pdf(input_bytes, options, progress_cb=None)

                        base = name[:-4] if name.lower().endswith(".pdf") else name
                        out_name = f"{base}-sepia-{ts}.pdf"
                        zf.writestr(out_name, output_bytes)

                        overall.progress(int(file_idx / total_files * 100))

            except Exception as e:
                st.error(f"Batch conversion failed: {e}")
            else:
                overall.progress(100)
                status.write("Done.")
                st.download_button(
                    label="Download ZIP",
                    data=out_zip.getvalue(),
                    file_name=f"sepia-batch-{ts}.zip",
                    mime="application/zip",
                )

elif mode == "PDF":
    options = SepiaOptions(
        intensity=float(intensity),
        dpi=int(dpi),
        image_format=str(image_format),
        jpeg_quality=int(jpeg_quality),
        background_only=bool(background_only),
        background_threshold=float(background_threshold),
    )

    st.caption(
        f"File: {uploaded.name} | dpi={options.dpi} | intensity={options.intensity:.2f} | format={options.image_format}"
    )

    if st.button("Convert PDF to sepia", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        def cb(cur: int, total: int) -> None:
            progress.progress(int(cur / total * 100))
            status.write(f"Processing page {cur}/{total}…")

        with st.spinner("Converting…"):
            try:
                input_bytes = uploaded.getvalue()
                output_bytes = pdf_to_sepia_pdf(input_bytes, options, progress_cb=cb)
            except Exception as e:
                st.error(f"Conversion failed: {e}")
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                base = uploaded.name[:-4] if uploaded.name.lower().endswith(".pdf") else uploaded.name
                out_name = f"{base}-sepia-{ts}.pdf"
                progress.progress(100)
                status.write("Done.")
                st.download_button(
                    label="Download sepia PDF",
                    data=output_bytes,
                    file_name=out_name,
                    mime="application/pdf",
                )

elif is_epub_batch:
    epub_options = EpubPdfOptions(
        page_format=str(page_format),
        margin_mm=int(margin_mm),
        use_merriweather=bool(use_merriweather),
    )

    st.caption(
        f"Files: {len(epub_inputs)} | page={epub_options.page_format} | margin={epub_options.margin_mm}mm | Merriweather={epub_options.use_merriweather}"
    )

    if st.button("Convert EPUBs to styled PDFs (ZIP)", type="primary"):
        overall = st.progress(0)
        status = st.empty()
        out_zip = io.BytesIO()
        ts = time.strftime("%Y%m%d-%H%M%S")

        with st.spinner("Converting batch…"):
            try:
                with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    total_files = len(epub_inputs)
                    for file_idx, (name, epub_bytes) in enumerate(epub_inputs, start=1):
                        status.write(f"Converting {file_idx}/{total_files}: {name}…")
                        pdf_bytes = epub_bytes_to_styled_pdf(epub_bytes, epub_options, progress_cb=None)
                        base = name[:-5] if name.lower().endswith(".epub") else name
                        out_name = f"{base}-styled-{ts}.pdf"
                        zf.writestr(out_name, pdf_bytes)
                        overall.progress(int(file_idx / total_files * 100))
            except Exception as e:
                st.error(f"Batch conversion failed: {e}")
            else:
                overall.progress(100)
                status.write("Done.")
                st.download_button(
                    label="Download ZIP",
                    data=out_zip.getvalue(),
                    file_name=f"styled-epub-batch-{ts}.zip",
                    mime="application/zip",
                )

else:
    epub_options = EpubPdfOptions(
        page_format=str(page_format),
        margin_mm=int(margin_mm),
        use_merriweather=bool(use_merriweather),
    )

    st.caption(
        f"File: {uploaded.name} | page={epub_options.page_format} | margin={epub_options.margin_mm}mm | Merriweather={epub_options.use_merriweather}"
    )

    st.info("EPUB conversion is generated as a new PDF with sepia background and black text.")

    if st.button("Convert EPUB to styled PDF", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        def cb(cur: int, total: int) -> None:
            progress.progress(int(cur / total * 100))
            status.write(f"Extracting content {cur}/{total}…")

        with st.spinner("Converting…"):
            try:
                epub_bytes = uploaded.getvalue()
                pdf_bytes = epub_bytes_to_styled_pdf(epub_bytes, epub_options, progress_cb=cb)
            except Exception as e:
                st.error(f"Conversion failed: {e}")
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                base = uploaded.name[:-5] if uploaded.name.lower().endswith(".epub") else uploaded.name
                out_name = f"{base}-styled-{ts}.pdf"
                progress.progress(100)
                status.write("Done.")
                st.download_button(
                    label="Download styled PDF",
                    data=pdf_bytes,
                    file_name=out_name,
                    mime="application/pdf",
                )
