import time

import streamlit as st

from epub_ops import EpubPdfOptions, epub_bytes_to_styled_pdf
from pdf_ops import SepiaOptions, pdf_to_sepia_pdf


st.set_page_config(page_title="PDF → Sepia", layout="centered")

st.title("Document to sepia")
st.write("Upload a PDF or EPUB, convert to a sepia-styled PDF, then download the result.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Input type", options=["PDF", "EPUB"], horizontal=True)

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


uploaded = st.file_uploader(
    "Upload a file",
    type=["pdf", "epub"],
    help="PDF: rendered per page then sepia-applied. EPUB: converted to a styled PDF via a browser renderer.",
)

if uploaded is None:
    st.info("Choose a PDF or EPUB file to get started.")
    st.stop()

ext = uploaded.name.lower().rsplit(".", 1)[-1] if "." in uploaded.name else ""

if mode == "PDF" and ext != "pdf":
    st.warning("Sidebar is set to PDF, but you uploaded a non-PDF. Switch to EPUB or upload a PDF.")
if mode == "EPUB" and ext != "epub":
    st.warning("Sidebar is set to EPUB, but you uploaded a non-EPUB. Switch to PDF or upload an EPUB.")

if mode == "PDF":
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
