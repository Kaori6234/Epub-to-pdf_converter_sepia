# pdfsepia

Small web app: upload a PDF → convert pages to sepia tone → download a new PDF.

## Run

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force



python -m venv .venv
# install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# run the app

.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Notes
- This implementation rasterizes each PDF page to an image, applies a sepia effect, then rebuilds a PDF from those images.
- That means selectable text/searchability is not preserved (output is image-based).

- EPUB → PDF is generated with a sepia page background and black text.
- Images found inside the EPUB are embedded as-is (no sepia applied). They are only scaled down if needed to fit the PDF page; pixel data is not altered.
- If the EPUB has a cover image, the output PDF uses it as a full-bleed first page, and the Table of Contents starts on the next page.
- For non-Latin scripts (e.g. Japanese/Arabic), the converter falls back to fonts that support those characters when available.
