# pdfsepia

Small web app: upload a PDF → convert pages to sepia tone → download a new PDF.

## Run

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

# activate venv
.\.venv\Scripts\Activate.ps1

# install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# run the app
streamlit run app.py
```

## Notes
- This implementation rasterizes each PDF page to an image, applies a sepia effect, then rebuilds a PDF from those images.
- That means selectable text/searchability is not preserved (output is image-based).
