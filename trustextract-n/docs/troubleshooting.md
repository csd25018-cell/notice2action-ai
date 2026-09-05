# Troubleshooting & Common Issues

## 1. Tesseract / OCR Engine Not Found

### Symptom:
`pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your PATH`

### Resolution:
- **macOS**: Run `brew install tesseract`
- **Ubuntu/Debian**: Run `sudo apt-get install -y tesseract-ocr libgl1`
- **Windows**:
  1. Download Tesseract installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
  2. Install to `C:\Program Files\Tesseract-OCR`.
  3. Add `C:\Program Files\Tesseract-OCR` to System PATH environment variable.

---

## 2. PyMuPDF (`fitz`) Installation Warning / Issue

### Symptom:
`ModuleNotFoundError: No module named 'fitz'`

### Resolution:
Install `PyMuPDF`:
```bash
pip install --upgrade PyMuPDF
```

---

## 3. PyTorch Memory Error / Out of CUDA Memory

### Symptom:
`RuntimeError: CUDA out of memory`

### Resolution:
Set `DEVICE=cpu` in your `.env` or run with CPU fallback:
```bash
DEVICE=cpu python run_app.py
```

---

## 4. Port 8501 Already in Use

### Symptom:
`Port 8501 is already in use`

### Resolution:
Launch on a custom port:
```bash
APP_PORT=8502 python run_app.py
```
Or kill existing Streamlit processes:
- **macOS / Linux**: `pkill -f streamlit`
- **Windows**: `taskkill /IM streamlit.exe /F`
