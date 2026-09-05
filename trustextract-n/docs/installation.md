# Installation & Setup Guide for TrustExtract-N

This guide provides step-by-step instructions for installing and running **TrustExtract-N** on **macOS**, **Linux**, and **Windows**.

---

## 📋 Prerequisites

- **Python**: `3.9`, `3.10`, or `3.11`
- **Git**: Installed
- *(Optional)* **Tesseract OCR**: Installed for scanned image PDF processing

---

## 🍏 macOS Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/trustextract-n.git
cd trustextract-n

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Install Tesseract via Homebrew for OCR support
brew install tesseract

# 5. Run System Diagnostic & Smoke Test
python scripts/system_info.py
python scripts/smoke_test.py

# 6. Launch Application
python run_app.py
```

---

## 🐧 Linux (Ubuntu / Debian) Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/trustextract-n.git
cd trustextract-n

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install Tesseract & Poppler for PDF OCR rendering
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libgl1

# 5. Run System Diagnostic & Smoke Test
python scripts/system_info.py
python scripts/smoke_test.py

# 6. Launch Application
python run_app.py
```

---

## 🪟 Windows Installation (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/your-username/trustextract-n.git
cd trustextract-n

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run System Diagnostic & Smoke Test
python scripts\system_info.py
python scripts\smoke_test.py

# 5. Launch Application
python run_app.py
```

---

## 🐳 Docker Setup (Alternative)

If you prefer containerized deployment:

```bash
# Build Docker image
docker build -t trustextract-n .

# Run container
docker run -p 8501:8501 trustextract-n
```
Or using docker-compose:
```bash
docker-compose up --build
```
Access the application at `http://localhost:8501`.
