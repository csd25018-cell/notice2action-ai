#!/usr/bin/env python3
"""
system_info.py – System Diagnostic Script for TrustExtract-N
============================================================
Displays Python, OS, PyTorch, Transformers, OCR dependencies,
CUDA/MPS hardware acceleration, and directory health status.
"""

import sys
import os
import platform
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def main():
    print("=" * 60)
    print("      TRUSTEXTRACT-N SYSTEM INFORMATION & DIAGNOSTICS      ")
    print("=" * 60)
    
    # 1. Operating System & Python
    print(f"  OS Platform       : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python Version    : {sys.version.split()[0]} ({sys.executable})")
    print(f"  Project Root      : {root_dir}")

    # 2. PyTorch & Hardware Acceleration
    try:
        import torch
        print(f"  PyTorch Version   : {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  Hardware Device   : CUDA GPU ({gpu_name})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  Hardware Device   : Apple Silicon MPS (GPU Acceleration)")
        else:
            print(f"  Hardware Device   : CPU (CPU Fallback Mode)")
    except ImportError:
        print("  PyTorch Version   : ❌ NOT INSTALLED")

    # 3. Transformers
    try:
        import transformers
        print(f"  Transformers      : {transformers.__version__}")
    except ImportError:
        print("  Transformers      : ❌ NOT INSTALLED")

    # 4. PDF Reader (PyMuPDF)
    try:
        import fitz
        print(f"  PyMuPDF (fitz)    : {fitz.__version__}")
    except ImportError:
        print("  PyMuPDF (fitz)    : ❌ NOT INSTALLED")

    # 5. OCR Engine Status
    ocr_status = "Available"
    try:
        import pytesseract
        print(f"  PyTesseract OCR   : Installed")
    except ImportError:
        print("  PyTesseract OCR   : ⚠️ Not Installed (Fallback active)")

    try:
        import paddleocr
        print(f"  PaddleOCR         : Installed")
    except ImportError:
        print("  PaddleOCR         : ⚠️ Not Installed (Standard OCR active)")

    # 6. Streamlit
    try:
        import streamlit
        print(f"  Streamlit UI      : {streamlit.__version__}")
    except ImportError:
        print("  Streamlit UI      : ❌ NOT INSTALLED")

    # 7. Check Key Project Directories
    print("\n── DIRECTORY HEALTH CHECK ─────────────────────────────────")
    dirs = [
        ("models/trustextract", root_dir / "models" / "trustextract"),
        ("output/baseline_model", root_dir / "output" / "baseline_model"),
        ("data/samples", root_dir / "data" / "samples"),
        ("app/app.py", root_dir / "app" / "app.py"),
    ]
    for label, path in dirs:
        status = "✓ Ready" if path.exists() else "⚠️ Not Found (Will auto-initialize)"
        print(f"  {label:<25} : {status}")

    print("=" * 60)
    print("  System Diagnostic Complete.\n")

if __name__ == "__main__":
    main()
