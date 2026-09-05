"""
config.py – Centralized Configuration for TrustExtract-N
=========================================================
Handles project-relative paths, device selection, model configurations,
field thresholds, and environment variable overrides.
"""

import os
import torch
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Core Paths
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "trustextract"))
BASELINE_MODEL_PATH = Path(os.getenv("BASELINE_MODEL_PATH", PROJECT_ROOT / "output" / "baseline_model"))
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data"))
SAMPLES_PATH = DATA_PATH / "samples"
UPLOAD_PATH = Path(os.getenv("UPLOAD_PATH", PROJECT_ROOT / "data" / "uploads"))
CACHE_PATH = Path(os.getenv("CACHE_PATH", PROJECT_ROOT / ".cache"))
CONFIGS_PATH = PROJECT_ROOT / "configs"
DOCS_PATH = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
for path in [MODEL_PATH, BASELINE_MODEL_PATH, DATA_PATH, SAMPLES_PATH, UPLOAD_PATH, CACHE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Device Selection Logic (CUDA -> MPS -> CPU Fallback)
def get_device() -> str:
    env_device = os.getenv("DEVICE", "auto").lower()
    if env_device != "auto":
        return env_device
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()

# Pretrained Model Configurations
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "google/muril-base-cased")
CALIBRATION_FILE = Path(os.getenv("CALIBRATION_FILE", MODEL_PATH / "calibration.json"))

# Confidence Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

FIELD_THRESHOLDS = {
    "title": float(os.getenv("THRESHOLD_TITLE", "0.75")),
    "authority": float(os.getenv("THRESHOLD_AUTHORITY", "0.80")),
    "audience": float(os.getenv("THRESHOLD_AUDIENCE", "0.70")),
    "eligibility": float(os.getenv("THRESHOLD_ELIGIBILITY", "0.75")),
    "document": float(os.getenv("THRESHOLD_DOCUMENT", "0.75")),
    "required_documents": float(os.getenv("THRESHOLD_DOCUMENT", "0.75")),
    "date": float(os.getenv("THRESHOLD_DATE", "0.80")),
    "last_date": float(os.getenv("THRESHOLD_LAST_DATE", "0.85")),
    "contact": float(os.getenv("THRESHOLD_CONTACT", "0.70")),
    "notice_number": float(os.getenv("THRESHOLD_NOTICE_NUMBER", "0.75")),
}

# Logging Level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
