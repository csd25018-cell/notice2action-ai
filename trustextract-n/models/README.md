# TrustExtract-N Models Directory

This directory contains fine-tuned model checkpoints, tokenizers, and temperature scaling calibration files.

---

## 📁 Directory Contents

```
models/
├── README.md                 <- Model documentation & download guidelines
└── trustextract/
    └── calibration.json      <- Learned temperature (T) & field-specific threshold parameters
```

---

## ⚙️ Calibration Configuration (`calibration.json`)

Contains post-hoc temperature scaling parameters learned on validation set logits:
- `temperature`: `1.50` (softens overconfident probabilities)
- `thresholds`: Field-specific acceptance cutoff parameters:
  - `last_date`: `0.85`
  - `authority`: `0.80`
  - `title`: `0.75`
  - `eligibility`: `0.75`
  - `required_documents`: `0.75`
  - `notice_number`: `0.75`
  - `contact`: `0.70`
  - `audience`: `0.70`

---

## 🚀 Pretrained Transformer Weights

By default, TrustExtract-N downloads pretrained transformer weights (`google/muril-base-cased`) automatically on first run via Hugging Face.

To run model initialization manually:
```bash
python scripts/download_models.py
```
