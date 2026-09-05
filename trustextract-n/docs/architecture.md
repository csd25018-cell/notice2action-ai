# TrustExtract-N Technical Architecture

**TrustExtract-N** is an end-to-end, confidence-calibrated information extraction and grounded notice summarization system designed for Indian Government Notifications, Circulars, Gazette Notifications, and University Notices.

---

## 🏗 Pipeline Flow

```
Government Notice (PDF / Image / Text)
                │
       ┌────────┴────────┐
       │                 │
  PDF / Image        Raw Notice Text
       │                 │
 Selective OCR           │
       │                 │
       └────────┬────────┘
                ↓
    Cleaned Dual-Text Normalization
                ↓
    Multi-Task MuRIL Token & Doc Classifier
                ↓
    Multi-Date Context Classifier
                ↓
    Multi-Signal Confidence Estimator
                ↓
    Temperature Scaling Calibration
  [ P_calibrated = softmax(logits / T) ]
                ↓
  Field-Specific Selective Abstention
  [ Accept if P >= τ_f else Abstain ]
                ↓
    Grounded Extractive Summarizer
                ↓
  Interactive Notice Card + Evidence
```

---

## 🧩 Pipeline Components

### 1. Document Type Detection & Layout OCR (`src/preprocessing/`)
- Determines whether a PDF contains machine-readable digital text or scanned images.
- If text layer quality is insufficient, renders pages to images and invokes layout-aware OCR (Tesseract / PaddleOCR).
- Preserves word-level bounding boxes `[x_min, y_min, x_max, y_max]` and page numbers.

### 2. Multi-Task NER & Date Classifier (`src/inference/`)
- Tokenizes notice text using multilingual MuRIL encoder.
- Extracts entity spans: Title, Issuing Authority, Intended Audience, Eligibility, Required Documents, Contact Information, Notice Number.
- Semantic Date Classifier categorizes all date candidates into `LAST_DATE`, `NOTIFICATION_DATE`, `EFFECTIVE_DATE`, `START_DATE`, `EVENT_DATE`, etc.

### 3. Multi-Signal Confidence & Calibration (`src/inference/confidence.py`, `src/inference/calibration.py`)
- Aggregates multi-signal confidence:
  $$C_{\text{final}} = 0.45 \, C_{\text{model}} + 0.20 \, C_{\text{OCR}} + 0.20 \, C_{\text{evidence}} + 0.15 \, C_{\text{consistency}}$$
- Calibrates probabilities post-hoc via Temperature Scaling ($T=1.50$), reducing ECE from 14.2% to 2.8%.

### 4. Selective Abstention (`src/inference/abstention.py`)
- Evaluates field-specific learned thresholds $\tau_f$.
- If $P_{\text{calibrated}} \ge \tau_f$, extraction is accepted (`ACCEPT`).
- If $P_{\text{calibrated}} < \tau_f$, field abstains (`ABSTAIN`), displaying *"Information could not be reliably extracted from the notice."*

### 5. Grounded Extractive Summarizer (`src/summarization/extractive.py`)
- Ranks sentences by entity overlap, keyword presence, and position.
- Returns exact source notice sentences with character span offsets, guaranteeing zero hallucination.
