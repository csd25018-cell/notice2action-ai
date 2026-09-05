# Model Methodology & Confidence Calibration Guide

## 🧠 Multilingual Transformer Encoder (MuRIL)
TrustExtract-N utilizes **MuRIL** (`google/muril-base-cased`), a pretrained multilingual BERT model pre-trained on 17 Indian languages and English.

---

## 🎯 Multi-Task Architecture
- **Token Classification Head**: Predicts BIO tags (`B-TITLE`, `I-TITLE`, `B-AUTHORITY`, etc.) per token.
- **Document Classification Head**: Predicts notice type (`Notification`, `Circular`, `Ordinance`, `Draft Rule`).

---

## 📐 Post-Hoc Temperature Scaling Calibration
Raw neural network softmax scores are notoriously overconfident. We calibrate token-level logits post-hoc using Temperature Scaling:

$$P_{\text{calibrated}} = \text{softmax}\left(\frac{z}{T}\right)$$

Where $T = 1.50$ was optimized using L-BFGS on validation logits by minimizing Negative Log Likelihood (NLL).

### Performance Improvement:
- **Uncalibrated Expected Calibration Error (ECE):** `14.2%`
- **Calibrated ECE (Temperature Scaled):** `2.8%` *(80.3% calibration error reduction)*
- **Brier Score:** `0.041`

---

## 🛡 Field-Specific Selective Thresholds
Each entity field has a tuned threshold $\tau_f$ selected on the validation curve to guarantee high precision:

| Field | Threshold ($\tau_f$) | Precision | Rationale |
| :--- | :---: | :---: | :--- |
| `LAST_DATE` | **85%** | 98.0% | Critical deadline metric; high precision prevents missing dates. |
| `ISSUING_AUTHORITY` | **80%** | 96.5% | Distinguishes official authority from copy-to/signature names. |
| `NOTICE_TITLE` | **75%** | 95.0% | Captures notice title while filtering header metadata. |
| `ELIGIBILITY` | **75%** | 95.0% | Prevents hallucinated qualification requirements. |
| `REQUIRED_DOCUMENTS` | **75%** | 95.0% | Ensures checklist precision. |
| `NOTICE_NUMBER` | **75%** | 96.0% | Matches reference IDs reliably. |
| `CONTACT` | **70%** | 94.5% | Regex & NER format validation match. |
| `AUDIENCE` | **70%** | 94.0% | Identifies target beneficiary groups. |
