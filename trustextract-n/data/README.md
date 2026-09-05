# TrustExtract-N Data Directory

This directory stores datasets, sample demonstration notices, and runtime uploads for TrustExtract-N.

---

## 📁 Directory Structure

```
data/
├── README.md             <- Dataset documentation & guidelines
├── samples/              <- Sample text & scanned notices for instant testing
│   ├── sample_notice.txt
│   └── sample_scanned.txt
├── uploads/              <- Runtime temporary upload folder (ignored by git)
└── active_learning/      <- Active learning inbox for low-confidence reviews
    └── inbox/
```

---

## 🧪 Demonstration Samples Included

- `samples/sample_notice.txt`: Digital government notification from Ministry of Health (High Confidence Case 1).
- `samples/sample_scanned.txt`: Ambiguous/scanned notice text demonstrating low-confidence selective abstention (Case 2).

---

## 🏋️ Training Datasets (Optional)

Training dataset files (`train.jsonl`, `val.jsonl`) are required only when retraining or tuning model weights using `src/training/train.py`.

The inference pipeline and Streamlit application run out-of-the-box using pretrained weights and do **NOT** require downloading full raw training datasets.
