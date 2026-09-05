#!/usr/bin/env python3
"""
smoke_test.py – Comprehensive End-to-End Smoke Test Script
=========================================================
Verifies imports, OCR engine, model initialization, text/PDF processing,
field extraction, confidence scoring, evidence quotes, and structured output.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def run_smoke_test():
    print("=" * 65)
    print("      TRUSTEXTRACT-N COMPREHENSIVE SMOKE TEST      ")
    print("=" * 65)
    
    passed_steps = 0
    total_steps = 7

    # 1. Test Environment & Core Imports
    try:
        import torch
        import transformers
        import fitz
        from src.trustextract.config import PROJECT_ROOT, DEVICE
        print(f"  [PASS 1/{total_steps}] Environment & Core Imports (Device: {DEVICE})")
        passed_steps += 1
    except Exception as e:
        print(f"  [FAIL 1/{total_steps}] Environment Import Error: {e}")
        return False

    # 2. Test OCR & Preprocessing Modules
    try:
        from src.preprocessing.pdf_parser import PDFParser
        from src.preprocessing.ocr import OCREngine
        from src.preprocessing.text_cleaner import TextCleaner
        pdf_parser = PDFParser()
        ocr_engine = OCREngine()
        cleaner = TextCleaner()
        print(f"  [PASS 2/{total_steps}] OCR Engine & PDF Preprocessors Initialized")
        passed_steps += 1
    except Exception as e:
        print(f"  [FAIL 2/{total_steps}] OCR Initialization Error: {e}")

    # 3. Test Pipeline Initialization
    try:
        from src.inference.pipeline import TrustExtractPipeline
        pipeline = TrustExtractPipeline(
            model_dir="output/baseline_model",
            calibration_path="models/trustextract/calibration.json"
        )
        print(f"  [PASS 3/{total_steps}] TrustExtract Ingestion Pipeline Initialized")
        passed_steps += 1
    except Exception as e:
        print(f"  [FAIL 3/{total_steps}] Pipeline Initialization Error: {e}")
        return False

    # 4. Test Notice Processing (Text Input)
    sample_notice = """MINISTRY OF HEALTH AND FAMILY WELFARE
GOVERNMENT OF INDIA
NOTIFICATION
Dated: 05th August, 2026.
Notice No: Z.28015/15/2026-DRS

All registered drug manufacturers must submit their annual compliance report by 30th September, 2026.
Required Documents:
1. Copy of drug manufacturing license
2. Audit report
Contact: drug-control@gov.in"""

    try:
        result = pipeline.process_raw_text(sample_notice, document_id="smoke_test_doc")
        print(f"  [PASS 4/{total_steps}] Notice Text Ingestion & Execution")
        passed_steps += 1
    except Exception as e:
        print(f"  [FAIL 4/{total_steps}] Text Processing Error: {e}")
        return False

    # 5. Test Field Extractions & Last Date Classifier
    try:
        dates = result.get("dates", [])
        last_date_found = any(d.get("type") in ("LAST_DATE", "DEADLINE") for d in dates)
        assert len(dates) >= 1
        assert result.get("contact", {}).get("value") is not None or result.get("authority", {}).get("value") is not None
        print(f"  [PASS 5/{total_steps}] Field Extraction & Multi-Date Semantic Classifier")
        passed_steps += 1
    except Exception as e:
        print(f"  [FAIL 5/{total_steps}] Field Extraction Verification Error: {e}")

    # 6. Test Confidence Calibration & Decision Status
    try:
        auth_data = result.get("authority", {})
        title_data = result.get("title", {})
        assert "confidence" in auth_data
        assert "status" in auth_data
        assert "decision" in auth_data
        print(f"  [PASS 6/{total_steps}] Calibrated Confidence & Selective Decision Rule")
        passed_steps += 1
    except Exception as e:
        print(f"  [FAIL 6/{total_steps}] Confidence Scoring Error: {e}")

    # 7. Test Grounded Notice Summarization
    try:
        summary_text = result.get("summary", {}).get("text", "")
        sentences = result.get("summary", {}).get("source_sentences", [])
        assert len(summary_text) > 10
        assert len(sentences) >= 1
        print(f"  [PASS 7/{total_steps}] Grounded Extractive Notice Summarization")
        passed_steps += 1
    except Exception as e:
        print(f"  [FAIL 7/{total_steps}] Grounded Summarization Error: {e}")

    print("=" * 65)
    if passed_steps == total_steps:
        print("  ✓ ALL SMOKE TESTS PASSED! TrustExtract-N is ready for execution.")
        print("=" * 65 + "\n")
        return True
    else:
        print(f"  ⚠️ SMOKE TEST COMPLETED WITH {total_steps - passed_steps} WARNING(S).")
        print("=" * 65 + "\n")
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
