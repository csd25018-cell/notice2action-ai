from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List, Dict, Any

from backend.app.models.schemas import NoticeInput, NoticeAnalysisResult, UserProfile
from backend.app.engine.trust_extract import TrustExtractNEngine
from backend.app.data.sample_notices import SAMPLE_NOTICES

router = APIRouter(prefix="/api", tags=["Notice2Action TrustExtract-N"])

@router.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "TrustExtract-N Multi-Task Engine v1.0",
        "features": [
            "Evidence Grounding & Offsets",
            "Temporal Intelligence Deadline Engine",
            "User-Conditioned Applicability Evaluator",
            "Trust Verification Score (0-100)"
        ]
    }

@router.get("/samples")
def get_sample_notices():
    return SAMPLE_NOTICES

@router.post("/analyze", response_model=NoticeAnalysisResult)
def analyze_notice(input_data: NoticeInput):
    if not input_data.raw_text or len(input_data.raw_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Notice text is too short or empty. Please provide full notice text.")

    try:
        result = TrustExtractNEngine.analyze_notice(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TrustExtract-N processing error: {str(e)}")

@router.post("/upload")
async def upload_notice_file(
    file: UploadFile = File(...),
    notice_title: Optional[str] = Form(None),
    turnover_lakhs: Optional[float] = Form(120.0),
    industry_sector: Optional[str] = Form("IT & Software"),
    entity_type: Optional[str] = Form("Pvt Ltd"),
    tax_registered: Optional[bool] = Form(True)
):
    try:
        content_bytes = await file.read()
        raw_text = content_bytes.decode("utf-8", errors="ignore")

        profile = UserProfile(
            entity_name="Acme Technologies Pvt Ltd",
            entity_type=entity_type,
            turnover_lakhs=turnover_lakhs,
            industry_sector=industry_sector,
            tax_registered=tax_registered
        )

        input_data = NoticeInput(
            raw_text=raw_text,
            notice_title=notice_title or file.filename,
            user_profile=profile
        )

        result = TrustExtractNEngine.analyze_notice(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File parsing error: {str(e)}")
