import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.schemas import NoticeInput, UserProfile
from backend.app.engine.trust_extract import TrustExtractNEngine
from backend.app.engine.temporal_intel import TemporalIntelligenceEngine
from backend.app.engine.applicability import ApplicabilityEngine
from backend.app.engine.trust_score import TrustVerificationEngine
from backend.app.data.sample_notices import SAMPLE_NOTICES

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "TrustExtract-N" in data["engine"]

def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 5
    assert samples[0]["id"] == "sample-gst-01"

def test_trust_extract_analysis():
    sample_text = SAMPLE_NOTICES[0]["text"]
    input_data = NoticeInput(
        raw_text=sample_text,
        notice_title="GST Audit Notice",
        user_profile=UserProfile(
            entity_name="Test Pvt Ltd",
            turnover_lakhs=150.0,
            industry_sector="IT & Software"
        )
    )

    result = TrustExtractNEngine.analyze_notice(input_data)
    assert result.notice_id.startswith("N2A-")
    assert result.metadata.notice_ref_no is not None
    assert len(result.obligations) > 0
    assert result.trust_metrics.overall_trust_score >= 0.0
    assert result.trust_metrics.overall_trust_score <= 100.0
    assert len(result.action_steps) == 3
    assert "FORMAL WRITTEN SUBMISSION" in result.compliance_response_draft

def test_temporal_intel_parser():
    calc_date, days_rem = TemporalIntelligenceEngine.parse_deadline("within 15 days", "2026-09-01")
    assert calc_date == "2026-09-16"

def test_applicability_evaluation():
    profile_small = UserProfile(turnover_lakhs=10.0, tax_registered=False)
    profile_large = UserProfile(turnover_lakhs=200.0, tax_registered=True)

    text_notice = "Notice for GST registered persons having turnover exceeding 50 lakhs"

    eval_small = ApplicabilityEngine.evaluate(text_notice, profile_small)
    eval_large = ApplicabilityEngine.evaluate(text_notice, profile_large)

    assert eval_small.is_applicable is False
    assert len(eval_small.exemptions_applied) > 0

    assert eval_large.is_applicable is True
    assert len(eval_large.matched_criteria) > 0

def test_api_analyze_route():
    payload = {
        "raw_text": SAMPLE_NOTICES[1]["text"],
        "notice_title": "MCA Annual Return Default",
        "user_profile": {
            "entity_name": "Acme Tech",
            "turnover_lakhs": 120.0
        }
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["issuing_authority"] is not None
    assert len(data["obligations"]) > 0
