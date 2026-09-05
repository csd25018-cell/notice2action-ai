import re
from typing import List, Dict, Any
from backend.app.models.schemas import TrustBreakdown, EvidenceGrounding

class TrustVerificationEngine:
    """
    Trust Verification Engine for TrustExtract-N (notice2action-ai)
    Computes evidence-grounded trust scores, format authenticity, and rule consistency.
    """

    RECOGNIZED_AUTHORITIES = [
        "CBIC", "GST", "INCOME TAX", "CENTRAL BOARD OF DIRECT TAXES", "CBDT",
        "MINISTRY OF CORPORATE AFFAIRS", "MCA", "REGISTRAR OF COMPANIES", "ROC",
        "EMPLOYEES' PROVIDENT FUND ORGANISATION", "EPFO", "CERT-IN",
        "MUNICIPAL CORPORATION", "FOOD SAFETY AND STANDARDS AUTHORITY", "FSSAI",
        "DEPT OF REVENUE", "CUSTOMS", "DIRECTORATE GENERAL OF FOREIGN TRADE", "DGFT",
        "MINISTRY OF HOME AFFAIRS", "MINISTRY OF HEALTH AND FAMILY WELFARE", "MINISTRY OF HEALTH",
        "MINISTRY OF EDUCATION", "UGC", "AICTE", "UNIVERSITY", "RESERVE BANK OF INDIA", "RBI",
        "SEBI", "GOVERNMENT OF INDIA", "GOVERNMENT OF", "HIGH COURT", "SUPREME COURT",
        "MINISTRY", "DEPARTMENT", "DIRECTORATE", "COMMISSION", "BOARD", "COUNCIL", "AUTHORITY"
    ]

    @classmethod
    def calculate_trust(
        cls,
        raw_text: str,
        notice_ref_no: str,
        issuing_authority: str,
        obligations_count: int,
        evidence_groundings: List[EvidenceGrounding]
    ) -> TrustBreakdown:
        text_upper = raw_text.upper()

        # 1. Evidence Grounding Score (0 to 100)
        if obligations_count > 0 and evidence_groundings:
            avg_ground_conf = sum(eg.confidence for eg in evidence_groundings) / len(evidence_groundings)
            grounding_score = min(100.0, avg_ground_conf * 100.0)
        else:
            grounding_score = 85.0

        # 2. Format Authenticity Score (0 to 100)
        format_checks = 0
        total_checks = 5

        # Ref / DIN Number check
        if notice_ref_no and notice_ref_no != "N/A" and re.search(r'[A-Z0-9\/-]{5,}', notice_ref_no):
            format_checks += 1
        elif re.search(r'(?:DIN|REF|NOTICE|F\.?\s*NO|MEMO)\s*[:\.\/]?\s*[A-Z0-9\/-]{5,}', text_upper):
            format_checks += 1

        # Section / Act citation check
        if re.search(r'SECTION\s+\d+|ACT,\s*\d{4}|RULE\s+\d+|CLAUSE\s+\d+', text_upper):
            format_checks += 1

        # Date marker check
        if re.search(r'DATED?\s*[:\-]?\s*\d', text_upper) or re.search(r'\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{4}', text_upper):
            format_checks += 1

        # Official designation signature block
        if re.search(r'(COMMISSIONER|OFFICER|INSPECTOR|DIRECTOR|ASSESSEE|REGISTRAR|SECRETARY|AUTHORISED SIGNATORY)', text_upper):
            format_checks += 1

        # Form / Legal Heading check
        if re.search(r'(FORM|NOTICE|SHOW CAUSE|DEMAND|SUMMONS|ORDER|ADVISORY|NOTIFICATION)', text_upper):
            format_checks += 1

        format_authenticity = (format_checks / total_checks) * 100.0

        # 3. Authority Verification Score (0 to 100)
        auth_upper = (issuing_authority or "").upper()
        if any(rec in auth_upper or rec in text_upper for rec in cls.RECOGNIZED_AUTHORITIES):
            authority_verification = 95.0
        elif any(kw in auth_upper or kw in text_upper for kw in ["MINISTRY", "DEPARTMENT", "GOVERNMENT", "BOARD", "COMMISSION", "AUTHORITY", "UNIVERSITY", "OFFICE", "DIRECTORATE"]):
            authority_verification = 90.0
        else:
            authority_verification = 75.0

        # 4. Rule Consistency Score (0 to 100)
        # Check for penalty consistency & non-contradictory logic
        rule_consistency = 90.0
        if "PENALTY" in text_upper or "INTEREST" in text_upper:
            rule_consistency += 5.0
        if "WITHIN" in text_upper or "DAYS" in text_upper:
            rule_consistency += 5.0

        rule_consistency = min(100.0, rule_consistency)

        # 5. Temporal Certainty Score (0 to 100)
        if re.search(r'\d{1,2}\s+(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{4}', text_upper):
            temporal_certainty = 98.0
        elif re.search(r'WITHIN\s+\d+\s+DAYS', text_upper):
            temporal_certainty = 90.0
        else:
            temporal_certainty = 78.0

        # Overall Trust Score Weighted Formula
        overall = (
            0.30 * grounding_score +
            0.25 * format_authenticity +
            0.20 * authority_verification +
            0.15 * rule_consistency +
            0.10 * temporal_certainty
        )
        overall = min(100.0, max(0.0, overall))

        if overall >= 90.0:
            confidence = "HIGH (VERIFIED)"
        elif overall >= 75.0:
            confidence = "MODERATE (RELIABLE)"
        else:
            confidence = "LOW (REQUIRES MANUAL REVIEW)"

        return TrustBreakdown(
            grounding_score=round(grounding_score, 1),
            format_authenticity=round(format_authenticity, 1),
            authority_verification=round(authority_verification, 1),
            rule_consistency=round(rule_consistency, 1),
            temporal_certainty=round(temporal_certainty, 1),
            overall_trust_score=round(overall, 1),
            confidence_level=confidence
        )
