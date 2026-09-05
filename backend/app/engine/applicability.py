import re
from typing import Dict, Any, List
from backend.app.models.schemas import UserProfile, UserApplicabilityResult

class ApplicabilityEngine:
    """
    User-Conditioned Applicability Engine for notice2action-ai (TrustExtract-N)
    Evaluates whether a notice directive applies to a given user profile.
    """

    @staticmethod
    def evaluate(raw_text: str, profile: UserProfile) -> UserApplicabilityResult:
        text_lower = raw_text.lower()
        matched = []
        exemptions = []
        score = 80.0

        # Check turnover threshold in text (e.g., "turnover exceeding 50 lakhs", "turnover > 1 crore")
        turnover_regex = re.search(r'turnover\s+(?:exceeding|above|over|more than|>)\s+₹?\s*(\d+(?:\.\d+)?)\s*(lakh|crore|lakhs|crores)?', text_lower)
        if turnover_regex:
            val = float(turnover_regex.group(1))
            unit = (turnover_regex.group(2) or 'lakh').lower()
            threshold_lakhs = val * 100.0 if 'crore' in unit else val

            if profile.turnover_lakhs >= threshold_lakhs:
                matched.append(f"Entity Turnover (₹{profile.turnover_lakhs} Lakhs) satisfies threshold >= ₹{threshold_lakhs} Lakhs")
                score += 10
            else:
                exemptions.append(f"Turnover Exemption: Entity turnover (₹{profile.turnover_lakhs} Lakhs) is below notice threshold (₹{threshold_lakhs} Lakhs)")
                score -= 40

        # Check GSTIN requirement
        if "gstin" in text_lower or "gst registered" in text_lower or "registered person" in text_lower:
            if profile.tax_registered:
                matched.append("GST Registration status: Active Taxpayer")
                score += 5
            else:
                exemptions.append("GST Exemption: Notice targets GST registered entities; profile is non-registered")
                score -= 35

        # Check Industry Sector keywords
        sector_lower = profile.industry_sector.lower()
        if any(sec in text_lower for sec in [sector_lower, "all taxpayers", "all entities", "registered persons"]):
            matched.append(f"Industry Sector ({profile.industry_sector}) matches target scope")
            score += 5
        elif any(sec in text_lower for sec in ["manufacturing", "factory", "industrial", "export", "cyber", "it/ites"]):
            # Specific sector mentioned
            if sector_lower in text_lower:
                matched.append(f"Sector match: {profile.industry_sector} directly specified in notice")
                score += 15
            else:
                matched.append(f"Notice specifies sector guidelines that apply to {profile.industry_sector}")

        # Check Employee threshold (e.g. EPFO/Labor notice > 20 employees)
        emp_regex = re.search(r'(\d+)\s+or more employees|establishments employing (\d+)', text_lower)
        if emp_regex:
            emp_thresh = int(emp_regex.group(1) or emp_regex.group(2))
            if profile.employee_count >= emp_thresh:
                matched.append(f"Employee Count ({profile.employee_count}) satisfies threshold >= {emp_thresh}")
                score += 10
            else:
                exemptions.append(f"Employee Exemption: Establishment has {profile.employee_count} employees (threshold is {emp_thresh})")
                score -= 30

        # Normalize score
        final_score = max(0.0, min(100.0, score))
        is_applicable = len(exemptions) == 0 or final_score >= 50.0

        if is_applicable:
            reason = f"Applicable to {profile.entity_name} ({profile.entity_type}, {profile.industry_sector}). Satisfies statutory applicability criteria."
        else:
            reason = f"Exempt/Non-applicable to {profile.entity_name}. Applied exemptions: {'; '.join(exemptions)}"

        return UserApplicabilityResult(
            is_applicable=is_applicable,
            applicability_score=round(final_score, 1),
            reason=reason,
            matched_criteria=matched if matched else ["General entity jurisdiction applies"],
            exemptions_applied=exemptions
        )
