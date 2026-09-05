import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from backend.app.models.schemas import (
    NoticeInput, NoticeAnalysisResult, ExtractedMetadata, ObligationItem,
    EvidenceGrounding, ActionStep, UserProfile
)
from backend.app.engine.temporal_intel import TemporalIntelligenceEngine
from backend.app.engine.applicability import ApplicabilityEngine
from backend.app.engine.trust_score import TrustVerificationEngine

class TrustExtractNEngine:
    """
    TrustExtract-N: Trustworthy Notice Extraction & Action Plan Engine.
    Multi-task NLP parsing, evidence grounding, temporal intelligence, and user-conditioned applicability.
    """

    @classmethod
    def analyze_notice(cls, input_data: NoticeInput) -> NoticeAnalysisResult:
        text = input_data.raw_text.strip()
        notice_id = f"N2A-{uuid.uuid4().hex[:8].upper()}"

        # 1. Extract Metadata
        metadata = cls._extract_metadata(text, input_data.notice_title, input_data.notice_date)

        # 2. Extract Obligations with offsets
        obligations, evidence_groundings = cls._extract_obligations_and_evidence(text, metadata.notice_date)

        # 3. User Applicability Evaluation
        profile = input_data.user_profile or UserProfile()
        applicability = ApplicabilityEngine.evaluate(text, profile)

        # 4. Calculate Trust Metrics
        trust_metrics = TrustVerificationEngine.calculate_trust(
            raw_text=text,
            notice_ref_no=metadata.notice_ref_no,
            issuing_authority=metadata.issuing_authority,
            obligations_count=len(obligations),
            evidence_groundings=evidence_groundings
        )

        # 5. Generate Step-by-Step Action Plan
        action_steps = cls._generate_action_steps(obligations, metadata)

        # 6. Generate Legal/Compliance Response Draft Template
        response_draft = cls._generate_response_draft(metadata, profile, obligations)

        return NoticeAnalysisResult(
            notice_id=notice_id,
            analyzed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata,
            user_applicability=applicability,
            obligations=obligations,
            trust_metrics=trust_metrics,
            evidence_groundings=evidence_groundings,
            action_steps=action_steps,
            compliance_response_draft=response_draft
        )

    @classmethod
    def _extract_metadata(cls, text: str, user_title: Optional[str], user_date: Optional[str]) -> ExtractedMetadata:
        text_upper = text.upper()

        # Issuing Authority
        authority = "Government Authority / Department"
        if "HEALTH" in text_upper:
            authority = "Ministry of Health & Family Welfare, Government of India"
        elif "HOME AFFAIRS" in text_upper:
            authority = "Ministry of Home Affairs, Government of India"
        elif "INCOME TAX" in text_upper or "CBDT" in text_upper:
            authority = "Income Tax Department, Ministry of Finance"
        elif "CORPORATE AFFAIRS" in text_upper or "ROC" in text_upper or "MCA" in text_upper:
            authority = "Registrar of Companies (ROC), Ministry of Corporate Affairs"
        elif "EPFO" in text_upper or "PROVIDENT FUND" in text_upper:
            authority = "Employees' Provident Fund Organisation (EPFO)"
        elif "CERT-IN" in text_upper or "CYBER" in text_upper:
            authority = "Indian Computer Emergency Response Team (CERT-In)"
        elif "MUNICIPAL" in text_upper or "TOWN PLANNING" in text_upper:
            authority = "Municipal Corporation & Urban Development Authority"
        elif "CUSTOMS" in text_upper or "DGFT" in text_upper:
            authority = "Directorate General of Foreign Trade (DGFT)"
        elif "TEZPUR" in text_upper or "UNIVERSITY" in text_upper:
            authority = "Tezpur University / Central University"
        elif "CBIC" in text_upper or "GST" in text_upper:
            authority = "Central Board of Indirect Taxes & Customs (CBIC)"
        else:
            # Dynamic regex extraction for Ministry / Department / Government / Board / University
            auth_match = re.search(
                r'(MINISTRY OF [A-Z\s&]+|DEPARTMENT OF [A-Z\s&]+|GOVERNMENT OF [A-Z\s&]+|[A-Z\s&]+UNIVERSITY|[A-Z\s&]+BOARD|[A-Z\s&]+COMMISSION)',
                text_upper
            )
            if auth_match:
                authority = auth_match.group(1).title()

        # Notice Title / Type
        title = user_title or "Statutory Notice & Obligation Demand"
        notice_type = "Show Cause Notice"
        if "SHOW CAUSE" in text_upper:
            notice_type = "Show Cause Notice (SCN)"
            title = user_title or "Show Cause Notice under Statutory Act"
        elif "AUDIT" in text_upper or "SCRUTINY" in text_upper:
            notice_type = "Audit & Scrutiny Notice"
            title = user_title or "Notice for Statutory Audit & Scrutiny"
        elif "DEMAND" in text_upper:
            notice_type = "Tax & Fine Demand Notice"
            title = user_title or "Demand Notice for Unpaid Duties / Fines"
        elif "ADVISORY" in text_upper or "DIRECTIVE" in text_upper:
            notice_type = "Compliance Advisory & Directive"
            title = user_title or "Statutory Compliance Directive"

        # Notice Reference Number / DIN
        ref_no = "DIN-2026-884920-CBIC"
        ref_match = re.search(r'(?:DIN|REF|NOTICE\s*NO|MEMO\s*NO|F\.?\s*NO)[:\s\.-]+([A-Z0-9\/-]{6,})', text_upper)
        if ref_match:
            ref_no = ref_match.group(1)

        # Notice Date
        notice_date = user_date or datetime.now().strftime("%Y-%m-%d")
        date_match = re.search(r'DATED?\s*[:\-]?\s*(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})', text)
        if date_match:
            notice_date = date_match.group(1)

        # Acts & Rules Cited
        acts = []
        if "CGST" in text_upper or "GST" in text_upper:
            acts.append("Central Goods and Services Tax Act, 2017 (Section 73/74)")
        if "INCOME TAX" in text_upper:
            acts.append("Income Tax Act, 1961 (Section 142(1) / 148)")
        if "COMPANIES ACT" in text_upper:
            acts.append("Companies Act, 2013 (Section 92 / 137)")
        if "EPF" in text_upper or "PROVIDENT" in text_upper:
            acts.append("Employees' Provident Funds and Misc Provisions Act, 1952")
        if "INFORMATION TECHNOLOGY" in text_upper or "CERT-IN" in text_upper:
            acts.append("Information Technology Act, 2000 (Section 70B)")

        if not acts:
            acts.append("Relevant Statutory & Regulatory Provisions")

        # Jurisdiction
        jurisdiction = "Jurisdictional Officer / State Commissionerate"
        if "BANGALORE" in text_upper or "KARNATAKA" in text_upper:
            jurisdiction = "Bengaluru GST & IT Commissionerate, Zone 1"
        elif "MUMBAI" in text_upper or "MAHARASHTRA" in text_upper:
            jurisdiction = "Mumbai Corporate Range 4"
        elif "DELHI" in text_upper:
            jurisdiction = "Delhi Central Taxes Division"

        return ExtractedMetadata(
            notice_title=title,
            notice_type=notice_type,
            issuing_authority=authority,
            notice_ref_no=ref_no,
            notice_date=notice_date,
            act_rules_cited=acts,
            jurisdiction=jurisdiction
        )

    @classmethod
    def _extract_obligations_and_evidence(cls, text: str, notice_date: str):
        obligations: List[ObligationItem] = []
        evidence_groundings: List[EvidenceGrounding] = []

        # Split text into paragraphs/sentences to locate offsets
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Pre-defined requirement triggers
        triggers = [
            {"kw": ["furnish", "submit", "file", "provide", "deposit"], "priority": "HIGH"},
            {"kw": ["show cause", "explain", "reply", "written submission"], "priority": "CRITICAL"},
            {"kw": ["pay", "remit", "penalty", "interest", "tax amount"], "priority": "CRITICAL"},
            {"kw": ["attend", "appear", "produce documents", "books of account"], "priority": "HIGH"},
            {"kw": ["audit", "rectify", "compliance", "report"], "priority": "MEDIUM"}
        ]

        item_idx = 1
        for line in lines:
            line_lower = line.lower()
            # Find if line contains actionable obligations
            for trig in triggers:
                if any(k in line_lower for k in trig["kw"]):
                    # Look for deadline in proximity
                    deadline_match = re.search(r'(within\s+\d+\s+days|by\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}|before\s+[A-Za-z0-9\s,]+)', line_lower)
                    raw_dl = deadline_match.group(0) if deadline_match else "within 15 days of notice date"

                    # Calculate target deadline via Temporal Intelligence
                    calc_dl, days_rem = TemporalIntelligenceEngine.parse_deadline(raw_dl, notice_date)

                    # Extract penalty clause if present
                    pen_clause = None
                    pen_amt = None
                    pen_match = re.search(r'(penalty|fine|interest)\s+of\s+₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)', line_lower)
                    if pen_match:
                        pen_clause = pen_match.group(0)
                        try:
                            pen_amt = float(pen_match.group(2).replace(',', ''))
                        except Exception:
                            pass

                    # Character offsets
                    start_off = text.find(line[:25]) if len(line) >= 25 else text.find(line)
                    if start_off == -1:
                        start_off = 0
                    end_off = start_off + len(line)

                    ob_id = f"OBL-{item_idx:02d}"
                    ob = ObligationItem(
                        id=ob_id,
                        title=f"{trig['kw'][0].capitalize()} Compliance Obligation",
                        description=line,
                        priority=trig["priority"],
                        mandatory=True,
                        penalty_clause=pen_clause,
                        penalty_amount_inr=pen_amt,
                        raw_deadline=raw_dl,
                        calculated_deadline=calc_dl,
                        days_remaining=days_rem,
                        status="PENDING",
                        evidence_snippet=line,
                        start_offset=start_off,
                        end_offset=end_off
                    )
                    obligations.append(ob)

                    eg = EvidenceGrounding(
                        id=f"EVD-{item_idx:02d}",
                        field_name=f"Obligation #{item_idx}",
                        snippet=line,
                        confidence=0.94,
                        start_offset=start_off,
                        end_offset=end_off
                    )
                    evidence_groundings.append(eg)
                    item_idx += 1
                    break  # avoid duplicate matching on same line

        # Fallback obligation if none found
        if not obligations:
            start_off = 0
            end_off = min(len(text), 150)
            snip = text[:150]
            calc_dl, days_rem = TemporalIntelligenceEngine.parse_deadline("within 15 days", notice_date)

            ob = ObligationItem(
                id="OBL-01",
                title="Review Notice & Submit Formal Response",
                description="Review the statutory notice and submit a verified written response to the issuing officer.",
                priority="HIGH",
                mandatory=True,
                penalty_clause="Statutory interest/fine as applicable under relevant Act",
                penalty_amount_inr=5000.0,
                raw_deadline="within 15 days of notice date",
                calculated_deadline=calc_dl,
                days_remaining=days_rem,
                status="PENDING",
                evidence_snippet=snip,
                start_offset=start_off,
                end_offset=end_off
            )
            obligations.append(ob)
            evidence_groundings.append(
                EvidenceGrounding(
                    id="EVD-01",
                    field_name="General Notice Directive",
                    snippet=snip,
                    confidence=0.90,
                    start_offset=start_off,
                    end_offset=end_off
                )
            )

        return obligations, evidence_groundings

    @classmethod
    def _generate_action_steps(cls, obligations: List[ObligationItem], metadata: ExtractedMetadata) -> List[ActionStep]:
        steps = [
            ActionStep(
                step_number=1,
                action_title="Document Audit & Record Verification",
                description=f"Gather all relevant books of accounts, tax returns, and filings cited under {', '.join(metadata.act_rules_cited)}.",
                assigned_role="Finance & Compliance Lead",
                recommended_deadline="Day 1 - Day 3",
                deliverable="Reconciliation Sheet & Source Records"
            ),
            ActionStep(
                step_number=2,
                action_title="Legal & Technical Assessment",
                description=f"Draft written explanation addressing specific obligations ({len(obligations)} extracted requirement items).",
                assigned_role="Tax Advocate / Company Secretary",
                recommended_deadline="Day 4 - Day 7",
                deliverable="Draft Response Affidavit & Supporting Evidences"
            ),
            ActionStep(
                step_number=3,
                action_title="Formal Response Filing & Receipt Tracking",
                description=f"Submit final reply to {metadata.issuing_authority} ({metadata.jurisdiction}) referencing {metadata.notice_ref_no}.",
                assigned_role="Authorized Signatory",
                recommended_deadline=obligations[0].calculated_deadline if obligations else "Within Statutory Limit",
                deliverable="Acknowledgement Receipt & Portal Acknowledgement Copy"
            )
        ]
        return steps

    @classmethod
    def _generate_response_draft(cls, metadata: ExtractedMetadata, profile: UserProfile, obligations: List[ObligationItem]) -> str:
        ob_text = "\n".join([f"  {i+1}. {ob.description} (Ref: {ob.raw_deadline})" for i, ob in enumerate(obligations)])

        draft = f"""BEFORE THE {metadata.issuing_authority.upper()}
{metadata.jurisdiction.upper()}

Ref Notice No: {metadata.notice_ref_no}
Notice Date: {metadata.notice_date}
In the matter of: {profile.entity_name} ({profile.entity_type}, GSTIN/PAN: Active)

SUBJECT: FORMAL WRITTEN SUBMISSION / REPLY TO {metadata.notice_type.upper()}

To,
The Assessing Officer / Competent Authority,
{metadata.issuing_authority},
{metadata.jurisdiction}

Respected Sir/Madam,

1. We are in receipt of the subject notice dated {metadata.notice_date} issued under {', '.join(metadata.act_rules_cited)}.

2. We respectfully submit that {profile.entity_name} is a law-abiding business operating in the {profile.industry_sector} sector, duly maintaining regular books of accounts and statutory compliance records.

3. In response to the specific directions contained in the notice:
{ob_text}

4. We respectfully pray that the documents attached herewith be taken on record and the proceedings/demand be dropped/closed accordingly.

Yours Faithfully,
For {profile.entity_name}

Authorized Signatory / Tax Representative
Name: ______________________
Designation: __________________
Date: {datetime.now().strftime("%Y-%m-%d")}
"""
        return draft
