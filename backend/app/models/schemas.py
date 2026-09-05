from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date

class UserProfile(BaseModel):
    entity_name: str = Field("Acme Technologies Pvt Ltd", description="Name of the user entity")
    entity_type: str = Field("Pvt Ltd", description="Pvt Ltd, LLP, Partnership, Proprietorship, Individual")
    turnover_lakhs: float = Field(120.0, description="Annual turnover in Lakhs INR")
    industry_sector: str = Field("IT & Software", description="IT & Software, Manufacturing, Retail, Healthcare, Construction")
    location_state: str = Field("Karnataka", description="State of operation")
    tax_registered: bool = Field(True, description="GST registered status")
    employee_count: int = Field(25, description="Number of employees")

class ExtractedMetadata(BaseModel):
    notice_title: str
    notice_type: str  # Show Cause Notice, Audit Notice, Demand Notice, Advisory, Statutory Return
    issuing_authority: str
    notice_ref_no: str
    notice_date: str
    act_rules_cited: List[str]
    jurisdiction: str

class ObligationItem(BaseModel):
    id: str
    title: str
    description: str
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    mandatory: bool
    penalty_clause: Optional[str] = None
    penalty_amount_inr: Optional[float] = None
    raw_deadline: str
    calculated_deadline: str
    days_remaining: int
    status: str  # PENDING, IN_PROGRESS, COMPLIED
    evidence_snippet: str
    start_offset: int
    end_offset: int

class TrustBreakdown(BaseModel):
    grounding_score: float  # 0 to 100
    format_authenticity: float  # 0 to 100
    authority_verification: float  # 0 to 100
    rule_consistency: float  # 0 to 100
    temporal_certainty: float  # 0 to 100
    overall_trust_score: float  # Weighted overall T score 0 to 100
    confidence_level: str  # HIGH, VERIFIED, MODERATE, UNCERTAIN

class EvidenceGrounding(BaseModel):
    id: str
    field_name: str
    snippet: str
    confidence: float
    start_offset: int
    end_offset: int

class UserApplicabilityResult(BaseModel):
    is_applicable: bool
    applicability_score: float  # 0 to 100
    reason: str
    matched_criteria: List[str]
    exemptions_applied: List[str]

class ActionStep(BaseModel):
    step_number: int
    action_title: str
    description: str
    assigned_role: str
    recommended_deadline: str
    deliverable: str

class NoticeAnalysisResult(BaseModel):
    notice_id: str
    analyzed_at: str
    metadata: ExtractedMetadata
    user_applicability: UserApplicabilityResult
    obligations: List[ObligationItem]
    trust_metrics: TrustBreakdown
    evidence_groundings: List[EvidenceGrounding]
    action_steps: List[ActionStep]
    compliance_response_draft: str

class NoticeInput(BaseModel):
    raw_text: str
    notice_title: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    notice_date: Optional[str] = None
