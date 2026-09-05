import json
import uuid

# Document 1 (Ministry of Home Affairs)
doc1_text = """MINISTRY OF HOME AFFAIRS
NOTIFICATION
New Delhi, the 26th May, 1998
S.O. 457 (E).—In pursuance of clause (1) of Article 239 of the Constitution, the President hereby directs that the Administrator (whether known as Administrator or Lieutenant Governor) of a Union territory shall, subject to the control of the President and until further orders, exercise the powers and discharge the functions of a State Government under the provisions of the Lotteries (Regulation) Ordinance, 1998 (Ord. 6 of 1998), within that Union Territory."""

doc1 = {
    "document_id": str(uuid.uuid4()),
    "document_type": "Notification",
    "document_title": "Lotteries (Regulation) Ordinance",
    "issuing_authority": "MINISTRY OF HOME AFFAIRS",
    "issue_date": "26th May, 1998",
    "document_number": "S.O. 457 (E)",
    "text": doc1_text,
    "entities": [
        {"label": "AUTHORITY", "start": doc1_text.find("MINISTRY OF HOME AFFAIRS"), "end": doc1_text.find("MINISTRY OF HOME AFFAIRS") + len("MINISTRY OF HOME AFFAIRS")},
        {"label": "DATE", "start": doc1_text.find("26th May, 1998"), "end": doc1_text.find("26th May, 1998") + len("26th May, 1998")},
        {"label": "DOCUMENT", "start": doc1_text.find("S.O. 457 (E)"), "end": doc1_text.find("S.O. 457 (E)") + len("S.O. 457 (E)")},
        {"label": "TITLE", "start": doc1_text.find("Lotteries (Regulation) Ordinance"), "end": doc1_text.find("Lotteries (Regulation) Ordinance") + len("Lotteries (Regulation) Ordinance")},
    ],
    "source": "CustomHackathonInjection"
}

# Document 2 (Ministry of Health)
doc2_text = """MINISTRY OF HEALTH AND FAMILY WELFARE
(Department of Health and Family Welfare)
NOTIFICATION
New Delhi, the 19th August, 2026
G.S.R. 745(E).— The following draft of certain rules further to amend the Drugs Rules, 1945, which the Central Government proposes to make, in exercise of the powers conferred by sub-section (1) of section 12 and sub-section (1) of section 33 of the Drugs and Cosmetics Act, 1940 (23 of 1940), after consultation with the Drugs Technical Advisory Board is hereby published for information of all persons likely to be affected thereby, and notice is hereby given that the said draft rules shall be taken into consideration on or after the expiry of a period of thirty days from the date on which the copies of the Gazette of India containing these draft rules are made available to the public.

Objections and suggestions which may be received from any person within the period specified above will be considered by the Central Government.

Objections and suggestions, if any, may be addressed to the Under Secretary (Drugs), Ministry of Health and Family Welfare, Government of India, U-6, Work Hall- C Wing, First Floor, Kartavya Bhawan-1, New Delhi, 110001 or emailed at drugsdiv-mohfw@gov.in.

DRAFT RULES
1. (i) These rules may be called the Drugs (...... Amendment) Rules, 2026."""

doc2 = {
    "document_id": str(uuid.uuid4()),
    "document_type": "Draft Rule",
    "document_title": "draft of certain rules further to amend the Drugs Rules, 1945",
    "issuing_authority": "MINISTRY OF HEALTH AND FAMILY WELFARE",
    "issue_date": "19th August, 2026",
    "document_number": "G.S.R. 745(E)",
    "audience": "all persons likely to be affected thereby",
    "contact": "drugsdiv-mohfw@gov.in",
    "text": doc2_text,
    "entities": [
        {"label": "AUTHORITY", "start": doc2_text.find("MINISTRY OF HEALTH AND FAMILY WELFARE"), "end": doc2_text.find("MINISTRY OF HEALTH AND FAMILY WELFARE") + len("MINISTRY OF HEALTH AND FAMILY WELFARE")},
        {"label": "DATE", "start": doc2_text.find("19th August, 2026"), "end": doc2_text.find("19th August, 2026") + len("19th August, 2026")},
        {"label": "DOCUMENT", "start": doc2_text.find("G.S.R. 745(E)"), "end": doc2_text.find("G.S.R. 745(E)") + len("G.S.R. 745(E)")},
        {"label": "TITLE", "start": doc2_text.find("draft of certain rules further to amend the Drugs Rules, 1945"), "end": doc2_text.find("draft of certain rules further to amend the Drugs Rules, 1945") + len("draft of certain rules further to amend the Drugs Rules, 1945")},
        {"label": "AUDIENCE", "start": doc2_text.find("all persons likely to be affected thereby"), "end": doc2_text.find("all persons likely to be affected thereby") + len("all persons likely to be affected thereby")},
        {"label": "CONTACT", "start": doc2_text.find("drugsdiv-mohfw@gov.in"), "end": doc2_text.find("drugsdiv-mohfw@gov.in") + len("drugsdiv-mohfw@gov.in")},
    ],
    "source": "CustomHackathonInjection"
}

corpus_path = "trustextract-n/data/annotations/annotation_corpus.jsonl"
with open(corpus_path, "a") as f:
    f.write(json.dumps(doc1) + "\n")
    f.write(json.dumps(doc2) + "\n")
    
print("Successfully injected 2 documents into annotation_corpus.jsonl")
