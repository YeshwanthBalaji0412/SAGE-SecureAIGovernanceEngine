"""
Document ingestion and ChromaDB RAG pipeline.
Works with PDF, TXT, or plain-text strings.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.utils import embedding_functions

# Optional PDF support
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file (bytes)."""
    if not HAS_PDF:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Dispatch to PDF or plain-text extractor based on filename."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    return file_bytes.decode("utf-8", errors="replace")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _infer_policy_id(text: str, filename: str) -> str:
    """Guess a short policy ID from content or filename."""
    m = re.search(r'\b(POL-[A-Z]+-\d{4})\b', text)
    if m:
        return m.group(1)
    stem = Path(filename).stem.upper()[:20].replace(" ", "-")
    return stem or "POL-UNKNOWN"


def chunk_text(text: str, policy_id: str, policy_name: str) -> List[Dict]:
    """
    Split a policy document into section-level chunks.
    Falls back to paragraph-level chunking for non-structured documents.
    """
    chunks = []

    # Strategy 1: split on "Section N:" / "Section N.N:" headers
    section_splits = re.split(r'(?=(?:Section|SECTION|Article|ARTICLE)\s+\d[\d.]*\s*[:\-—])', text)

    if len(section_splits) > 2:
        for idx, sec in enumerate(section_splits):
            sec = sec.strip()
            if len(sec) < 40:
                continue
            m = re.match(r'(?:Section|SECTION|Article|ARTICLE)\s+(\d[\d.]*)', sec)
            section_num = m.group(1) if m else str(idx)
            chunks.append({
                "text":        sec,
                "policy_id":   policy_id,
                "policy_name": policy_name,
                "section":     section_num,
                "chunk_id":    f"{policy_id}-S{section_num}",
            })
        return chunks

    # Strategy 2: paragraph-level chunking (300-word windows, 50-word overlap)
    words = text.split()
    WIN, OVERLAP = 300, 50
    for i, start in enumerate(range(0, len(words), WIN - OVERLAP)):
        segment = " ".join(words[start: start + WIN])
        if len(segment) < 40:
            continue
        chunks.append({
            "text":        segment,
            "policy_id":   policy_id,
            "policy_name": policy_name,
            "section":     str(i),
            "chunk_id":    f"{policy_id}-P{i}",
        })

    return chunks


# ── Section index (for CitationVerifier) ─────────────────────────────────────

def build_section_lookup(text: str, policy_id: str) -> Dict[str, str]:
    """
    Build a {policy_id§X.X → line_text} index from raw policy text.
    Used by CitationVerifier to check groundedness of cited sections.
    """
    lookup: Dict[str, str] = {}
    for line in text.split("\n"):
        m = re.match(r'^\s*(\d+\.\d+)', line.strip())
        if m:
            key = f"{policy_id}§{m.group(1)}"
            lookup[key] = line.strip()
    return lookup


# ── ChromaDB collection ───────────────────────────────────────────────────────

def build_chromadb_collection(
    chunks: List[Dict],
    api_key: str,
    collection_name: str = "sage_policies",
):
    """
    Create (or recreate) a ChromaDB in-memory collection from policy chunks.
    Returns the collection object or None on failure.
    """
    try:
        cc = chromadb.Client()
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
        # Delete existing collection if it exists
        try:
            cc.delete_collection(collection_name)
        except Exception:
            pass

        col = cc.create_collection(collection_name, embedding_function=ef)
        col.add(
            documents=[c["text"] for c in chunks],
            metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
            ids=[c["chunk_id"] for c in chunks],
        )
        return col
    except Exception as e:
        print(f"[ChromaDB] Failed to build collection: {e}")
        return None


# ── Document validation ───────────────────────────────────────────────────────

_POLICY_SIGNALS = [
    r'\bpolicy\b', r'\bpolicies\b', r'\bprocedure\b', r'\bguideline\b',
    r'\bcompliance\b', r'\bemployee(s)?\b', r'\bmust\b', r'\bshall\b',
    r'\bprohibited\b', r'\brequired\b', r'\bapproval\b', r'\bsection\s+\d',
    r'\beffective\s+date\b', r'\bscope\b', r'\bpurpose\b',
    r'\bviolation\b', r'\bmanagement\b', r'\bauthorization\b',
]

def validate_policy_document(text: str) -> dict:
    """
    Check if a document looks like a policy document.
    Returns {"is_policy": bool, "score": int, "signals_found": list}.
    """
    tl = text.lower()
    found = [p for p in _POLICY_SIGNALS if re.search(p, tl)]
    score = len(found)
    return {
        "is_policy":     score >= 4,
        "score":         score,
        "signals_found": found,
        "total_signals": len(_POLICY_SIGNALS),
    }


# ── High-level ingestion ──────────────────────────────────────────────────────

def ingest_documents(
    files: List[Tuple[bytes, str]],   # [(file_bytes, filename), ...]
    api_key: str,
) -> Tuple[List[Dict], Dict[str, str], object, List[dict]]:
    """
    Ingest a list of (file_bytes, filename) tuples.
    Returns (chunks, section_lookup, chromadb_collection, validation_results).
    """
    all_chunks: List[Dict] = []
    section_lookup: Dict[str, str] = {}
    validations: List[dict] = []

    for file_bytes, filename in files:
        text        = extract_text_from_file(file_bytes, filename)
        validation  = validate_policy_document(text)
        validation["filename"] = filename
        validations.append(validation)

        policy_id   = _infer_policy_id(text, filename)
        policy_name = Path(filename).stem.replace("-", " ").replace("_", " ").title()

        doc_chunks = chunk_text(text, policy_id, policy_name)
        all_chunks.extend(doc_chunks)
        section_lookup.update(build_section_lookup(text, policy_id))

    collection = build_chromadb_collection(all_chunks, api_key)
    return all_chunks, section_lookup, collection, validations


# ── Built-in TechNova demo corpus ─────────────────────────────────────────────

TECHNOVA_POLICIES_TEXT = {
    "POL-RW-2025": (
        "Remote Work Policy",
        """REMOTE WORK POLICY (POL-RW-2025)
Effective Date: January 1, 2025

Section 1: Purpose
Establishes guidelines for remote work arrangements at TechNova Inc.

Section 2: Eligibility
Full-time employees who completed their 90-day probationary period are eligible.
Contractors and temporary employees are NOT covered by this policy.

Section 3: Domestic Remote Work
3.1 Remote work from any US location requires prior written manager approval.
3.2 Core hours (10 AM–3 PM ET) must be maintained.
3.3 Working remotely more than 3 days/week requires a dedicated ergonomic workspace.
3.4 Temporary remote work (coffee shop, airport) is permitted without formal approval
    for short durations PROVIDED VPN is active.

Section 4: International Remote Work
4.1 Requires prior written manager approval.
4.2 Exceeding 30 consecutive days requires ADDITIONAL approval from HR AND Legal
    due to tax, employment law, and regulatory implications.
4.3 Must comply with POL-DP-2025 and POL-IS-2025 at all times.
4.4 Benefits (including health insurance) may NOT extend internationally.
    Employee must verify coverage before departure.
4.5 Extended arrangements (>30 days) are discouraged without documented business justification.

Section 5: Equipment and Reimbursement
5.1 Company provides standard equipment for employees approved 3+ days/week.
5.2 Home office reimbursement up to $500/year with manager approval and receipts.

Section 6: Termination of Remote Work
6.1 Arrangements may be revoked with 30 days written notice.
""",
    ),
    "POL-DP-2025": (
        "Data Privacy Policy",
        """DATA PRIVACY POLICY (POL-DP-2025)
Effective Date: January 1, 2025

Section 1: Purpose
Governs collection, processing, storage, and transfer of personal data.

Section 2: Scope
Applies to all employees, contractors, and third-party processors handling personal
data on behalf of TechNova Inc.

Section 3: Definitions
3.1 Personal Data: information relating to an identified or identifiable person.
3.2 Sensitive Personal Data: health, biometric, financial data, and similar.
3.3 EEA: European Economic Area, subject to GDPR-equivalent standards.

Section 4: Data Retention
4.1 Personal data collected only for specified, explicit, legitimate purposes.
4.2 Customer PII retained no longer than 7 years after end of relationship.
4.3 Employee data retained for the employee lifecycle plus 5 years post-termination.

Section 5: Cross-Border Data Transfer
5.1 Transfers outside the EEA require appropriate safeguards: SCCs or BCRs.
5.2 Non-EEA transfers must be documented and approved by the DPO.
5.3 Sensitive personal data transfers require EXPLICIT WRITTEN CONSENT from data
    subjects in addition to §5.1 safeguards.
5.4 DPO must be consulted for ALL new data flows involving customer PII.

Section 6: Data Breach Notification
6.1 Suspected breach must be reported to InfoSec within 24 hours.
6.2 Confirmed breach involving personal data: notify DPO and Legal within 72 hours.
6.3 Regulatory notification (where applicable) within 72 hours of confirmation.

Section 7: Data Subject Rights
7.1 Access requests fulfilled within 30 days.
7.2 Erasure requests fulfilled within 30 days unless legitimate grounds for retention.
""",
    ),
    "POL-IS-2025": (
        "Information Security Policy",
        """INFORMATION SECURITY POLICY (POL-IS-2025)
Effective Date: January 1, 2025

Section 1: Purpose
Information security requirements for all TechNova systems, data, and personnel.

Section 2: Access Controls
2.1 Principle of least privilege applies to all system access.
2.2 MFA is MANDATORY for all remote access to company systems.
2.3 Credentials must NOT be shared under any circumstances.
2.4 Inactive accounts disabled within 30 days.

Section 3: Software and Systems
3.1 Only company-approved software on company devices.
3.2 All software installations require IT Security approval.
3.3 Critical patches applied within 14 days of release.
3.4 Personal software on company devices is prohibited.

Section 4: Network Security
4.1 Company VPN required when accessing resources from external networks.
4.2 Public Wi-Fi without VPN is STRICTLY PROHIBITED.
4.3 Unauthorized proxies or VPNs are prohibited.

Section 5: Data Handling
5.1 All sensitive data encrypted in transit and at rest (AES-256 or equivalent).
5.2 Data classification labels must be applied to all documents.
5.3 Confidential/Restricted data MUST NOT be stored locally on personal devices or
    unmanaged storage. Cloud-only access required. Encryption does NOT exempt this rule.
5.4 Printing Restricted data requires explicit manager AND IT Security approval.

Section 6: Personal Devices (BYOD)
6.1 Personal devices used for work must be enrolled in MDM.
6.2 MDM enrollment requires IT Security approval and remote-wipe consent.
6.3 Personal devices must NOT store Confidential or Restricted data locally.

Section 7: Security Training
7.1 Mandatory security awareness training within 30 days of hire.
7.2 Annual refresher training required.
7.3 Role-specific training for employees handling sensitive data.

Section 8: Incident Response
8.1 Security incidents reported to InfoSec immediately.
8.2 Employees must preserve evidence and not self-remediate.
""",
    ),
}


def load_technova_demo(api_key: str) -> Tuple[List[Dict], Dict[str, str], object, List[dict]]:
    """Load the built-in TechNova demo policy corpus."""
    files = [
        (text.encode("utf-8"), f"{pid}.txt")
        for pid, (_, text) in TECHNOVA_POLICIES_TEXT.items()
    ]
    return ingest_documents(files, api_key)
