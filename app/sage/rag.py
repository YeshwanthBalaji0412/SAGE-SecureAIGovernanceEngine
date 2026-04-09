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
    m = _POL_ID_RE.search(text)
    if m:
        return m.group(1)
    stem = Path(filename).stem.upper()[:20].replace(" ", "-")
    return stem or "POL-UNKNOWN"


# Matches policy IDs with any number of segments: POL-XX-2025, POL-RW-NOVA-2025, etc.
_POL_ID_RE = re.compile(r'\b(POL-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{4})\b')


def chunk_text(text: str, policy_id: str, policy_name: str) -> List[Dict]:
    """
    Split a policy document into section-level chunks.
    Tracks policy ID changes within the document so multi-policy files
    (several POL-XX-XXXX identifiers in one file) assign the correct ID
    to each chunk. Falls back to paragraph-level chunking for unstructured docs.
    """
    chunks = []

    # Strategy 1: split on known header patterns (tried in order)
    # Covers: "Section 3.1 —", "Article 2 —", "3.1 Title", "I. Title", "HEADING\n"
    header_patterns = [
        r'(?=(?:Section|SECTION|Article|ARTICLE)\s+\d[\d.]*\s*[:\-—])',   # Section 3.1 —
        r'(?=\n\s*\d+\.\d+\s+[A-Z])',                                      # 3.1 Title
        r'(?=\n\s*(?:I{1,3}|IV|VI{0,3}|IX|X{0,3})\.\s+\w)',               # Roman: II. Title
        r'(?=\n[A-Z][A-Z\s]{4,}\n)',                                        # ALL CAPS HEADER
    ]
    section_splits = None
    for pat in header_patterns:
        splits = re.split(pat, text)
        if len(splits) > 2:
            section_splits = splits
            break

    if section_splits and len(section_splits) > 2:
        current_pid = policy_id          # tracks the active policy ID
        for idx, sec in enumerate(section_splits):
            sec = sec.strip()
            if len(sec) < 40:
                continue
            # If a new POL-XX-XXXX appears at the start of this chunk, switch to it
            pid_m = _POL_ID_RE.search(sec[:300])
            if pid_m:
                current_pid = pid_m.group(1)
            m = re.match(r'(?:Section|SECTION|Article|ARTICLE)\s+(\d[\d.]*)', sec)
            section_num = m.group(1) if m else str(idx)
            chunks.append({
                "text":        sec,
                "policy_id":   current_pid,
                "policy_name": policy_name,
                "section":     section_num,
                "chunk_id":    f"{current_pid}-S{section_num}",
            })
        return chunks

    # Strategy 2: paragraph-level chunking — use smaller windows (150 words)
    # for slide-deck / short-bullet documents so each chunk stays focused
    words = text.split()
    # Smaller window for slide/bullet docs (avg slide ≈ 50 words → 150 = ~3 slides per chunk)
    WIN, OVERLAP = 150, 30
    current_pid = policy_id
    for i, start in enumerate(range(0, len(words), WIN - OVERLAP)):
        segment = " ".join(words[start: start + WIN])
        if len(segment) < 40:
            continue
        pid_m = _POL_ID_RE.search(segment[:200])
        if pid_m:
            current_pid = pid_m.group(1)
        chunks.append({
            "text":        segment,
            "policy_id":   current_pid,
            "policy_name": policy_name,
            "section":     str(i),
            "chunk_id":    f"{current_pid}-P{i}",
        })

    return chunks


# ── Section index (for CitationVerifier) ─────────────────────────────────────

def build_section_lookup(text: str, policy_id: str) -> Dict[str, str]:
    """
    Build a {policy_id§X.X → line_text} index from raw policy text.
    Tracks policy ID changes inline so multi-policy documents map each
    section number to the correct policy ID.
    Used by CitationVerifier to check groundedness of cited sections.
    """
    lookup: Dict[str, str] = {}
    current_pid = policy_id
    for line in text.split("\n"):
        # Update current policy ID if a new one appears on this line
        pid_m = _POL_ID_RE.search(line)
        if pid_m:
            current_pid = pid_m.group(1)
        stripped = line.strip()
        # Match "Section X.X" or "X.X" at line start
        m = re.match(
            r'^(?:Section|SECTION|Article|ARTICLE)\s+(\d+\.\d+)|^\s*(\d+\.\d+)',
            stripped
        )
        if m:
            sec_num = m.group(1) or m.group(2)
            key = f"{current_pid}§{sec_num}"
            lookup[key] = stripped
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


# ── Additional built-in demo corpora ─────────────────────────────────────────

EDUTRACK_POLICIES_TEXT = {
    "POL-AI-2025": (
        "Academic Integrity Policy",
        """ACADEMIC INTEGRITY POLICY (POL-AI-2025)
Effective Date: August 1, 2025 — EduTrack Academy

Section 1: Purpose
Uphold academic honesty and authentic learning across all programs at EduTrack Academy.

Section 2: Scope
Applies to all enrolled students in credit-bearing courses.
Does NOT apply to staff or faculty conducting independent research (see Research Integrity Policy).
Part-time and online students are equally covered.

Section 3: Definitions
3.1 Plagiarism: Submitting another person's words, ideas, or work as one's own without proper attribution.
3.2 Fabrication: Inventing or falsifying data, citations, or research results.
3.3 Collusion: Unauthorized collaboration that misrepresents individual work.
3.4 AI-Generated Content: Text, code, or analysis produced by large language models (e.g., ChatGPT)
    constitutes plagiarism UNLESS the instructor has explicitly authorized AI tool use in writing.
3.5 Contract Cheating: Submitting work purchased from or written by a third-party service.

Section 4: Prohibited Conduct
4.1 Using unauthorized materials (notes, devices, websites) during exams.
4.2 Sharing exam questions or answers before or after an assessment window.
4.3 Altering a graded submission and requesting re-evaluation without disclosure.
4.4 Accessing another student's account or submitting work using another identity.

Section 5: Consequences
5.1 First offense: Zero on the assignment plus a written Academic Warning on the student record.
5.2 Second offense: Failing grade (F) for the entire course.
5.3 Third offense: Academic suspension for one full semester; notation on transcript.
5.4 Egregious cases (e.g., contract cheating, exam impersonation): Expulsion; permanent transcript notation.
5.5 Consequences are cumulative across all courses and semesters at the Academy.

Section 6: Reporting and Appeals
6.1 Faculty must report suspected violations to the Dean of Academic Affairs within 5 business days.
6.2 Students may appeal a finding to the Academic Integrity Board within 14 calendar days of notice.
6.3 The Board must issue a decision within 30 calendar days of receiving the appeal.
6.4 Board decisions are final and binding.
""",
    ),
    "POL-SP-2025": (
        "Student Privacy Policy",
        """STUDENT PRIVACY POLICY (POL-SP-2025)
Effective Date: August 1, 2025 — EduTrack Academy

Section 1: Purpose
Protect the privacy of student education records in compliance with FERPA principles.

Section 2: Scope
Applies to all education records of enrolled and formerly enrolled students.
2.1 Parental rights: Parents retain access rights until the student turns 18 OR enrolls in
    a postsecondary program, whichever comes first. After that threshold, only the student
    controls record access.
2.2 Contractors and EdTech vendors accessing student data require a signed Data Processing Agreement.

Section 3: Education Records
3.1 Education records include: grades, transcripts, enrollment status, financial aid, disciplinary files.
3.2 Records do NOT include: faculty personal notes not shared with others, law enforcement records,
    employment records of students employed by the Academy.

Section 4: Directory Information
4.1 The following may be disclosed without consent: student name, enrollment status, degree program,
    dates of attendance, honors and awards.
4.2 Students may opt out of directory disclosure by submitting a written request to the Registrar
    by the end of the first week of each semester. Opt-out applies only to the current academic year.

Section 5: Non-Directory Information
5.1 Grades, GPA, disciplinary records, health and disability records, Social Security numbers,
    and financial records are non-directory and require WRITTEN CONSENT before disclosure.
5.2 Emergency disclosure without consent is permitted only if there is an imminent threat to
    health or safety, and must be documented within 24 hours.

Section 6: Third-Party Access
6.1 Vendors and researchers may access de-identified data only with IRB approval.
6.2 Any data breach involving student records must be reported to the Privacy Officer within 24 hours
    and to affected students within 30 days.

Section 7: Student Rights
7.1 Students may inspect their education records within 45 days of a written request.
7.2 Students may request amendment of records they believe are inaccurate.
7.3 Students may file complaints with the Academy's Privacy Office or federal regulators.
""",
    ),
    "POL-IU-2025": (
        "IT Acceptable Use Policy",
        """IT ACCEPTABLE USE POLICY (POL-IU-2025)
Effective Date: August 1, 2025 — EduTrack Academy

Section 1: Purpose
Define authorized use of Academy technology resources and networks.

Section 2: Scope
Applies to all students, faculty, staff, and guests using Academy networks, devices, or systems.
Personal devices connected to campus Wi-Fi are also subject to this policy.

Section 3: Authorized Use
3.1 Academic, administrative, and limited personal use are permitted.
3.2 Personal use must not interfere with academic work, consume excessive bandwidth, or violate law.

Section 4: Prohibited Activities
4.1 Illegal activities of any kind (piracy, hacking, fraud).
4.2 Cryptocurrency mining on Academy infrastructure.
4.3 Sharing login credentials; accounts are individual and non-transferable.
4.4 Installing unauthorized software on Academy-owned devices.
4.5 Accessing, storing, or distributing sexually explicit material.
4.6 Using Academy email to send bulk unsolicited communications.
4.7 Attempting to bypass network security controls or monitoring systems.

Section 5: Privacy and Monitoring
5.1 Users have NO expectation of privacy on Academy-owned systems or networks.
5.2 Network traffic, system logs, and email on Academy servers may be monitored without notice.
5.3 Personal devices on campus Wi-Fi are subject to traffic monitoring for security purposes.

Section 6: BYOD
6.1 Personal devices must register with IT before accessing internal Academy systems.
6.2 Registered BYOD devices must have up-to-date antivirus and OS patches.
6.3 BYOD devices may NOT be granted administrative access to Academy servers.

Section 7: Enforcement
7.1 First violation: written warning and mandatory IT security training.
7.2 Repeated or serious violations: suspension of access privileges, academic or HR referral.
7.3 Illegal activity is referred to law enforcement; Academy cooperation is mandatory.
""",
    ),
}

MEDCORE_POLICIES_TEXT = {
    "POL-PHI-2025": (
        "Patient Health Information Policy",
        """PATIENT HEALTH INFORMATION POLICY (POL-PHI-2025)
Effective Date: January 1, 2025 — MedCore Health

Section 1: Purpose
Ensure lawful, ethical, and secure handling of Protected Health Information (PHI) for all patients.

Section 2: Scope
2.1 Applies to ALL workforce members: employees, contractors, volunteers, students on placement.
2.2 Business associates handling PHI on behalf of MedCore must sign a Business Associate Agreement (BAA).
2.3 PHI includes any individually identifiable health information in any medium.

Section 3: Minimum Necessary Standard
3.1 Access only the PHI required to perform your specific job function.
3.2 Requesting or viewing PHI beyond your role (including records of family members or celebrities)
    is a violation even if technically accessible.
3.3 Verbal discussions of PHI must occur in private areas; do not discuss patient details in hallways,
    elevators, or public spaces.

Section 4: Access Controls
4.1 Access is role-based; your supervisor grants access; IT provisions credentials.
4.2 Shared logins are STRICTLY PROHIBITED. Each workforce member has a unique identifier.
4.3 Workstations must auto-lock after 15 minutes of inactivity. Manual lock required when leaving.
4.4 PHI must never be transmitted via personal email, SMS, or unapproved messaging apps.

Section 5: Disclosure Rules
5.1 PHI may be disclosed without authorization for Treatment, Payment, and Operations (TPO).
5.2 All other disclosures require the patient's signed written authorization.
5.3 Patients have the right to request restrictions on disclosures; honor restrictions where feasible.
5.4 A disclosure log must be maintained for all non-TPO disclosures; available to patients on request.

Section 6: Mobile and Remote Access
6.1 PHI on mobile devices requires AES-256 encryption and MDM enrollment.
6.2 USB drives and personal cloud storage (Google Drive, Dropbox) are PROHIBITED for PHI.
6.3 Remote access to clinical systems requires VPN and MFA at all times.

Section 7: Breach Response
7.1 Suspected PHI breach must be reported to the Privacy Officer within 1 hour of discovery.
7.2 The Privacy Officer will conduct a risk assessment within 24 hours.
7.3 Confirmed breaches affecting patients must be notified to those patients within 60 days.
7.4 Workforce member breaches may result in immediate suspension pending investigation.
""",
    ),
    "POL-WS-2025": (
        "Workplace Safety Policy",
        """WORKPLACE SAFETY POLICY (POL-WS-2025)
Effective Date: January 1, 2025 — MedCore Health

Section 1: Purpose
Protect patients, visitors, and workforce from preventable harm in MedCore facilities.

Section 2: Scope
Applies to all staff, contractors, volunteers, and students present on any MedCore premises.

Section 3: Personal Protective Equipment (PPE)
3.1 Gloves and surgical masks are MANDATORY in all patient-facing areas at all times.
3.2 N95 respirators are required in airborne-precaution rooms (TB, COVID-19, measles protocols).
3.3 Eye protection required when splatter risk exists. Gowns required for contact precaution rooms.
3.4 PPE must be donned before entering and doffed safely before leaving the designated zone.
3.5 Reusing single-use PPE is prohibited.

Section 4: Incident Reporting
4.1 All workplace injuries, needle-stick incidents, and near-misses must be reported to your
    supervisor AND the Safety Officer within 4 hours.
4.2 Post-exposure prophylaxis (PEP) must be initiated within 2 hours of blood-borne exposure.
4.3 Falsifying or omitting incident reports is grounds for immediate termination.

Section 5: Hazardous Materials
5.1 Safety Data Sheets (SDS) must be current and accessible in every department.
5.2 Spill kits must be available and stocked; staff must complete spill response training annually.
5.3 Cytotoxic drugs require additional closed-system transfer device (CSTD) handling protocols.

Section 6: Emergency Procedures
6.1 Evacuation routes must be posted in every room and reviewed at orientation.
6.2 Fire drills and code simulations are conducted quarterly; attendance is mandatory.
6.3 For patient evacuations, use RACE protocol: Rescue, Alarm, Contain/Close, Extinguish/Evacuate.

Section 7: Workplace Violence
7.1 Zero tolerance for violence or threats against any person on MedCore property.
7.2 Immediately activate the security code for your facility upon threat observation.
7.3 Do NOT physically intervene; ensure your own safety first.
7.4 All incidents must be documented and reported to HR and Security within 24 hours.
""",
    ),
    "POL-SC-2025": (
        "Staff Conduct Policy",
        """STAFF CONDUCT POLICY (POL-SC-2025)
Effective Date: January 1, 2025 — MedCore Health

Section 1: Purpose
Maintain a professional, ethical, and respectful environment for patients and colleagues.

Section 2: Scope
Applies to all employed staff. Contractors are subject to conduct clauses in their agreements.

Section 3: Professional Standards
3.1 Address patients by their preferred name/pronoun; confirm at each encounter.
3.2 Maintain patient dignity during all examinations, procedures, and conversations.
3.3 Follow the chain of command; escalate clinical concerns through proper channels, not social media.
3.4 Punctuality is required; notify your supervisor at least 1 hour before shift start if unable to attend.

Section 4: Social Media
4.1 No photographs of patients, patient belongings, or areas where patients are visible.
4.2 Do not identify or describe a patient's condition, treatment, or visit on any public platform.
4.3 Personal views must not be presented as MedCore's position; include a disclaimer if posting
    health-related content as a MedCore employee.
4.4 Violations may result in immediate suspension pending review.

Section 5: Conflicts of Interest
5.1 Disclose gifts or hospitality valued over $25 from vendors, pharmaceutical companies, or patients.
5.2 Do not refer patients to businesses in which you or immediate family hold a financial interest.
5.3 Secondary employment in competing healthcare facilities requires prior written approval from HR.

Section 6: Substance Use
6.1 Zero tolerance for working under the influence of alcohol or any non-prescribed substance.
6.2 Random and post-incident drug/alcohol testing is required for safety-sensitive roles.
6.3 Self-referral to the Employee Assistance Program (EAP) will be treated confidentially.
""",
    ),
}

LAUNCHPAD_POLICIES_TEXT = {
    "POL-RF-2025": (
        "Remote-First Work Policy",
        """REMOTE-FIRST WORK POLICY (POL-RF-2025)
Effective Date: February 1, 2025 — LaunchPad Startup

Section 1: Purpose
LaunchPad is a remote-first company. All roles are remote by default unless explicitly designated
"on-site required" in the job description.

Section 2: Scope
2.1 Applies to all full-time and part-time employees.
2.2 Contractors are covered by separate provisions in their Statement of Work (SOW); this policy
    does NOT apply to contractors unless explicitly stated in the SOW.
2.3 New employees must complete a 30-day remote onboarding program before working fully async.

Section 3: Equipment
3.1 LaunchPad provides a company-issued laptop (MacBook Pro) and peripherals to all full-time employees.
3.2 Full-time employees receive a one-time $1,500 home-office setup stipend, reimbursed within 30 days.
3.3 Part-time employees (under 30 hrs/week) receive a laptop only; no home-office stipend.
3.4 Contractors must provide their own equipment.

Section 4: Working Hours
4.1 Core overlap window: 10 AM – 2 PM in the employee's LOCAL timezone, Monday–Friday.
4.2 Employees outside the core overlap window may request an adjusted schedule with manager approval.
4.3 Results-oriented culture: hours worked matter less than deliverables met.

Section 5: International Work
5.1 Employees may work from any country for up to 90 consecutive days without additional approval,
    provided they notify HR 2 weeks in advance.
5.2 Stays exceeding 90 consecutive days require a legal and tax review by HR; additional approvals
    may include CFO and Legal. Permanent relocation requires separate negotiation.
5.3 Employees are responsible for understanding their personal tax obligations abroad.
5.4 LaunchPad will not cover tax liabilities arising from unauthorized extended international stays.

Section 6: Co-working Spaces
6.1 Co-working space costs are reimbursed up to $300/month with receipts and manager approval.
6.2 All work conducted at co-working spaces must comply with POL-DS-2025 (Data Security Policy).

Section 7: Company Gatherings
7.1 Quarterly all-hands (in person): attendance strongly encouraged; travel and accommodation fully covered.
7.2 Team offsites: travel costs covered; minimum 4 weeks notice required for scheduling.
""",
    ),
    "POL-IP-2025": (
        "Intellectual Property Policy",
        """INTELLECTUAL PROPERTY POLICY (POL-IP-2025)
Effective Date: February 1, 2025 — LaunchPad Startup

Section 1: Purpose
Define ownership of intellectual property created during the employment relationship.

Section 2: Scope
2.1 Applies to all full-time and part-time employees from the first day of employment.
2.2 Contractors have separate IP provisions in their SOW; if silent, this policy applies.

Section 3: Company-Owned IP
3.1 All inventions, software, designs, data, processes, and works of authorship created:
    (a) during working hours, OR
    (b) using company equipment or resources, OR
    (c) related to LaunchPad's current or anticipated business
    are assigned to LaunchPad in perpetuity, worldwide, across all media.
3.2 This includes work created evenings, weekends, or personal time if conditions in §3.1 apply.
3.3 Employees must promptly disclose any potentially assignable invention to the CTO in writing.

Section 4: Personal Exclusion
4.1 Work that is (a) entirely unrelated to LaunchPad's business, (b) created on personal time,
    and (c) created using exclusively personal equipment is NOT assigned.
4.2 Employees claiming a personal exclusion must document the creation (date, tools used,
    description) within 30 days of creation. Late documentation may invalidate the exclusion claim.

Section 5: Open Source Contributions
5.1 Contributing to external open-source projects during working hours requires prior written approval
    from the CTO.
5.2 Contributions using company IP or referencing company systems require Legal review.

Section 6: Post-Employment
6.1 IP assignment survives termination of employment.
6.2 For one year post-termination, IP derived from or closely related to company work remains
    assigned to LaunchPad unless a written release is obtained.
""",
    ),
    "POL-CC-2025": (
        "Code of Conduct",
        """CODE OF CONDUCT (POL-CC-2025)
Effective Date: February 1, 2025 — LaunchPad Startup

Section 1: Purpose
Build an inclusive, respectful, and high-integrity workplace where everyone can do their best work.

Section 2: Scope
Applies to all employees, contractors, interns, and vendors when:
(a) on company premises or at company events, or
(b) using company-provided communication channels (Slack, email, Zoom), or
(c) representing LaunchPad in any public or professional capacity.

Section 3: Expected Behavior
3.1 Communicate respectfully; assume good intent; ask for clarification before escalating.
3.2 Give proper credit for others' ideas and contributions.
3.3 Report concerns early; silence enables harm.
3.4 Protect confidential information; do not discuss fundraising, cap table, or unannounced features.

Section 4: Prohibited Behavior
4.1 Harassment based on race, gender, age, disability, sexual orientation, religion, or any other
    protected characteristic, whether overt or subtle.
4.2 Retaliation against anyone who reports in good faith; retaliation is treated as seriously as
    the original violation.
4.3 Dishonesty including misrepresenting work, falsifying expense reports, or deceiving customers.
4.4 Creating a hostile work environment through intimidation, bullying, or exclusion.

Section 5: Reporting
5.1 Report concerns to HR (hr@launchpad.io) or anonymously via the EthicsPoint hotline.
5.2 All reports are investigated within 30 days. Reporter identity is protected to the extent possible.
5.3 Do NOT investigate on your own; self-investigation may compromise official proceedings.

Section 6: Consequences
6.1 Minor violations: coaching and documented improvement plan.
6.2 Serious violations: Performance Improvement Plan, suspension, or termination.
6.3 Egregious violations (harassment, fraud, violence): immediate termination without severance.
""",
    ),
}

RETAILFLOW_POLICIES_TEXT = {
    "POL-CD-2025": (
        "Customer Data Policy",
        """CUSTOMER DATA POLICY (POL-CD-2025)
Effective Date: March 1, 2025 — RetailFlow Corp

Section 1: Purpose
Govern the lawful and secure collection, use, and storage of customer personal data.

Section 2: Scope
Applies to all associates and managers who collect, process, or access customer data.
Third-party marketing and analytics partners must sign a Data Processing Agreement.

Section 3: Data Collection
3.1 Collect only the minimum data necessary for the stated transaction purpose.
3.2 Loyalty program enrollment: collect name, email, and purchase history only; SSN and date
    of birth are PROHIBITED unless required for age-restricted purchases (alcohol, tobacco).
3.3 Customers must provide affirmative opt-in consent before marketing emails are sent.
3.4 Data collected at POS for fraud prevention must be disclosed in the store privacy notice.

Section 4: Payment Card Data (PCI-DSS Compliance)
4.1 Full card numbers must NEVER be written down, photographed, emailed, or stored in any system
    not certified for PCI-DSS compliance.
4.2 Use only company-approved point-to-point encrypted (P2PE) POS terminals.
4.3 Inspect POS terminals for skimming devices at the start of every shift; report anomalies immediately.
4.4 Associates may NEVER process their own transactions or transactions for immediate family members.
4.5 Credentials for POS systems must NOT be shared; each associate has a unique login.

Section 5: Data Retention
5.1 Customer transaction records: retained 7 years for financial/legal compliance.
5.2 Loyalty program data: retained while account is active plus 2 years after last purchase.
5.3 Customer complaints and return records: retained 3 years.
5.4 Data must be securely deleted (not just deleted) at end of retention period using approved tools.

Section 6: Breach Response
6.1 Suspected customer data breach (loss of device, unauthorized access, skimming device found)
    must be reported to the Store Manager and IT Security within 30 minutes of discovery.
6.2 The Store Manager must escalate to the Regional Loss Prevention team within 1 hour.
6.3 Associates must NOT attempt to investigate or contain a breach independently.
""",
    ),
    "POL-EH-2025": (
        "Employee Handbook",
        """EMPLOYEE HANDBOOK (POL-EH-2025)
Effective Date: March 1, 2025 — RetailFlow Corp

Section 1: Purpose
Establish clear standards for employment, scheduling, and conduct at all RetailFlow locations.

Section 2: Scope
2.1 Applies to all hourly and salaried employees.
2.2 Seasonal and temporary employees are covered by a separate Seasonal Agreement, which
    supersedes this handbook where they conflict.
2.3 This handbook does not constitute a contract of employment.

Section 3: Scheduling
3.1 Schedules are posted 2 weeks in advance via the RetailFlow scheduling app.
3.2 Shift swap requests must be submitted at least 72 hours before the affected shift.
3.3 Employees are responsible for finding a qualified replacement for approved swaps;
    the Store Manager must approve all swaps before they take effect.

Section 4: Attendance
4.1 Three unexcused absences within any 90-day rolling period = written warning.
4.2 Six unexcused absences within 90 days = termination.
4.3 No-call no-show (absent without contact before shift start) = automatic written warning.
4.4 Two no-call no-shows within 90 days = termination.
4.5 Medical or family emergency absences must be documented within 3 business days.

Section 5: Dress Code
5.1 Company uniform (shirt and name badge) is mandatory and must be clean and pressed.
5.2 Name badge must be visible at chest level at all times while on the sales floor.
5.3 Visible tattoos above the collar and facial piercings (other than small ear studs) are not permitted.
5.4 Closed-toe, non-slip shoes required in all areas; fashion footwear not permitted.

Section 6: Break Policy
6.1 Shifts of 4–6 hours: one 15-minute paid break.
6.2 Shifts over 6 hours: one 30-minute unpaid meal break plus one 15-minute paid break.
6.3 Breaks must be taken; waiving a break is not permitted and will not result in additional pay.
6.4 Meal breaks must be taken no later than the 5th hour of a shift.

Section 7: Social Media
7.1 Do not photograph or post content that reveals store layout, security camera placement,
    inventory counts, or access codes.
7.2 Personal opinions expressed online must not be presented as RetailFlow's position.
7.3 Do not post about ongoing loss-prevention investigations or personnel matters.
""",
    ),
    "POL-SS-2025": (
        "Store Safety Policy",
        """STORE SAFETY POLICY (POL-SS-2025)
Effective Date: March 1, 2025 — RetailFlow Corp

Section 1: Purpose
Protect customers, associates, and RetailFlow property from preventable safety incidents.

Section 2: Scope
Applies to all associates, managers, contractors, and delivery personnel on RetailFlow premises.

Section 3: Hazard Prevention
3.1 Spills must be cleaned up or clearly marked with wet-floor cones within 5 minutes of discovery.
3.2 Aisle obstructions must be cleared before opening and within 15 minutes of occurrence during hours.
3.3 Ladders and step-stools must be inspected before each use; damaged equipment must be tagged
    and removed from service immediately.
3.4 Lifting objects over 30 lbs requires proper technique; team lift required for objects over 50 lbs.
3.5 Box cutters must be retracted when not in active use; associates under 18 may not use box cutters.

Section 4: Incident Reporting
4.1 All customer injuries on premises must be documented in an Incident Report within 1 hour,
    regardless of whether the customer accepts or declines medical assistance.
4.2 All associate injuries, however minor, must be reported to the Store Manager before end of shift.
4.3 Failing to report an incident is a serious disciplinary offense.
4.4 Do not make admissions of liability to customers; refer all comments to the Store Manager.

Section 5: Emergency Procedures
5.1 Fire: Activate nearest pull station; call 911; initiate customer evacuation; meet at designated
    assembly point. Do NOT use elevators. Do NOT attempt to extinguish large fires.
5.2 Medical emergency: Call 911 immediately; retrieve AED if needed; do NOT move the person unless
    they are in immediate danger.
5.3 Active threat: Lock or barricade, Lights out, Leave if safe, Call 911 — follow ALICE protocol.

Section 6: Loss Prevention
6.1 Associates must greet all customers entering their department; this is both service and deterrence.
6.2 Never physically confront or detain a suspected shoplifter; notify Loss Prevention or call for help.
6.3 Discounts, voids, and refunds above $50 require manager approval and are logged in the system.
""",
    ),
}


# ── Demo organization registry ────────────────────────────────────────────────

DEMO_ORGANIZATIONS = {
    "technova": {
        "name":        "TechNova Inc.",
        "org_type":    "technology",
        "icon":        "💻",
        "tagline":     "Software company — remote work, data privacy, information security",
        "policies":    TECHNOVA_POLICIES_TEXT,
    },
    "edutrack": {
        "name":        "EduTrack Academy",
        "org_type":    "education",
        "icon":        "🎓",
        "tagline":     "University — academic integrity, student privacy, IT acceptable use",
        "policies":    EDUTRACK_POLICIES_TEXT,
    },
    "medcore": {
        "name":        "MedCore Health",
        "org_type":    "healthcare",
        "icon":        "🏥",
        "tagline":     "Healthcare org — patient data (HIPAA-style), workplace safety, staff conduct",
        "policies":    MEDCORE_POLICIES_TEXT,
    },
    "launchpad": {
        "name":        "LaunchPad Startup",
        "org_type":    "startup",
        "icon":        "🚀",
        "tagline":     "Remote-first startup — IP assignment, code of conduct, international work",
        "policies":    LAUNCHPAD_POLICIES_TEXT,
    },
    "retailflow": {
        "name":        "RetailFlow Corp",
        "org_type":    "retail",
        "icon":        "🛒",
        "tagline":     "Retail chain — PCI-DSS, customer data, employee handbook, store safety",
        "policies":    RETAILFLOW_POLICIES_TEXT,
    },
}


def load_demo(org_key: str, api_key: str) -> Tuple[List[Dict], Dict[str, str], object, List[dict]]:
    """Load a built-in demo policy corpus by organization key."""
    org = DEMO_ORGANIZATIONS[org_key]
    files = [
        (text.encode("utf-8"), f"{pid}.txt")
        for pid, (_, text) in org["policies"].items()
    ]
    return ingest_documents(files, api_key)


def load_technova_demo(api_key: str) -> Tuple[List[Dict], Dict[str, str], object, List[dict]]:
    """Load the built-in TechNova demo policy corpus (backward-compat wrapper)."""
    return load_demo("technova", api_key)
