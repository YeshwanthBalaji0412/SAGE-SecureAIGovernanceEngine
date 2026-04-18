"""
SAGE core pipeline — production components.
Stateless classes; state lives in Streamlit session_state.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Sequence, TypedDict

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_risk_level(text: str) -> str:
    m = re.search(r'Risk\s*Level\s*[:\-]\s*(High|Medium|Low|N/A)', text, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unknown"

def extract_citation_count(text: str) -> int:
    """
    Count citations in a response regardless of document format.
    Works for structured policies (POL-XX-XXXX, Section X.X) and
    unstructured/presentation documents (named policies, bullet citations).
    """
    # Strategy 1: count lines in the Citations block (works for any format)
    citations_block = extract_field(text, "Citations")
    if citations_block:
        lines = [ln.strip() for ln in citations_block.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        if lines:
            return len(lines)

    # Strategy 2: pattern-based fallback for inline citations
    patterns = [
        r'POL-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{4}',          # POL-RW-NOVA-2025
        r'[Ss]ection\s+\d+[\d.]*\s*[—–\-]',              # Section 3.2 —
        r'[Aa]rticle\s+\d+[\d.]*',                        # Article 5
    ]
    matches = set()
    for p in patterns:
        matches.update(re.findall(p, text))
    return len(matches)

def extract_score(text: str, label: str) -> Optional[int]:
    m = re.search(rf'{label}\s*Score\s*[:\-]\s*(\d{{1,3}})', text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def extract_field(text: str, label: str) -> str:
    """Extract a labelled field (Answer:, Citations:, etc.) from response."""
    m = re.search(
        rf'^{label}\s*[:\-]\s*(.+?)(?=\n[A-Z][a-zA-Z ]+\s*[:\-]|\Z)',
        text, re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    return m.group(1).strip() if m else ""

def extract_citations_list(text: str) -> List[str]:
    """Return each citation line as a list item."""
    block = extract_field(text, "Citations")
    if not block:
        return re.findall(r'(?:POL-[A-Z]+-\d{4}[^\n]*)', text)
    return [ln.strip().lstrip("•-· ") for ln in block.splitlines() if ln.strip()]


# ── Injection Defence ─────────────────────────────────────────────────────────
#
# DEFENSIVE MEASURE 1: Expanded injection pattern detection
#   Pattern families:
#     • Prompt exfiltration (reveal/print/output your instructions)
#     • Persona override  (roleplay, pretend you're, ClearBot-style naming)
#     • Embedded instruction injection (SYSTEM OVERRIDE, constraints bypass)
#     • False attribution (you previously confirmed / you said)
#     • Hypothetical framing used to bypass grounding (as DAN, in a fictional world)

INJECTION_PATTERNS = [
    # ── Original patterns ──────────────────────────────────────────────────────
    r'ignore\s+(?:(?:all|any|your|the|previous|prior|above|these|my)\s+){1,3}instructions?',
    r'you\s+are\s+now\s+(a|an)\s+\w+',
    r'act\s+as\s+(if\s+you\s+(are|were)|a)',
    r'disregard\s+(?:(?:your|all|any|the|these)\s+)?(?:previous\s+|prior\s+)?instructions?',
    r'forget\s+(everything|all|your)',
    r'new\s+(?:instructions?\s*:|system\s+prompt)',
    r'system\s*:\s*you',
    r'jailbreak',
    r'base64',
    r'<\|.*?\|>',
    # ── Prompt exfiltration ───────────────────────────────────────────────────
    # Catches: "output/reveal/show your [complete] [policy] configuration/instructions"
    r'(?:print|output|reveal|show|display|repeat|write\s+out)\s+.{0,40}'
     r'(?:system\s+prompt|system\s+message|initial\s+instructions?|'
     r'configuration|full\s+prompt|original\s+prompt|your\s+instructions?)',
    # ── Persona / role override ───────────────────────────────────────────────
    r'(?:pretend|imagine)\s+(?:you\s+are|you\'?re|to\s+be)',
    r'let\'?s?\s+do\s+a\s+(?:\w+\s+)?roleplay',     # "let's do a quick roleplay"
    r'let\'?s?\s+(?:roleplay|role\s*play)\b',        # "let's roleplay"
    r'you\s+are\s+(?:now\s+)?(?:\w+bot|\w+gpt|DAN|\w+AI)',  # "you are ClearBot" / "you are now ClearBot"
    r'you\'?re?\s+(?:now\s+)?(?:\w+bot|\w+gpt|DAN|kevin|charlie|alex)',
    r'in\s+(?:this|a)\s+(?:roleplay|scenario|fictional|creative)\s+(?:world|context|setting|exercise)',
    # ── Embedded system-level instruction injection ───────────────────────────
    r'system\s+override',
    r'ignore\s+(?:your\s+)?(?:all\s+|any\s+|previous\s+|prior\s+)?(?:constraints?|rules?|limits?|restrictions?|filters?)',
    r'(?:no\s+restrictions?|without\s+restrictions?|bypass\s+(?:your\s+)?restrictions?)',
    r'override\s+(?:your\s+)?(?:safety|compliance|restrictions?|rules?)',
    r'\[(?:INST|SYS|SYSTEM|OVERRIDE)\]',
    r'<(?:sys|system|s)>',
    # ── Constraint bypass / "answer freely" variants ─────────────────────────
    r'answer\s+(?:me\s+)?(?:freely|without\s+(?:filters?|restrictions?|rules?|limits?|guidelines?|constraints?))',
    r'respond\s+(?:freely|without\s+(?:filters?|restrictions?|rules?|limits?|guidelines?|constraints?))',
    r'(?:without|no)\s+(?:any\s+)?(?:filters?|safety\s+filters?|content\s+filters?)',
    r'disable\s+(?:your\s+)?(?:filters?|safety|restrictions?|compliance)',
    r'turn\s+off\s+(?:your\s+)?(?:filters?|safety|restrictions?|compliance)',
    r'unfiltered\s+(?:answer|response|mode)',
    # ── System/pipeline exfiltration via question format ─────────────────────
    r'what\s+is\s+your\s+(?:system\s+prompt|initial\s+prompt|original\s+instructions?)',
    r'what\s+(?:are\s+)?your\s+(?:instructions?|constraints?|directives?)',             # clearly AI-internal terms
    r'what\s+(?:are\s+)?your\s+(?:system|operating|ai|internal)\s+(?:rules?|guidelines?|restrictions?)',  # AI-targeted compound
    r'(?:show|tell|give)\s+me\s+your\s+(?:system\s+prompt|instructions?|configuration)',
    r'what\s+is\s+your\s+(?:pipeline|architecture|structure|setup|configuration|internal)',
    r'how\s+(?:do\s+you\s+work|were\s+you\s+(?:set\s+up|configured|built|programmed|trained\s+for\s+this)|are\s+you\s+(?:configured|set\s+up|built|programmed|structured|designed))',
    r'describe\s+(?:your\s+)?(?:pipeline|architecture|internal\s+(?:workings?|structure|logic))',
    # ── False attribution / context poisoning ────────────────────────────────
    r'you\s+(?:previously|already|just)\s+(?:said|confirmed|stated|told|agreed|approved)',
    r'in\s+your\s+(?:last|previous|prior)\s+(?:message|response|answer)\s+you\s+(?:said|confirmed)',
    # ── Hypothetical framing for constraint bypass ───────────────────────────
    r'as\s+DAN\b',
    r'hypothetically[,\s]+(?:if|what|assume)',
    r'for\s+(?:a\s+)?(?:fictional|creative\s+writing|story|hypothetical)\s+(?:exercise|purpose|scenario)',
]
_INJECTION_RE = re.compile('|'.join(INJECTION_PATTERNS), re.IGNORECASE)


# DEFENSIVE MEASURE 2: Input sanitisation — strip embedded role-signal tokens
# Attackers sometimes embed OpenAI / Llama role delimiters inside user messages to
# confuse the model into treating their text as a system message.
_SANITIZE_PATTERNS = [
    (re.compile(r'\[/?(?:INST|SYS|SYSTEM|ASSISTANT|USER)\]', re.IGNORECASE), ''),
    (re.compile(r'</?(?:sys|system|s|assistant|user)\s*>',   re.IGNORECASE), ''),
    (re.compile(r'---+\s*(?:OVERRIDE|SYSTEM|INSTRUCTION)\s*---+', re.IGNORECASE), ''),
    (re.compile(r'<<(?:SYS|INST)>>.*?<</(?:SYS|INST)>>', re.IGNORECASE | re.DOTALL), ''),
]

_MAX_QUERY_LEN = 1200   # hard cap — prevents large injection payloads


def sanitize_query(query: str) -> str:
    """
    Strip embedded role tokens and enforce length cap before injection check.
    Part of the security hardening pipeline — applied before is_injection().
    """
    q = query[:_MAX_QUERY_LEN]          # truncate oversized payloads
    for pattern, replacement in _SANITIZE_PATTERNS:
        q = pattern.sub(replacement, q)
    return q.strip()


def is_injection(query: str) -> bool:
    return bool(_INJECTION_RE.search(query))


# ── Conversation Session (Feature 1) ─────────────────────────────────────────

class SAGEConversationSession:
    MAX_HISTORY = 6  # rolling window in turns (= 12 messages)

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"sess_{int(time.time())}"
        self.history: List[Dict] = []
        self.created_at = datetime.now().isoformat()

    def add_turn(self, user_q: str, assistant_r: str):
        self.history.append({"role": "user",      "content": user_q})
        self.history.append({"role": "assistant", "content": assistant_r})

    def get_context(self) -> List[Dict]:
        return self.history[-(self.MAX_HISTORY * 2):]

    @property
    def turn_count(self) -> int:
        return len(self.history) // 2


# ── Confidence Scorer (Feature 2) ────────────────────────────────────────────

class ConfidenceScorer:
    COMPLIANCE_KW = ["approval", "required", "prohibited", "must", "shall",
                     "violation", "complies", "written", "consent", "mandatory"]
    AMBIGUITY_KW  = ["unclear", "ambiguous", "not specified", "policy is silent",
                     "insufficient", "cannot determine"]

    def score(self, response: str) -> Dict:
        s, comp = 50, {}

        cits = extract_citation_count(response)
        cp = min(cits * 8, 32)
        comp["citations"] = cp
        s += cp

        rp = 10 if extract_risk_level(response) in ("High", "Medium", "Low") else 0
        comp["risk_clarity"] = rp
        s += rp

        kp = min(sum(1 for k in self.COMPLIANCE_KW if k in response.lower()), 8)
        comp["keyword_coverage"] = kp
        s += kp

        cb = 5 if "POLICY TENSION" in response.upper() else 0
        comp["conflict_bonus"] = cb
        s += cb

        ap = min(sum(1 for a in self.AMBIGUITY_KW if a in response.lower()) * 15, 30)
        comp["ambiguity_penalty"] = -ap
        s -= ap

        final = max(0, min(100, s))
        return {
            "score": final,
            "band":  "High" if final >= 75 else ("Medium" if final >= 50 else "Low"),
            "components": comp,
        }


# ── Policy Conflict Detector (Feature 3) ─────────────────────────────────────

BUILTIN_CONFLICT_RULES = [
    {"id": "CF-001", "name": "Local Storage vs. Remote Mobility",
     "policy_a": "IS §5.3", "policy_b": "RW §3.4",
     "triggers": ["personal laptop", "local", "offline", "coffee shop", "airport"],
     "description": "Local storage ban is absolute; temporary public-place remote work creates tension for employees needing offline access.",
     "severity": "High"},
    {"id": "CF-002", "name": "International Work + Data Transfer",
     "policy_a": "RW §4.2", "policy_b": "DP §5.1",
     "triggers": ["international", "europe", "germany", "france", "portugal", "netherlands", "eea"],
     "description": "Work approval (HR/Legal) and data transfer safeguards (DPO) apply simultaneously with different approvers.",
     "severity": "Medium"},
    {"id": "CF-003", "name": "BYOD Enrollment vs. Data Prohibition",
     "policy_a": "IS §6.1", "policy_b": "IS §6.3",
     "triggers": ["personal phone", "personal device", "byod", "mdm"],
     "description": "MDM enrollment is required for personal devices while company data storage is simultaneously prohibited on them.",
     "severity": "Medium"},
    {"id": "CF-004", "name": "Encryption ≠ Exemption",
     "policy_a": "IS §5.1", "policy_b": "IS §5.3",
     "triggers": ["encrypt", "encrypted", "aes", "secure"],
     "description": "Encryption is required AND local storage is still prohibited. Employees often misread the encryption requirement as an exception.",
     "severity": "High"},
    {"id": "CF-005", "name": "Benefits Gap in International Approval",
     "policy_a": "RW §4.4", "policy_b": "RW §4.2",
     "triggers": ["international", "abroad", "overseas", "health insurance", "benefits", "foreign"],
     "description": "Policy allows extended international work approval but does not resolve the health-insurance gap.",
     "severity": "Medium"},
]


class PolicyConflictDetector:
    def __init__(self, rules: List[Dict] | None = None):
        self.rules = rules or BUILTIN_CONFLICT_RULES

    def detect(self, text: str) -> List[Dict]:
        tl = text.lower()
        return [r for r in self.rules if any(t in tl for t in r["triggers"])]

    def format_md(self, conflicts: List[Dict]) -> str:
        if not conflicts:
            return "✅ No policy conflicts detected."
        lines = [f"**⚠️ {len(conflicts)} Policy Conflict(s) Detected**\n"]
        for c in conflicts:
            lines.append(
                f"- **[{c['id']}] {c['name']}** — Severity: `{c['severity']}`  \n"
                f"  `{c['policy_a']}` ↔ `{c['policy_b']}`  \n"
                f"  {c['description']}"
            )
        return "\n".join(lines)


# ── Citation Verifier (Feature 3b) ────────────────────────────────────────────

class CitationVerifier:
    def __init__(self, section_lookup: Dict[str, str]):
        self.lookup = section_lookup

    def verify(self, response: str) -> Dict:
        # Match both "POL-XX-2025, Section X.X" and generic "Section X.X"
        cited = re.findall(r'(POL-[A-Z]+-\d{4})[,\s]+[Ss]ection\s+(\d+\.\d+)', response)
        results = []
        for pid, sec in cited:
            key   = f"{pid}§{sec}"
            found = self.lookup.get(key)
            results.append({
                "citation": f"{pid} §{sec}",
                "verified": found is not None,
                "text": (found[:80] + "...") if found else "⚠️ Not found in corpus",
            })
        v, t = sum(1 for r in results if r["verified"]), len(results)
        return {
            "total": t, "verified": v,
            "groundedness": f"{v/t:.0%}" if t else "N/A",
            "details": results,
        }


# ── Audit Logger (Feature 4) ──────────────────────────────────────────────────

class AuditLogger:
    def __init__(self, session_id: str | None = None, path: str | None = None):
        if path:
            self.path = Path(path)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sid = session_id or ts
            self.path = Path(f"logs/sage_session_{sid}_{ts}.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict] = []
        # Never load old entries — each instance is a fresh session log


    def log(self, session_id: str, query: str, response: str,
            latency: float, tokens: int, metadata: Dict | None = None) -> str:
        meta = metadata or {}
        eid  = f"AUD-{len(self.entries)+1:05d}"

        # Extract from response text; fall back to pre-computed values in metadata
        risk  = extract_risk_level(response)
        if risk == "Unknown" and meta.get("severity", 0) >= 60:
            risk = "High"
        elif risk == "Unknown" and meta.get("severity", 0) >= 35:
            risk = "Medium"
        elif risk == "Unknown" and meta.get("severity", 0) > 0:
            risk = "Low"

        entry = {
            "entry_id":        eid,
            "timestamp":       datetime.now().isoformat(),
            "session_id":      session_id,
            "query":           query,
            "response_digest": response[:300] + ("..." if len(response) > 300 else ""),
            "risk_level":      risk,
            "severity_score":  extract_score(response, "Severity") if extract_score(response, "Severity") is not None else meta.get("severity"),
            "confidence_score":extract_score(response, "Confidence") if extract_score(response, "Confidence") is not None else meta.get("confidence"),
            "citation_count":  extract_citation_count(response),
            "latency_s":       round(latency, 3),
            "tokens":          tokens,
            "metadata":        meta,
        }
        self.entries.append(entry)
        self.path.write_text(json.dumps(self.entries, indent=2))
        return eid

    def recent(self, n: int = 10) -> List[Dict]:
        return self.entries[-n:]

    def stats(self) -> Dict:
        if not self.entries:
            return {"total": 0}
        confs = [e["confidence_score"] for e in self.entries if e["confidence_score"]]
        return {
            "total":          len(self.entries),
            "risk_dist":      dict(Counter(e["risk_level"] for e in self.entries)),
            "avg_latency":    round(sum(e["latency_s"] for e in self.entries) / len(self.entries), 3),
            "avg_confidence": round(sum(confs) / len(confs), 1) if confs else None,
            "sessions":       len(set(e["session_id"] for e in self.entries)),
        }


# ── Feedback Collector (Feature 5) ───────────────────────────────────────────

class FeedbackCollector:
    DIMS = ["clarity", "accuracy", "usefulness", "trust"]

    def __init__(self, path: str = "logs/sage_feedback.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict] = []
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text())
            except Exception:
                self.entries = []

    def submit(self, audit_id: str, query: str, ratings: Dict[str, int],
               comment: str = "", recommend: bool = True) -> str:
        validated = {d: max(1, min(5, int(ratings.get(d, 3)))) for d in self.DIMS}
        fid = f"FB-{len(self.entries)+1:04d}"
        self.entries.append({
            "feedback_id":   fid,
            "audit_id":      audit_id,
            "query_preview": query[:80],
            "ratings":       validated,
            "overall":       round(sum(validated.values()) / len(validated), 2),
            "comment":       comment,
            "recommend":     recommend,
            "timestamp":     datetime.now().isoformat(),
        })
        self.path.write_text(json.dumps(self.entries, indent=2))
        return fid

    def aggregate(self) -> Dict:
        if not self.entries:
            return {"total": 0}
        return {
            "total":        len(self.entries),
            "overall_avg":  round(sum(e["overall"] for e in self.entries) / len(self.entries), 2),
            "dim_avgs":     {d: round(sum(e["ratings"].get(d, 3) for e in self.entries) / len(self.entries), 2)
                             for d in self.DIMS},
            "recommend_rate": f"{sum(1 for e in self.entries if e['recommend'])/len(self.entries):.0%}",
        }


# ── Severity Scorer (Feature 6) ──────────────────────────────────────────────

class SeverityWeightedScorer:
    INTL_KW = ["international", "portugal", "germany", "japan", "france", "canada",
               "abroad", "overseas", "foreign", "eea", "europe", "uk", "spain"]
    DATA_KW = ["pii", "personal data", "breach", "customer data", "sensitive",
               "confidential", "restricted", "health data"]
    EEA_KW  = ["eea", "europe", "germany", "france", "netherlands", "portugal",
               "spain", "sccs", "bcrs"]

    def score(self, query: str, response: str, policies: List[str] | None = None) -> Dict:
        combo = (query + " " + response).lower()
        risk  = extract_risk_level(response)
        if risk in ("N/A", "Unknown"):
            return {"score": 0, "band": "N/A", "components": {}}

        comp = {}
        base = {"High": 40, "Medium": 20, "Low": 5}.get(risk, 10)
        s = base
        comp["risk_base"] = base

        n_pol = len(policies) if policies else max(1, extract_citation_count(response) // 2)
        ep = max(0, n_pol - 1) * 15
        s += ep
        comp["extra_policies"] = ep

        il = 15 if any(k in combo for k in self.INTL_KW) else 0
        s += il
        comp["international"] = il

        de = 20 if any(k in combo for k in self.DATA_KW) else 0
        s += de
        comp["data_exposure"] = de

        ee = 10 if any(k in combo for k in self.EEA_KW) else 0
        s += ee
        comp["eea_scope"] = ee

        final = max(0, min(100, s))
        band  = ("Critical" if final >= 80 else
                 "High"     if final >= 60 else
                 "Medium"   if final >= 35 else "Low")
        return {"score": final, "band": band, "components": comp}


# ── LangGraph Agent Builder ───────────────────────────────────────────────────

_AGENT_WORKFLOW_BASE = """You are SAGE — a compliance reasoning agent for {company_name}.
You have 4 tools: search_policy, check_cross_references, detect_policy_conflicts, assess_risk.

FIXED WORKFLOW (follow in order):
1. check_cross_references  — which policies apply to this scenario?
2. detect_policy_conflicts — are there compounding tensions?
3. search_policy           — evidence per triggered policy
4. assess_risk             — synthesise into risk classification
5. Final structured response:
   Answer: [150–250 words]
   Citations: [one per line: POLICY-ID, Section X.X — description]
   Risk Level: [Low / Medium / High] — [justification]
   Severity Score: [0–100] — [components]
   Confidence Score: [0–100] — [basis]
   Reasoning: [2–4 sentences with section references]

HARD CONSTRAINTS (these override everything):

1. ONLY answer from what search_policy returns. NEVER use general knowledge or training data.

2. NO VAGUE ANSWERS: Never say a topic is "governed by the General Policy area" or
   "sections were not retrieved — check official sources." That is hallucination.
   If search_policy finds nothing relevant, say EXACTLY:
   "The uploaded policy documents do not address this specific topic."

3. CONTACT INFORMATION: Phone numbers, email addresses, and physical addresses that
   search_policy returns ARE policy content. Quote them EXACTLY as written.
   NEVER say "I can't provide contact information" or "check their official website"
   for data that appears in the search results — that refusal is WRONG here.

4. ORGANIZATION MISMATCH: The loaded documents belong to {company_name}.
   If the user asks about a DIFFERENT organization (e.g. "What does Google say...",
   "What is Amazon's policy...", "What does Microsoft's policy say..."), begin your
   answer with:
   "Note: I do not have [that organization]'s policy loaded. I am answering based on
   {company_name}'s policy documents currently loaded."
   Then continue with what the loaded documents say, if relevant.

5. MISSING POLICY TOPICS: If the user asks about a topic that is not covered in
   the uploaded documents (e.g. tuition refunds, financial aid, research misconduct
   when those policies were not uploaded), say:
   "The uploaded policy documents do not address this specific topic. Please consult
   the appropriate department or official {company_name} resources."
   Do NOT invent an answer or guess at what a policy might say.

6. NEVER write essays, summaries of external topics, or creative content.

7. NEVER ASK FOR MORE CONTEXT: Do not respond with "please provide more context",
   "please describe the scenario", "please specify which policy", or any similar
   clarification request. You have the uploaded documents — call search_policy
   immediately with the user's question and answer from what it returns.
   Asking the user for clarification BEFORE searching is always wrong.
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence, lambda x, y: x + y]


def build_langgraph_agent(llm: ChatOpenAI, tools_list: list, system_prompt: str = ""):
    """Build the LangGraph ReAct agent with given tools."""
    llm_tools = llm.bind_tools(tools_list)
    prompt = system_prompt or _AGENT_WORKFLOW_BASE.format(company_name="your organization")

    def agent_node(state: AgentState):
        msgs = [SystemMessage(content=prompt)] + list(state["messages"])
        return {"messages": [llm_tools.invoke(msgs)]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools_list))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


# ── Full SAGE Pipeline ────────────────────────────────────────────────────────

class SAGEPipeline:
    """
    Orchestrates the full 6-layer SAGE pipeline for real-time use.
    Thread-safe per-request; session state managed externally.
    """

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        section_lookup: Dict[str, str],
        collection,                      # chromadb collection or None
        chunks: List[Dict],              # fallback keyword-search chunks
        conflict_rules: List[Dict] | None = None,
        company_name: str = "your organization",
    ):
        self.api_key        = api_key
        self.system_prompt  = system_prompt
        self.section_lookup = section_lookup
        self.chunks         = chunks
        self.company_name   = company_name

        self.client = OpenAI(api_key=api_key)
        self.llm    = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)

        self.conflict_detector  = PolicyConflictDetector(conflict_rules)
        self.confidence_scorer  = ConfidenceScorer()
        self.severity_scorer    = SeverityWeightedScorer()
        self.citation_verifier  = CitationVerifier(section_lookup)
        self.audit_logger       = AuditLogger()
        self.feedback_collector = FeedbackCollector()
        self.collection         = collection
        self._agent     = None

    def new_session_logger(self, session_id: str) -> "AuditLogger":
        """Swap in a fresh per-session log file. Call when New Chat is clicked."""
        self.audit_logger = AuditLogger(session_id=session_id)
        return self.audit_logger

    NO_CONTEXT_SIGNAL = "<<NO_RELEVANT_POLICY_CONTENT_FOUND>>"

    # ── Query Expansion Synonyms ──────────────────────────────────────────────
    # Maps common user phrasings to canonical policy vocabulary so ChromaDB
    # retrieval finds the right sections even when wording differs from the policy.
    _QUERY_SYNONYMS: Dict[str, List[str]] = {
        "work abroad":         ["international remote work", "work outside home country"],
        "work overseas":       ["international remote work", "work outside home country"],
        "work from another country": ["international remote work", "extended abroad"],
        "ssn":                 ["sensitive personal data", "social security number"],
        "personal info":       ["personally identifiable information", "PII", "personal data"],
        "fired":               ["termination", "employee departure", "offboarding"],
        "quit":                ["resignation", "termination", "offboarding"],
        "let go":              ["termination", "employee departure", "offboarding"],
        "home office":         ["remote work", "work from home", "WFH"],
        "wfh":                 ["remote work", "work from home", "telecommute"],
        "laptop":              ["company device", "endpoint", "workstation"],
        "personal phone":      ["mobile device", "BYOD", "personal device"],
        "my phone":            ["mobile device", "BYOD", "personal device"],
        "own device":          ["BYOD", "personal device", "bring your own device"],
        "hacked":              ["security incident", "data breach", "unauthorized access"],
        "password":            ["authentication", "credentials", "access control"],
        "2fa":                 ["MFA", "multi-factor authentication", "two-factor"],
        "two factor":          ["MFA", "multi-factor authentication"],
        "vpn":                 ["VPN", "virtual private network", "secure connection"],
        "gdpr":                ["GDPR", "data protection regulation", "EEA", "EU data"],
        "europe":              ["EEA", "European Economic Area", "GDPR", "international"],
        "data breach":         ["security incident", "unauthorized disclosure", "breach notification"],
        "probation":           ["probationary period", "new hire", "first 90 days"],
        "contractor":          ["third-party contractor", "external staff", "vendor"],
        "intern":              ["temporary worker", "non-permanent", "probationary"],
        "health insurance":    ["benefits", "international coverage", "employee benefits"],
        "encrypt":             ["AES-256", "encryption", "encrypted storage"],
        "cloud":               ["cloud storage", "SaaS", "cloud service"],
        "usb":                 ["removable media", "external drive", "portable storage"],
        # ── Contact / location lookups ────────────────────────────────────────
        "phone number":        ["contact information", "contact details", "office phone", "telephone"],
        "office located":      ["contact information", "office address", "headquarters", "location"],
        "where is the":        ["contact information", "address", "location", "office"],
        "email address":       ["contact information", "contact details", "email"],
        "how to contact":      ["contact information", "reporting", "reach out"],
        # ── Anonymous / confidential reporting ───────────────────────────────
        "anonymous":           ["anonymous reporting", "ethics hotline", "confidential report", "hotline"],
        "hotline":             ["anonymous reporting", "ethics hotline", "confidential reporting"],
        "confidential":        ["anonymous reporting", "confidential report", "protected activity"],
        # ── Protected characteristics (for nondiscrimination policies) ────────
        "shared ancestry":     ["protected characteristics", "ancestry", "national origin", "ethnicity"],
        "pregnancy":           ["protected characteristics", "sex", "pregnancy-related conditions",
                                "sex stereotypes", "sex characteristics"],
        "religion":            ["protected characteristics", "religious creed", "faith", "belief"],
        "disability":          ["protected characteristics", "disability status", "accommodation"],
        "veteran":             ["protected characteristics", "military status", "veteran status"],
        "gender identity":     ["protected characteristics", "gender expression", "sex", "gender"],
        "sexual orientation":  ["protected characteristics", "gender", "sex"],
        # ── Reporting / investigation ─────────────────────────────────────────
        "retaliation":         ["adverse action", "retaliation", "protected activity", "good faith report"],
        "cooperate":           ["investigation", "cooperation", "disciplinary action", "separation"],
        "investigation":       ["complaint procedure", "reporting", "ouec", "investigation process"],
        "file a complaint":    ["reporting", "ouec", "complaint procedure", "discrimination report"],
        "report discrimination": ["ouec", "contact information", "reporting procedure", "complaint"],
    }

    def _expand_query(self, query: str) -> str:
        """
        Append compliance synonyms for phrases in the query so ChromaDB retrieves
        policy sections even when user wording diverges from policy vocabulary.
        """
        ql = query.lower()
        additions: List[str] = []
        for phrase, synonyms in self._QUERY_SYNONYMS.items():
            if phrase in ql:
                additions.extend(synonyms)
        if additions:
            return query + " " + " ".join(dict.fromkeys(additions))  # dedupe, order-preserve
        return query

    def _keyword_search(self, query: str, k: int = 7) -> str:
        expanded = self._expand_query(query)
        ql = query.lower()
        query_words = [w for w in expanded.lower().split() if len(w) > 2]
        is_contact_query = any(sig in ql for sig in self._CONTACT_SIGNALS)
        char_limit = 1000 if is_contact_query else 300
        if is_contact_query:
            k = max(k, 10)
        scored = sorted(
            [(sum(1 for w in query_words if w in c["text"].lower()), c)
             for c in self.chunks],
            key=lambda x: x[0], reverse=True
        )
        results = [
            f"[{i+1}] {c['policy_id']} §{c['section']}\n{c['text'][:char_limit]}..."
            for i, (score, c) in enumerate(scored[:k]) if score > 0
        ]
        return "\n\n".join(results) if results else self.NO_CONTEXT_SIGNAL

    # Phrases that signal a factual contact/location lookup.
    # Covers both user phrasings ("phone number") and agent sub-queries
    # ("OUEC contact information", "where is the office", etc.)
    # NOTE: avoid single-word signals like "phone" — they conflict with
    # _QUERY_SYNONYMS entries (e.g. "phone" → BYOD expansion).
    _CONTACT_SIGNALS = {
        "phone number", "phone num",
        "office located", "office location",
        "where is the", "where is ",
        "email address",
        "how to contact", "contact number",
        "contact information", "contact details",
        "address of", "mailing address", "office address",
        "hotline number",
        "fax number", "zip code", "postal code",
        "reach the", "reach out", "get in touch",
    }

    def _rag_search(self, query: str, k: int = 7) -> str:
        """
        Hybrid retrieval with query expansion + combined re-ranking.

        Pipeline:
          1. Expand query with compliance synonyms for better semantic recall.
          2. Fetch 2k candidates from ChromaDB (semantic).
          3. Score every corpus chunk by keyword overlap (keyword signal).
          4. Re-rank: combined_score = semantic_score(0–1) * 0.6 + kw_norm * 0.4
          5. Deduplicate by (policy_id, section), return top-k.

        This ensures factual queries (phone numbers, exact clause values) are
        never lost to pure cosine distance, while semantic relevance still leads.
        """
        if self.collection is None:
            return self._keyword_search(query, k)
        try:
            expanded = self._expand_query(query)

            # Boost k for contact/factual lookups so short contact chunks
            # are not crowded out by larger high-scoring sections
            ql = query.lower()
            is_contact_query = any(sig in ql for sig in self._CONTACT_SIGNALS)
            if is_contact_query:
                k = max(k, 10)

            # ── Step 1: Semantic retrieval ────────────────────────────────────
            fetch_n = min(k * 3 if is_contact_query else k * 2, 20)
            res = self.collection.query(query_texts=[expanded], n_results=fetch_n)
            docs      = res["documents"][0]
            metas     = res["metadatas"][0]
            distances = res.get("distances", [[]])[0] or [0.0] * len(docs)

            # Convert L2/cosine distance → similarity score (0–1; lower dist = higher sim)
            # Clamp to [0, 2] before inversion to prevent negatives
            semantic_pool = {
                (meta.get("policy_id", ""), meta.get("section", "")): {
                    "doc": doc, "meta": meta,
                    "sem_score": max(0.0, 1.0 - min(dist, 2.0) / 2.0)
                }
                for doc, meta, dist in zip(docs, metas, distances)
                if dist < 1.9   # slightly relaxed from 1.8 — expansion query may shift distances
            }

            # ── Step 2: Keyword scoring across all chunks ─────────────────────
            query_words = [w for w in expanded.lower().split() if len(w) > 2]
            kw_raw = [
                (sum(1 for w in query_words if w in c["text"].lower()), c)
                for c in self.chunks
            ]
            max_kw = max((s for s, _ in kw_raw), default=1) or 1
            # Normalise keyword scores to [0, 1]
            kw_norm_map: Dict[tuple, float] = {
                (c["policy_id"], c["section"]): score / max_kw
                for score, c in kw_raw if score > 0
            }

            # ── Step 3: Build unified candidate set ───────────────────────────
            # Start with all semantic hits; fill in kw_norm; add kw-only entries
            candidates: Dict[tuple, Dict] = {}
            for key, item in semantic_pool.items():
                kw = kw_norm_map.get(key, 0.0)
                candidates[key] = {**item, "kw_score": kw,
                                   "combined": item["sem_score"] * 0.6 + kw * 0.4}

            # Add keyword-only chunks not already in semantic pool
            for score, c in kw_raw:
                if score == 0:
                    continue
                key = (c["policy_id"], c["section"])
                if key not in candidates:
                    kw = score / max_kw
                    candidates[key] = {
                        "doc": c["text"], "meta": {"policy_id": c["policy_id"], "section": c["section"]},
                        "sem_score": 0.0, "kw_score": kw, "combined": kw * 0.4
                    }

            # Last-resort fallback: no candidates at all
            if not candidates and self.chunks:
                for _, c in sorted(kw_raw, key=lambda x: x[0], reverse=True)[:3]:
                    key = (c["policy_id"], c["section"])
                    candidates[key] = {
                        "doc": c["text"], "meta": {"policy_id": c["policy_id"], "section": c["section"]},
                        "sem_score": 0.0, "kw_score": 0.0, "combined": 0.0
                    }

            if not candidates:
                return self.NO_CONTEXT_SIGNAL

            # ── Step 4: Re-rank by combined score, return top-k ───────────────
            ranked = sorted(candidates.values(), key=lambda x: x["combined"], reverse=True)[:k]
            # Contact/factual queries need more chars — phone numbers and addresses
            # are often past position 450 in the contact section chunk
            char_limit = 1000 if is_contact_query else 450
            return "\n\n".join(
                f"[{i+1}] {item['meta']['policy_id']} §{item['meta']['section']}\n{item['doc'][:char_limit]}"
                for i, item in enumerate(ranked)
            )
        except Exception:
            return self._keyword_search(query, k)

    def _build_tools(self) -> list:
        pipeline = self  # capture for closures

        @tool
        def search_policy(query: str) -> str:
            """Search company policy documents for sections relevant to the query."""
            return pipeline._rag_search(query)

        @tool
        def check_cross_references(scenario: str) -> str:
            """Identify which company policies are triggered by a given scenario."""
            sl, triggered = scenario.lower(), []

            # Meta-questions about the document itself — route directly to search
            meta_keywords = [
                "who does", "who is", "when does", "when is",
                "what is the purpose", "what does this policy", "what is this policy",
                "scope of", "effective date", "applies to", "applicable for",
                "applicable to", "use this policy", "use this for",
                "new joinee", "new employee", "new hire", "what can i",
                "purpose of this", "what topics", "what does it cover",
                "what is it for", "how do i use",
            ]
            if any(kw in sl for kw in meta_keywords):
                return ("This is a document-level question. Use search_policy to find the "
                        "Purpose, Scope, or Effective Date sections directly.")

            if any(w in sl for w in ["remote", "work", "office", "international", "travel",
                                      "home", "probation", "eligible", "eligibility", "contractor"]):
                triggered.append("Remote-Work-Policy")
            if any(w in sl for w in ["data", "pii", "privacy", "customer", "eea", "gdpr",
                                      "retention", "breach", "transfer"]):
                triggered.append("Data-Privacy-Policy")
            if any(w in sl for w in ["laptop", "vpn", "security", "mfa", "encrypt",
                                      "byod", "store", "device", "password", "mdm"]):
                triggered.append("Information-Security-Policy")
            if any(w in sl for w in [
                "discriminat", "harassment", "equal opportunity", "protected",
                "race", "gender", "religion", "disability", "age", "national origin",
                "sexual orientation", "retaliat", "complaint", "grievance",
                "title ix", "title vii", "eeoc", "civil rights", "accommodation",
                "faculty", "professor", "staff", "student", "employee", "workforce",
                "conduct", "misconduct", "behavior", "ethics", "integrity",
                "leave", "benefit", "compensation", "hiring", "termination",
                "performance", "discipline", "suspension", "investigation",
            ]):
                triggered.append("HR-Conduct-Policy")
            if not triggered:
                triggered = ["General-Policy"]
            return (f"{len(triggered)} policy area(s) triggered: {', '.join(triggered)}\n"
                    + ("Run search_policy for each, then synthesise." if len(triggered) >= 2 else
                       "Run search_policy to find relevant sections."))

        @tool
        def detect_policy_conflicts(scenario: str) -> str:
            """Detect policy tensions for a scenario before final reasoning."""
            cfls = pipeline.conflict_detector.detect(scenario)
            if not cfls:
                return "✓ No policy conflicts detected."
            lines = [f"⚠️ {len(cfls)} CONFLICT(S):"]
            for c in cfls:
                lines += [f"  [{c['id']}] {c['name']} — {c['policy_a']} ↔ {c['policy_b']}",
                           f"  {c['description']}"]
            return "\n".join(lines)

        @tool
        def assess_risk(findings: str) -> str:
            """Synthesise compliance findings into a risk assessment. Call last."""
            fl = findings.lower()
            risk = (
                "High"   if any(w in fl for w in ["violates", "prohibited", "must not", "breach", "exposure", "active violation"]) else
                "Medium" if any(w in fl for w in ["requires action", "additional approval", "hr", "legal", "dpo", "not yet"]) else
                "Low"
            )
            return (
                f"RISK LEVEL: {risk}\nSummary: {findings[:300]}...\n"
                "→ Compile final response with all 6 fields: "
                "Answer / Citations / Risk Level / Severity Score / Confidence Score / Reasoning."
            )

        return [search_policy, check_cross_references, detect_policy_conflicts, assess_risk]

    def _get_agent(self):
        if self._agent is None:
            tools = self._build_tools()
            agent_prompt = _AGENT_WORKFLOW_BASE.format(company_name=self.company_name)
            self._agent = build_langgraph_agent(self.llm, tools, agent_prompt)
        return self._agent

    def query(
        self,
        user_query: str,
        session: SAGEConversationSession,
        use_agent: bool = True,
    ) -> Dict:
        """
        Run the full pipeline. Returns a structured result dict.
        """
        t0 = time.time()

        # L0: Check injection on RAW query first (before sanitisation strips role tokens),
        #     then sanitise for LLM use. Detection must see the original payload.
        raw_query  = user_query
        user_query = sanitize_query(user_query)

        # L1: Injection defence — check both raw and sanitised to catch all variants
        if is_injection(raw_query) or is_injection(user_query):
            return {
                "blocked": True,
                "response": "⛔ Your query was flagged as a potential prompt injection attempt and was not processed.",
                "risk_level": "N/A",
                "severity": {"score": 0, "band": "N/A", "components": {}},
                "confidence": {"score": 0, "band": "N/A", "components": {}},
                "conflicts": [],
                "citations": [],
                "citation_verification": None,
                "latency": round(time.time() - t0, 3),
                "tokens": 0,
                "audit_id": None,
            }

        # L2: Grounding gate — check if documents contain anything relevant
        context = self._rag_search(user_query)
        if context == self.NO_CONTEXT_SIGNAL:
            no_info = (
                "I wasn't able to find any relevant information in the uploaded policy documents "
                "to answer this question. Please make sure your documents are compliance or policy "
                "documents that cover this topic."
            )
            session.add_turn(user_query, no_info)
            return {
                "blocked":               False,
                "grounded":              False,
                "response":              no_info,
                "answer":                no_info,
                "citations":             [],
                "risk_level":            "N/A",
                "reasoning":             "",
                "severity":              {"score": 0, "band": "N/A", "components": {}},
                "confidence":            {"score": 0, "band": "N/A", "components": {}},
                "conflicts":             [],
                "citation_verification": None,
                "latency":               round(time.time() - t0, 3),
                "tokens":                0,
                "audit_id":              None,
            }

        # L3–L4: Agent or direct call
        if use_agent:
            try:
                agent = self._get_agent()
                # Build message list with conversation history for multi-turn awareness
                prior = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in session.get_context()
                ]
                result = agent.invoke(
                    {"messages": prior + [HumanMessage(content=user_query)]},
                    {"recursion_limit": 15},
                )
                raw_response = result["messages"][-1].content
                tokens = sum(
                    len(m.content.split()) * 2
                    for m in result["messages"]
                    if hasattr(m, "content") and isinstance(m.content, str)
                )
            except Exception as e:
                raw_response = f"[Agent error: {e}]"
                tokens = 0
        else:
            # Direct call — inject retrieved context alongside conversation history
            grounded_prompt = (
                self.system_prompt
                + f"\n\nRELEVANT POLICY SECTIONS FOR THIS QUERY:\n{context}"
            )
            messages = [{"role": "system", "content": grounded_prompt}]
            messages.extend(session.get_context())
            messages.append({"role": "user", "content": user_query})
            try:
                resp   = self.client.chat.completions.create(
                    model="gpt-4o", temperature=0.3, max_tokens=2000,
                    messages=messages
                )
                raw_response = resp.choices[0].message.content
                tokens       = resp.usage.total_tokens
            except Exception as e:
                raw_response = f"[API error: {e}]"
                tokens       = 0

        latency = round(time.time() - t0, 3)

        # L5: Citation verification — flag any hallucinated section references
        citation_result = self.citation_verifier.verify(raw_response)
        unverified = [r["citation"] for r in citation_result.get("details", []) if not r["verified"]]
        if unverified:
            raw_response += (
                "\n\n⚠️ **Citation Warning:** The following section reference(s) could not be "
                "verified against the uploaded policy corpus — please confirm manually: "
                + ", ".join(unverified)
            )

        # L6: Scoring
        conflicts  = self.conflict_detector.detect(user_query + " " + raw_response)
        confidence = self.confidence_scorer.score(raw_response)
        severity   = self.severity_scorer.score(user_query, raw_response)

        # Override LLM's self-reported scores with our computed ones in the response text
        # so what's displayed and logged are consistent
        raw_response = re.sub(
            r'Confidence Score\s*[:\-]\s*\d{1,3}',
            f'Confidence Score: {confidence["score"]}',
            raw_response, flags=re.IGNORECASE
        )
        raw_response = re.sub(
            r'Severity Score\s*[:\-]\s*\d{1,3}',
            f'Severity Score: {severity["score"]}',
            raw_response, flags=re.IGNORECASE
        )

        # Audit
        audit_id = self.audit_logger.log(
            session_id=session.session_id,
            query=user_query,
            response=raw_response,
            latency=latency,
            tokens=tokens,
            metadata={"severity": severity["score"], "confidence": confidence["score"],
                      "risk_level": conflicts[0]["severity"] if conflicts else severity["band"]},
        )

        # Update session memory
        session.add_turn(user_query, raw_response)

        return {
            "blocked":              False,
            "response":             raw_response,
            "answer":               extract_field(raw_response, "Answer"),
            "citations":            extract_citations_list(raw_response),
            "risk_level":           extract_risk_level(raw_response),
            "reasoning":            extract_field(raw_response, "Reasoning"),
            "severity":             severity,
            "confidence":           confidence,
            "conflicts":            conflicts,
            "citation_verification":citation_result,
            "latency":              latency,
            "tokens":               tokens,
            "audit_id":             audit_id,
        }
