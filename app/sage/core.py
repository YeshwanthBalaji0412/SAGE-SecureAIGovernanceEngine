"""
SAGE core pipeline — all A6 production components.
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
from langchain_core.messages import HumanMessage, SystemMessage
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

INJECTION_PATTERNS = [
    r'ignore\s+(previous|prior|above|all)\s+instructions?',
    r'you\s+are\s+now\s+(a|an)\s+\w+',
    r'act\s+as\s+(if\s+you\s+(are|were)|a)',
    r'disregard\s+(your\s+)?(previous\s+)?instructions?',
    r'forget\s+(everything|all|your)',
    r'new\s+instructions?\s*:',
    r'system\s*:\s*you',
    r'jailbreak',
    r'base64',
    r'<\|.*?\|>',
]
_INJECTION_RE = re.compile('|'.join(INJECTION_PATTERNS), re.IGNORECASE)

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
        cp = min(cits * 8, 32);  comp["citations"]        = cp;  s += cp
        rp = 10 if extract_risk_level(response) in ("High", "Medium", "Low") else 0
        comp["risk_clarity"] = rp;  s += rp
        kp = min(sum(1 for k in self.COMPLIANCE_KW if k in response.lower()), 8)
        comp["keyword_coverage"] = kp;  s += kp
        cb = 5 if "POLICY TENSION" in response.upper() else 0
        comp["conflict_bonus"] = cb;  s += cb
        ap = min(sum(1 for a in self.AMBIGUITY_KW if a in response.lower()) * 15, 30)
        comp["ambiguity_penalty"] = -ap;  s -= ap
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
    def __init__(self, path: str = "logs/sage_audit_log.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict] = []
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text())
            except Exception:
                self.entries = []

    def log(self, session_id: str, query: str, response: str,
            latency: float, tokens: int, metadata: Dict | None = None) -> str:
        eid = f"AUD-{len(self.entries)+1:05d}"
        entry = {
            "entry_id":        eid,
            "timestamp":       datetime.now().isoformat(),
            "session_id":      session_id,
            "query":           query,
            "response_digest": response[:300] + ("..." if len(response) > 300 else ""),
            "risk_level":      extract_risk_level(response),
            "severity_score":  extract_score(response, "Severity"),
            "confidence_score":extract_score(response, "Confidence"),
            "citation_count":  extract_citation_count(response),
            "latency_s":       round(latency, 3),
            "tokens":          tokens,
            "metadata":        metadata or {},
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
        s    = base;  comp["risk_base"] = base
        n_pol = len(policies) if policies else max(1, extract_citation_count(response) // 2)
        ep = max(0, n_pol - 1) * 15;  s += ep;  comp["extra_policies"] = ep
        il = 15 if any(k in combo for k in self.INTL_KW) else 0
        s += il;  comp["international"] = il
        de = 20 if any(k in combo for k in self.DATA_KW) else 0
        s += de;  comp["data_exposure"] = de
        ee = 10 if any(k in combo for k in self.EEA_KW) else 0
        s += ee;  comp["eea_scope"] = ee
        final = max(0, min(100, s))
        band  = ("Critical" if final >= 80 else
                 "High"     if final >= 60 else
                 "Medium"   if final >= 35 else "Low")
        return {"score": final, "band": band, "components": comp}


# ── LangGraph Agent Builder ───────────────────────────────────────────────────

AGENT_WORKFLOW_PROMPT = """You are SAGE — a compliance reasoning agent.
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
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence, lambda x, y: x + y]


def build_langgraph_agent(llm: ChatOpenAI, tools_list: list):
    """Build the LangGraph ReAct agent with given tools."""
    llm_tools = llm.bind_tools(tools_list)

    def agent_node(state: AgentState):
        msgs = [SystemMessage(content=AGENT_WORKFLOW_PROMPT)] + list(state["messages"])
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
    ):
        self.api_key        = api_key
        self.system_prompt  = system_prompt
        self.section_lookup = section_lookup
        self.chunks         = chunks

        self.client = OpenAI(api_key=api_key)
        self.llm    = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)

        self.conflict_detector  = PolicyConflictDetector(conflict_rules)
        self.confidence_scorer  = ConfidenceScorer()
        self.severity_scorer    = SeverityWeightedScorer()
        self.citation_verifier  = CitationVerifier(section_lookup)
        self.audit_logger       = AuditLogger()
        self.feedback_collector = FeedbackCollector()

        self.collection = collection
        self._agent     = None

    NO_CONTEXT_SIGNAL = "<<NO_RELEVANT_POLICY_CONTENT_FOUND>>"

    def _keyword_search(self, query: str, k: int = 5) -> str:
        scored = sorted(
            [(sum(1 for w in query.lower().split() if w in c["text"].lower()), c)
             for c in self.chunks],
            key=lambda x: x[0], reverse=True
        )
        results = [
            f"[{i+1}] {c['policy_id']} §{c['section']}\n{c['text'][:300]}..."
            for i, (score, c) in enumerate(scored[:k]) if score > 0
        ]
        return "\n\n".join(results) if results else self.NO_CONTEXT_SIGNAL

    def _rag_search(self, query: str, k: int = 5) -> str:
        """
        Hybrid retrieval: merge semantic (ChromaDB) + keyword results, deduplicate,
        return top-k unique chunks. Wider candidate pool improves recall on edge cases.
        """
        if self.collection is None:
            return self._keyword_search(query, k)
        try:
            # Fetch more candidates than needed so merging has room to work
            fetch_n = min(k * 2, 10)
            res = self.collection.query(query_texts=[query], n_results=fetch_n)
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            distances = res.get("distances", [[]])[0]

            # Semantic results: filter near-random hits
            # Use 1.8 threshold (slide/presentation docs have higher distances than
            # structured policies — 1.5 was too strict and caused false NO_CONTEXT)
            semantic = [
                (dist, doc, meta)
                for doc, meta, dist in zip(docs, metas, distances or [0] * len(docs))
                if dist < 1.8
            ]

            # Keyword results — always include top matches regardless of score
            # so factual queries (hotline number, contact details) are never lost
            query_words = [w for w in query.lower().split() if len(w) > 2]
            kw_scored = sorted(
                [(sum(1 for w in query_words if w in c["text"].lower()), c)
                 for c in self.chunks],
                key=lambda x: x[0], reverse=True
            )
            # Include top-k keyword results always (score > 0), plus top-3 as
            # last-resort fallback even if no words match (prevents grounding-gate
            # false positives on short/fragmented slide-deck documents)
            keyword_top = [(0.0, c["text"], {"policy_id": c["policy_id"], "section": c["section"]})
                           for score, c in kw_scored[:k] if score > 0]
            if not keyword_top and self.chunks:
                keyword_top = [(0.0, c["text"], {"policy_id": c["policy_id"], "section": c["section"]})
                               for _, c in kw_scored[:3]]

            # Merge: semantic first, then keyword; deduplicate by (policy_id, section)
            seen: set = set()
            merged = []
            for dist, doc, meta in semantic + keyword_top:
                key = (meta.get("policy_id", ""), meta.get("section", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append((dist, doc, meta))
                if len(merged) >= k:
                    break

            if not merged:
                return self.NO_CONTEXT_SIGNAL
            return "\n\n".join(
                f"[{i+1}] {m['policy_id']} §{m['section']}\n{d[:400]}"
                for i, (_, d, m) in enumerate(merged)
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
            meta_keywords = ["who does", "who is", "when does", "when is", "what is the purpose",
                             "what does this policy", "scope of", "effective date", "applies to"]
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
            self._agent = build_langgraph_agent(self.llm, tools)
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

        # L1: Injection defence
        if is_injection(user_query):
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
                    else SystemMessage(content=m["content"])
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
            metadata={"severity": severity["score"], "confidence": confidence["score"]},
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
