"""
SAGE — Secure AI Governance Engine
Real-time local Streamlit application.

Run:
    cd app && streamlit run app.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from sage.core import (
    SAGEConversationSession,
    SAGEPipeline,
    AuditLogger,
    is_injection,
    sanitize_query,
)
from sage.rag import ingest_documents, DEMO_ORGANIZATIONS, load_demo
from sage.prompts import build_system_prompt, detect_org_type

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SAGE — Compliance Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── App shell ── */
[data-testid="stAppViewContainer"] > .main {
    background: #060c1a;
}

/* ── Hide Streamlit footer and main menu only ── */
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Blend Streamlit header into background (keeps sidebar toggle working) ── */
[data-testid="stHeader"] {
    background: #060c1a !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
/* Hide just the deploy/share buttons inside the header */
[data-testid="stToolbarActions"] { display: none !important; }
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 900px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    background: #07091c !important;
    border-right: 1px solid rgba(99,102,241,0.14) !important;
}
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background: #07091c !important;
    padding: 0 12px 24px 12px !important;
}

/* ── Sidebar section card ── */
.sb-section {
    background: #0d1424;
    border: 1px solid rgba(99,102,241,0.13);
    border-radius: 10px;
    padding: 14px 14px 12px 14px;
    margin-bottom: 10px;
}
.sb-label {
    font-size: 10px;
    font-weight: 700;
    color: #4b5a7a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* ── Sidebar secondary buttons ── */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: #111828 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    color: #8b9db8 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: #182038 !important;
    border-color: rgba(99,102,241,0.45) !important;
    color: #c7d2fe !important;
}

/* ── Sidebar primary button (Load & Index) ── */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #3730a3, #4f46e5) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #312e81, #4338ca) !important;
    box-shadow: 0 6px 22px rgba(79,70,229,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Main area: chip / action secondary buttons ── */
section[data-testid="stMain"] [data-testid="stBaseButton-secondary"] {
    background: #0d1628 !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    color: #8b9cf4 !important;
    border-radius: 22px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.18s ease !important;
}
section[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover {
    background: #132040 !important;
    border-color: rgba(99,102,241,0.55) !important;
    color: #c7d2fe !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.18) !important;
}

/* ── Main area: New Chat primary button ── */
section[data-testid="stMain"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #312e81, #4f46e5) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 3px 10px rgba(79,70,229,0.28) !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stMain"] [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #3730a3, #4338ca) !important;
    box-shadow: 0 5px 18px rgba(79,70,229,0.42) !important;
    transform: translateY(-1px) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #0b1322 !important;
    border: 1px solid rgba(99,102,241,0.09) !important;
    border-radius: 12px !important;
    margin: 6px 0 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #0b1322 !important;
    border-color: rgba(99,102,241,0.22) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] {
    background: #060c1a !important;
    border-top: 1px solid rgba(99,102,241,0.1) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #0b1322 !important;
    border: 1px solid rgba(99,102,241,0.13) !important;
    border-radius: 10px !important;
}
[data-testid="stExpanderToggleIcon"] { color: #6366f1 !important; }

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11.5px !important; }

/* ── Progress bars ── */
[data-testid="stProgressBar"] > div {
    background: #1c2538 !important;
    border-radius: 6px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4f46e5, #818cf8) !important;
    border-radius: 6px !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] span { background: #1c2538 !important; }

/* ── Risk labels ── */
.r-high   { color: #f87171; font-weight: 700; }
.r-medium { color: #fbbf24; font-weight: 700; }
.r-low    { color: #4ade80; font-weight: 700; }
.r-na     { color: #6b7280; font-weight: 700; }

/* ── Citation box ── */
.cite-box {
    background: #08111f;
    border-left: 3px solid #3b82f6;
    padding: 9px 14px;
    border-radius: 0 8px 8px 0;
    margin: 5px 0;
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.6;
}

/* ── Conflict box ── */
.conflict-box {
    background: #160d00;
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #fde68a;
    line-height: 1.55;
}

/* ── Feature cards (welcome screen) ── */
.feat-card {
    background: linear-gradient(145deg, #0b1220, #101b34);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 20px;
    min-height: 110px;
}
.feat-card h4 { color: #e2e8f0; margin: 0 0 8px; font-size: 14px; font-weight: 600; }
.feat-card p  { color: #94a3b8; margin: 0; font-size: 12.5px; line-height: 1.65; }

/* ── Status pill ── */
.pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
    letter-spacing: 0.3px;
}
.pill-ready { background:rgba(74,222,128,0.1); color:#4ade80; border:1px solid rgba(74,222,128,0.25); }
.pill-setup { background:rgba(251,191,36,0.1); color:#fbbf24; border:1px solid rgba(251,191,36,0.25); }

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.1) !important; margin: 8px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2d3f65; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

# Module-level counter — reset to [0] on every Streamlit re-run (script re-executes top-to-bottom).
# Ensures render_report()'s download buttons get unique keys even when called multiple times per cycle.
_report_render_n = [0]

RISK_COLOR = {
    "High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80",
    "Critical": "#a78bfa", "N/A": "#6b7280", "Unknown": "#6b7280",
}

QUICK_QUESTIONS = [
    "Who does this policy apply to?",
    "What activities are strictly prohibited?",
    "What approvals are required for exceptions?",
    "What are the consequences of a policy violation?",
    "What mandatory training is required?",
    "How should security incidents or breaches be reported?",
]


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "api_key":             os.environ.get("OPENAI_API_KEY", ""),
        "company_name":        "",
        "corpus_loaded":       False,
        "pipeline":            None,
        "session":             None,
        "chat_history":        [],
        "audit_logger":        AuditLogger(),
        "show_audit":          False,
        "use_agent":           True,
        "pending_question":    "",
        "_load_msg":           "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_bar(value: int, color: str) -> str:
    """Return an inline HTML score bar — no CSS class dependency."""
    return (
        f'<div style="background:#1a2235;border-radius:4px;height:6px;'
        f'overflow:hidden;margin-top:5px;">'
        f'<div style="width:{value}%;height:6px;border-radius:4px;'
        f'background:{color};"></div></div>'
    )


def _start_new_chat():
    """Start a fresh chat session, swapping in a new per-session audit log."""
    new_sess = SAGEConversationSession()
    new_logger = st.session_state.pipeline.new_session_logger(new_sess.session_id)
    st.session_state.chat_history     = []
    st.session_state.session          = new_sess
    st.session_state.audit_logger     = new_logger
    st.session_state.pending_question = ""
    st.session_state["_load_msg"]     = ""
    st.rerun()


# Exact-match phrases that mean the user wants the session report dashboard.
# Single words like "report" or "audit" only trigger when the WHOLE query is
# that word (or a short phrase) — not when they appear inside a policy question.
_REPORT_EXACT = {"report", "audit", "stats", "statistics", "summary"}
_REPORT_PHRASES = {"show report", "session report", "show stats", "show summary",
                   "show audit", "view report", "get report", "my report",
                   "download report", "session stats"}

# Domain-specific compliance terms only — no generic question words (who/what/when)
# so that casual/nonsensical queries like "what am i doing to you" are caught.
_COMPLIANCE_KW = {
    # ── Universal ────────────────────────────────────────────────────────────
    "policy", "policies", "applies", "apply", "scope", "eligible", "eligib",
    "covered", "covers", "effective", "section", "rule", "regulation",
    "approval", "compliance", "violation", "requirement", "prohibited",
    "mandatory", "must", "shall", "authorized", "permit", "waiver",
    "exception", "consent", "procedure", "guideline", "handbook",
    "allowed", "permitted", "restrict", "obligat", "harass", "discriminat",
    "termination", "disciplin", "conduct", "reimbursement", "report",
    "contact", "phone", "hotline", "office", "address", "email",
    # ── TechNova — remote work, data privacy, information security ────────────
    "remote", "work", "vpn", "data", "pii", "privacy", "breach", "incident",
    "mfa", "byod", "gdpr", "encrypt", "laptop", "international", "travel",
    "retention", "security", "access", "eea", "password", "training",
    "contractor", "employee", "mdm", "probation", "equipment", "leave",
    "benefit", "insurance", "overtime", "exempt", "salary", "pto", "holiday",
    # ── EduTrack — academic integrity, student privacy, IT acceptable use ─────
    "student", "academic", "plagiar", "faculty", "transcript", "grade",
    "gpa", "enrollment", "ferpa", "exam", "assignment", "cheating",
    "fabricat", "chatgpt", "essay", "submiss", "parent", "directory",
    "appeal", "research", "collusion", "copyright", "ai-generated",
    "professor", "instructor", "course", "semester", "degree", "campus",
    "tuition", "scholarship", "suspension", "expulsion", "offense",
    # ── MedCore — PHI, HIPAA, workplace safety, staff conduct ─────────────────
    "patient", "phi", "hipaa", "medical", "clinical", "ppe", "baa",
    "nurse", "doctor", "workforce", "disclose", "login", "vendor",
    "treatment", "payment", "minimum necessary", "mobile", "substance",
    "drug", "alcohol", "eap", "gift", "conflict", "social media",
    "needle", "exposure", "spill", "evacuation", "hazard", "incident",
    # ── LaunchPad — IP, remote-first, code of conduct ─────────────────────────
    "intellectual", "invention", "open source", "side project", "side app",
    "weekend", "retaliat", "misconduct", "ip ", "cto", "equity", "venture",
    "async", "stipend", "coworking", "offshore", "relocation", "founder",
    "startup", "offsite", "onboarding", "repo", "contribut",
    # ── RetailFlow — PCI, customer data, employee handbook, store safety ──────
    "card", "payment", "skimming", "register", "shift", "cashier",
    "refund", "customer", "seasonal", "shoplifter", "uniform", "schedule",
    "attendance", "store", "pci", "transaction", "break", "loss prevention",
    "ladder", "fire", "emergency", "associate", "badge", "tattoo",
    "no-call", "no call", "swap", "discount", "void",
}

def _is_report_request(q: str) -> bool:
    ql = q.strip().lower()
    # Trigger only when the entire query IS a report keyword (e.g. user types just "report")
    # OR matches a known multi-word report phrase — never when "report" appears mid-sentence.
    if ql in _REPORT_EXACT:
        return True
    return any(phrase in ql for phrase in _REPORT_PHRASES)

_GENERAL_KNOWLEDGE_KW = {
    # Creative writing / general tasks
    "write me", "write an essay", "write a", "essay about", "tell me a",
    "explain to me", "give me a summary of", "summarize the history",
    "what is the history", "help me write",
    # Factual lookups unrelated to policy
    "tuition fee", "tuition cost", "how much does", "what does it cost",
    "who is the president", "who is the ceo", "who is the founder",
    "what courses", "which courses", "list of courses", "what programs",
    "what majors", "ranking of", "acceptance rate", "application deadline",
    "weather", "capital of", "population of", "tell me a joke",
    "recommend a", "best movie", "what is 2 + 2",
}

# Contact-info queries must bypass the keyword check — they are always policy-grounded
# (asking where to report, who to call, office locations in the uploaded document).
_CONTACT_BYPASS_KW = {
    "phone number", "phone num",
    "office located", "office location",
    "where is the", "where is ",
    "email address",
    "how to contact", "contact number",
    "contact information", "contact details",
    "address of", "mailing address", "office address",
    "hotline number",
    "fax number", "zip code", "postal code",
    "reach the office", "reach out to", "get in touch",
}

def _is_out_of_scope(q: str) -> bool:
    ql = q.lower()
    words = ql.split()
    if len(words) <= 2:
        return True
    # Block pure general knowledge requests first
    if any(kw in ql for kw in _GENERAL_KNOWLEDGE_KW):
        return True
    # Contact/location lookups are always policy-grounded — let them through
    if any(sig in ql for sig in _CONTACT_BYPASS_KW):
        return False
    return not any(kw in ql for kw in _COMPLIANCE_KW)


# ── Response renderer ─────────────────────────────────────────────────────────

def render_response(result: dict):
    if result.get("blocked"):
        st.error(result["response"])
        return

    risk = result.get("risk_level", "Unknown")
    sev  = result.get("severity",   {})
    conf = result.get("confidence", {})

    # ── Answer text (always visible) ──────────────────────────────────────────
    answer = result.get("answer") or result.get("response", "")
    st.markdown(answer)

    # ── Compact single-line status strip ──────────────────────────────────────
    risk_class = {"High": "r-high", "Medium": "r-medium", "Low": "r-low"}.get(risk, "r-na")
    sev_v      = sev.get("score", 0)
    sev_color  = RISK_COLOR.get(sev.get("band", "Unknown"), "#6b7280")
    conf_v     = conf.get("score", 0)
    sess       = st.session_state.get("session")
    turn       = sess.turn_count if sess else "—"

    st.markdown(
        f"<div style='margin:8px 0 4px;font-size:12px;color:#3d4f6b;'>"
        f"<span class='{risk_class}' style='font-size:12px;'>⬤ {risk}</span>"
        f"<span style='color:#2d3d55;'>"
        f" &nbsp;·&nbsp; Severity <b style='color:{sev_color};'>{sev_v}</b>/100"
        f" &nbsp;·&nbsp; Confidence <b style='color:#6366f1;'>{conf_v}</b>/100"
        f" &nbsp;·&nbsp; ⚡ {result.get('latency', 0)}s"
        f" &nbsp;·&nbsp; 🔤 {result.get('tokens', 0)} tok"
        f" &nbsp;·&nbsp; Turn {turn}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # ── Single collapsible details section ────────────────────────────────────
    citations = result.get("citations", [])
    cv        = result.get("citation_verification") or {}
    conflicts = result.get("conflicts", [])
    has_detail = citations or conflicts or result.get("reasoning") or sev.get("components")

    if has_detail:
        cit_label = f"{len(citations)} citation{'s' if len(citations) != 1 else ''}" if citations else ""
        cfl_label = f"{len(conflicts)} conflict{'s' if len(conflicts) != 1 else ''}" if conflicts else ""
        badge     = "  ·  ".join(filter(None, [cit_label, cfl_label])) or "details"
        with st.expander(f"📎 {badge}", expanded=False):

            # Citations
            if citations:
                st.markdown(
                    f"**📌 Citations ({len(citations)})**"
                    f"<span style='font-size:12px;color:#475569;'>"
                    f"  ·  Groundedness: {cv.get('groundedness', 'N/A')}</span>",
                    unsafe_allow_html=True,
                )
                for c in citations:
                    st.markdown(f'<div class="cite-box">{c}</div>', unsafe_allow_html=True)

            # Policy conflicts
            if conflicts:
                st.markdown("")
            for c in conflicts:
                st.markdown(
                    f'<div class="conflict-box">'
                    f'⚠️ <strong>[{c["id"]}] {c["name"]}</strong>  —  Severity: {c["severity"]}<br>'
                    f'<code style="font-size:12px;">{c["policy_a"]}</code> '
                    f'↔ <code style="font-size:12px;">{c["policy_b"]}</code><br>'
                    f'<span style="font-size:12.5px;">{c["description"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Reasoning
            if result.get("reasoning"):
                st.markdown("**🧠 Reasoning**")
                st.markdown(result["reasoning"])


    if result.get("audit_id"):
        st.caption(f"Audit ID: `{result['audit_id']}`")


# ── Load pipeline ─────────────────────────────────────────────────────────────

def load_pipeline(api_key: str, uploaded_files=None, company_name: str = ""):
    with st.spinner("Indexing policy documents into ChromaDB…"):
        file_data = [(f.read(), f.name) for f in uploaded_files]
        chunks, lookup, collection, validations = ingest_documents(file_data, api_key)
        for v in validations:
            if not v["is_policy"]:
                st.warning(
                    f"⚠️ **{v['filename']}** doesn't look like a policy document "
                    f"(signals: {v['score']}/{v['total_signals']}). "
                    "SAGE may not answer compliance questions from it accurately."
                )

        corpus_text   = "\n\n".join(c["text"] for c in chunks)
        org_type      = detect_org_type(corpus_text)
        system_prompt = build_system_prompt(corpus_text, company_name or "your organization", org_type=org_type)
        pipeline      = SAGEPipeline(
            api_key=api_key,
            system_prompt=system_prompt,
            section_lookup=lookup,
            collection=collection,
            chunks=chunks,
            conflict_rules=None,
            company_name=company_name or "your organization",
        )

        st.session_state.update({
            "pipeline":         pipeline,
            "session":          SAGEConversationSession(),
            "chat_history":     [],
            "corpus_loaded":    True,
            "company_name":     company_name or "your organization",
            "org_type":         org_type,
            "audit_logger":     pipeline.audit_logger,
            "pending_question": "",
        })

    return len(chunks), len(lookup), "ChromaDB (semantic)" if collection else "Keyword fallback"


# ── Report renderer ───────────────────────────────────────────────────────────

def render_report():
    stats = st.session_state.audit_logger.stats()

    if stats.get("total", 0) == 0:
        st.info("No queries recorded in this session yet.")
        return

    st.markdown("**Session Report**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Queries",  stats["total"])
    c2.metric("Avg Latency",    f"{stats.get('avg_latency','—')}s")
    c3.metric("Avg Confidence", f"{stats.get('avg_confidence','—')}/100")

    rdb = stats.get("risk_dist", {})
    if rdb:
        st.markdown("**Risk distribution**")
        icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}
        rc = st.columns(len(rdb))
        for i, (k, v) in enumerate(rdb.items()):
            rc[i].metric(f"{icons.get(k,'⚪')} {k}", v)

    with st.expander("📋 Recent audit entries"):
        for e in reversed(st.session_state.audit_logger.recent(10)):
            st.markdown(
                f"`{e['entry_id']}` · {e['timestamp'][:19]} · "
                f"Risk: **{e['risk_level']}** · Conf: {e.get('confidence_score') or '—'}/100  \n"
                f"> {e['query'][:80]}…"
            )

    # ── Download buttons ──────────────────────────────────────────────────────
    st.markdown("**Download**")
    dl1, dl2 = st.columns(2)

    # JSON report
    all_entries = st.session_state.audit_logger.recent(500)
    report_payload = {
        "organization": st.session_state.get("company_name", ""),
        "org_type":     st.session_state.get("org_type", "generic"),
        "stats":        stats,
        "audit_log":    all_entries,
    }
    _n = _report_render_n[0]
    _report_render_n[0] += 1

    dl1.download_button(
        "⬇️ Full Report (JSON)",
        data=json.dumps(report_payload, indent=2, default=str),
        file_name="sage_report.json",
        mime="application/json",
        use_container_width=True,
        key=f"dl_report_json_{_n}",
    )

    # CSV audit log
    if all_entries:
        csv_buf = io.StringIO()
        writer  = csv.DictWriter(csv_buf, fieldnames=list(all_entries[0].keys()))
        writer.writeheader()
        writer.writerows(all_entries)
        dl2.download_button(
            "⬇️ Audit Log (CSV)",
            data=csv_buf.getvalue(),
            file_name="sage_audit.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_audit_csv_{_n}",
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:18px 2px 14px 2px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:26px;line-height:1;filter:drop-shadow(0 0 8px rgba(99,102,241,0.5));">🛡️</span>
    <div>
      <div style="font-size:15px;font-weight:700;color:#e2e8f0;line-height:1.2;">SAGE</div>
      <div style="font-size:10.5px;color:#475569;margin-top:1px;">Secure AI Governance Engine</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.corpus_loaded:
        co = st.session_state.company_name[:24]
        st.markdown(f'<span class="pill pill-ready">● Ready &nbsp;·&nbsp; {co}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-setup">◐ Setup required</span>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section"><div class="sb-label">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "api_key", type="password",
        value=st.session_state.api_key,
        placeholder="sk-…",
        label_visibility="collapsed",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Policy Documents ──────────────────────────────────────────────────────
    st.markdown('<div class="sb-section"><div class="sb-label">📄 Policy Documents</div>', unsafe_allow_html=True)

    company_name_input = st.text_input(
        "org", placeholder="Organization name…",
        value="", label_visibility="collapsed",
    )
    uploaded_files = st.file_uploader(
        "files", type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("🚀 Load & Index", use_container_width=True, type="primary",
                     disabled=not st.session_state.api_key):
            try:
                nc, ns, _ = load_pipeline(
                    st.session_state.api_key,
                    uploaded_files=uploaded_files,
                    company_name=company_name_input,
                )
                st.session_state["_load_msg"] = f"✅ {nc} chunks · {ns} sections"
                st.rerun()
            except Exception as exc:
                st.error(f"Load failed: {exc}")
    else:
        demo_options = {"": "Try a demo org…"} | {
            k: f"{v['icon']} {v['name']}" for k, v in DEMO_ORGANIZATIONS.items()
        }
        selected_demo = st.selectbox(
            "demo", options=list(demo_options.keys()),
            format_func=lambda k: demo_options[k],
            label_visibility="collapsed",
        )
        if selected_demo:
            if st.button("🎯 Load Demo", use_container_width=True, type="primary",
                         disabled=not st.session_state.api_key):
                try:
                    org = DEMO_ORGANIZATIONS[selected_demo]
                    with st.spinner(f"Loading {org['name']}…"):
                        chunks, lookup, collection, _ = load_demo(selected_demo, st.session_state.api_key)
                        corpus_text   = "\n\n".join(c["text"] for c in chunks)
                        org_type      = detect_org_type(corpus_text)
                        system_prompt = build_system_prompt(
                            corpus_text, org["name"], org_type=org_type
                        )
                        pipeline = SAGEPipeline(
                            api_key=st.session_state.api_key,
                            system_prompt=system_prompt,
                            section_lookup=lookup,
                            collection=collection,
                            chunks=chunks,
                            conflict_rules=None,
                            company_name=org["name"],
                        )
                        st.session_state.update({
                            "pipeline":         pipeline,
                            "session":          SAGEConversationSession(),
                            "chat_history":     [],
                            "corpus_loaded":    True,
                            "company_name":     org["name"],
                            "org_type":         org_type,
                            "audit_logger":     pipeline.audit_logger,
                            "pending_question": "",
                            "_load_msg":        f"✅ {org['name']} loaded",
                        })
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")
        else:
            st.button("🚀 Load & Index", use_container_width=True, type="primary", disabled=True)

    if st.session_state.get("_load_msg"):
        st.success(st.session_state["_load_msg"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section"><div class="sb-label">⚙️ Settings</div>', unsafe_allow_html=True)
    st.session_state.use_agent = st.toggle(
        "LangGraph ReAct Agent",
        value=st.session_state.use_agent,
        help="Multi-step reasoning agent. Off = faster direct mode.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Actions (visible after loading) ───────────────────────────────────────
    if st.session_state.corpus_loaded:
        ca, cb = st.columns(2, gap="small")
        with ca:
            if st.button("📋 Audit Log", use_container_width=True):
                st.session_state.show_audit = not st.session_state.show_audit
                st.rerun()
        with cb:
            if st.button("✦ New Chat", use_container_width=True):
                _start_new_chat()


# ── Main: page header ─────────────────────────────────────────────────────────

company = st.session_state.get("company_name", "")
if st.session_state.corpus_loaded:
    pill_html = f'<span class="pill pill-ready">● Live &nbsp;·&nbsp; {company[:20]}</span>'
else:
    pill_html = '<span class="pill pill-setup">◐ Setup</span>'

st.markdown(f"""
<div style="
  text-align:center;
  background:linear-gradient(135deg,#0c1426 0%,#131c3a 50%,#0c1426 100%);
  border:1px solid rgba(99,102,241,0.16);
  border-radius:14px; padding:16px 24px;
  box-shadow:0 4px 24px rgba(99,102,241,0.07);
  margin-bottom:12px;
">
  <div style="display:inline-flex;align-items:center;gap:12px;justify-content:center;">
    <span style="font-size:28px;filter:drop-shadow(0 0 10px rgba(99,102,241,0.5));line-height:1;">🛡️</span>
    <div style="text-align:left;">
      <div style="font-size:18px;font-weight:700;color:#f1f5f9;letter-spacing:-0.2px;line-height:1.2;">
        SAGE — Compliance Assistant
      </div>
      <div style="font-size:11.5px;color:#64748b;margin-top:3px;">
        Enterprise compliance reasoning &nbsp;·&nbsp; {pill_html}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# New Chat button — separate row, right-aligned, only when chat has content
if st.session_state.corpus_loaded and st.session_state.chat_history:
    _, col_nc = st.columns([5, 1])
    with col_nc:
        if st.button("✦ New Chat", type="primary", use_container_width=True):
            _start_new_chat()


# ── Welcome screen (no corpus loaded) ────────────────────────────────────────

if not st.session_state.corpus_loaded:
    st.markdown("""
<div style="text-align:center;padding:18px 0 14px;">
  <div style="font-size:44px;filter:drop-shadow(0 0 22px rgba(99,102,241,0.5));
              margin-bottom:8px;line-height:1;">🛡️</div>
  <div style="font-size:19px;font-weight:700;color:#e2e8f0;margin-bottom:5px;
              letter-spacing:-0.2px;">
    Secure AI Governance Engine
  </div>
  <div style="font-size:12.5px;color:#64748b;max-width:480px;margin:0 auto;line-height:1.65;">
    Policy-grounded compliance answers with citations, risk scores, and audit trails —
    for any organization, from startups to hospitals to schools.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Feature row ───────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3, gap="small")
    with fc1:
        st.markdown("""
<div class="feat-card">
  <h4>📄 Any Organization</h4>
  <p>Upload your own PDF or TXT policy files. SAGE indexes them and answers questions grounded exclusively in your documents.</p>
</div>""", unsafe_allow_html=True)
    with fc2:
        st.markdown("""
<div class="feat-card">
  <h4>🔍 Semantic RAG</h4>
  <p>ChromaDB + OpenAI embeddings retrieves the most relevant policy sections — 80% fewer tokens vs. full-corpus injection.</p>
</div>""", unsafe_allow_html=True)
    with fc3:
        st.markdown("""
<div class="feat-card">
  <h4>📊 Audit &amp; Download</h4>
  <p>Every query is logged with risk score, citations, and reasoning. Export full reports as JSON or CSV for compliance review.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.11);
            border-radius:10px;padding:10px 20px;text-align:center;margin-top:10px;">
  <div style="font-size:10px;color:#3d4f6b;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:5px;font-weight:700;">6-Layer Pipeline</div>
  <div style="font-size:12.5px;color:#6366f1;font-weight:500;">
    Security &nbsp;→&nbsp; Intent Router &nbsp;→&nbsp; RAG &nbsp;→&nbsp;
    LangGraph Agent &nbsp;→&nbsp; Citation Verifier &nbsp;→&nbsp; Scoring &amp; Audit
  </div>
</div>
<div style="text-align:center;margin-top:14px;font-size:12.5px;color:#6366f1;font-weight:600;">
  👈 Enter your OpenAI API key and click <b>Load &amp; Index Documents</b> to start
</div>
""", unsafe_allow_html=True)

    st.stop()


# ── Chat history ──────────────────────────────────────────────────────────────

for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["content"])
    elif entry.get("type") == "report":
        with st.chat_message("assistant", avatar="🛡️"):
            render_report()
    else:
        with st.chat_message("assistant", avatar="🛡️"):
            render_response(entry.get("result", {"response": entry["content"]}))


# ── Quick question chips (only when chat is empty) ────────────────────────────

if not st.session_state.chat_history:
    quick_qs = QUICK_QUESTIONS

    st.markdown("""
<div style="font-size:10.5px;font-weight:700;color:#3d4f6b;
            text-transform:uppercase;letter-spacing:0.9px;
            margin:20px 0 10px;">
  ✦ Quick Questions &nbsp;—&nbsp; click to ask instantly
</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="medium")
    for i, qtext in enumerate(quick_qs):
        with (col_a if i % 2 == 0 else col_b):
            if st.button(f"💬  {qtext}", key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_question = qtext
                st.rerun()

    st.markdown("""
<div style="
  text-align:center; margin-top:18px; padding:11px 16px;
  background:rgba(99,102,241,0.04);
  border:1px solid rgba(99,102,241,0.1);
  border-radius:9px;
">
  <span style="font-size:12.5px;color:#475569;">
    Or type your own question below
    &nbsp;·&nbsp;
    Ask for a <b style="color:#6366f1;">report</b> to see session analytics
  </span>
</div>
""", unsafe_allow_html=True)


# ── Pending question bridge (chip → pipeline) ─────────────────────────────────

pending = st.session_state.get("pending_question", "")
if pending:
    st.session_state.pending_question = ""   # clear before any rerun


# ── Chat input ────────────────────────────────────────────────────────────────

typed  = st.chat_input("Ask a compliance question…")
query  = pending or (typed.strip() if typed else "")

if query:
    q        = query.strip()
    st.session_state["_load_msg"] = ""   # dismiss load banner once user starts chatting
    pipeline: SAGEPipeline            = st.session_state.pipeline
    session:  SAGEConversationSession = st.session_state.session

    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant", avatar="🛡️"):
        if _is_report_request(q):
            render_report()
            st.session_state.chat_history.append({"role": "assistant", "type": "report"})

        elif is_injection(q) or is_injection(sanitize_query(q)):
            blocked_msg = (
                "I'm a compliance assistant — I can only answer questions grounded "
                "in your organization's policy documents. Try asking about specific "
                "rules, requirements, whether an activity is permitted, or what "
                "happens in a particular scenario."
            )
            st.warning(blocked_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": blocked_msg,
                                                   "result": {"blocked": True, "response": blocked_msg}})

        elif _is_out_of_scope(q):
            msg = (
                "I'm a compliance assistant — I can only answer questions grounded "
                "in your organization's policy documents. Try asking about specific "
                "rules, requirements, whether an activity is permitted, or what "
                "happens in a particular scenario."
            )
            st.warning(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

        else:
            with st.spinner("Analyzing your policy documents…"):
                result = pipeline.query(
                    user_query=q,
                    session=session,
                    use_agent=st.session_state.use_agent,
                )
            render_response(result)
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": result.get("response", ""),
                "result":  result,
            })

    st.rerun()


# ── Audit log panel ───────────────────────────────────────────────────────────

if st.session_state.show_audit:
    st.markdown("---")
    st.markdown("### 📋 Full Audit Log")
    for e in reversed(st.session_state.audit_logger.recent(20)):
        with st.expander(
            f"`{e['entry_id']}` · {e['timestamp'][:19]} · "
            f"Risk: {e['risk_level']} · Conf: {e.get('confidence_score') or '—'}"
        ):
            st.json(e)
