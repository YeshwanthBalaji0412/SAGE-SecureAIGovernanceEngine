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
    FeedbackCollector,
)
from sage.rag import ingest_documents
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
.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 5rem !important;
    max-width: 960px;
}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #07091c !important;
    border-right: 1px solid rgba(99,102,241,0.12) !important;
}
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background: #07091c !important;
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
    min-height: 120px;
    height: 100%;
}
.feat-card h4 { color: #e2e8f0; margin: 0 0 8px; font-size: 15px; font-weight: 600; }
.feat-card p  { color: #8892a4; margin: 0; font-size: 12.5px; line-height: 1.6; }

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
        "company_name":        "TechNova Inc.",
        "corpus_loaded":       False,
        "pipeline":            None,
        "session":             None,
        "chat_history":        [],
        "audit_logger":        AuditLogger(),
        "feedback_collector":  FeedbackCollector(),
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

# Exact-match phrases that mean the user wants the session report dashboard.
# Single words like "report" or "audit" only trigger when the WHOLE query is
# that word (or a short phrase) — not when they appear inside a policy question.
_REPORT_EXACT = {"report", "audit", "stats", "statistics", "summary", "feedback"}
_REPORT_PHRASES = {"show report", "session report", "show stats", "show summary",
                   "show audit", "view report", "get report", "my report",
                   "download report", "session stats"}

# Domain-specific compliance terms only — no generic question words (who/what/when)
# so that casual/nonsensical queries like "what am i doing to you" are caught.
_COMPLIANCE_KW = {
    "policy", "policies", "applies", "apply", "scope", "eligible", "eligib",
    "covered", "covers", "effective", "section", "rule", "regulation",
    "remote", "work", "vpn", "data", "pii", "privacy", "breach", "incident",
    "mfa", "byod", "gdpr", "encrypt", "approval", "compliance", "laptop",
    "international", "travel", "retention", "security", "access", "eea",
    "password", "training", "contractor", "employee", "mdm", "probation",
    "equipment", "reimbursement", "violation", "requirement", "report to",
    "harass", "discriminat", "termination", "disciplin", "conduct", "leave",
    "benefit", "insurance", "overtime", "exempt", "salary", "pto", "holiday",
    "handbook", "procedure", "guideline", "prohibited", "mandatory", "must",
    "shall", "authorized", "permit", "waiver", "exception", "consent",
}

def _is_report_request(q: str) -> bool:
    ql = q.strip().lower()
    # Trigger only when the entire query IS a report keyword (e.g. user types just "report")
    # OR matches a known multi-word report phrase — never when "report" appears mid-sentence.
    if ql in _REPORT_EXACT:
        return True
    return any(phrase in ql for phrase in _REPORT_PHRASES)

def _is_out_of_scope(q: str) -> bool:
    words = q.lower().split()
    if len(words) <= 2:
        return True
    return not any(kw in q.lower() for kw in _COMPLIANCE_KW)


# ── Response renderer ─────────────────────────────────────────────────────────

def render_response(result: dict):
    if result.get("blocked"):
        st.error(result["response"])
        return

    risk = result.get("risk_level", "Unknown")
    sev  = result.get("severity",   {})
    conf = result.get("confidence", {})

    # Answer text
    answer = result.get("answer") or result.get("response", "")
    st.markdown(answer)

    # Score / metadata strip
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 2])

    risk_class = {"High": "r-high", "Medium": "r-medium", "Low": "r-low"}.get(risk, "r-na")
    c1.markdown(
        f"<div style='font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:4px;'>Risk Level</div>"
        f"<span class='{risk_class}' style='font-size:15px;'>{risk}</span>",
        unsafe_allow_html=True,
    )

    sev_v     = sev.get("score", 0)
    sev_color = RISK_COLOR.get(sev.get("band", "Unknown"), "#6b7280")
    c2.markdown(
        f"<div style='font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:4px;'>Severity</div>"
        f"<span style='font-size:14px;font-weight:700;color:{sev_color};'>{sev_v}</span>"
        f"<span style='font-size:11px;color:#475569;'>/100</span>"
        f"{_score_bar(sev_v, sev_color)}",
        unsafe_allow_html=True,
    )

    conf_v = conf.get("score", 0)
    c3.markdown(
        f"<div style='font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:4px;'>Confidence</div>"
        f"<span style='font-size:14px;font-weight:700;color:#818cf8;'>{conf_v}</span>"
        f"<span style='font-size:11px;color:#475569;'>/100</span>"
        f"{_score_bar(conf_v, '#6366f1')}",
        unsafe_allow_html=True,
    )

    sess = st.session_state.get("session")
    turn = sess.turn_count if sess else "—"
    c4.markdown(
        f"<div style='font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:6px;'>Metrics</div>"
        f"<span style='font-size:12px;color:#64748b;'>"
        f"⚡ {result.get('latency',0)}s &nbsp;·&nbsp; "
        f"🔤 {result.get('tokens',0)} tok &nbsp;·&nbsp; "
        f"Turn {turn}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

    # Citations
    citations = result.get("citations", [])
    cv        = result.get("citation_verification") or {}
    if citations:
        with st.expander(
            f"📌 Citations ({len(citations)})  ·  "
            f"Groundedness: {cv.get('groundedness', 'N/A')}"
        ):
            for c in citations:
                st.markdown(f'<div class="cite-box">{c}</div>', unsafe_allow_html=True)

    # Policy conflicts
    for c in result.get("conflicts", []):
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
        with st.expander("🧠 Reasoning chain"):
            st.markdown(result["reasoning"])

    # Score breakdown
    if sev.get("components"):
        with st.expander("📊 Score breakdown"):
            b1, b2 = st.columns(2, gap="large")
            b1.markdown("**Severity components**")
            for k, v in sev["components"].items():
                b1.markdown(f"- `{k}`: +{v}")
            b2.markdown("**Confidence components**")
            for k, v in (conf.get("components") or {}).items():
                b2.markdown(f"- `{k}`: {v:+d}")

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
        )

        st.session_state.update({
            "pipeline":           pipeline,
            "session":            SAGEConversationSession(),
            "chat_history":       [],
            "corpus_loaded":      True,
            "company_name":       company_name or "your organization",
            "audit_logger":       pipeline.audit_logger,
            "feedback_collector": pipeline.feedback_collector,
            "pending_question":   "",
        })

    return len(chunks), len(lookup), "ChromaDB (semantic)" if collection else "Keyword fallback"


# ── Report renderer ───────────────────────────────────────────────────────────

def render_report():
    stats = st.session_state.audit_logger.stats()
    fb    = st.session_state.feedback_collector.aggregate()

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

    if fb.get("total", 0):
        st.markdown("**Feedback summary**")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Responses Rated", fb["total"])
        fc2.metric("Overall Score",   f"{fb['overall_avg']}/5")
        fc3.metric("Recommend Rate",  fb["recommend_rate"])
        for dim, score in fb.get("dim_avgs", {}).items():
            st.progress(int(score * 20), text=f"{dim.capitalize()}: {score}/5")
    else:
        st.caption("No feedback submitted yet.")

    with st.expander("📋 Recent audit entries"):
        for e in reversed(st.session_state.audit_logger.recent(10)):
            st.markdown(
                f"`{e['entry_id']}` · {e['timestamp'][:19]} · "
                f"Risk: **{e['risk_level']}** · Conf: {e.get('confidence_score','—')}/100  \n"
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
        "feedback":     fb,
        "audit_log":    all_entries,
    }
    dl1.download_button(
        "⬇️ Full Report (JSON)",
        data=json.dumps(report_payload, indent=2, default=str),
        file_name="sage_report.json",
        mime="application/json",
        use_container_width=True,
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
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand header
    st.markdown("""
<div style="padding:14px 0 10px;">
  <div style="font-size:28px;filter:drop-shadow(0 0 10px rgba(99,102,241,0.5));">🛡️</div>
  <div style="font-size:17px;font-weight:700;color:#e2e8f0;margin-top:4px;letter-spacing:-0.3px;">SAGE</div>
  <div style="font-size:11.5px;color:#4b5568;margin-top:2px;">Secure AI Governance Engine</div>
</div>
""", unsafe_allow_html=True)

    # Status
    if st.session_state.corpus_loaded:
        co = st.session_state.company_name[:22]
        st.markdown(f'<span class="pill pill-ready">● Ready &nbsp;·&nbsp; {co}</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-setup">◐ Setup required</span>',
                    unsafe_allow_html=True)

    st.divider()

    # API Key
    st.markdown('<div style="font-size:10.5px;font-weight:700;color:#3d4f6b;text-transform:uppercase;letter-spacing:0.9px;margin-bottom:6px;">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "key", type="password",
        value=st.session_state.api_key,
        placeholder="sk-…",
        label_visibility="collapsed",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.divider()

    # Policy documents
    st.markdown('<div style="font-size:10.5px;font-weight:700;color:#3d4f6b;text-transform:uppercase;letter-spacing:0.9px;margin-bottom:6px;">📄 Policy Documents</div>', unsafe_allow_html=True)

    company_name_input = st.text_input(
        "Organization name",
        value=st.session_state.get("company_name", ""),
        placeholder="e.g. Acme Corp, City Hospital, Lincoln High School",
    )
    uploaded_files = st.file_uploader(
        "Upload policy files (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload your HR, IT, data privacy, safety, or any other policy documents.",
    )

    if st.button(
        "🚀 Load & Index Documents",
        use_container_width=True, type="primary",
        disabled=not st.session_state.api_key,
    ):
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one policy document.")
        else:
            try:
                nc, ns, rag = load_pipeline(
                    st.session_state.api_key,
                    uploaded_files=uploaded_files,
                    company_name=company_name_input,
                )
                st.session_state["_load_msg"] = f"✅ {nc} chunks · {ns} sections · {rag}"
                st.rerun()
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    # Show load result message (persists after rerun so pill has time to update)
    if st.session_state.get("_load_msg"):
        st.success(st.session_state["_load_msg"])

    st.divider()

    # Settings
    st.markdown('<div style="font-size:10.5px;font-weight:700;color:#3d4f6b;text-transform:uppercase;letter-spacing:0.9px;margin-bottom:6px;">⚙️ Settings</div>', unsafe_allow_html=True)
    st.session_state.use_agent = st.toggle(
        "LangGraph ReAct Agent",
        value=st.session_state.use_agent,
        help="4-tool reasoning agent. Disable for faster direct-prompt mode.",
    )

    # Session stats + actions (only when loaded)
    if st.session_state.corpus_loaded:
        st.divider()
        stats = st.session_state.audit_logger.stats()
        if stats.get("total", 0) > 0:
            st.markdown('<div style="font-size:10.5px;font-weight:700;color:#3d4f6b;text-transform:uppercase;letter-spacing:0.9px;margin-bottom:8px;">📊 Session Stats</div>', unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:13px;color:#94a3b8;line-height:2.1;'>"
                f"Queries: <b style='color:#e2e8f0;'>{stats['total']}</b><br>"
                f"Avg latency: <b style='color:#e2e8f0;'>{stats.get('avg_latency','—')}s</b><br>"
                f"Avg confidence: <b style='color:#e2e8f0;'>{stats.get('avg_confidence','—')}/100</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
            rdb = stats.get("risk_dist", {})
            if rdb:
                badges = "".join(
                    f'<span style="display:inline-block;padding:2px 8px;border-radius:12px;'
                    f'background:rgba(99,102,241,0.1);color:#a5b4fc;'
                    f'border:1px solid rgba(99,102,241,0.22);font-size:11px;'
                    f'font-weight:600;margin:3px 3px 0 0;">{k} {v}</span>'
                    for k, v in rdb.items()
                )
                st.markdown(
                    f'<div style="display:flex;flex-wrap:wrap;margin-top:6px;">{badges}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

            # Confidence trend — extract per-turn confidence scores from chat history
            conf_scores = [
                entry["result"]["confidence"]["score"]
                for entry in st.session_state.chat_history
                if entry.get("role") == "assistant"
                and isinstance(entry.get("result"), dict)
                and isinstance(entry["result"].get("confidence"), dict)
                and entry["result"]["confidence"].get("score") is not None
            ]
            if len(conf_scores) >= 2:
                st.markdown(
                    '<div style="font-size:10.5px;font-weight:700;color:#3d4f6b;'
                    'text-transform:uppercase;letter-spacing:0.9px;margin-bottom:4px;">'
                    '📈 Confidence Trend</div>',
                    unsafe_allow_html=True,
                )
                st.line_chart(
                    {"Confidence": conf_scores},
                    height=80,
                    use_container_width=True,
                )
                st.markdown("")

        ca, cb = st.columns(2, gap="small")
        with ca:
            if st.button("📋 Audit Log", use_container_width=True):
                st.session_state.show_audit = not st.session_state.show_audit
                st.rerun()
        with cb:
            if st.button("✦ New Chat", use_container_width=True):
                st.session_state.chat_history     = []
                st.session_state.session          = SAGEConversationSession()
                st.session_state.pending_question = ""
                st.rerun()

    st.divider()
    st.caption("SAGE · INFO7375 Final Project · Group 1")


# ── Main: page header ─────────────────────────────────────────────────────────

company = st.session_state.get("company_name", "")
if st.session_state.corpus_loaded:
    pill_html = f'<span class="pill pill-ready">● Live &nbsp;·&nbsp; {company[:20]}</span>'
else:
    pill_html = '<span class="pill pill-setup">◐ Setup</span>'

st.markdown(f"""
<div style="
  display:flex; align-items:center; gap:16px;
  background:linear-gradient(135deg,#0c1426 0%,#131c3a 50%,#0c1426 100%);
  border:1px solid rgba(99,102,241,0.16);
  border-radius:14px; padding:18px 24px;
  box-shadow:0 4px 24px rgba(99,102,241,0.07);
  margin-bottom:10px;
">
  <span style="font-size:32px;flex-shrink:0;
               filter:drop-shadow(0 0 12px rgba(99,102,241,0.5));">🛡️</span>
  <div style="flex:1;min-width:0;">
    <div style="font-size:19px;font-weight:700;color:#f1f5f9;letter-spacing:-0.3px;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
      SAGE — Compliance Assistant
    </div>
    <div style="font-size:12px;color:#64748b;margin-top:3px;">
      Enterprise compliance reasoning &nbsp;·&nbsp; {pill_html}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# New Chat button — separate row, right-aligned, only when chat has content
if st.session_state.corpus_loaded and st.session_state.chat_history:
    _, col_nc = st.columns([5, 1])
    with col_nc:
        if st.button("✦ New Chat", type="primary", use_container_width=True):
            st.session_state.chat_history     = []
            st.session_state.session          = SAGEConversationSession()
            st.session_state.pending_question = ""
            st.rerun()


# ── Welcome screen (no corpus loaded) ────────────────────────────────────────

if not st.session_state.corpus_loaded:
    st.markdown("""
<div style="text-align:center;padding:24px 0 16px;">
  <div style="font-size:48px;filter:drop-shadow(0 0 24px rgba(99,102,241,0.5));
              margin-bottom:10px;">🛡️</div>
  <div style="font-size:20px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">
    Secure AI Governance Engine
  </div>
  <div style="font-size:13px;color:#64748b;max-width:500px;margin:0 auto;line-height:1.65;">
    Policy-grounded compliance answers with citations, risk scores, and audit trails —
    for any organization, from startups to hospitals to schools.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Feature row ───────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3, gap="medium")
    with fc1:
        st.markdown("""
<div class="feat-card">
  <h4>📄 Any Organization</h4>
  <p>Upload your own PDF or TXT policy files. SAGE indexes them and answers
  questions grounded exclusively in your documents.</p>
</div>""", unsafe_allow_html=True)
    with fc2:
        st.markdown("""
<div class="feat-card">
  <h4>🔍 Semantic RAG</h4>
  <p>ChromaDB + OpenAI embeddings retrieves the most relevant policy sections —
  80% fewer tokens vs. full-corpus injection.</p>
</div>""", unsafe_allow_html=True)
    with fc3:
        st.markdown("""
<div class="feat-card">
  <h4>📊 Audit &amp; Download</h4>
  <p>Every query is logged with risk score, citations, and reasoning.
  Export full reports as JSON or CSV for compliance review.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.11);
            border-radius:10px;padding:12px 20px;text-align:center;margin-top:12px;">
  <div style="font-size:10.5px;color:#3d4f6b;text-transform:uppercase;
              letter-spacing:0.9px;margin-bottom:6px;font-weight:700;">6-Layer Pipeline</div>
  <div style="font-size:13px;color:#6366f1;font-weight:500;">
    Security &nbsp;→&nbsp; Intent Router &nbsp;→&nbsp; RAG &nbsp;→&nbsp;
    LangGraph Agent &nbsp;→&nbsp; Citation Verifier &nbsp;→&nbsp; Scoring &amp; Audit
  </div>
</div>
<div style="text-align:center;margin-top:18px;font-size:13px;color:#6366f1;font-weight:600;">
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

        elif _is_out_of_scope(q):
            msg = (
                "I'm a compliance assistant — I can only answer questions about your "
                "company's policies. Try asking about remote work, data privacy, "
                "security requirements, or BYOD rules."
            )
            st.markdown(msg)
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
            f"Risk: {e['risk_level']} · Conf: {e.get('confidence_score','—')}"
        ):
            st.json(e)
