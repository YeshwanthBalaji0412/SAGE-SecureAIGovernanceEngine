"""
SAGE — Secure AI Governance Engine
Real-time local Streamlit application.

Run:
    cd app
    streamlit run app.py
"""
from __future__ import annotations

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
    BUILTIN_CONFLICT_RULES,
)
from sage.rag import ingest_documents, load_technova_demo
from sage.prompts import build_system_prompt

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SAGE — Compliance Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.risk-high    { color:#ef4444; font-weight:700; }
.risk-medium  { color:#f59e0b; font-weight:700; }
.risk-low     { color:#22c55e; font-weight:700; }
.risk-na      { color:#6b7280; font-weight:700; }
.cite-box     { background:#1e293b; border-left:3px solid #3b82f6;
                padding:10px 14px; border-radius:6px; margin:6px 0;
                font-size:13px; }
.conflict-box { background:#1c0a00; border:1px solid #f59e0b;
                padding:10px 14px; border-radius:8px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defs: dict = {
        "api_key":           os.environ.get("OPENAI_API_KEY", ""),
        "company_name":      "TechNova Inc.",
        "corpus_loaded":     False,
        "pipeline":          None,
        "session":           None,
        "chat_history":      [],
        "audit_logger":      AuditLogger(),
        "feedback_collector":FeedbackCollector(),
        "show_audit":        False,
        "use_agent":         True,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

RISK_COLOR = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e",
              "Critical": "#7c3aed", "N/A": "#6b7280", "Unknown": "#6b7280"}


# ── Response renderer ─────────────────────────────────────────────────────────

def render_response(result: dict):
    """Render one SAGE response with all cards."""
    if result.get("blocked"):
        st.error(result["response"])
        return

    risk  = result.get("risk_level", "Unknown")
    sev   = result.get("severity",   {})
    conf  = result.get("confidence", {})

    # ── Answer ────────────────────────────────────────────────────────────────
    answer = result.get("answer") or result.get("response", "")
    st.markdown(answer)

    # ── Score strip ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    risk_css = {"High": "risk-high", "Medium": "risk-medium",
                "Low": "risk-low"}.get(risk, "risk-na")
    c1.markdown(f"**Risk Level**\n<span class='{risk_css}'>{risk}</span>",
                unsafe_allow_html=True)

    sev_v = sev.get("score", 0)
    sev_color = RISK_COLOR.get(sev.get("band", "Unknown"), "#6b7280")
    c2.markdown(f"**Severity** `{sev_v}/100`")
    c2.markdown(
        f'<div style="background:#1e2535;border-radius:6px;height:8px;">'
        f'<div style="width:{sev_v}%;height:8px;border-radius:6px;background:{sev_color}"></div>'
        f'</div>', unsafe_allow_html=True)

    conf_v = conf.get("score", 0)
    c3.markdown(f"**Confidence** `{conf_v}/100`")
    c3.markdown(
        f'<div style="background:#1e2535;border-radius:6px;height:8px;">'
        f'<div style="width:{conf_v}%;height:8px;border-radius:6px;background:#3b82f6"></div>'
        f'</div>', unsafe_allow_html=True)

    sess: SAGEConversationSession | None = st.session_state.get("session")
    turn = sess.turn_count if sess else "—"
    c4.caption(
        f"⚡ `{result.get('latency', 0)}s` · "
        f"🔤 `{result.get('tokens', 0)} tokens` · "
        f"Turn `{turn}`"
    )

    # ── Citations ─────────────────────────────────────────────────────────────
    citations = result.get("citations", [])
    cv        = result.get("citation_verification") or {}
    if citations:
        with st.expander(f"📌 Citations ({len(citations)}) — groundedness: {cv.get('groundedness','N/A')}"):
            for c in citations:
                st.markdown(f'<div class="cite-box">{c}</div>', unsafe_allow_html=True)

    # ── Policy conflicts ──────────────────────────────────────────────────────
    for c in result.get("conflicts", []):
        st.markdown(
            f'<div class="conflict-box">'
            f'⚠️ <b>[{c["id"]}] {c["name"]}</b> — Severity: {c["severity"]}<br>'
            f'<code>{c["policy_a"]}</code> ↔ <code>{c["policy_b"]}</code><br>'
            f'{c["description"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Reasoning ────────────────────────────────────────────────────────────
    reasoning = result.get("reasoning", "")
    if reasoning:
        with st.expander("🧠 Reasoning chain"):
            st.markdown(reasoning)

    # ── Severity components ───────────────────────────────────────────────────
    if sev.get("components"):
        with st.expander("📊 Score breakdown"):
            cc1, cc2 = st.columns(2)
            cc1.markdown("**Severity components**")
            for k, v in sev["components"].items():
                cc1.markdown(f"- `{k}`: +{v}")
            cc2.markdown("**Confidence components**")
            for k, v in (conf.get("components") or {}).items():
                cc2.markdown(f"- `{k}`: {v:+d}")

    # ── Audit ID ─────────────────────────────────────────────────────────────
    if result.get("audit_id"):
        st.caption(f"Audit: `{result['audit_id']}`")


# ── Load pipeline helper ──────────────────────────────────────────────────────

def load_pipeline(api_key: str, uploaded_files=None, company_name: str = "TechNova Inc."):
    with st.spinner("Indexing documents into ChromaDB…"):
        if uploaded_files:
            file_data = [(f.read(), f.name) for f in uploaded_files]
            chunks, lookup, collection, validations = ingest_documents(file_data, api_key)
            is_technova = False
            # Warn about non-policy documents
            for v in validations:
                if not v["is_policy"]:
                    st.warning(
                        f"⚠️ **{v['filename']}** doesn't look like a policy document "
                        f"(policy signals found: {v['score']}/{v['total_signals']}). "
                        "SAGE may not be able to answer compliance questions from it."
                    )
        else:
            chunks, lookup, collection, _ = load_technova_demo(api_key)
            is_technova = True

        corpus_text   = "\n\n".join(c["text"] for c in chunks)
        system_prompt = build_system_prompt(corpus_text, company_name, is_technova)

        pipeline = SAGEPipeline(
            api_key=api_key,
            system_prompt=system_prompt,
            section_lookup=lookup,
            collection=collection,
            chunks=chunks,
            conflict_rules=BUILTIN_CONFLICT_RULES if is_technova else None,
        )
        st.session_state.pipeline            = pipeline
        st.session_state.session             = SAGEConversationSession()
        st.session_state.chat_history        = []
        st.session_state.corpus_loaded       = True
        st.session_state.company_name        = company_name
        st.session_state.audit_logger        = pipeline.audit_logger
        st.session_state.feedback_collector  = pipeline.feedback_collector

    n_chunks   = len(chunks)
    n_sections = len(lookup)
    rag_method = "ChromaDB (semantic)" if collection else "Keyword fallback"
    return n_chunks, n_sections, rag_method


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ SAGE")
    st.markdown("*Secure AI Governance Engine*")
    st.divider()

    st.markdown("### 🔑 OpenAI API Key")
    api_key_input = st.text_input(
        "Enter your key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.divider()

    st.markdown("### 📄 Policy Documents")
    doc_source = st.radio(
        "Source",
        ["TechNova Demo (built-in)", "Upload my company's documents"],
        index=0,
    )

    uploaded_files     = None
    company_name_input = "TechNova Inc."

    if doc_source == "Upload my company's documents":
        company_name_input = st.text_input(
            "Company name",
            value=st.session_state.company_name,
            placeholder="Acme Corp",
        )
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT policy files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

    load_disabled = not st.session_state.api_key
    if st.button("🚀 Load & Index Documents",
                 use_container_width=True, type="primary", disabled=load_disabled):
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API key first.")
        elif doc_source == "Upload my company's documents" and not uploaded_files:
            st.warning("Please upload at least one policy document.")
        else:
            try:
                n_chunks, n_secs, rag = load_pipeline(
                    st.session_state.api_key,
                    uploaded_files=uploaded_files if doc_source == "Upload my company's documents" else None,
                    company_name=company_name_input,
                )
                st.success(f"✅ {n_chunks} chunks · {n_secs} sections · {rag}")
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    st.divider()

    st.markdown("### ⚙️ Settings")
    st.session_state.use_agent = st.toggle(
        "LangGraph ReAct Agent",
        value=st.session_state.use_agent,
        help="Uses the 4-tool reasoning agent. Disable for faster direct-prompt mode.",
    )

    st.divider()

    if st.button("📋 Toggle Audit Log", use_container_width=True):
        st.session_state.show_audit = not st.session_state.show_audit

    if st.session_state.corpus_loaded:
        stats = st.session_state.audit_logger.stats()
        if stats.get("total", 0) > 0:
            st.markdown(f"""
**Session stats**
- Queries: `{stats['total']}`
- Avg latency: `{stats.get('avg_latency', '—')}s`
- Avg confidence: `{stats.get('avg_confidence', '—')}/100`
""")
            rdb = stats.get("risk_dist", {})
            if rdb:
                st.markdown(" · ".join(f"`{k}:{v}`" for k, v in rdb.items()))

    if st.session_state.corpus_loaded:
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.session      = SAGEConversationSession()
            st.rerun()

    st.divider()
    st.caption("SAGE · INFO7375 Final Project · Group 1")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🛡️ SAGE — Secure AI Governance Engine")
st.caption(f"Enterprise compliance reasoning · {st.session_state.get('company_name','')}")

# Welcome screen
if not st.session_state.corpus_loaded:
    st.info("👈 Enter your OpenAI API key in the sidebar, then click **Load & Index Documents** to start.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
#### 📄 Any Company
Upload your own PDF or TXT policy files. SAGE indexes them automatically and answers questions grounded exclusively in your documents.
""")
    with col2:
        st.markdown("""
#### 🔍 Semantic RAG
ChromaDB + `text-embedding-3-small` retrieves the 5 most relevant policy sections per query — 80% token reduction vs. full-corpus injection.
""")
    with col3:
        st.markdown("""
#### 🔒 Injection-Safe
Multi-layer security blocks prompt injection, persona-switching, and obfuscation attacks before they reach the LLM.
""")

    st.markdown("---")
    st.markdown("""
**6-Layer Pipeline**
`Security` → `Intent Router` → `RAG` → `LangGraph Agent` → `Citation Verifier` → `Scoring & Audit`

**6 Production Features**
Multi-turn memory · Confidence scoring · Policy conflict detection ·
Citation verification · Audit logging · Severity-weighted risk scoring
""")
    st.stop()


# ── Report renderer ───────────────────────────────────────────────────────────

def render_report():
    """Show audit stats and feedback summary inline in the chat."""
    stats = st.session_state.audit_logger.stats()
    fb    = st.session_state.feedback_collector.aggregate()

    if stats.get("total", 0) == 0:
        st.info("No queries recorded in this session yet.")
        return

    st.markdown("**Session Report**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Queries",   stats["total"])
    c2.metric("Avg Latency",     f"{stats.get('avg_latency','—')}s")
    c3.metric("Avg Confidence",  f"{stats.get('avg_confidence','—')}/100")

    risk_dist = stats.get("risk_dist", {})
    if risk_dist:
        st.markdown("**Risk distribution**")
        rc = st.columns(len(risk_dist))
        colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}
        for i, (k, v) in enumerate(risk_dist.items()):
            rc[i].metric(f"{colors.get(k,'⚪')} {k}", v)

    if fb.get("total", 0):
        st.markdown("**Feedback summary**")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Responses Rated", fb["total"])
        fc2.metric("Overall Score",   f"{fb['overall_avg']}/5")
        fc3.metric("Recommend Rate",  fb["recommend_rate"])
        for dim, score in fb.get("dim_avgs", {}).items():
            st.progress(int(score * 20), text=f"{dim.capitalize()}: {score}/5")
    else:
        st.caption("No feedback submitted yet. Ask SAGE to 'rate' a response to log feedback.")

    with st.expander("📋 Recent audit entries"):
        for e in reversed(st.session_state.audit_logger.recent(10)):
            st.markdown(
                f"`{e['entry_id']}` · {e['timestamp'][:19]} · "
                f"Risk: **{e['risk_level']}** · Conf: {e.get('confidence_score','—')}/100  \n"
                f"> {e['query'][:80]}…"
            )


# ── Intent helpers ────────────────────────────────────────────────────────────

_REPORT_KW = {"report", "audit", "stats", "statistics", "summary",
              "feedback", "show report", "session report"}

_COMPLIANCE_KW = {
    # Policy / document meta-questions
    "policy", "applies", "apply", "scope", "purpose", "effective", "eligib",
    "covered", "covers", "who", "when", "what",
    # Compliance topics
    "remote", "work", "home", "vpn", "data", "pii", "privacy", "breach",
    "mfa", "byod", "gdpr", "encrypt", "approval", "compliance", "laptop",
    "international", "travel", "retention", "security", "access", "eea",
    "password", "incident", "training", "contractor", "employee", "mdm",
    "probation", "equipment", "reimbursement", "violation", "requirement",
}

def _is_report_request(q: str) -> bool:
    ql = q.lower()
    return any(kw in ql for kw in _REPORT_KW)

def _is_out_of_scope(q: str) -> bool:
    ql = q.lower().split()
    # Greetings / single tokens always out of scope
    if len(ql) <= 2:
        return True
    # Partial-word match so "eligibility" matches "eligib", etc.
    q_lower = q.lower()
    has_compliance = any(kw in q_lower for kw in _COMPLIANCE_KW)
    return not has_compliance


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


# ── Query input ───────────────────────────────────────────────────────────────

query = st.chat_input("Ask a compliance question…")

if query and query.strip():
    q = query.strip()
    pipeline: SAGEPipeline            = st.session_state.pipeline
    session:  SAGEConversationSession = st.session_state.session

    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": q})

    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant", avatar="🛡️"):

        # ── Report request ────────────────────────────────────────────────────
        if _is_report_request(q):
            render_report()
            st.session_state.chat_history.append({"role": "assistant", "type": "report"})

        # ── Out of scope ──────────────────────────────────────────────────────
        elif _is_out_of_scope(q):
            msg = "I'm a compliance assistant — I can only answer questions about your company's policies. Try asking about remote work, data privacy, security requirements, or BYOD rules."
            st.markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

        # ── Compliance query ──────────────────────────────────────────────────
        else:
            with st.spinner("Reasoning…"):
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


# ── Audit log panel (sidebar toggle) ─────────────────────────────────────────

if st.session_state.show_audit:
    st.markdown("---")
    st.markdown("### 📋 Full Audit Log")
    for e in reversed(st.session_state.audit_logger.recent(20)):
        with st.expander(
            f"`{e['entry_id']}` · {e['timestamp'][:19]} · "
            f"Risk: {e['risk_level']} · Conf: {e.get('confidence_score','—')}"
        ):
            st.json(e)
