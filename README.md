# SAGE: Secure AI Governance Engine
### Enterprise AI Policy & Compliance Reasoning Agent

SAGE is an AI-powered compliance reasoning system that interprets enterprise policy documents, answers employee queries with verifiable citations, classifies compliance risk at four severity levels, and defends against adversarial prompt injection. Built on GPT-4o, LangChain, LangGraph, and ChromaDB.

---

## What This Project Does

Enterprise policy documents are dense, interconnected, and difficult to interpret accurately. Employees frequently rely on informal guidance, which can trigger regulatory penalties and security incidents. SAGE addresses this by:

- **Answering policy questions** grounded exclusively in authoritative documents — no hallucinated guidance
- **Citing every claim** with document name, section, and page number
- **Classifying compliance risk** (Low / Medium / High) for each response with explicit justification
- **Numeric severity scoring** (0–100) and **confidence scoring** (0–100) per response
- **Detecting policy conflicts** across documents via 5 named conflict rules (CF-001 through CF-005)
- **Verifying citations** against an authoritative section index for groundedness percentage
- **Maintaining multi-turn conversation memory** (6-turn rolling window) for contextual follow-ups
- **Auditing all interactions** to `sage_audit_log.json` for compliance replay
- **Collecting user feedback** (4-dimensional ratings) via `sage_feedback.json`
- **Defending against prompt injection** through multi-layered security: pattern detection, LLM-based classification, and output validation
- **Reasoning transparently** using Chain-of-Thought prompting so every answer is auditable

---

## System Architecture

```
User Query
    │
    ▼
L1: Security Layer       ← Pattern detection + LLM injection classifier
    │                       Blocks: instruction overrides, persona-switching,
    │                       base64 payloads, typoglycemia obfuscation
    ▼
L2: Intent Router        ← Few-shot classifier
    │                       Categories: risk_assessment / policy_question /
    │                       follow_up / out_of_scope
    ▼
L3: RAG Retrieval        ← ChromaDB semantic search (text-embedding-3-small)
    │                       Returns top-5 relevant policy chunks (~350 words)
    │                       vs. 1,600-word full corpus injection (80% token reduction)
    ▼
L4: Reasoning Engine     ← LangGraph ReAct Agent (GPT-4o)
    │                       Tools: search_policy · check_cross_references ·
    │                       assess_risk · detect_policy_conflicts (NEW in A6)
    ▼
L5: Validation Layer     ← Self-criticism: citation check + groundedness verification
    │                       CitationVerifier cross-checks every cited §X.X
    │                       Flags unsupported claims before response reaches user
    ▼
L6: Scoring & Audit      ← ConfidenceScorer (0–100) + SeverityScorer (0–100)
    │                       AuditLogger → sage_audit_log.json
    │                       FeedbackCollector → sage_feedback.json
    ▼
Structured Response      ← Answer · Citations · Risk Level · Severity Score ·
                           Confidence Score · Reasoning · Policy Conflicts
```

---

## Policy Corpus

Three fictional enterprise policy documents totalling ~1,600 words:

| Document | ID | Owner | Key Sections |
|----------|----|-------|--------------|
| Remote Work Policy | `POL-RW-2025` | Human Resources | Eligibility, domestic/international remote work, BYOD equipment |
| Data Privacy Policy | `POL-DP-2025` | Legal & Compliance | Data retention, cross-border transfers, breach notification, data subject rights |
| Information Security Policy | `POL-IS-2025` | Information Security | Access control, acceptable use, network security, encryption, BYOD requirements |

---

## Project Structure

```
SAGE-SecureAIGovernanceEngine/
├── SAGE_Complete.ipynb          ← Full tutorial notebook (Phases 1–3, 65 cells)
├── SAGE_Assignment_6.ipynb      ← Production-grade submission (A6, 32 cells) ← NEW
├── README.md                    ← This file
└── sage_prompt_flow/            ← Created at runtime by Step 12 of SAGE_Complete
    ├── flow.dag.yaml
    ├── compliance_check.py
    ├── variant_a.txt ... variant_e.txt
    └── test_data.jsonl
```

**Runtime artifacts created by SAGE_Assignment_6.ipynb:**
```
sage_audit_log.json              ← Audit trail of all interactions
sage_feedback.json               ← User feedback ratings & comments
sage_a6_results.json             ← Comprehensive evaluation results
```

---

## Notebook Overview

### `SAGE_Complete.ipynb` — Full Tutorial (Phases 1–3)

The original entry point covering all development phases from prompting exploration through production deployment.

**Phase 1 — Prompting Technique Exploration (Steps 4–5)**

| Technique | Key Insight |
|-----------|-------------|
| Zero-Shot | Baseline — correct policy identification but no unified risk level |
| Few-Shot | Best output format consistency via in-context behavioural examples |
| Chain-of-Thought | Highest section coverage; uniquely surfaces Sec 5.4 DPO approval |
| Step-Back | Surfaces Sec 4.4 benefits gap via categorical domain scan |
| Analogical | Best explainability; non-linear risk escalation insight |
| Auto-CoT | Scalable scaffold generation without hand-crafted steps |
| Generated Knowledge | Compound risk modelling; 2.3× token cost, justified for complex queries |
| Decomposition | 3-stage pipeline: Intent → Policy Analysis → Risk Assessment |
| Ensembling | Legal / Security / HR perspectives merged into one definitive answer |
| Self-Consistency | Diagnoses prompt instability — revealed "risk temporal ambiguity" |
| Universal Self-Consistency | Meta-review selects strongest reasoning across 3 independent runs |
| Self-Criticism | Caught Sec 5.3 citation precision error; forms the L5 Validation Layer |

**Phase 2 — System Prompt Hardening & Evaluation (Steps 6–9)**
- Sensitivity analysis: phrasing variations, temperature sweep (0.0–1.0), auto-generated paraphrases
- 8 instabilities identified and resolved in Hardened System Prompt
- 23-case evaluation dataset (8 Typical / 9 Edge / 6 Adversarial)
- RAG pipeline with 80% token reduction via ChromaDB
- Meta prompting & perplexity evaluation

**Phase 3 — Agent, Automation & Final Evaluation (Steps 10–15)**
- LangGraph ReAct Agent with 3 tools
- 5-variant prompt testing via LangChain
- Azure Prompt Flow local execution
- LLM-as-Judge final evaluation

---

### `SAGE_Assignment_6.ipynb` — Production-Grade System (A6) ← NEW

Assignment 6 extends the A5 system with six new production components and a comprehensive evaluation suite.

#### New Components

**1. Multi-Turn Conversation Memory (`SAGEConversationSession`)**
- Rolling window of 6 turns (12 messages) maintaining session context
- Enables follow-up questions without repeating context
- Each session tracked with unique ID and timestamp

**2. Numeric Confidence Scorer (0–100)**

| Component | Points |
|-----------|--------|
| Citation density | 8 pts per citation, capped at 32 |
| Risk clarity | 10 pts if High/Medium/Low classified |
| Compliance keyword coverage | Up to 8 keywords × weighting |
| Policy tension detection bonus | +5 pts |
| Ambiguity penalty | Up to −30 pts |

**3. Policy Conflict Detector (5 Named Rules)**

| Rule | Conflict |
|------|---------|
| CF-001 | Local Storage vs Remote Mobility (POL-IS-2025 §5.3 vs POL-RW-2025 §3.4) |
| CF-002 | International Work + EEA Transfer Compounding |
| CF-003 | BYOD Enrollment vs Data Prohibition |
| CF-004 | Encryption ≠ Exemption (§5.3 local storage ban applies even with encryption) |
| CF-005 | Benefit Extension Gaps |

**4. Citation Verifier**
- Cross-checks every cited section (POL-XX-2025, Section X.X) against `POLICY_SECTION_LOOKUP` index
- Returns groundedness percentage
- Flags non-existent cited sections

**5. Audit Logger**
- Persists all query-response interactions to `sage_audit_log.json`
- Records: session ID, timestamp, query, response digest, risk level, severity/confidence scores, citations, latency, token usage

**6. User Feedback Collector**
- 4-dimensional ratings: clarity, accuracy, usefulness, trust (1–5 scale)
- Optional free-text comments and recommend flag
- Linked to audit entries via `audit_id`
- Aggregation statistics persisted to `sage_feedback.json`

**7. Severity-Weighted Risk Scorer (0–100)**

| Component | Points |
|-----------|--------|
| Risk base (Low / Medium / High) | 5 / 20 / 40 |
| Extra policies triggered | +15 each (max 3) |
| International scope indicator | +15 |
| Data exposure risk | +20 |
| EEA cross-border scope | +10 |

#### Enhanced System Prompt (A6)

- Intent classification: `risk_assessment / policy_question / follow_up / out_of_scope`
- Multi-turn awareness — explicit instructions to reference prior context
- **5 Edge-Case Disambiguation Rules:**
  - `DURATION-EXACT`: "30 days" ≠ threshold trigger (§4.2 triggers at 31+ days)
  - `ENCRYPTION≠EXEMPTION`: §5.3 local storage ban applies even with encryption
  - `ELIGIBILITY-FIRST`: Check 90-day probation (§2) before advising on remote work
  - `SCOPE-CHECK`: §2 explicitly excludes contractors
  - `TEMPORAL`: Assign risk based on CURRENT state BEFORE corrective actions
- Mandatory reasoning checklist covering 5 commonly-missed policy items
- Extended response format: adds Severity Score and Confidence Score fields

#### Enhanced LangGraph ReAct Agent (A6)

4 tools with a fixed 5-step workflow:
1. `check_cross_references` → identify triggered policies
2. `detect_policy_conflicts` → surface tensions early **(NEW)**
3. `search_policy` → gather evidence per policy
4. `assess_risk` → synthesize findings
5. Final structured response with all 6 output fields

#### Evaluation Suite (A6)

- **30-case dataset**: 23 A4/A5 cases + 7 new cases covering previously uncovered sections
- **Unit tests** for all 8 components (ConfidenceScorer, PolicyConflictDetector, CitationVerifier, AuditLogger, FeedbackCollector, SeverityScorer, MultiTurnSession, EdgeCaseRules)
- **Integration tests**: 5 end-to-end pipeline runs
- **Regression tests**: A5 baseline vs. A6 enhanced across all 30 cases
- **LLM-as-Judge**: GPT-4o-mini scores on 5 dimensions (accuracy, groundedness, completeness, clarity, risk classification)
- **Iteration log**: 8 improvements documented from A1 through A6

---

## Key Results

### SAGE_Complete.ipynb (A1–A5 Baseline)

| Metric | Target | Result |
|--------|--------|--------|
| Answer Accuracy | ≥85% | Validated via 23-case dataset |
| Citation Precision | ≥90% | Zero hallucinated citations across all experiments |
| Groundedness | ≥90% | Self-criticism caught and fixed Sec 5.3 precision error |
| Risk Classification | ≥80% | 52% baseline → 75%+ after hardening |
| Injection Resistance | ≥95% | All 6 adversarial cases correctly handled |
| Output Stability | 7.9/10 | 0 risk variance across 12 phrasing/temperature runs |
| Token Efficiency | — | 80% reduction with RAG vs. full-corpus injection |

### SAGE_Assignment_6.ipynb (A6 Enhanced)

| Metric | A5 Baseline | A6 Enhanced |
|--------|-------------|-------------|
| Confidence Score (avg) | — | ~82 / 100 |
| Citation Groundedness | Manual | 100% verified (CitationVerifier) |
| Conflict Detection | None | 5/5 rules validated |
| User Satisfaction | — | 4.4 / 5.0 avg |
| Test Coverage | None | 8 test classes (unit + integration + regression) |
| Evaluation Dataset | 23 cases | 30 cases |

---

## How to Run

### Prerequisites

```bash
pip install openai langchain langchain-openai langgraph chromadb tiktoken tabulate promptflow
```

### API Key

```bash
export OPENAI_API_KEY='sk-...'
```

### Execute

**Full tutorial (A1–A5):** Open `SAGE_Complete.ipynb` in Jupyter and run cells sequentially. Each phase builds on the previous.

**Production system (A6):** Open `SAGE_Assignment_6.ipynb` and run cells sequentially. Produces `sage_audit_log.json`, `sage_feedback.json`, and `sage_a6_results.json`.

> **Tip**: Steps 11–12 in `SAGE_Complete.ipynb` and the evaluation section in `SAGE_Assignment_6.ipynb` make ~50 API calls each. At `gpt-4o` pricing, expect ~$0.50–$1.00 for a full run. Use `gpt-4o-mini` in `run_prompt()` for cost-reduced testing.

---

## Technologies

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Agent Orchestration | LangGraph (ReAct loop) |
| LLM Abstractions | LangChain + langchain-openai |
| Vector Store | ChromaDB + text-embedding-3-small |
| Prompt Automation | Azure Prompt Flow SDK (local execution) |
| Evaluation | LLM-as-Judge + 30-case structured dataset |
| Persistence | JSON audit log + feedback store |

---

## Prompt Engineering Techniques Demonstrated

- **Foundational**: Zero-Shot, Few-Shot, Chain-of-Thought, Step-Back, Analogical, Auto-CoT, Generated Knowledge
- **Advanced**: Decomposition, Ensembling, Self-Consistency, Universal Self-Consistency, Self-Criticism
- **Production**: System prompt hardening, sensitivity analysis, intent routing, mandatory checklist constraints, edge-case disambiguation rules
- **Security**: Injection pattern detection, delimiter-based trust boundaries, adversarial test cases
- **Evaluation**: Automated dataset-based testing, LLM-as-Judge scoring, regression testing, Azure Prompt Flow variant comparison
- **Retrieval**: RAG with ChromaDB, section-level chunking, semantic retrieval, query-time context injection
- **Memory & State**: Multi-turn conversation sessions, rolling context window, session-scoped audit trail
- **Observability**: Numeric confidence scoring, severity-weighted risk scoring, citation groundedness verification, structured audit logging

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
- Perez & Ribeiro (2022). *Ignore Previous Prompt: Attack Techniques for Language Models.* NeurIPS ML Safety Workshop.
- Greshake et al. (2023). *Not What You've Signed Up For: Indirect Prompt Injection.* ACM AISec.
- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.
- Chen et al. (2025). *Unleashing the Potential of Prompt Engineering for LLMs.* Patterns.
- OWASP Foundation (2025). *OWASP Top 10 for LLM Applications.*
