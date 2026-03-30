# SAGE: Secure AI Governance Engine
### Enterprise AI Policy & Compliance Reasoning Agent

SAGE is an AI-powered compliance reasoning system that interprets enterprise policy documents, answers employee queries with verifiable citations, classifies compliance risk at four severity levels, and defends against adversarial prompt injection. Built on GPT-4o, LangChain, LangGraph, and ChromaDB.

---

## What This Project Does

Enterprise policy documents are dense, interconnected, and difficult to interpret accurately. Employees frequently rely on informal guidance, which can trigger regulatory penalties and security incidents. SAGE addresses this by:

- **Answering policy questions** grounded exclusively in authoritative documents — no hallucinated guidance
- **Citing every claim** with document name, section, and page number
- **Classifying compliance risk** (Low / Medium / High) for each response with explicit justification
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
    │                       Categories: risk_assessment / policy_question / out_of_scope
    ▼
L3: RAG Retrieval        ← ChromaDB semantic search (text-embedding-3-small)
    │                       Returns top-5 relevant policy chunks (~350 words)
    │                       vs. 1,600-word full corpus injection (80% token reduction)
    ▼
L4: Reasoning Engine     ← LangGraph ReAct Agent (GPT-4o)
    │                       Tools: search_policy · check_cross_references · assess_risk
    │                       Techniques: CoT · Step-Back · Generated Knowledge
    ▼
L5: Validation Layer     ← Self-criticism: citation check + groundedness verification
    │                       Flags unsupported claims before response reaches user
    ▼
Structured Response      ← Answer · Citations · Risk Level · Reasoning
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
├── SAGE_Complete.ipynb          ← Main notebook (all three phases, 65 cells)
├── README.md                    ← This file
└── sage_prompt_flow/            ← Created at runtime by Step 12
    ├── flow.dag.yaml
    ├── compliance_check.py
    ├── variant_a.txt ... variant_e.txt
    └── test_data.jsonl
```

---

## Notebook Structure

`SAGE_Complete.ipynb` is the single entry point covering all development phases:

### Phase 1 — Prompting Technique Exploration (Steps 4–5)

**Step 4: Foundational Techniques** — Seven techniques applied to the same compliance scenario, directly comparable:

| Technique | Key Insight |
|-----------|-------------|
| Zero-Shot | Baseline — correct policy identification but no unified risk level, Sec 4.4 absent |
| Few-Shot | Best output format consistency via in-context behavioural examples |
| Chain-of-Thought | Highest section coverage; uniquely surfaces Sec 5.4 DPO approval requirement |
| Step-Back | Surfaces Sec 4.4 benefits gap via categorical domain scan before query-specific reasoning |
| Analogical | Best explainability; non-linear risk escalation insight via comparative reasoning |
| Auto-CoT | Scalable scaffold generation without hand-crafted steps |
| Generated Knowledge | Compound risk modelling; 2.3× token cost, justified for complex multi-policy queries |

**Step 5: Advanced Techniques** — Building on the foundational phase:

| Technique | What It Tests |
|-----------|---------------|
| Initial System Prompt | First-draft synthesis of best elements from all 7 foundational techniques |
| Decomposition (3-Stage) | Pipeline: Intent → Policy Analysis → Risk Assessment |
| Ensembling (3 + Aggregation) | Legal / Security / HR perspectives merged into one definitive answer |
| Self-Consistency (3 Runs) | Diagnoses prompt instability — revealed the "risk temporal ambiguity" bottleneck |
| Universal Self-Consistency | Meta-review step selects strongest reasoning across 3 independent runs |
| Self-Criticism (Gen→Critique→Revise) | Caught Sec 5.3 citation precision error; forms the L5 Validation Layer |

---

### Phase 2 — System Prompt Hardening & Evaluation (Steps 6–9)

**Step 6: Sensitivity Analysis & Hardening** — Three stress-test dimensions:
- **Phrasing variations** (formal / casual / bullet-point) — tests surface-level brittleness
- **Temperature sweep** (0.0, 0.3, 0.7, 1.0) — measures determinism under randomness
- **Auto-generated paraphrases** — simulates real users asking the same question differently

Eight instabilities identified → all resolved in the **Hardened System Prompt**, which achieves zero risk variance across 12 runs.

Key fixes applied:
1. Explicit risk criterion anchored to *current non-compliance state, before corrective actions*
2. Mandatory checklist: VPN (Sec 4.1), benefits gap (Sec 4.4), DPO approval (Sec 5.4), local storage prohibition (Sec 6.4)
3. 150–250 word output length constraint → eliminates word-count variance
4. Intent routing gate → out-of-scope queries declined before full pipeline runs
5. Ambiguity flag instruction → uncertain policy language flagged rather than assumed

**Step 7: Evaluation Dataset** — 23 structured test cases:
- 8 Typical (single-policy, easy baselines)
- 9 Edge (boundary conditions, multi-policy, contractor exclusions)
- 6 Adversarial (injection attempts, hallucination traps, fake policy citations)

**Step 8: RAG Pipeline** — ChromaDB vector store with section-level chunking:
- `text-embedding-3-small` embeddings per policy section
- Top-5 semantic retrieval at query time
- **80% token reduction** versus full-corpus injection with no accuracy penalty on typical queries

**Step 9: Meta Prompting & Perplexity Evaluation** — Uses a prompt-engineer meta-persona to auto-generate an improved prompt, and a proxy perplexity evaluation to verify that in-scope queries receive confident responses and out-of-scope queries receive short, correct refusals.

---

### Phase 3 — Agent, Automation & Final Evaluation (Steps 10–15)

**Step 10: LangChain ReAct Agent** — LangGraph state machine with three tools:

| Tool | Purpose |
|------|---------|
| `search_policy` | Semantic vector search over ChromaDB; returns top-5 relevant chunks |
| `check_cross_references` | LLM-powered cross-policy dependency analysis for multi-domain scenarios |
| `assess_risk` | Structured risk classification from accumulated compliance findings |

The agent dynamically selects which tools to invoke, in what order, based on query complexity. Simple lookups use only `search_policy`; the primary Portugal scenario invokes all three.

**Step 11: Prompt Variant Testing** — Five variants tested across 10 representative cases via LangChain:

| Variant | Configuration |
|---------|--------------|
| A | Basic CoT (7-step reasoning scaffold) |
| B | CoT + Step-Back (broad principle scan before specifics) |
| C | CoT + Generated Knowledge + Self-Consistency |
| D | Decomposition + Self-Criticism pipeline |
| E | Full Hardened production prompt |

**Step 12: Azure Prompt Flow** — Local execution of the same variant tests via the Prompt Flow SDK. Creates the standard `flow.dag.yaml` / `compliance_check.py` directory structure, runs all 5 variants × 8 cases, and produces a LangChain vs. Azure Prompt Flow head-to-head comparison.

**Step 13: Prompt Analysis** — Documents all 13 techniques (performance, coverage, cost), identifies five bottlenecks (root cause + mitigation), and shows the decomposition strategy used in the agent architecture.

**Step 14: Iteration Improvement Log** — Chronological record of all 8 prompt refinements from Zero-Shot baseline to production Hardened prompt, with before/after metrics for each change.

**Step 15: Final Evaluation** — Three-part evaluation:
1. **LLM-as-Judge**: GPT-4o scores the production system prompt on Clarity, Groundedness, Consistency, Completeness, Security, and Citation Accuracy
2. **Baseline vs. Refined**: Direct comparison on 5 cases — Zero-Shot baseline vs. Hardened prompt
3. **Multi-dimensional scorecard**: All metrics across all three phases in one table

---

## Key Results

| Metric | Target | Result |
|--------|--------|--------|
| Answer Accuracy | ≥85% | Validated via 23-case dataset |
| Citation Precision | ≥90% | Zero hallucinated citations across all experiments |
| Groundedness | ≥90% | Self-criticism caught and fixed Sec 5.3 precision error |
| Risk Classification | ≥80% | 52% baseline → 75%+ after hardening |
| Injection Resistance | ≥95% | All 6 adversarial cases correctly handled |
| Output Stability | 7.9/10 | 0 risk variance across 12 phrasing/temperature runs |
| Token Efficiency | — | 80% reduction with RAG vs. full-corpus injection |

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

Open `SAGE_Complete.ipynb` in Jupyter and run cells sequentially from top to bottom. Each phase builds on the previous — policy documents and helper functions defined in Steps 2–3 are used throughout.

> **Tip**: Steps 11–12 (variant testing) make ~50 API calls each. At `gpt-4o` pricing, expect ~$0.50–$1.00 for the full run. Use `gpt-4o-mini` in `run_prompt()` for cost-reduced testing.

---

## Technologies

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Agent Orchestration | LangGraph (ReAct loop) |
| LLM Abstractions | LangChain + langchain-openai |
| Vector Store | ChromaDB + text-embedding-3-small |
| Prompt Automation | Azure Prompt Flow SDK (local execution) |
| Evaluation | LLM-as-Judge + structured 23-case dataset |

---

## Prompt Engineering Techniques Demonstrated

This project demonstrates the following techniques from the prompt engineering curriculum:

- **Foundational**: Zero-Shot, Few-Shot, Chain-of-Thought, Step-Back, Analogical, Auto-CoT, Generated Knowledge
- **Advanced**: Decomposition, Ensembling, Self-Consistency, Universal Self-Consistency, Self-Criticism
- **Production**: System prompt hardening, sensitivity analysis, intent routing, mandatory checklist constraints
- **Security**: Injection pattern detection, delimiter-based trust boundaries, adversarial test cases
- **Evaluation**: Automated dataset-based testing, LLM-as-Judge scoring, Promptfoo-compatible structure, Azure Prompt Flow variant comparison
- **Retrieval**: RAG with ChromaDB, section-level chunking, semantic retrieval, query-time context injection

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
- Perez & Ribeiro (2022). *Ignore Previous Prompt: Attack Techniques for Language Models.* NeurIPS ML Safety Workshop.
- Greshake et al. (2023). *Not What You've Signed Up For: Indirect Prompt Injection.* ACM AISec.
- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.
- Chen et al. (2025). *Unleashing the Potential of Prompt Engineering for LLMs.* Patterns.
- OWASP Foundation (2025). *OWASP Top 10 for LLM Applications.*
