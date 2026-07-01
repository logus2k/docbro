# Phase 2 — Design & Architecture

### How AI is applied to system design, architecture, and API contracts (mid‑2026)

---

## 1. What this phase is for

This phase turns an agreed specification into a **technical design**: how the system is decomposed into services, what the API and data contracts are, which technologies and models you'll use, and — crucially — whether the design will actually hold up under your real constraints. The output is the architecture and the interface contracts that will constrain everything the build phase does.

There's a twist in 2026 that reshapes how you plan this phase. Coding agents don't wait for a finished architecture — they make architectural decisions *implicitly*, the moment they start generating code. Choose your framing carefully or the agents will choose for you, silently and without a record. So this phase is no longer just "produce a design document"; it's also "**govern the design decisions the agents are already making.**"

## 2. Where AI genuinely helps (and where it doesn't)

**What works.** AI is strong at the concrete, lower‑level parts of design. It produces credible first drafts of service decompositions, API contracts, database schemas, event‑driven patterns, and architecture diagrams — the kind of scaffolding an architect then refines rather than invents from a blank page. Its single most valuable use here is **feasibility analysis against your actual stack** (see the activities below): asking not "is this a good architecture?" but "given *my* specific tools and limits, show me exactly where this will break."

**Where it falls short — plan around this.** AI is weak precisely where architecture is hardest: large‑scale trade‑offs, whole‑system reasoning, and the organisational and legacy constraints that don't appear in any prompt. The academic consensus is blunt — for real architecture work, generative AI is *assistive, not autonomous,* and still proof‑of‑concept‑grade. Treat every AI design as a first draft to be reviewed like a proposal from a promising but junior architect.

**The new failure mode: "vibe architecting."** When an agent scaffolds a system from a prompt, it quietly decides the framework, the persistence layer, the integration protocol, the module boundaries — with **no ADR and no rationale.** Worse, small differences in prompt wording produce entirely different structures (in one study the same task through three prompts yielded 141, 472, and 827 lines of code across 2, 4, and 6 files, with different storage and dependency graphs). The architecture ships in minutes; auditing what was actually decided takes hours. If you don't govern this deliberately, you end up with an undocumented, inconsistent, unreviewed architecture — and, because agents default to the same popular libraries, a concentration of the same vulnerabilities across many systems.

## 3. The activities — what to actually do

**a) Make an AI feasibility analysis a required part of every significant design decision.** This is the highest‑value habit in the phase. For each meaningful architecture choice, have a model — primed via RAG with your internal tech‑stack documentation — stress‑test the proposal against your real constraints. Drive it with stepwise prompts: *identify the risks → the sensitivity points → the quality‑attribute trade‑offs → the scenarios where this fails.* Done well, it surfaces bottlenecks a human review misses: for a RAG pipeline built on a hosted embedding API behind a queue, the model flags the embedding rate limit as the true bottleneck, predicts the reprocessing loops that will follow, and prescribes batching plus a persistent vector store. Capture the output as an appendix to the ADR. One caveat to plan for: it also produces false positives and out‑of‑scope suggestions, so a human must confirm or refute each flagged risk — this is decision *support*, not a verdict.

**b) Use AI to draft the contracts and diagrams — then refine.** Let the model produce the first‑pass API/message schemas, state‑management plans, ERDs, and component diagrams. This is genuinely useful for communication and review and saves real time; just don't mistake a generated diagram for a decision.

**c) Feed agents your real architecture as context, and encode the rules persistently.** Agents are only as good as the architectural context they can see. Keep internal docs and API boundaries accurate and current, and — this is the operational lever — encode your architectural constraints as **persistent agent instructions** in files like `AGENTS.md` or `.cursorrules`: "follow the hexagonal architecture pattern," "never call the database directly from a controller," explicit "never‑allow" security rules. These rules travel with every design and build agent, so they generate aligned, non‑duplicative work instead of drifting.

**d) Govern "vibe architecting" with a three‑layer model.** Make the implicit explicit:
- **Constraints** — bound the allowed technology up front (`AGENTS.md`, architecture‑decision language) so agents can't reach for whatever is trendy.
- **Conformance** — insert checkpoints: plan‑before‑build workflows, post‑generation hooks, and complexity thresholds that trigger a human review when a change crosses a size or dependency limit.
- **Knowledge** — after the fact, **extract an ADR from what the agent actually chose**, so decisions that were made implicitly become part of the record.
Reinforce this with **architectural impact statements**: require the agent (or a wrapping hook) to declare the structural cost of a change before it lands — "adds a vector database and an embedding pipeline; +330 LoC" — so reviewers see infrastructure consequences, not just a diff.

**e) Design for nondeterminism — containment over elimination.** You cannot make an LLM‑backed component deterministic, so stop trying to eliminate the variance and instead **contain** it. Isolate AI‑driven functionality behind explicit boundaries, and put a validation gate at every point where a nondeterministic component hands off to a deterministic one. The specific failure to architect against is *"acceptable but wrong"* — output that compiles, passes, and looks fine on the dashboard yet is subtly incorrect and therefore invisible to your current monitoring.

**f) Contain the blast radius of hallucinations.** Complementary to (e): since you can't guarantee the model won't fabricate, design so that a fabricated or incorrect output *cannot cause outsized harm.* Scope and rate‑limit what any AI‑touched path is allowed to do; keep a deterministic or human check on anything irreversible or high‑impact; fail safe. This is the design‑time sibling of the runtime guardrails in [Operate](06_operate.md) and the actuation control plane in [Release](05_release.md).

**g) Make the AI‑feature decisions: RAG vs fine‑tuning, and the retrieval layer.** For features that are themselves AI‑powered, this phase is where you decide the model strategy — whether to ground the model with retrieval (RAG) or adapt it with fine‑tuning (LoRA/QLoRA), and how to build the knowledge/retrieval layer (a vector store, or a knowledge graph for relationship‑aware reasoning). Decide it here, deliberately, rather than letting the build agent improvise it.

## 4. How to pilot this

Start advisory and low‑risk. The natural entry point is **feasibility analysis on the next significant architecture decision** — it's purely advisory, carries no delivery risk, and immediately demonstrates value by catching a real bottleneck. From there, introduce the **architecture‑as‑context discipline** (`AGENTS.md` rules) so build agents inherit your constraints, and only then layer on the **vibe‑architecting governance** (impact statements, ADR extraction, complexity‑triggered review) as your agent usage grows. Design is best treated as a Phase‑2 capability that you turn on once your build and test pilots are working — not the first thing you automate.

## 5. Guardrails & what to watch for

- **Humans stay authoritative on trade‑offs.** AI drafts; architects decide. The systemic and organisational judgment is yours.
- **Don't let advisory feasibility become rubber‑stamping.** The value is in confirming or refuting each flagged risk — not in nodding at the output.
- **Put security and design‑system constraints in the design artefact itself** (and the agent's rule files), so they're enforced downstream automatically rather than remembered later.
- **Don't accept an architecture you didn't review.** "Vibe architecting" means the phase no longer ends at a human ADR by default — you have to *make* it end there, via impact statements and ADR extraction, or accept an undocumented and converging architecture.

## 6. How you'll know it's working

- **Architecture‑related rework is falling** — fewer "we built the wrong structure" reversals downstream.
- **Feasibility predictions prove out** — the bottlenecks the model flagged are the ones that actually materialised (or were pre‑empted).
- **Design cycle time** from concept to an approved, documented design.
- **Coverage of implicit decisions** — the share of agent‑chosen architecture that ends up captured in an ADR rather than shipping silently.

## 7. Tools to reach for

| Need | Options |
| --- | --- |
| **Spec/design tooling** | AWS Kiro (`design.md` generation with contradiction checking), GitHub Spec Kit (the `Plan` step) — keep design artefacts version‑controlled with the spec |
| **Diagramming / prototyping** | Eraser.io, Whimsical AI, Mermaid‑via‑LLM, v0 (frontend component trees), **Claude Design** (conversational UI/UX prototyping before you commit to implementation) — good for communication, not for making the decision |
| **Context engines** | Semantic code indexing (Cursor/Continue/Cline embeddings; Aider's repo‑map; Claude Code's agentic search) over the existing codebase, so the design model reasons about your real architecture rather than guessing |
| **Models** | Frontier reasoning models (Claude Opus 4.8, GPT‑5.5, Gemini 3.1 Pro) with large context for whole‑subsystem reasoning; RAG over your internal ADRs/wiki for stack‑specific feasibility. Reserve the expensive model for the genuinely hard reasoning |

## 8. Evidence & sources

*Reliability tags: [PRIMARY] · [RCT/PEER] · [ANALYST] · [VENDOR] (treat as a ceiling) · [3P].*

- **Maturity:** AI is effective at low‑level design generation and weak at high‑level systemic trade‑offs; the "diagram/IaC‑from‑text" capability is mainstream while architectural judgment stays human‑led. The academic survey is explicit that GenAI *"struggles with capturing organizational constraints, legacy dependencies, and non‑functional requirements."*
- **"Vibe architecting"** — the finding that agents make consequential architecture decisions implicitly, and that the same task through three prompts produced 141/472/827 LoC across 2/4/6 files — comes from the "Architecture Without Architects" study.
- **Grounding sources** (full reviews in [references/sdlc_phases.md](../references/sdlc_phases.md)):
  1. **"Generative AI for Software Architecture: Applications, Challenges, and Future Directions"** (arXiv, Mar 2025) — the systematic, unhyped survey; best for realism and the task taxonomy. https://arxiv.org/abs/2503.13310
  2. **"Architecture Without Architects: How AI Coding Agents Shape Software Architecture"** (arXiv, 2026) — "vibe architecting" and the three‑layer governance model (Constraints / Conformance / Knowledge). https://arxiv.org/abs/2604.04990
  3. **"Supporting architecture evaluation for ATAM scenarios with LLMs"** (arXiv, Jun 2025) — validates the RAG + stepwise‑prompt feasibility method (with honest caveats about false positives). https://arxiv.org/abs/2506.00150
  4. **Thoughtworks — "Beyond vibe coding: the five building blocks of AI‑native engineering"** — architecture‑as‑context via persistent `AGENTS.md`/`.cursorrules` rules. https://www.thoughtworks.com/en-au/insights/blog/generative-ai/beyond-vibe-coding-the-five-building-blocks-of-aI-native-engineering
  5. **LTM — "SDLC AI Radar 2026"** [VENDOR] — the *nondeterminism → containment* framing, *Nondeterministic Dependency Design*, *Hallucination Containment*, and *Verifiability‑as‑Architecture* trends. `../articles/ltm_sdlc_ai_radar_2026.md`

---

*Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md) · [Technical Architecture](../analysis/02_technical_architecture.md) · [Implementation Planning](../analysis/03_implementation_planning.md) · previous → [Requirements & Planning](01_requirements.md) · next → [Build](03_build.md).*
