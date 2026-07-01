# Phase 1 - Requirements & Planning

### How AI is applied to requirements engineering, planning, and specification (mid‑2026)

*Audience: technical leadership. Part of the per‑phase SDLC series. Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md), [Technical Architecture](../analysis/02_technical_architecture.md), [Implementation Planning](../analysis/03_implementation_planning.md). Reliability tags: **[PRIMARY]** · **[RCT/PEER]** · **[ANALYST]** · **[VENDOR]** · **[3P]** · ⚠️ contested.*

---

## 1. What this phase covers

Turning intent into a structured, agreed, machine‑readable definition of *what to build*: eliciting and clarifying requirements, synthesizing unstructured inputs (meetings, tickets, docs) into epics/user stories/acceptance criteria, drafting Product Requirements Documents (PRDs), and — the genuinely novel 2026 practice — **spec‑driven development (SDD)**, where a version‑controlled specification becomes the source of truth that downstream agents implement against.

This phase historically one of the weakest in the SDLC (ambiguous requirements, incomplete stories, inconsistent acceptance criteria) and is now one of the highest‑leverage places to apply AI — *but it is also one of the least trusted for autonomy.*

## 2. Adoption status & evidence

- **Lagging coding, but rising.** Requirements/design AI use sits around **~53%** (Techreviewer 2025 **[3P]**) versus ~72% for code generation. Stack Overflow 2025 found **69.2% would not use AI for project planning** **[PRIMARY]** — the second‑least‑trusted task after deployment.
- **Specification quality is becoming "the new bottleneck and control plane."** As agents get better at implementation, the precision of the spec increasingly determines output quality — a recurring theme across the source analyses.
- **Spec‑driven development emerged in 2025** as the industry's answer to "vibe coding" (agents producing plausible code that drifts from intent). The recurring slogan: *"intent is the source of truth"* / *"the spec is the prompt."* By mid‑2026 every major coding tool ships an SDD flavor.

## 3. What works

- **Requirement synthesis from unstructured inputs.** Converting meeting transcripts, Slack threads, emails, and legacy docs into structured epics, user stories, and acceptance criteria — including de‑duplication and conflict detection between stakeholders. Concrete pattern from the source corpus: expand a thin story ("As a user I can upload files") into a complete one with size limits, formats, duplicate detection, virus scan, and explicit acceptance criteria.
- **PRD first‑drafting.** AI drafts the PRD; humans curate and validate rather than author from scratch. McKinsey cites ~40% PM‑productivity improvement on AI‑assisted planning **[ANALYST]** (directional).
- **Requirements‑gap analysis as a gate.** Before a ticket moves to "Ready for Dev," an LLM cross‑references the new story against existing ADRs and API docs and flags contradictions (e.g., a request that violates a newly established security protocol) — preventing costly late‑stage rework.
- **Multi‑variant exploration.** Because the spec is decoupled from any one implementation, you can cheaply ask for several approaches and compare — a practical way to de‑risk decisions early.

## 4. What doesn't work (yet)

*The sharpest field evidence here is Birgitta Böckeler's Thoughtworks write‑up of actually running Kiro, Spec Kit, and Tessl on real code ([grounding source ①](#10-grounding--further-reading)) and the arXiv "nine pitfalls" catalog ([source ②](#10-grounding--further-reading)).*

- **Large‑scale trade‑off and business‑context judgment.** AI is weak on org constraints, prioritization politics, and genuine product strategy. Keep humans authoritative here.
- **Spec drift / spec rot.** Specs drift out of sync with code unless tooling actively maintains them; most tools still treat specs as static documents. This is SDD's primary failure mode (one of the arXiv "nine pitfalls").
- **The verbosity tax.** Spec Kit generated 8+ markdown files per spec plus repetitive research notes; Böckeler's verdict: *"I'd rather review code than all these markdown files."* Reviewing the artifacts can cost more than reviewing the code — the opposite of the intended benefit.
- **Problem‑size misfit.** Heavyweight SDD workflows over‑specify small work: in Böckeler's test a **1‑point bug fix ballooned into 4 user stories with 16 acceptance criteria.** Current tooling has no flexible scope handling — *you* must decide when to skip it.
- **Functional/technical blurring.** Practitioners (and the agents) repeatedly lose track of *"when to stay on the functional level and when to add technical details,"* muddying the what/how separation that gives SDD its power.
- **Waterfall regression.** Writing the whole spec before implementation encodes the assumption that *you won't learn anything during implementation that changes the spec* — conflicting with proven iterative, small‑batch delivery. ThoughtWorks flags "reverting to waterfall" as a real risk.
- **False confidence.** Per the arXiv paper: *"a passing spec test doesn't guarantee correct software — it only guarantees that the software matches the spec."* A wrong spec faithfully implemented is still wrong.
- **Hallucinated requirements & instruction non‑compliance.** AI invents plausible‑but‑wrong acceptance criteria, and agents observably *ignore* constraints in one place while *over‑following* them elsewhere — undermining the control SDD promises. Human review remains mandatory.

## 5. Tools, models & frameworks

| Category | Options | Notes |
| --- | --- | --- |
| **Spec‑driven dev** | **GitHub Spec Kit** (MIT, agent‑agnostic: `Constitution → Specify → Clarify → Plan → Tasks → Implement`), **AWS Kiro** (EARS notation, generates `requirements.md`/`design.md`/`tasks.md`), **Tessl** (spec‑as‑source + Spec Registry), **BMAD‑METHOD** (MIT, multi‑agent) | Keep specs version‑controlled and CI‑validated |
| **Requirements notation** | **EARS** (Easy Approach to Requirements Syntax): Ubiquitous `The <system> shall…`, Event‑driven **WHEN**, State‑driven **WHILE**, Optional **WHERE**, Unwanted **IF/THEN** | Rigid templates → atomic, testable requirements LLMs translate reliably |
| **Synthesis tools** | General LLMs (Claude, Gemini, GPT) for PRD drafting; transcription (Otter.ai, Gong) feeding LLM extraction; Notion AI; ChatPRD | Pair with RAG over internal wiki/ADRs for grounding |
| **Models** | 1M‑token context models (Claude Opus 4.8, GPT‑5.5, Gemini 3.1 Pro) ingest a full requirements set at once; local‑first option: Qwen3‑Coder / DeepSeek via llama.cpp/vLLM for privacy | Stable formats (OpenAPI, JSON Schema, structured PRDs) parse best |

## 6. Concrete patterns to adopt

1. **Choose a rigor level deliberately — "minimum rigor that removes ambiguity."** The arXiv taxonomy ([source ②](#10-grounding--further-reading)) gives a usable decision model:
   - **Spec‑first** — spec guides initial generation, then may be discarded. Good for prototypes / AI‑assisted one‑offs with low maintenance burden. *(Where essentially all current tools actually operate.)*
   - **Spec‑anchored** — spec persists and evolves with the code; tests enforce alignment (BDD scenarios exemplify it). Described as *"the sweet spot for most production systems."*
   - **Spec‑as‑source** — humans edit only the spec; code is fully generated/regenerated (Tessl's aspiration, still beta). Eliminates drift by construction but needs mature, trusted generation tooling — and risks repeating Model‑Driven Development's "too much overhead" failure. Don't reach for it yet.
   The rule: *use the minimum level that removes ambiguity for your context* — over‑specification is itself a listed pitfall.
2. **`specs/` as a first‑class, version‑controlled folder.** Store OpenAPI/JSON Schema/EARS requirements alongside code; validate in CI against the implementation to catch drift. The single most important structural decision for this phase.
3. **Spec‑first for the right work only.** Apply SDD to greenfield zero‑to‑one work and legacy modernization (where original intent is lost); skip it for small fixes — the 1‑point‑bug → 16‑acceptance‑criteria failure is what happens otherwise.
4. **Persistent context as repository artifacts (AI‑DLC).** Treat requirements, assumptions, design decisions, and test plans as *versioned files in the repo, not disposable chat transcripts* ([source ③](#10-grounding--further-reading)) — enabling session continuity, auditability, and better future prompts.
5. **"Mob Elaboration" + an explicit no‑assumptions rule (AI‑DLC).** Run requirements as a synchronous loop where *AI proposes a plan and asks clarifying questions, and the whole team validates before anything proceeds.* Give the agent a standing instruction to **ask rather than assume** — LLMs rush to outcomes and must be told to defer business decisions to humans.
6. **Bake non‑functional requirements into the spec, not the review — gate compliance/security at *spec time*, not release time.** Security requirements, design‑system constraints, and config/parameter contracts become part of the plan the agent reads — enforced from day one rather than audited at the end. Release‑time gates get overwhelmed as AI raises deployment velocity; encoding governance into specs and templates upfront is what keeps them from becoming the bottleneck.
7. **"Requirements Gap Analyzer" webhook.** On ticket transition, an LLM checks the story against ADRs/API docs/security protocols and comments conflicts automatically.
8. **Mandatory AI feasibility appendix in ADRs** (see [design.md](02_design.md)) — generated by a model primed with your internal tech‑stack docs.

## 7. Implementation priorities (80/20)

- **Value/effort:** Medium‑high value, medium effort. Not a *first* pilot (lower verifiability than testing), but a strong **Phase 2** investment once coding/testing pilots prove out.
- **Quick win:** PRD/user‑story drafting with human curation — low risk, immediate quality‑of‑backlog improvement.
- **Higher‑leverage:** SDD on the next greenfield service or a legacy‑modernization effort.
- **Target (planning‑first):** deliberately spend **~30–40% of task time in specification before code generation**, and *track planning time / reward spec quality alongside code quality* — a vendor‑briefing benchmark for "planning is the new coding" **[VENDOR]**, consistent with our finding that spec quality gates downstream ROI.
- **Metrics:** spec completeness / ambiguity reduction, downstream rework attributable to unclear requirements, first‑pass agent success rate under SDD vs ad‑hoc prompting.

## 8. Risks & governance

- **Embed security in the spec/constitution files** so it propagates to every downstream agent.
- **Keep humans authoritative on trade‑offs and prioritization** — AI drafts, humans decide.
- **Guard against spec drift** with CI validation; treat an out‑of‑date spec as a build failure.
- **Avoid waterfall regression** — keep specs lightweight and iterative, not big‑upfront‑design.

## 9. Key takeaways

1. Requirements is a high‑leverage, low‑trust phase — **AI drafts and synthesizes; humans clarify, decide, and own**.
2. **Spec‑driven development is the defining 2026 practice here** — but selectively, on greenfield and modernization, with specs as version‑controlled, CI‑validated artifacts.
3. The payoff is a higher‑quality, less‑ambiguous backlog that makes every downstream phase (especially [build](03_build.md) and [test](04_test.md)) more reliable — the spec is the contract the rest of the pipeline depends on.

## 10. Grounding & further reading

*Curated, quality‑reviewed sources behind this phase (full review in [references/sdlc_phases.md](../references/sdlc_phases.md)).*

① **Birgitta Böckeler / Martin Fowler — "Understanding Spec‑Driven Development: Kiro, spec‑kit, and Tessl"** (Thoughtworks, 2025/26) — https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html
   *Primary realism source.* A practitioner running the actual tools on real code; honest, concrete failure modes (verbosity tax, problem‑size misfit, waterfall risk, MDD parallel). Best for the "what doesn't work" rigor.

② **"Spec‑Driven Development: From Code to Contract in the Age of AI"** (arXiv, Feb 2026) — https://arxiv.org/html/2602.00180v1
   *Citable framework.* The spec‑first / spec‑anchored / spec‑as‑source taxonomy, the "minimum rigor that removes ambiguity" rule, and nine pitfalls. Caveat: thin primary evidence (leans on a cited "up to 50% error reduction"); use for framing, not hard ROI.

③ **AWS — "AI‑Driven Development Life Cycle (AI‑DLC)"** + companions — https://aws.amazon.com/blogs/devops/ai-driven-development-life-cycle/ · [Building with AI‑DLC using Amazon Q](https://aws.amazon.com/blogs/devops/building-with-ai-dlc-using-amazon-q-developer/) · [open‑sourced adaptive workflows](https://aws.amazon.com/blogs/devops/open-sourcing-adaptive-workflows-for-ai-driven-development-life-cycle-ai-dlc/) · [sample repo](https://github.com/aws-samples/sample-ai-driven-development-lifecycle-platform/)
   *Methodology inspiration.* Named primitives: Inception, Mob Elaboration, ask‑don't‑assume rule, persistent context as repo artifacts. Caveat: the flagship blog is conceptual and omits limitations (vendor methodology) — the repo and the Amazon Q walkthrough hold the substance.

*Also referenced across the series:* GitHub Spec Kit (github/spec-kit), AWS Kiro docs (kiro.dev), Alistair Mavin's EARS guide (alistairmavin.com/ears).
