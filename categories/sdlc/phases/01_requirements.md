# Phase 1 — Requirements & Planning

### How AI is applied to requirements engineering, planning, and specification (mid‑2026)

---

## 1. What this phase is for

The job of this phase is to turn a rough intention — a feature idea, a business ask, a pile of meeting notes — into a **clear, agreed, machine‑readable definition of what to build.** That has always mattered, but in an AI‑accelerated lifecycle it becomes the single highest‑leverage thing you do, for one blunt reason: **the specification is now the contract that every downstream agent implements against.** When a coding agent turns your requirements into code in minutes, the quality of the result is bounded almost entirely by the quality of the spec it was given. Ambiguity that used to be caught by a developer thinking it through is now faithfully implemented as a bug.

This is what people mean when they say *"planning is the new coding."* The bottleneck has moved off the keyboard and onto the question of *knowing what to build and stating it precisely.* Historically this was one of the weakest phases in most teams' SDLC — vague stories, missing acceptance criteria, requirements that contradict each other. AI makes that weakness far more expensive, and also gives you the best tools yet to fix it.

The output you are aiming for is a **living, version‑controlled specification** — user stories with real acceptance criteria, API and data contracts, and the non‑functional constraints (security, performance, compliance) written down — that the design and build phases can consume directly.

## 2. Where AI genuinely helps (and where it doesn't)

**What works today.** AI is very good at the mechanical, high‑volume parts of requirements work that humans find tedious and therefore skip. It can take unstructured inputs — a meeting transcript, a Slack thread, a legacy document — and turn them into structured epics, user stories, and acceptance criteria, flagging duplicates and contradictions between stakeholders along the way. It can draft a first‑pass PRD that a product manager then curates rather than authors from scratch. And it can act as a **gate**: before a ticket is marked "ready for development," an LLM can cross‑check it against your existing architecture‑decision records and API docs and flag conflicts — for example, a new request that quietly violates a security rule you established last month.

**What doesn't work yet — and this is important for planning.** Do not expect AI to make the hard calls. Prioritisation, business trade‑offs, and genuine product strategy stay firmly human; the model has no view of your org's politics, constraints, or intent. Two failure modes are worth designing around from the start:

- **Spec drift.** A spec that isn't actively kept in sync with the code rots quickly, and most tools still treat specs as write‑once documents. If you don't plan to maintain the spec, don't build heavy process around it.
- **Over‑engineering small work.** Heavyweight spec‑driven workflows badly over‑specify trivial tasks — in one practitioner's test a one‑point bug fix ballooned into four user stories and sixteen acceptance criteria. The tooling won't tell you when to stop; *you* have to decide when a spec is worth the effort.

There is also a subtler trap: **a passing spec test proves only that the software matches the spec, not that the spec was right.** A wrong requirement, faithfully implemented, is still wrong. So human review of the specification itself remains non‑negotiable.

## 3. The activities — what to actually do

This is the core of the plan. These are the concrete pieces of work to stand up, roughly in the order you'd introduce them.

**a) Create a version‑controlled home for specifications.** Make a `specs/` folder a first‑class part of the repository, sitting next to the code. Put the machine‑readable artefacts there — OpenAPI definitions, JSON Schemas, structured requirements — and validate them in CI against the implementation so drift shows up as a failed build rather than a surprise in production. This one structural decision does more than any tool choice to make the phase work.

**b) Decide, deliberately, how much rigour each piece of work gets.** Not everything deserves a full specification. A useful model distinguishes three levels: **spec‑first** (the spec guides the first generation, then can be discarded — fine for prototypes), **spec‑anchored** (the spec persists and evolves with the code, with tests enforcing alignment — the sweet spot for most production systems), and **spec‑as‑source** (humans edit only the spec and the code is fully regenerated — powerful in theory, still immature, don't rely on it yet). The governing rule is *use the minimum rigour that removes the ambiguity for this particular piece of work.* Reserve real spec‑driven development for greenfield builds and legacy‑modernisation efforts, where capturing intent pays off; skip it for small fixes.

**c) Use AI to synthesise and expand requirements — then have a human clarify.** The everyday workflow: feed the model the raw material (transcripts, tickets, existing docs), ask it to produce structured stories with explicit acceptance criteria, and then review. The value is easiest to see in the expansion. A thin story like *"As a user I can upload files"* becomes something a team can actually build against:

> *As a recruiter, I can upload candidate CVs up to 20 MB in PDF or DOCX, so that candidate profiles are created automatically.*
> **Acceptance criteria:** upload succeeds for valid files · duplicates are detected · files are virus‑scanned · extraction is validated · oversized/wrong‑format files are rejected with a clear message.

**d) Run requirements as a team loop with a "don't assume — ask" rule.** Instead of one person prompting an AI in isolation, run a short synchronous session where the AI proposes a plan *and asks clarifying questions*, and the team validates before anything proceeds. Give the agent a standing instruction to **ask rather than assume**, because models rush toward an answer and will invent a plausible requirement rather than admit a gap. This is where ambiguity gets caught cheaply. (Amazon's AI‑DLC calls this "Mob Elaboration.")

**e) Gate compliance and security at spec time, not release time.** Write the non‑functional requirements — security rules, design‑system constraints, performance budgets, configuration contracts — *into the spec itself*, so they become part of the plan every downstream agent reads and are enforced from day one. This matters more as velocity rises: if you leave governance to a release‑time gate, that gate gets overwhelmed by the volume of AI‑generated changes. Encoding it upstream keeps it from becoming the bottleneck. A concrete pattern: a **"requirements‑gap analyser"** that runs on ticket transition and automatically comments when a story conflicts with an ADR, an API contract, or a security protocol.

**f) Write requirements in a form that's testable — consider EARS.** The *Easy Approach to Requirements Syntax* uses a handful of constrained‑English templates (`The <system> shall…`; `WHEN <trigger> the <system> shall…`; `WHILE <state>…`; `WHERE <feature>…`; `IF <unwanted condition> THEN…`). The point isn't ceremony — it's that rigid templates produce **atomic, unambiguous, testable** requirements that an LLM can translate reliably into both design and tests. It's a small habit with a large downstream payoff.

## 4. How to pilot this

Don't try to do all of the above at once. A sensible sequence:

1. **Quick win first:** PRD and user‑story drafting with human curation. Low risk, immediate improvement in backlog quality, and it builds the team's trust in the workflow.
2. **Then the structural move:** stand up the `specs/` folder with CI validation.
3. **Then go deeper, selectively:** apply full spec‑driven development to the next greenfield service or a legacy‑modernisation effort — not to routine tickets.

A useful target to set and track: aim to spend roughly **30–40% of a task's time in specification before generation begins**, and reward spec quality alongside code quality. That ratio is the practical expression of "planning is the new coding." It's a directional benchmark from industry commentary rather than a hard rule, but it's a good forcing function.

## 5. Guardrails & what to watch for

- **Humans own the trade‑offs and priorities.** AI drafts; people decide. Keep that line bright.
- **Treat spec drift as a build failure.** If the spec and the implementation diverge, CI should complain.
- **Don't slide back into waterfall.** Writing the entire specification up front assumes you'll learn nothing during implementation — which is false. Keep specs lightweight and iterative, revised as you learn.
- **Watch the verbosity tax.** If reviewing the generated spec artefacts costs more than reviewing the code would have, you've over‑specified — dial the rigour down.

## 6. How you'll know it's working

Track outcomes, not activity:

- **Ambiguity / completeness of specs** going into development (fewer clarifying questions mid‑build).
- **Downstream rework attributable to unclear requirements** — this should fall.
- **First‑pass agent success rate** on work done under spec‑driven development versus ad‑hoc prompting.
- **Share of task time spent planning** (moving toward the 30–40% target), watched alongside delivered quality so it doesn't become planning for its own sake.

## 7. Tools to reach for

| Need | Options |
| --- | --- |
| **Spec‑driven development** | GitHub Spec Kit (agent‑agnostic: `Constitution → Specify → Clarify → Plan → Tasks → Implement`), AWS Kiro (generates `requirements.md`/`design.md`/`tasks.md`, uses EARS), Tessl (spec‑as‑source), BMAD‑METHOD (multi‑agent) |
| **Requirements notation** | EARS templates (see activity f) |
| **Synthesis & drafting** | General frontier LLMs (Claude, Gemini, GPT) for PRDs; transcription (Otter.ai, Gong) feeding LLM extraction; Notion AI, ChatPRD — pair with RAG over your wiki/ADRs for grounding |
| **Models** | 1M‑token‑context models can ingest a full requirements set at once; a local model (Qwen3‑Coder / DeepSeek via vLLM) is fine for privacy‑sensitive drafting |

## 8. Evidence & sources

*Reliability tags: [PRIMARY] primary survey/report · [RCT/PEER] randomized or peer‑reviewed · [ANALYST] Gartner/McKinsey · [VENDOR] vendor‑sourced (treat as a ceiling) · [3P] aggregator.*

- **Adoption / trust:** Requirements‑and‑design AI use is around ~53% versus ~72% for code generation [3P]; **69.2% of developers would not use AI for project planning** — the second‑least‑trusted task after deployment (Stack Overflow 2025) [PRIMARY]. AI‑assisted planning has been credited with ~40% PM‑productivity gains [ANALYST], directional.
- **The 30–40% planning‑time target** is a vendor‑briefing benchmark (LTM SDLC AI Radar) [VENDOR], consistent with our broader finding that spec quality gates downstream ROI.
- **Grounding sources** (full reviews in [references/sdlc_phases.md](../references/sdlc_phases.md)):
  1. **Böckeler / Fowler — "Understanding Spec‑Driven Development: Kiro, spec‑kit, and Tessl"** (Thoughtworks) — the honest, hands‑on account of the failure modes (verbosity tax, problem‑size misfit, waterfall risk). https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html
  2. **"Spec‑Driven Development: From Code to Contract in the Age of AI"** (arXiv, Feb 2026) — the spec‑first/anchored/as‑source taxonomy and the "minimum rigour" rule. https://arxiv.org/html/2602.00180v1
  3. **AWS — "AI‑Driven Development Life Cycle (AI‑DLC)"** — the Mob Elaboration and "ask‑don't‑assume" / persistent‑context practices. https://aws.amazon.com/blogs/devops/ai-driven-development-life-cycle/
  - Also: GitHub Spec Kit, AWS Kiro docs, Alistair Mavin's EARS guide.

---

*Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md) · [Technical Architecture](../analysis/02_technical_architecture.md) · [Implementation Planning](../analysis/03_implementation_planning.md) · next phase → [Design & Architecture](02_design.md).*
