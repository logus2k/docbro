# Phase 2 - Design & Architecture

### How AI is applied to system design, architecture, and API contracts (mid‑2026)

*Audience: technical leadership. Part of the per‑phase SDLC series. Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md), [Technical Architecture](../analysis/02_technical_architecture.md), [Implementation Planning](../analysis/03_implementation_planning.md). Reliability tags: **[PRIMARY]** · **[ANALYST]** · **[VENDOR]** · **[3P]** · ⚠️ contested.*

---

## 1. What this phase covers

Translating an agreed specification into a technical design: system/service decomposition, API and data‑contract definition, state‑management plans, technology selection, and **feasibility analysis** against a concrete stack. The output is the architecture and interface contracts that constrain the [build](03_build.md) phase.

## 2. Adoption status & evidence

- **Rapidly growing, mid‑maturity.** One of the most conceptually novel areas, but rigor has "relocated from writing algorithms to defining precise criteria and guardrails." AI is surprisingly effective at low‑level design generation and weak at high‑level systemic trade‑offs.
- **Architecture‑as‑context is the enabler.** Agents are only as good as the architectural context they can access; teams that maintain accurate internal docs and clear API boundaries get materially better generated designs (and less duplicated/misaligned work).
- **The diagram/IaC‑from‑text capability is mainstream** (ERDs, cloud architecture diagrams, component trees from a prompt), while genuine architectural judgment remains human‑led.
- **"Vibe architecting" is the new governance gap.** The most important 2026 insight for this phase ([grounding source ②](#10-grounding--further-reading)): coding agents now make consequential architectural decisions — framework selection, persistence, integration protocols, module boundaries — *implicitly, from the prompt, without review or an ADR.* In a controlled study the **same task expressed through three prompts produced 141, 472, and 827 LoC across 2, 4, and 6 files with entirely different storage and dependency graphs.** Architecture is being decided whether or not anyone is architecting; the academic consensus is still that GenAI is **assistive, not autonomous, and proof‑of‑concept‑stage** for real architecture work ([source ①](#10-grounding--further-reading)).

## 3. What works

- **Microservice/bounded‑context decomposition, API design, schema proposals, event‑driven patterns.** AI produces credible first drafts that architects refine — e.g., "Design a platform handling 10M candidates" → bounded contexts, service diagrams, caching strategy, messaging patterns, scalability considerations.
- **API‑contract and state‑plan generation.** The source‑notes LSP example is representative: from raw intent, draft the WebSocket/JSON‑RPC message schemas (`textDocument/didOpen`, `didChange`, custom `collaboration/cursorMove`) and a state‑store shape (e.g., document/editor/tree slices), *before any code is written*.
- **Feasibility analysis against a specific stack** — the highest‑value pattern. Instead of "is my architecture good?", ask "show me exactly how it will fail given *my* tools." Example from the corpus: a RAG pipeline (LangChain + ChromaDB + a hosted embedding API on an SQS queue) → the model flags the embedding‑API rate limit as the bottleneck, predicts SQS visibility‑timeout re‑processing loops, and prescribes token‑bucket rate‑limiting + batch sizing + a persistent‑disk vector store.
- **Diagram generation** (ERDs, architecture diagrams, component trees) for communication and review.

## 4. What doesn't work (yet)

*Grounded in the GenAI‑for‑architecture survey ([source ①](#10-grounding--further-reading)) and the "Architecture Without Architects" coupling‑pattern study ([source ②](#10-grounding--further-reading)).*

- **Large‑scale trade‑off decisions and complex organizational constraints** — AI lacks the systemic and political context; this remains firmly human. The survey is blunt: GenAI *"struggles with capturing organizational constraints, legacy dependencies, and non‑functional requirements"* and architecture remains *"a synthesis task requiring human expertise."*
- **High‑level systemic reasoning** — least‑mature sub‑area; AI excels at low‑level design, struggles to hold a whole‑system model.
- **Hallucinated / unvalidated guidance** — models produce *"plausible but incorrect architectural guidance without grounding mechanisms,"* and verifying suggestions against domain constraints is hard — a reliability problem in high‑stakes decisions.
- **The speed‑review gap.** Agents scaffold a system in minutes; auditing its implicit architecture takes hours. Without controls, the architecture ships unreviewed.
- **Opacity & implicit coupling.** Agent‑chosen infrastructure arrives with *no ADR or rationale*; small prompt wording differences silently produce different dependency graphs and failure modes ("vibe architecting").
- **Converging stacks = concentrated risk.** When agents default to the same narrow framework/library choices (training‑data priors), they concentrate vulnerability exposure across many systems.
- **Unconstrained generation** drifts from existing patterns and duplicates work without strong architecture‑as‑context.

## 5. Tools, models & frameworks

| Category | Options | Notes |
| --- | --- | --- |
| **Spec/design tools** | AWS Kiro (`design.md` generation, SMT‑style contradiction checking), GitHub Spec Kit (`Plan` step) | Design artifacts version‑controlled with the spec |
| **Diagramming / prototyping** | Eraser.io, Whimsical AI, Mermaid‑via‑LLM, v0 (frontend component trees), **Claude Design** (conversational UI/UX prototyping before committing to implementation) | Good for review/communication, not decisions |
| **Context engines** | Semantic code indexing (Cursor/Continue/Cline embeddings; Aider repo‑map; Claude Code agentic search) over the existing codebase for impact analysis | Feed the design model real architecture, not guesses |
| **Models** | Frontier reasoning models (Claude Opus 4.8, GPT‑5.5, Gemini 3.1 Pro) with 1M‑token context for whole‑subsystem reasoning; RAG over internal ADRs/wiki for stack‑specific feasibility | Reserve the expensive model for the hard reasoning |

## 6. Concrete patterns to adopt

1. **Mandatory "AI Feasibility Analysis" appendix in every ADR**, generated by a model primed (via RAG) with your internal tech‑stack documentation. The ATAM‑with‑LLMs study ([source ③](#10-grounding--further-reading)) validates the mechanics: feed the architecture doc via **RAG** and use **stepwise prompts** (identify risks → sensitivity points → quality‑attribute trade‑offs → scenarios). It surfaced *materially more* risks and trade‑offs than human evaluators (one group: 8 risks vs 4) — but **with false positives and out‑of‑scope suggestions**, so treat output as decision support requiring expert gatekeeping, not a verdict.
2. **Stack‑specific stress prompts.** Always include your actual tools, limits, and SLAs in the prompt — feasibility output is only useful when grounded in your constraints.
3. **Architecture‑as‑context discipline + persistent rules.** Keep internal docs and API boundaries accurate; encode architectural constraints as **persistent agent instructions** (`AGENTS.md` / `.cursorrules` — e.g., *"Follow the Hexagonal Architecture pattern,"* *"never‑allow"* security rules) so design and build agents generate aligned, non‑duplicative work ([source ④](#10-grounding--further-reading)).
4. **Govern "vibe architecting" with a three‑layer model** ([source ②](#10-grounding--further-reading)): **Constraints** (`AGENTS.md`/ADLs bounding allowed tech) → **Conformance** (plan‑before‑build workflows, post‑generation hooks, complexity thresholds that trigger human review) → **Knowledge** (extract an ADR from what the agent actually chose, so implicit decisions become explicit).
5. **Architectural impact statements.** Require the agent (or a wrapping hook) to declare the architectural cost of a change before it lands — e.g., *"adds vector database + embedding pipeline; +330 LoC"* — so reviewers see infrastructure consequences, not just diffs.
6. **Multi‑variant design exploration.** Generate two or three architectures (or implementations in different languages) from the same spec and compare — cheap de‑risking enabled by spec/implementation decoupling.
7. **Wrap generative design in deterministic validation** — schema validators, contract tests, and policy‑as‑code assert the design's boundaries rather than trusting the LLM.
8. **Design for nondeterminism — containment over elimination ([source ⑤](#10-grounding--further-reading)).** Treat nondeterminism as an architectural *reality*, not a bug to remove: isolate AI‑driven functionality behind explicit **containment boundaries**, and place a **validation gate at every deterministic↔nondeterministic crossing**. The failure mode to architect against is *"acceptable but wrong"* — output that compiles, passes, and satisfies existing dashboards yet is subtly incorrect and therefore invisible to current monitoring. ("Nondeterministic Dependency Design" is a named Trial‑ring trend in the LTM SDLC AI Radar.)
9. **Hallucination Containment — limit the *blast radius* of wrong outputs ([source ⑤](#10-grounding--further-reading)).** Since you can't guarantee the model won't fabricate, design so that a fabricated or incorrect output *cannot cause outsized harm*: scope and rate‑limit what any AI‑touched path can do, keep a deterministic/human check on irreversible or high‑blast‑radius actions, and fail safe. A named Scale‑ring trend — the design‑time sibling of the runtime guardrails in [Operate](06_operate.md) and the actuation control plane in [Release](05_release.md).

## 7. Implementation priorities (80/20)

- **Value/effort:** Medium‑high value, medium effort. Best as a **Phase 2** capability (feasibility analysis + ADR appendix) layered on once [build](03_build.md) and [test](04_test.md) pilots succeed.
- **Quick win:** feasibility analysis on the next significant architecture decision — low risk (advisory), high insight.
- **Metrics:** architecture‑related rework reduction, review‑cycle time, accuracy of feasibility predictions (did the flagged bottleneck materialize?), time from concept to approved design.

## 8. Risks & governance

- **Keep humans authoritative on trade‑offs** — treat AI designs as first drafts, reviewed like a senior architect's proposal.
- **Security and design‑system constraints belong in the design artifact** (and the agent's rules/constitution), enforced downstream automatically.
- **Don't let advisory feasibility become rubber‑stamping** — require the human to confirm or refute each flagged risk.
- **Avoid big‑upfront‑design regression** — keep design iterative and tied to the living spec.

## 9. Key takeaways

1. AI is strong at **low‑level design and API/contract generation**, weak at **systemic trade‑offs** — split the work accordingly.
2. **Feasibility analysis against your actual stack is the standout high‑value pattern** — it surfaces bottlenecks (rate limits, scaling, data‑flow) before code is written.
3. **Architecture‑as‑context is the multiplier** — accurate internal docs and clear boundaries are what make every design (and downstream build) agent effective.
4. **Govern the architecture agents are *already* deciding.** "Vibe architecting" means the design phase no longer ends at a human ADR — make implicit agent choices explicit (impact statements, ADR extraction, complexity‑triggered review) or accept an undocumented, converging, unreviewed architecture.

## 10. Grounding & further reading

*Curated, quality‑reviewed sources behind this phase (full review in [references/sdlc_phases.md](../references/sdlc_phases.md)).*

① **"Generative AI for Software Architecture: Applications, Challenges, and Future Directions"** (arXiv, Mar 2025) — https://arxiv.org/abs/2503.13310
   *Citable academic backbone.* Systematic review (100+ studies) across ADR generation, design generation, architecture analysis/evaluation, modernization. Honest on limits: assistive‑not‑autonomous, context loss, hallucination, proof‑of‑concept stage. Best for realism and taxonomy.

② **"Architecture Without Architects: How AI Coding Agents Shape Software Architecture"** (arXiv, 2026) — https://arxiv.org/abs/2604.04990
   *Best inspiration source — genuinely new framing.* Defines "vibe architecting," the five mechanisms by which agents shape architecture, six prompt→architecture coupling patterns, and a three‑layer governance model (Constraints / Conformance / Knowledge). Concrete case study (3 prompts → 141/472/827 LoC). Caveat: exploratory (single run, single tool); conceptual foundation, not a finished procedure.

③ **"Supporting architecture evaluation for ATAM scenarios with LLMs"** (arXiv, Jun 2025) — https://arxiv.org/abs/2506.00150
   *Empirical support for the feasibility/trade‑off pattern.* RAG + stepwise prompts surfaced more risks/sensitivity points/trade‑offs than students, with honest caveats (false positives, scope creep, single‑scenario brittleness; single‑LLM student pilot). Decision‑support requiring expert gatekeeping.

④ **Thoughtworks — "Beyond vibe coding: the five building blocks of AI‑native engineering"** — https://www.thoughtworks.com/en-au/insights/blog/generative-ai/beyond-vibe-coding-the-five-building-blocks-of-aI-native-engineering
   *Practitioner framing.* Agent / Model / Methodology / Spec / Context; architecture‑as‑context via persistent `AGENTS.md`/`.cursorrules` instructions and "never‑allow" rules. Moderately concrete (establishes vocabulary more than implementation detail).

⑤ **LTM — "SDLC AI Radar 2026" (Executive Briefing)** — `../articles/sdlc-ai-executive-slides-final.pdf`
   *Vendor foresight briefing* **[VENDOR]** (Tech‑Radar format: 33 trends × 4 quadrants × Scale/Trial/Assess/Hold). Source of the *"nondeterminism as architectural reality → containment over elimination"* framing and the *Verifiability‑as‑Architecture* trend. Directionally aligned with our findings; treat its specific figures as vendor‑sourced.
