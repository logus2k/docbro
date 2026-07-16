# Master Plan — Implementing a Modern, AI‑Augmented SDLC

### The main activities from Design to Operate, how AI improves each phase, and the concrete patterns that actually work (mid‑2026)

*Audience: technical leadership and engineering leads standing up an AI‑augmented software delivery lifecycle. This plan pulls together the whole research base: the [Business Benchmark](../analysis/01_business_benchmark.md) (what works and what doesn't), the [Technical Architecture](../analysis/02_technical_architecture.md) (tools, models, pipeline), the [Implementation Planning](../analysis/03_implementation_planning.md) guide (roadmap, cost, governance), the six per‑phase deep‑dives ([requirements](../phases/01_requirements.md) · [design](../phases/02_design.md) · [build](../phases/03_build.md) · [test](../phases/04_test.md) · [release](../phases/05_release.md) · [operate](../phases/06_operate.md)), the [Google SRE reference](../articles/ai_in_sre_google_full_extract.md), and the phase‑tagged [IdeaLab use‑case catalog](../idealab/sdlc_ideas.md).*

---

## The thesis this whole plan rests on

One finding organizes everything. It comes from Google's DORA research — the single most corroborated result in the [benchmark](../analysis/01_business_benchmark.md):

> **"AI doesn't fix a team; it amplifies what's already there."** AI is an **amplifier, not an autopilot.**

Six principles follow from that, and every phase below is an application of them:

1. **Fix the foundation first.** Returns scale with the engineering health you already have — clean architecture, real tests, fast feedback, good delivery measurement. AI on a brittle system just ships low‑quality work faster.
2. **Verifiability predicts the payoff.** Tasks that are bounded and cheap to check — generating tests, root‑cause analysis, triage — pay off. Open‑ended generative tasks like greenfield architecture or big refactors disappoint, and can actually *slow experts down* (a randomized trial by METR found experienced developers 19% slower while *feeling* 20% faster).
3. **Verification‑driven, not trust‑driven.** This is the one pattern that recurs in every phase. Never ship AI output on trust; ship only what *provably* passes a deterministic gate — a test that measurably improves coverage (Test), a real deployment check (Release), a human approval (Build and Release), an evaluation against known‑good data (Operate).
4. **The bottleneck moved downstream.** Generation is solved; review, integration, and stability are the new constraint. The winning move is to **slow the merge gate, not the keyboard.**
5. **Govern first, and expand autonomy on a trust ladder.** Suggest, then act‑with‑approval, then autonomous‑for‑low‑risk — one task category at a time, watching stability. The 40% of agentic projects Gartner expects to be canceled fail on governance, not on the model.
6. **Measure outcomes, not vanity metrics.** Track review latency, change‑failure rate, bugs and incidents per pull request, and rework — never lines of code. You cannot manage the productivity paradox you cannot see.

The bottom line for planning: **expect single‑digit realistic throughput gains (roughly 7–20% on suitable work), a downstream "verification tax," and a governance‑first rollout** — not the order‑of‑magnitude transformation the marketing promises. The competitive edge is *discipline*, not access to the tools.

---

## The reference pipeline (the shape of the machine)

The consensus architecture for 2026 is **spec‑driven, agent‑assisted development, where agents produce reviewable artifacts — pull requests — that flow through your *existing* delivery pipeline** (see [Technical Architecture §1](../analysis/02_technical_architecture.md)). The agents sit *upstream* of your DevOps tooling and are *gated by it*; they never bypass it.

```
 UPSTREAM            DESIGN          BUILD            TEST            RELEASE            OPERATE
 ┌────────┐        ┌────────┐     ┌────────┐       ┌────────┐       ┌────────┐         ┌────────┐
 │ Intent │  spec  │ Arch + │ API │ Code   │ tests │ AI test│ green │ IaC +  │ approve │ Observe│
 │ →Spec  │ ─────▶ │ contr. │ ──▶ │ (agent)│ ────▶ │ gen +  │ ────▶ │ RCA bot│ ──────▶ │ + RCA +│
 │ (EARS) │        │ +feas. │     │ +review│       │ SAST   │       │ canary │   ▣     │ mitigate│
 └────────┘        └────────┘     └────────┘       └────────┘       └────────┘  HUMAN  └────────┘
      ▲                                                                          GATE        │
      └──────────────  closed loop: production errors (Sentry/AIOps) → new tasks ◀───────────┘
```

Five foundation layers run *underneath* every phase (detailed in [Technical Architecture](../analysis/02_technical_architecture.md), summarized in the Cross‑cutting foundations section below): the **models and the routing between a cheap/local one and a frontier one**; the **agent harnesses** (the editor and command‑line tools that give a model hands); the **integration layer** (the Model Context Protocol, which connects agents to your systems through one adapter); the **orchestration** that coordinates multiple agents; and the **governance and observability** that sit across all of it. The two structural decisions that define your architecture are *(a) how you split work between a cheap/local model and a frontier model,* and *(b) where the human approval gates sit.*

**How to read each phase below:** the objective, the main activities (the actual work), how AI improves it (proven patterns), concrete examples that work (named tools plus the relevant [IdeaLab](../idealab/sdlc_ideas.md) use cases), the guardrails, and what to measure.

---

## Upstream input — Requirements & spec intake (feeds Design)

This isn't strictly a "Design‑to‑Operate" phase, but it's the input the whole pipeline depends on: a clear, version‑controlled specification is the contract every downstream agent implements against ([Requirements deep‑dive](../phases/01_requirements.md)).

- **Main activities:** elicit and clarify intent; turn unstructured inputs (meetings, tickets, documents) into epics, user stories, and acceptance criteria; write the specification into a version‑controlled `specs/` folder; and decide how much rigour each piece of work needs.
- **How AI improves it:** it drafts product‑requirements documents and user stories and expands thin ones (adding the size limits, formats, edge cases, and acceptance criteria a human tends to skip), and it can run a "requirements‑gap analyser" that flags contradictions against your existing architecture decision records and API documentation *before* a ticket is marked ready for development.
- **Concrete patterns that work:** keep the `specs/` folder as first‑class, pipeline‑validated artifacts; write requirements in the constrained‑English EARS templates (for example, "WHEN a trigger occurs THE SYSTEM SHALL…") so they're atomic and testable; apply the "minimum rigour that removes ambiguity" rule (spec‑first for prototypes, spec‑anchored for production — not the fully spec‑as‑source approach yet); run requirements as a team loop where the AI proposes a plan and *asks clarifying questions* rather than assuming (Amazon's "Mob Elaboration"). Tools: GitHub Spec Kit, AWS Kiro. **IdeaLab use cases:** *AI project management bot*, *AI strategy & roadmap engagements*.
- **Guardrails:** keep humans authoritative on trade‑offs and prioritisation; treat a specification that has drifted out of sync with the code as a build failure; avoid sliding back into big‑upfront‑design (keep specs lightweight and iterative); and write security and other non‑functional requirements into the spec so they propagate to every downstream agent.
- **Measure:** specification ambiguity and completeness; downstream rework caused by unclear requirements; and the first‑pass success rate of agents working from a spec versus ad‑hoc prompting.

---

## Phase 1 — Design & Architecture

**Objective:** turn the agreed specification into a technical design — service decomposition, API and data contracts, technology and model selection, and an architecture that constrains the Build phase — *with feasibility tested before any code exists* ([Design deep‑dive](../phases/02_design.md)).

**Main activities.** Decompose the system into services and define the API and data contracts and state‑management plans. Choose the technology and, for AI‑powered features, the model strategy (frontier versus local; grounding the model with retrieval versus adapting it with fine‑tuning; how to build the retrieval/knowledge layer). Record decisions as architecture decision records, and run feasibility and quality‑attribute analysis against your *actual* stack. And — new in 2026 — govern the architecture the agents are *already* deciding implicitly, a phenomenon nicknamed "vibe architecting."

**How AI improves it.** Its standout use is **feasibility analysis against your specific stack** — not "is my architecture good?" but "show me exactly how it fails given *my* tools, limits, and service‑level targets." (For example, it flags an embedding‑API rate limit as the real bottleneck in a retrieval pipeline and prescribes batching plus a persistent vector store.) It also produces credible **first drafts** of API contracts, schemas, event‑driven patterns, and diagrams that an architect then refines. And when you feed agents accurate internal documentation plus persistent rules — "architecture as context" — you get aligned, non‑duplicative designs.

**Concrete examples that work.**
- **Make an "AI feasibility analysis" a required appendix to every architecture decision record**, generated by a model primed (via retrieval over your internal tech‑stack docs) and driven with stepwise prompts (risks → sensitivity points → quality‑attribute trade‑offs). A study applying this to the ATAM architecture‑evaluation method found it surfaced *more* risks than human reviewers — but with false positives, so treat it as expert‑gated decision support, not a verdict.
- **Encode architectural constraints as persistent agent instructions** (in `AGENTS.md` or `.cursorrules` files — "follow the hexagonal architecture pattern," plus explicit "never‑allow" security rules) so design *and* build agents stay aligned.
- **Govern "vibe architecting" with a three‑layer model** (from the "Architecture Without Architects" study): **constraints** that bound the allowed technology, **conformance** checks (plan‑before‑build workflows and complexity thresholds that trigger human review), and **knowledge** capture (extracting an architecture decision record from what the agent actually chose). Reinforce it with **architectural impact statements** — the agent declares the structural cost of a change ("adds a vector database and an embedding pipeline; +330 lines") so reviewers see the infrastructure consequences, not just the diff.
- **IdeaLab use cases:** *domain‑specific model fine‑tuning* (adapting a model rather than training from scratch), *knowledge‑graph construction for retrieval* and *retrieval‑augmented generation* (the knowledge layer), *enterprise technology modernization*, *cloud + foundation‑model integration*, and *agentic architecture concerns* (tool registries, memory, observability, evaluations, human checkpoints) — all tagged `Design`.

**Guardrails:** keep humans authoritative on systemic trade‑offs (for real architecture work AI is assistive, not autonomous, and still proof‑of‑concept‑grade); don't let advisory feasibility become rubber‑stamping (confirm or refute each flagged risk); and wrap generative design in deterministic validation (schema validators, contract tests, policy‑as‑code).

**Measure:** the reduction in architecture‑related rework; the accuracy of the feasibility predictions (did the flagged bottleneck actually materialise?); and the time from concept to an approved design.

---

## Phase 2 — Build (Development & Coding)

**Objective:** implement against the specification at speed *without* letting the extra throughput turn into accelerated debt. This is the most mature AI phase, and the one where the tension between productivity and quality is sharpest ([Build deep‑dive](../phases/03_build.md)).

**Main activities.** Implement features (in‑editor completion, chat, multi‑file agentic edits); scaffold modules to your conventions; validate configuration; and keep the human review gate.

**How AI improves it.** It's reliable on **bounded generation** (boilerplate, create/read/update/delete services, REST endpoints, SQL, user‑interface components) and on well‑scoped **multi‑file edits** — you keep the good ~30% of suggestions and reject the rest. It's genuinely useful for **configuration validation** (point an agent at your layered config files in a pre‑commit hook to catch conflicts before they hit runtime). And a **disciplined operating routine** is what turns a coding agent from a toy into a supervised junior developer.

**Concrete examples that work.**
- **Operating practices for driving the agent** (from Anthropic's Claude Code guidance, which generalises across tools): explore, then plan, then implement, then commit — separating research from coding so the agent doesn't confidently solve the wrong problem; *always give the agent a check it can run* (a test, a build, a linter, a screenshot comparison) so it self‑corrects — "if you can't verify it, don't ship it"; manage the context window deliberately (load schemas and full error traces, clear it between unrelated tasks, delegate file‑heavy investigation to sub‑agents); keep a short, pruned conventions file (`CLAUDE.md` / `AGENTS.md`) in version control; and have a *second* agent review the diff in a fresh context.
- **Two‑tier routing:** send the routine work — completions, config checks, quick edits — to a cheap or local model (such as Qwen3‑Coder or DeepSeek served on vLLM), and reserve the frontier model for the hard ~40%. This is the single biggest lever for both cost and data residency.
- **The delegation quadrant:** auto‑accept the high‑benefit, cheap‑to‑verify work (boilerplate, tests, documentation); apply your strongest review to high‑verification‑cost work (architecture, security‑critical code); and simply avoid handing an agent over‑engineered or unfamiliar work on critical paths.
- **IdeaLab use cases:** *AI pair‑programmer rollout*, *measurable productivity gains*, *enterprise return window* (all `Build`); *no‑code AI agent platforms* (low‑code building by non‑engineers); *network‑as‑code developer platform* (AI‑assisted code generation plus provisioning); and *synthetic documents for model training*.

**Guardrails:** the human review gate is **non‑negotiable** (treat AI output like a junior's first pull request); apply **identical static‑analysis, dependency, and secret‑scanning gates to AI and human code** — the fix is to gate AI, not to slow it down; **never run agents with permission‑bypass flags against a repository that holds secrets** (the Nx supply‑chain lesson); use **enterprise or zero‑retention tiers** for proprietary code; and have async agents push to a sandbox branch and open a pull request, never to the main branch.

**Measure (outcomes, not lines of code):** review latency, change‑failure rate, bugs and incidents per pull request, and rework/churn rate. Anchor the leadership narrative with the perception gap (developers *felt* 20% faster while measuring 19% slower) so decisions weigh measured outcomes over felt speed.

> **The headline risk to plan for:** 45% of AI‑generated code introduces a top‑ten vulnerability (Veracode), 19.7% of the packages models recommend don't exist, and 24.2% of AI‑introduced issues survive to the latest revision. AI lowers syntax errors but raises architectural and security flaws — uniform gates are mandatory, not optional.

---

## Phase 3 — Test & QA

**Objective:** verify that the build meets the specification, and **absorb the extra change volume the Build phase now generates.** This is the highest‑return phase and the recommended *first pilot* ([Test deep‑dive](../phases/04_test.md)).

**Main activities.** Generate unit, integration, and end‑to‑end tests and test data; keep a self‑healing test suite; select tests by risk; run AI‑assisted code review and security scanning; and, for AI‑powered features, run an evaluation harness.

**How AI improves it.** Generating a test is **bounded and instantly verifiable** — it passes or it fails — and it replaces work developers dislike, which is exactly why it's the best risk‑adjusted bet. **Self‑healing** execution repairs broken user‑interface references automatically (a defensible 25–50% cut in maintenance). And **AI‑assisted code review** as a first pass gives a modest, useful speed‑up (a 10–20% improvement in review time) — as an augment, not a replacement.

**Concrete examples that work.**
- **Assured‑improvement filtration — the single most transferable pattern in the whole plan** (from Meta's TestGen‑LLM): don't surface generated tests on trust; surface only the ones that provably *(a)* build, *(b)* pass reliably, and *(c)* measurably increase coverage, and silently discard the rest. This proves the improvement before anyone sees it, which is also what neutralises hallucination. (Open‑source equivalent: Qodo Cover; a deterministic alternative for Java: Diffblue.) **This is verification‑driven deployment made concrete — apply the same shape everywhere.**
- **Self‑healing on the existing suite plus coverage gates at pull‑request time**, and then have the build agent write tests for the features it ships so coverage keeps pace with generation (Mabl, Functionize, testRigor).
- **An independent test harness:** the agent that writes the tests is isolated from the agent that wrote the code, which prevents shared blind spots (a Google‑SRE‑mandated practice).
- **An evaluation harness for AI features:** judge models, regression suites, and drift monitors that score groundedness, citation quality, and accuracy.
- **IdeaLab use cases:** *automated test generation & documentation*, *AI code review assistant*, *retrieval‑evaluation harness*, *conversation‑data generation* and *synthetic documents for model training* (synthetic test and training data), and *guardrails* for synthetic data.

**Guardrails:** make "AI authors and heals; a human approves" the explicit rule — the engineer accepts, modifies, or rejects each recommendation; review generated tests (they can be plausible‑but‑wrong); remember AI review augments rather than replaces human review (agent‑only pull requests merged at 45% versus 68% for human‑reviewed ones); and keep tests in your repository in portable formats to avoid lock‑in.

**Measure:** coverage, edge cases found, the reduction in production defects, the flaky‑test rate, the false‑positive rate, and the time to detect security issues — never the raw count of tests generated.

---

## Phase 4 — Release (Deployment, CI/CD & Infrastructure‑as‑Code)

**Objective:** get validated changes into production *safely*, under tight guardrails. This is the phase where AI moves from "recommend" to "act," and where the cost of a mistake is highest — the "autonomy gap" ([Release deep‑dive](../phases/05_release.md)).

**Main activities.** Generate and review infrastructure‑as‑code (Docker, Kubernetes, Terraform); run the delivery pipeline; do root‑cause analysis on pipeline failures; roll out progressively with canaries; verify deployment health; and handle rollback or fix‑forward.

**How AI improves it.** It's good at **infrastructure‑code review** when it stays "propose, then a human approves" — reordering Dockerfile layers for caching, checking Kubernetes resource limits and GPU passthrough, validating Terraform and Helm against policy at pull‑request time. It's good at **automatic root‑cause analysis on failures** — reading the build logs and the diff, pinpointing the misconfigured proxy or dependency conflict, and commenting the cause and fix on the failed pull request. And it enables **deployment‑risk prediction** and **adaptive canaries** that catch subtle, multi‑variable regressions static thresholds miss.

**Concrete examples that work.**
- **Validate generated infrastructure code iteratively, never in one shot** (from the IaCGen study): run it through escalating stages — format check, then schema/policy lint, then a live dry‑run or ephemeral deploy — feeding each failure back to the model to refine. The blunt truth is that *"it parses" is not "it deploys" is not "it's secure"*: one‑shot generation deploys less than a third of the time and is security‑complete less than a tenth of the time.
- **The Terraform verification ladder** (from the TerraFormer study): `terraform validate` for syntax, then **`terraform plan`** as the highest‑signal gate (it catches hallucinated resources and attributes against the provider API), then TFLint plus policy checks (OPA/Rego, Checkov) — wired into the pull‑request gate before ArgoCD or Flux applies anything. This loop lifts deployability from about 42% to about 73%.
- **A copyable control model for release‑capable agents** (GitHub's Copilot coding agent): the agent can push only to branches it created (never the default branch, and it can't merge its own pull request), runs on ephemeral runners, and the pipeline is blocked until a human with write access approves. Add CODEOWNERS protection on the agent's config and a tight list of allowed network destinations.
- **Confidence‑thresholded remediation** plus **feature flags and AI‑assisted fix‑forward** — granular switches beat a blunt rollback, especially given the "intervening pull request" problem (rolling back to "last known good" can unwind interim fixes).
- **IdeaLab use cases:** *AI bug triage in the pipeline*, *SaaS‑AI co‑deployment* (integrating into existing stacks rather than building from scratch), *zero‑day vulnerability remediation* (correlate a vulnerability with the asset graph and draft a patch playbook), and *network‑as‑code*.

**Guardrails:** **no unsupervised production action** (the Replit production‑database deletion is the canonical failure); a **mandatory dry‑run** on any infrastructure interface an agent can call; route changes through a **deterministic actuation control plane** rather than raw scripts; and keep **circuit breakers and a global "stop" button** for in‑flight agentic actions.

**Measure (the DORA keys):** deployment frequency, change lead time, change‑failure rate, mean time to recovery, and failed‑deployment recovery time.

---

## Phase 5 — Operate (Maintenance, Observability & SRE)

**Objective:** keep production healthy. This is the most production‑validated AI phase outside coding, and the one that *closes the loop* back to planning ([Operate deep‑dive](../phases/06_operate.md), grounded in the [Google SRE paper](../articles/ai_in_sre_google_full_extract.md)).

**Main activities.** Monitoring and semantic observability; correlating and enriching alerts; detecting, summarising, and diagnosing incidents; gated autonomous mitigation; proactive bug triage; securing the agent runtime; and monitoring for data and model drift.

**How AI improves it.** **Semantic observability** — anomaly detection that learns what "normal" looks like per metric, correlates a spike with the recent commit that caused it, and posts a plain‑language summary — beats static "CPU over 80%" thresholds. **Alert enrichment before a human is paged** — a read‑only agent queries monitoring, logs, change‑logs, and dependency graphs in parallel within a couple of minutes and attaches verifiable, source‑linked context. **Incident summarisation and root‑cause analysis** (Google reports a ~10% cut in time‑to‑mitigate from an incident‑hypothesis assist alone, ~44% from AI investigation dashboards) and **proactive bug triage** (deduplicate, score by severity, route) attack the toil that eats most of a team's reactive‑operations time.

**Concrete examples that work.**
- **Graduated autonomy behind a control plane** (the Google SRE model): the "Safety Trifecta" — transparency (every action logs its reasoning), real‑time risk evaluation (an action's risk is scored against the current production context), and progressive authorisation (agents earn autonomy) — plus explicit autonomy levels from fully manual to fully autonomous. An "AI Operator" acts as first responder at the "propose, human approves" level and graduates to acting alone only on minor, bounded incidents, with its chain of thought exposed. Every production change routes through a deterministic "actuation agent" that does standardised discovery, a mandatory dry‑run, safety checks, post‑action verification, and an emergency stop.
- **Evaluation grounded in real operational memory:** capture how humans actually resolved past incidents, stratify the evaluation data by quality (a Bronze/Silver/Gold scheme), run continuous nightly evaluations combining an LLM‑as‑judge with strict deterministic scoring, and promote an agent to more autonomy only when it beats the known‑good data with statistical significance. This is the Operate‑phase version of verification‑driven deployment.
- **Platforms:** Datadog Bits AI, Grafana Assistant/Investigations, PagerDuty AIOps, incident.io, Dynatrace Davis; dedicated site‑reliability agents (AWS DevOps Agent, Azure SRE Agent — both generally available in early 2026, propose‑and‑approve); a Model Context Protocol server that exposes your telemetry to a reasoning model; and Sentry feeding production errors back as new agent tasks (the closed loop).
- **IdeaLab use cases:** *data quality monitoring* (schema drift and anomalies), *retrieval‑evaluation harness* drift monitors, *AI agent identity & runtime governance* and *runtime data protection*, *zero‑day remediation*, and a *large‑scale machine‑learning data platform* (the operations‑platform pattern).

**Guardrails:** AI‑for‑operations **augments; it doesn't replace** monitoring or humans (the model is "AI crunches the data, an engineer validates and acts"); **explainability is mandatory** (expose the logs and metrics it reasoned over, and keep an immutable, attributable trace of every action); use **zero‑trust, safe‑by‑default actuation** with a unique, least‑privilege identity for each agent; and design deliberately against feedback loops and cascading failures.

**Measure:** mean time to detect, mean time to mitigate and recover, alert‑noise reduction, root‑cause accuracy, the share of incidents detected automatically, and the false‑positive rate.

**Close the loop:** production insight (from Sentry or your operations tooling) becomes new agent tasks that flow back into [Requirements](../phases/01_requirements.md) and [Build](../phases/03_build.md). This feedback is where the AI‑augmented lifecycle becomes a *system* rather than a collection of point tools.

---

## Cross‑cutting foundations (build these alongside every phase)

These are the layers that make the phases work; they are not optional add‑ons.

**Models & infrastructure** ([Technical Architecture](../analysis/02_technical_architecture.md)). Adopt **two‑tier routing** — a cheap or local model for the routine ~60%, a frontier model for the hard ~40% — as the single highest‑leverage cost and privacy decision (RouteLLM is a reference router; vLLM serves a local Qwen3‑Coder or DeepSeek model). Use the **Model Context Protocol** as the integration layer — one adapter per system (GitHub, Jira, Sentry, observability, Terraform), with reads open, writes gated, and the agent treated as an untrusted caller. Keep components **self‑hostable and model‑agnostic** where data residency or intellectual property demands it, which also gives you a credible exit from any single vendor. And **orchestrate simply** — a generator‑plus‑evaluator pair beats a six‑agent assembly line until you've proven the need (multi‑agent workflows cost roughly fifteen times the tokens).

**Governance & risk** ([Implementation Planning §5](../analysis/03_implementation_planning.md)). Use the NIST AI Risk Management Framework as the backbone (its Govern / Map / Measure / Manage functions) plus its Generative AI Profile. Set graded **autonomy thresholds** per task risk (this maps to the OWASP "excessive agency" risk), keep **human approval gates** before irreversible actions, and extend your incident‑response plan to AI‑specific attacks (model poisoning, prompt injection — the number‑one risk on the OWASP list for large‑language‑model applications). On data: "doesn't train on your data" is not the same as "doesn't retain it," so use enterprise or zero‑retention tiers, and remember that on‑premises models remove the question entirely. On regulation: internal development tooling is generally minimal‑risk under the EU AI Act, *but* piping AI development metrics into individual performance dashboards makes it high‑risk (the employment category), so measure teams and improvement, never individuals. **IdeaLab use cases:** *AI responsible‑use policy*, *AI security & safety guardrails*, *human‑in‑the‑loop oversight*, *internal AI‑tool portfolio governance*.

**Security / DevSecOps.** The same quality and security gates for AI and human code; the agent treated as an untrusted caller (authenticate and rate‑limit every tool call); no permission‑bypass on repositories with secrets; and security requirements written into the specification and the agent's rule files so they're enforced from day one.

**Observability & metrics** ([Implementation Planning §6](../analysis/03_implementation_planning.md)). Instrument the *whole pipeline*, not just generation throughput. Use the DORA five keys (deployment frequency, change lead time, change‑failure rate, failed‑deployment recovery time, and rework rate) and DX Core 4. Track outcomes — change‑failure rate, review latency, bugs and incidents per pull request, the share of time spent on new capabilities — never lines of code or the raw acceptance rate.

**People & organization** ([Implementation Planning §8](../analysis/03_implementation_planning.md)). Demand is bottom‑up; discipline is top‑down. Protect junior skill development (the "explain before you accept" norm); keep enough psychological safety that AI use isn't hidden; and plan the role shift toward orchestration, validation, and architecture. Adopt across the product‑management/engineering boundary, not just inside engineering, or you build faster silos. **IdeaLab use cases:** *AI‑aware org design*, *digital & AI transformation programs*, *AI strategy & roadmap engagements*.

---

## The rollout — crawl, walk, run (gated on stability)

Sequence the *implementation* of everything above so each step is gated on the previous step's stability metrics, not on the calendar ([Implementation Planning §7](../analysis/03_implementation_planning.md)).

| Stage | Timebox | What you turn on | Gate to advance |
| --- | --- | --- | --- |
| **0 — Foundation & governance** | weeks 0–6 | Assess foundation‑readiness; stand up governance (acceptable‑use policy, the NIST framework, autonomy thresholds, human approval gates, data‑residency rules, an AI incident‑response plan); capture a **baseline** on the DORA / DX Core 4 metrics; pick one or two healthy pilot teams | Governance live; baseline captured |
| **1 — High‑value, verifiable pilots** | months 1–3 | Test generation (behind coverage gates) → AI‑assisted code review (augment; humans merge) → a pipeline root‑cause bot (propose‑then‑approve); roll out gated coding assistants (enterprise tier, secret scanning on) | Stability metrics holding *or improving*, not just throughput up |
| **2 — Workflow integration** | months 3–9 | Self‑healing user‑interface tests; observability (alert correlation and bug triage); infrastructure‑code review; AI security scanning in the pull‑request pipeline; spec‑driven development (greenfield and modernization); two‑tier routing | Change‑failure rate and rework flat or down at wider scope |
| **3 — Orchestrated & agentic** | months 9–24 | Generator‑plus‑evaluator agent pairs; async sandbox‑to‑pull‑request agents where the evaluation discipline is proven; narrow, confidence‑thresholded auto‑remediation (restart, scale, roll back) behind an actuation control plane | Each autonomy step earns the next, watching stability |

> **The stage‑gate rule:** advance only when the previous stage's *stability* metrics held. Speed that arrives with falling stability is the failure mode, not progress.

How the rollout maps onto the phases: Stage 1 lands in Test, Build, and Release (the verifiable wins); Stage 2 broadens into Design (spec‑driven development), Release, and Operate; and Stage 3 is where Operate autonomy and multi‑agent orchestration mature.

---

## Realism — what to expect (so the plan survives contact)

- **Gains are modest and conditional:** roughly 7–20% throughput on suitable work, and *negative* on expert open‑ended work. Budget accordingly.
- **The hidden costs are real:** the senior‑engineer review tax (reviews take about 200% longer and land on your seniors), the reliability tax (43% of AI changes need debugging in production), and usage overruns (Gartner expects 40% of teams to exceed budget by more than 2×). The net gain is *generation savings minus (review tax + reliability tax + token and seat cost)* — and it can go negative on a weak foundation.
- **The "70% problem":** AI gets you 70% of the way fast; the last 30% — edge cases, error handling, security, integration, polish — is where the human effort concentrates, and it's exactly what the gates protect.
- **What doesn't work yet (don't pilot these):** fully autonomous testing, autonomous closed‑loop remediation, AI review replacing humans, and greenfield architecture by an agent (see [Benchmark §4](../analysis/01_business_benchmark.md)).

---

## The plan in one page

1. **Amplifier, not autopilot** — fix the foundation before scaling AI; on a weak one the return goes negative.
2. **The spec is the contract** — version‑controlled, pipeline‑validated specifications feed every phase; AI drafts, humans clarify and own.
3. **Design:** feasibility analysis against your real stack, plus governing "vibe architecting" (constraints → conformance → decision‑record extraction).
4. **Build:** gated coding assistants + two‑tier routing + the explore‑plan‑verify workflow; the same security gates for AI and human code; never bypass permissions on repositories with secrets.
5. **Test (pilot this first):** assured‑improvement filtration ("build, pass, increase coverage — or discard") — verification‑driven, not trust‑driven; AI authors and heals, a human approves; AI review augments, never replaces.
6. **Release:** propose‑then‑approve infrastructure‑code review, plus the "validate → plan → policy" ladder and iterative deploy‑validation; the copyable agent‑release gate (branch isolation, human approval before the pipeline runs, mandatory dry‑run, a stop button); and the reminder that "it parses" is not "it deploys" is not "it's secure."
7. **Operate:** semantic observability, alert enrichment, and proactive triage; graduated autonomy (the Safety Trifecta and autonomy levels) behind a deterministic control plane; evaluations grounded in known‑good data; and *close the loop* back to planning.
8. **Across all phases:** the Model Context Protocol for integration (reads open, writes gated), the NIST framework for governance, uniform security, DORA / DX Core 4 outcome metrics, and a trust ladder for autonomy.
9. **Roll out crawl‑walk‑run**, gated on *stability*, starting with the verifiable few (test generation → AI review → root‑cause bot).
10. **The win condition:** turn acceleration into *delivered, stable, secure* software — and instrument the pipeline well enough to prove you did.

---

## Source documents

- Evidence & realism: [Business Benchmark](../analysis/01_business_benchmark.md)
- Tools, models, pipeline: [Technical Architecture](../analysis/02_technical_architecture.md)
- Roadmap, cost, governance, SWOT: [Implementation Planning](../analysis/03_implementation_planning.md)
- Per‑phase deep‑dives and grounding sources: [Requirements](../phases/01_requirements.md) · [Design](../phases/02_design.md) · [Build](../phases/03_build.md) · [Test](../phases/04_test.md) · [Release](../phases/05_release.md) · [Operate](../phases/06_operate.md) · [grounding index](../references/sdlc_phases.md)
- Operate reference: [Google SRE — "AI in SRE"](../articles/ai_in_sre_google_full_extract.md)
- Concrete use‑case candidates (phase‑tagged): [IdeaLab SDLC ideas](../idealab/sdlc_ideas.md)

*This master plan synthesizes the project's research base as of June 2026. Treat fast‑moving figures, model names, and tool statuses as point‑in‑time; re‑verify before formal budgeting or procurement.*
