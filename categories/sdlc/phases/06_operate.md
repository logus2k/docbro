# Phase 6 — Operate (Maintenance, Observability & SRE)

### How AI is applied to production operations, observability, and incident response (mid‑2026)

---

## 1. What this phase is for

This phase keeps production healthy: monitoring and observability, anomaly detection, alert correlation, incident detection and triage, root‑cause analysis, increasingly some autonomous mitigation, and proactive bug triage. Two things make it strategically important and worth investing in early.

First, **it has outsized ROI.** Maintenance and operations account for an estimated 60–80% of the total engineering effort over a product's lifetime — the biggest and most under‑tooled slice of the SDLC — so even modest AI coverage (log triage, dependency updates, incident RCA) compounds into large returns. Outside of coding itself, this is the most production‑validated place to apply AI.

Second, **this is where the lifecycle closes the loop.** Production insight — the errors users actually hit, the incidents that actually fire — should feed back as new work for planning and build. When it does, the AI‑augmented SDLC stops being a collection of point tools and becomes a system that learns from its own operation.

## 2. Where AI genuinely helps (and where it doesn't)

**What works — this is the strongest non‑coding phase.**
- **Semantic observability.** Instead of static thresholds ("CPU > 80%"), AI learns what "normal" looks like per metric and per time window, detects a real deviation, and — the valuable part — traces it back to the specific commit that caused it and summarises the incident in plain language.
- **Alert enrichment before a human sees it.** A read‑only agent, working within a tight time budget, queries monitoring, logs, change‑logs, and dependency graphs in parallel and appends verifiable, source‑linked context to the alert — so the on‑call engineer starts with "latency spike likely caused by slow DB queries in service X since 10:23," not a wall of red.
- **Incident summarisation and RCA.** Synthesising logs, metrics, traces, recent deploys, and similar past incidents into a credible hypothesis meaningfully cuts time‑to‑mitigate (Google reports ~10% from a hypothesis assist alone, ~44% from AI‑curated investigation dashboards).
- **Detection from unstructured signals.** Clustering user feedback from support and forums catches novel outages that telemetry misses entirely.
- **Proactive bug triage.** Auto‑categorising incoming reports, deduplicating against existing issues, scoring by impact, and routing to the right team — mature, low‑risk, and it attacks the toil that eats most of a team's reactive‑ops time.

**Where it doesn't — plan for these limits.**
- **Fully autonomous closed‑loop remediation** only works in narrow, well‑known scenarios; over‑eager auto‑remediators cause cascading degradation.
- **AIOps sits *on top of* monitoring; it doesn't replace it.** The durable model is "AI crunches the data, the engineer validates and acts."
- **Opaque conclusions don't earn trust.** Teams need to see the logs and metrics the AI reasoned over — explainability here is a hard requirement, not a nicety.
- **Naive metric anomaly detection produces noise** — a statistical blip isn't user impact, and without an understanding of intent a normal launch reads as a failure.

## 3. The activities — what to actually do

**a) Build semantic observability on the telemetry you already have.** Add anomaly detection that learns per‑metric, per‑window normals; expose your telemetry to a reasoning model (an MCP server is the clean way to do this); and have it correlate a spike with recent commits and post a narrative summary. You don't need to replace your observability stack — you layer intelligence on top of it.

**b) Enrich alerts automatically, read‑only, before they page a human.** Stand up an agent that, on alert, gathers context from monitoring/logs/change‑logs/dependency graphs in parallel and attaches a source‑linked summary. Keep it strictly read‑only — its job is to accelerate the human responder, not to act.

**c) Run a proactive bug‑triage agent.** Wire it to your error tracker (Sentry/Bugsnag) and issue tracker (Jira/Linear) and let it own deduplication, severity scoring, and routing. This is one of the safest, highest‑leverage automations in the whole SDLC.

**d) Close the loop back to development.** Feed production errors (via Sentry, for example) back as new agent tasks, and let operational insight flow back into specs and planning. This feedback loop is the difference between "we bought some AIOps tools" and "our SDLC improves itself."

**e) Introduce autonomous mitigation only through a control plane, and only by degrees.** When you do let AI *act* on production, adopt the "Safety Trifecta" — **transparency** (every action logs its chain of thought), **real‑time risk evaluation** (an action's risk is scored against the current production context before it runs), and **progressive authorisation** (agents earn autonomy). Use explicit autonomy levels (L0 manual → L4 fully autonomous), keep agents at "propose, human approves" (L2) until they've proven themselves on bounded scenarios (L3), and route *every* production change through a deterministic **actuation agent** that does standardised discovery, a mandatory dry‑run, safety checks, and post‑action verification — with an emergency "red button" to pause all in‑flight agentic actions.

**f) Gate that autonomy on evaluation grounded in real operational memory.** Capture how humans actually resolved past incidents ("trajectories"), stratify your evaluation data by quality (Bronze/Silver/Gold), and run continuous nightly evaluations that combine an LLM‑as‑judge with strict deterministic scoring. Promote an agent to a higher autonomy level only when it beats the human‑verified "golden" data with statistical significance. This is the Operate‑phase form of *verification‑driven, not trust‑driven* — you prove the agent is good before you let it act.

## 4. How to pilot this

The safe, high‑value entry points are **alert correlation/enrichment, incident summarisation, and proactive bug triage** — all production‑validated and low‑risk, a strong second wave after your test‑first pilots. Sequence: alert enrichment + correlation → incident summarisation and RCA (as an assist) → proactive triage → and only then, narrow, confidence‑thresholded auto‑remediation (L2 → L3) behind the control plane described in activity (e).

## 5. Guardrails & what to watch for

- **Keep humans on the gate; expand autonomy on a trust ladder** (L0 → L4), one bounded scenario at a time.
- **Zero‑trust, safe‑by‑default actuation** — agents never hold standing production credentials or run raw scripts; they go through the control plane with deterministic checks and circuit breakers.
- **Explainability is mandatory** — expose the reasoning and persist an immutable, attributable trace of every autonomous action.
- **Design against feedback loops and cascading failures**, and keep the global "red button" within reach.

## 6. How you'll know it's working

- **Mean time to detect (MTTD)** and **mean time to mitigate/recover (MTTM/MTTR)** — the core signals.
- **Alert‑noise reduction** — how much of the wall of red the correlation layer removes.
- **RCA accuracy** — how often the flagged root cause was the real one.
- **Manual investigation time saved** and **% of incidents auto‑detected**.

## 7. Tools to reach for

| Need | Options |
| --- | --- |
| **AIOps platforms** | Datadog Bits AI, Grafana Assistant/Investigations, PagerDuty AIOps, incident.io AI SRE, Dynatrace Davis (causal) — they sit on top of telemetry; keep the audit trail |
| **SRE agents** | AWS DevOps Agent, Azure SRE Agent (both GA early 2026) — propose‑and‑approve beyond narrow, known‑pattern remediation |
| **Integration** | An MCP server exposing telemetry to a reasoning model; Sentry's MCP feeding errors back as new tasks — reads open, writes gated |
| **Kubernetes ops** | K8sGPT (open‑source) — deterministic analyzers first, optional LLM enrichment |
| **Models** | Frontier reasoning for correlation/summarisation; fine‑tuned on your internal tooling and failure patterns; RAG over runbooks and incident history |

## 8. Evidence & sources

*Reliability tags: [PRIMARY] · [ANALYST] · [VENDOR] (treat as a ceiling) · [3P].*

- **Market & maturity:** AIOps is a fast‑maturing market (~$11B in 2025, projected ~$33B by 2029); by 2026 ~84% of orgs have explored or piloted AI in observability, and Gartner projects >60% of large enterprises will run self‑healing systems [ANALYST].
- **Impact:** MTTR improvements cluster around a ~40% industry mid‑point (vendor band 30–92%) — use 40% as the planning figure. The strongest hard number in the whole SDLC is IBM's −80 days / −$1.9M on breach identify‑and‑contain with extensive AI/automation [PRIMARY]. Google's own A/B‑tested figures: ~10% MTTM reduction from an incident‑hypothesis assist, ~44% from AI investigation dashboards [VENDOR, primary‑internal].
- **Grounding source** (full review in [references/sdlc_phases.md](../references/sdlc_phases.md)):
  - **Google — "AI in SRE: How Google is Engineering the Future of Reliable Operations"** — the deep reference for this phase: the Safety Trifecta, autonomy levels L0–L4, the Actuation Agent and "red button", Detectr, the AI Operator (L2 → L3), and evaluation grounded in Bronze/Silver/Gold operational memory. Full extract: [../articles/ai_in_sre_google_full_extract.md](../articles/ai_in_sre_google_full_extract.md).

---

*Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md) · [Technical Architecture](../analysis/02_technical_architecture.md) · [Implementation Planning](../analysis/03_implementation_planning.md) · previous → [Release](05_release.md) · the loop closes back to [Requirements & Planning](01_requirements.md) and [Build](03_build.md).*
