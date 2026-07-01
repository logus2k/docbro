# Phase 4 — Test & Quality Assurance

### How AI is applied to testing, QA, and security scanning (mid‑2026)

---

## 1. What this phase is for

This phase verifies that the build meets the spec — unit, integration, and end‑to‑end tests, test data, self‑healing UI tests, risk‑based test selection, AI‑assisted code review, and security scanning. It has a second, strategic job in an AI‑accelerated lifecycle: **it's the release valve that absorbs the extra change volume the build phase now generates.** If generation speeds up and verification doesn't, the extra throughput just becomes faster‑arriving defects.

It's also the best place to *start* with AI, for a simple reason: the work here is **bounded and instantly verifiable** — a test either passes or it doesn't — and it replaces work developers dislike. That combination (clear value, cheap verification, small blast radius) makes it the highest‑ROI, lowest‑risk entry point in the whole SDLC. If you pilot AI in exactly one phase, pilot it here.

## 2. Where AI genuinely helps (and where it doesn't)

**What works.** AI generates unit, integration, regression, and API tests — including the null/negative/overflow/boundary edge cases a human tends to skip — from requirements, code, or even observed user sessions. **Self‑healing** test execution repairs broken UI locators automatically when an element changes, cutting maintenance meaningfully. **Risk‑based selection** runs only the tests a change is likely to affect, keeping suite runtime flat as coverage grows. And AI **code review** as a first pass gives a modest, useful speed‑up on PR triage.

**Where it doesn't — plan for this.** The most important thing to understand is that **raw generation is a funnel, not a faucet.** Of the tests a model generates, roughly 75% build, ~57% pass reliably, and only ~25% actually increase coverage — and across a large industrial run it improved barely 11% of the classes it touched. Most generated tests are *not* useful. The value comes from *filtering*, not volume — which is the whole design of activity (a) below. Two further limits:

- **Strong on benchmarks, weaker on your real code.** The same generators that score well on public benchmarks do poorly on real projects, because they struggle to reason about actual control flow and execution state.
- **AI review augments, it doesn't replace, human review.** In a large study, code‑review‑agent‑only PRs merged at 45% versus 68% for human‑reviewed, and about 60% of agent feedback was low‑signal. Use it as a first pass, not a gate.

Two traps worth naming: **plausible‑but‑wrong tests** that create false confidence, and **circular validation**, where tests quietly encode the same wrong assumptions as the code they're testing.

## 3. The activities — what to actually do

**a) Adopt "assured‑improvement filtration" — the single most transferable pattern in the whole SDLC.** Don't surface generated tests on trust. Run every candidate through deterministic filters and keep only the ones that **(1) build, (2) pass reliably (not flaky), and (3) measurably increase coverage** — and silently discard the rest. This inverts the usual risk: instead of trusting the model's output, the system *proves* the test is an improvement before a human ever sees it, which is also what neutralises hallucination. This is the concrete, provable form of the principle that runs through this entire project — **verification‑driven, not trust‑driven** — and you can apply the same shape to any AI‑generation task.

**b) Turn on self‑healing for the existing suite, behind PR‑time gates.** Enable self‑healing so a changed locator is repaired automatically, but surface every fix as a reviewable diff and block merges on failure. Then have the build agent author tests for the features it ships, so coverage tracks generation throughput rather than falling behind it. This is exactly what counters the review/quality bottleneck coming out of [Build](03_build.md).

**c) Make "AI authors, human approves" the explicit rule.** The engineer accepts, modifies, or rejects each recommended test — nothing enters the regression suite unreviewed. Under this model, industrial deployments see high acceptance rates *because* a human is curating; it's the discipline, not the autonomy, that makes it work.

**d) Keep the test‑writing agent separate from the code‑writing agent.** Use an **independent harness**: the agent that writes tests must not be the one that wrote the code. This prevents cross‑bias and mechanically catches correctness requirements the coding agent quietly assumed. (Google's SRE practice mandates exactly this for the agentic SDLC.)

**e) Run risk‑based test selection** to keep CI fast as the suite grows — execute only the tests a change is likely to affect rather than the whole suite on every push.

**f) Generate synthetic test data and pipeline‑resilience tests.** Use AI to produce large, varied, realistic datasets for stress and edge‑case testing, and — for data‑heavy systems — point an agent at your Airflow DAGs or DVC scripts to synthesise tests that simulate data anomalies and execution failures, posted as PR comments for review.

**g) Shift security left, and let AI accelerate threat modelling.** Put SAST and dependency/secret scanning on every PR, with AI explaining vulnerabilities and proposing patches for a human to accept. Add **AI‑augmented threat modelling**: an LLM reads the design or the diff, extracts the relevant context, and proposes STRIDE‑style attack scenarios for a human to triage — which makes the threat‑modelling step teams usually skip actually happen.

> **The umbrella name for all of this is "Harness Engineering"** — building the control systems *around* AI‑generated code (tests, monitors, architectural constraints, orchestration) so its output is safe to trust. It's the named form of this project's verification‑driven spine, and it's a Scale‑ring (proven, standardise‑now) practice.

## 4. How to pilot this

**Pilot this phase first.** The sequence that works: start with **test generation in CI behind coverage gates** (the assured‑improvement filter), add **AI‑assisted PR review as an augment** (humans still merge), and bring in **self‑healing UI tests** in a second wave. Because everything here is bounded and instantly verifiable, you get fast, low‑risk wins that build organisational confidence for the harder phases.

## 5. Guardrails & what to watch for

- **Review the generated tests** — they can be plausible‑but‑wrong, and an unreviewed bad test is worse than no test because it manufactures false confidence.
- **Keep tests in your repo, in portable formats.** Many platforms store non‑portable tests; that's a lock‑in risk you can avoid up front.
- **Don't let AI review replace humans** — it's a first pass, not a merge gate.
- **Treat security scanning as first‑class** — the same gates for AI and human code, and stay alert to hallucinated dependencies.

## 6. How you'll know it's working

- **Coverage** and, more tellingly, **edge cases found** that the team wasn't testing before.
- **Production‑defect reduction** and **flaky‑test rate**.
- **False‑positive rate** on AI review and security findings (a noisy tool gets ignored).
- **Mean time to detect** security issues.
- Not "number of tests generated" — volume is the wrong target here; *verified improvement* is the product.

## 7. Tools to reach for

| Need | Options |
| --- | --- |
| **Self‑healing E2E** | Mabl, Functionize, Testim (Tricentis), Katalon (auditable), testRigor — surface fixes as reviewable diffs |
| **Unit‑test generation** | Diffblue Cover (Java; symbolic/deterministic, reproducible), Qodo (multi‑language LLM; Qodo Cover is open‑source), Early (JS/TS/Python) — prefer deterministic where audit matters, LLM for breadth |
| **CI integration** | Qodo Cover as a PR‑triggered step, Diffblue as a build step; enforce coverage gates so AI fills the gaps the gate exposes |
| **Security scanning** | SAST + dependency/secret scanning on every PR; AI explain‑and‑patch (Snyk DeepCode, GitHub Advanced Security) |
| **Models** | Frontier for complex test reasoning; local models are fine for routine generation — kept in an independent harness from the build agent |

## 8. Evidence & sources

*Reliability tags: [PRIMARY] · [RCT/PEER] · [VENDOR] (treat as a ceiling) · [3P].*

- **Adoption:** ~89% of orgs are piloting or deploying GenAI in QA, but only ~15% at enterprise scale, with an average ~19% productivity boost (World Quality Report 2025) [PRIMARY‑ish survey]; 68% of DevOps teams use AI in delivery (Tricentis 2026) [VENDOR].
- **The funnel:** of generated tests ~75% build / ~57% pass reliably / ~25% increase coverage, improving ~11.5% of classes (Meta TestGen‑LLM) [VENDOR, industrial]. Raw LLM generation reaches ~70% statement / ~53% branch coverage on benchmarks but does poorly on real projects (TestPilot) [RCT/PEER].
- **AI review ≠ human review:** agent‑only PRs merged 45.2% vs 68.4% human‑only; ~60% of agent feedback low‑signal (MSR 2026) [RCT/PEER].
- **Grounding sources** (full reviews in [references/sdlc_phases.md](../references/sdlc_phases.md)):
  1. **"Automated Unit Test Improvement using LLMs at Meta" (TestGen‑LLM)** (arXiv, Feb 2024) — the assured‑improvement‑filter pattern and the honest funnel numbers. https://arxiv.org/abs/2402.09171
  2. **"An Empirical Evaluation of Using LLMs for Automated Unit Test Generation" (TestPilot)** (IEEE TSE / arXiv 2302.06527) — the coverage baseline and the "poor on real projects" caveat. https://arxiv.org/abs/2302.06527
  3. **Shiplight — "AI in Test Automation: The Complete 2026 Guide"** [VENDOR] — the "supervised autonomy" framing and the production‑ready‑vs‑maturing split. https://www.shiplight.ai/blog/ai-in-test-automation
  4. **LTM — "SDLC AI Radar 2026"** [VENDOR] — *Harness Engineering* (per Martin Fowler) and *AI‑Augmented Threat Modelling*. `../articles/ltm_sdlc_ai_radar_2026.md`

---

*Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md) · [Technical Architecture](../analysis/02_technical_architecture.md) · [Implementation Planning](../analysis/03_implementation_planning.md) · previous → [Build](03_build.md) · next → [Release](05_release.md).*
