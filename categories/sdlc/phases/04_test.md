# Phase 4 - Test & Quality Assurance

### How AI is applied to testing, QA, and security scanning (mid‑2026)

*Audience: technical leadership. Part of the per‑phase SDLC series. Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md), [Technical Architecture](../analysis/02_technical_architecture.md), [Implementation Planning](../analysis/03_implementation_planning.md). Reliability tags: **[RCT/PEER]** · **[PRIMARY]** · **[VENDOR]** · **[3P]** · ⚠️ contested.*

---

## 1. What this phase covers

Verifying that the build meets the spec: unit/integration/E2E test generation, synthetic test data, self‑healing UI tests, risk‑based test selection, flaky‑test detection, and AI‑assisted code review and security scanning. **This is the single highest‑ROI phase for AI** — and the recommended first pilot.

## 2. Adoption status & evidence

- **High and rising.** ~89% of orgs are piloting or deploying GenAI in QA (World Quality Report 2025, 2,000+ executives) but only **15% at enterprise scale**, with an average **19% productivity boost** **[PRIMARY‑ish survey]**. Tricentis 2026: **68%** of DevOps teams use AI in delivery **[VENDOR]**.
- **Why it's the best risk‑adjusted bet:** the task is **bounded and instantly verifiable** (a test passes or fails), and it replaces work developers dislike. Generative AI is now ranked the #1 skill for quality engineers.

## 3. What works

- **Test generation** from requirements, code, or observed user sessions — unit, integration, regression, API tests; edge cases (null/negative/overflow/boundary) a human might miss. Empirical grounding: the TestPilot study ([source ②](#10-grounding--further-reading)) reports **median 70.2% statement / 52.8% branch coverage** from raw LLM generation — beating prior automated tools — and Meta's industrial TestGen‑LLM ([source ①](#10-grounding--further-reading)) had **73% of its recommendations accepted by engineers** for production.
- **Self‑healing execution** (most production‑proven) — when a UI element changes, the tool repairs the broken locator automatically (rule‑based fallback or intent‑based re‑resolution). Defensible maintenance reduction **25–50%** (vendor claims of 85–95% ⚠️ are unaudited).
- **Risk‑based / intelligent test selection** — run only the tests a change is likely to affect, keeping suite runtime flat as coverage grows.
- **Synthetic test data** — generate large, realistic, varied datasets for stress and edge‑case testing.
- **Pipeline‑aware test generation** — point an agent at Airflow DAGs / DVC scripts to synthesize tests simulating data anomalies and execution failures (the source‑notes pattern), posted as PR comments for review.
- **AI code review (first pass)** — Microsoft internal: ~10–20% PR‑time improvement **[VENDOR]**; useful as an *augment*.

## 4. What doesn't work (yet)

- **Raw generation is a funnel, not a faucet.** Meta's TestGen‑LLM data is the honest yardstick: of generated tests, **~75% built, ~57% passed reliably, and only ~25% increased coverage** — and across a test‑a‑thon it improved just **11.5% of the classes** it was applied to ([source ①](#10-grounding--further-reading)). Most generated tests are *not* useful; the value comes from filtering, not volume.
- **Strong on benchmarks, weaker on your real code.** The same LLM test generators that score well on public benchmarks have *"poor performance for open‑source projects based on coverage"* ([source ②](#10-grounding--further-reading)) — deep reasoning about control flow and execution state is the limiter.
- **Fully autonomous testing with zero oversight** remains "conference‑demo magic." What is production‑ready in 2026 is **supervised autonomy** — agents that explore, generate, and *flag*, with humans reviewing before tests enter the regression suite; still‑maturing are fully autonomous test interpretation and complex business‑logic generation ([source ③](#10-grounding--further-reading)). The reliable model is **"AI authors and heals; human approves."**
- **Plausible‑but‑wrong tests** create false confidence — keep generated tests in your repo and review them.
- **AI review cannot replace human review:** MSR 2026 (3,109 PRs) — code‑review‑agent‑only PRs merged **45.2% vs 68.4% human‑only**; ~60% of agent feedback was low‑signal **[RCT/PEER]**.
- **"Circular validation"** — tests that mirror the same flawed assumptions as the code; mitigate with independent generation harnesses (the test‑authoring agent isolated from the code‑authoring agent).
- **Security false confidence** — Snyk: >75% of devs think AI code is more secure while ~48% is insecure; only ~10% scan most AI code.

## 5. Tools, models & frameworks

| Category | Options | Notes |
| --- | --- | --- |
| **Self‑healing E2E** | Mabl, Functionize, Testim (Tricentis), Katalon (auditable), testRigor | Commercial SaaS; surface fixes as reviewable diffs |
| **Unit‑test generation** | **Diffblue Cover** (Java; symbolic/deterministic, reproducible), **Qodo** (multi‑lang LLM; **Qodo Cover OSS**), Early (JS/TS/Py) | Deterministic where audit matters; LLM for breadth |
| **CI integration** | Qodo Cover as PR‑triggered step; Diffblue as build step; enforce **coverage gates** | AI fills gaps so gates pass |
| **Security scanning** | SAST + dependency/secret scanning on every PR; AI explains+patches (Snyk DeepCode, GitHub Advanced Security) | Same gates for AI and human code |
| **Models** | Frontier for complex test reasoning; local models fine for routine generation | Independent harness from the build agent |

## 6. Concrete patterns to adopt

1. **Assured‑improvement filtration — the single most transferable pattern (Meta TestGen‑LLM, [source ①](#10-grounding--further-reading)).** Don't surface generated tests on trust; surface only those that *provably improve* the suite. Run every candidate through deterministic filters — **(a) it builds, (b) it passes reliably (not flaky), (c) it measurably increases coverage** — and silently discard the rest. This *"inverts the typical LLM deployment risk: rather than trusting generated output, the system proves improvement before surfacing it,"* which is also what neutralizes hallucination. The principle generalizes to any AI‑generation task in the SDLC: **verification‑driven deployment, not trust‑driven deployment.**
2. **Self‑healing on the existing suite + PR‑time gates** — enable healing (fixes as reviewable PR diffs), block merge on failure, then let the build agent author tests for features it ships so coverage tracks generation throughput. Directly counters the review/quality bottleneck from [build](03_build.md).
3. **"AI authors, human approves"** as the explicit operating rule — the engineer accepts/modifies/rejects each recommendation (Meta saw 73% acceptance under this model).
4. **Independent test harness** — the agent that writes tests is isolated from the agent that writes code (prevents cross‑bias; catches untested correctness requirements mechanically). This is also a Google‑SRE‑mandated practice for the agentic SDLC.
5. **Risk‑based selection** to keep CI fast as suites grow.
6. **Pipeline‑resilience tests** for data workflows (Airflow/DVC anomaly + failure simulation).
7. **Shift security left** — continuous threat modeling in the IDE/PR; auto‑patch suggestions with human accept. **AI‑augmented threat modelling** (an LLM extracts context from the design/diff and proposes STRIDE‑style attack scenarios for a human to triage) accelerates the modelling step teams usually skip.

## 7. Implementation priorities (80/20)

- **Value/effort:** **High value, low effort — pilot this FIRST.** Bounded, instantly verifiable, replaces disliked work, small blast radius.
- **Sequence:** test generation in CI (coverage gates) → AI‑assisted PR review (augment, human merges) → self‑healing UI tests (Phase 2).
- **Metrics:** test coverage, manual test‑creation time saved, edge cases found, production‑defect reduction, flaky‑test rate, false‑positive rate, MTTD for security issues. Avoid treating raw "tests generated" as success.

## 8. Risks & governance

- **Review generated tests** — they can be plausible‑but‑wrong.
- **Avoid vendor lock‑in** — keep tests in your repo in portable formats (many platforms store non‑portable tests).
- **Don't let AI review replace humans** — augment only.
- **Treat security scanning as first‑class** — uniform gates, scan AI suggestions, watch hallucinated dependencies.

## 9. Key takeaways

1. **Testing/QA is the best risk‑adjusted place to start with AI** — bounded, verifiable, high‑ROI.
2. **"AI authors and heals; human approves"** — self‑healing + coverage gates is the proven operating model, and it directly absorbs the extra change volume AI generates upstream.
3. **Independent harnesses and uniform security gates** keep generated tests and code honest — verifiability is what makes this phase win.
4. **Filter, don't trust.** The defining lesson from the best industrial deployment: generate many candidates, keep only the ones that provably build, pass, and raise coverage. Volume is cheap; *verified* improvement is the product.
5. **This discipline has a name: "Harness Engineering"** — building the control systems *around* AI‑generated code (tests, monitors, architectural constraints, orchestration) so its output is safe to trust. Coined by Martin Fowler; a Scale‑ring trend in the LTM SDLC AI Radar — and effectively the named form of this project's *verification‑driven, not trust‑driven* spine.

## 10. Grounding & further reading

*Curated, quality‑reviewed sources behind this phase (full review in [references/sdlc_phases.md](../references/sdlc_phases.md)).*

① **"Automated Unit Test Improvement using LLMs at Meta" (TestGen‑LLM)** (arXiv, Feb 2024) — https://arxiv.org/abs/2402.09171
   *Gold‑standard industrial primary source.* First report of industrial‑scale deployment of LLM‑generated tests with assured‑improvement guarantees. The transferable principle — **verification‑driven, not trust‑driven deployment** (build → pass reliably → increase coverage filters) — plus honest funnel numbers (75%/57%/25%; 11.5% of classes improved; 73% engineer acceptance). Open‑source reimplementation: Qodo Cover.

② **"An Empirical Evaluation of Using LLMs for Automated Unit Test Generation" (TestPilot)** (IEEE TSE 2023 / arXiv 2302.06527) — https://arxiv.org/abs/2302.06527
   *Citable academic baseline.* Median 70.2% statement / 52.8% branch coverage, beating prior tools — with the honest caveat that performance is *poor on real open‑source projects* and that complete suites need deep control‑flow/execution‑state reasoning. (See also the 2026 survey arXiv 2511.21382 for achievements/challenges/opportunities.)

③ **Shiplight — "AI in Test Automation: The Complete 2026 Guide"** — https://www.shiplight.ai/blog/ai-in-test-automation
   *Practitioner state‑of‑the‑art, unusually balanced for a vendor.* Clear "production‑ready vs still‑maturing" split (self‑healing/test‑gen/intent‑authoring work; fully autonomous interpretation + complex business logic don't), "supervised autonomy" framing, measurable benefits (5–10× authoring throughput, maintenance 40–60%→<5%), and limitations (hallucinated tests, opaque failures, false confidence). ⚠️ Vendor blog — treat throughput/maintenance figures as directional.
