# Phase 3 - Build (Development & Coding)

### How AI is applied to implementation and coding (mid‑2026)

*Audience: technical leadership. Part of the per‑phase SDLC series. Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md), [Technical Architecture](../analysis/02_technical_architecture.md), [Implementation Planning](../analysis/03_implementation_planning.md). Reliability tags: **[RCT/PEER]** · **[PRIMARY]** · **[ANALYST]** · **[VENDOR]** · **[3P]** · ⚠️ contested.*

---

## 1. What this phase covers

Generating, completing, refactoring, and reviewing source code: inline completion, chat‑assisted coding, multi‑file agentic edits, boilerplate/scaffolding, configuration validation, and the human review gate. This is the **most mature and most saturated** AI phase — and the one where the productivity‑vs‑quality tension is sharpest.

## 2. Adoption status & evidence

- **Saturated.** ~90% of developers use AI at work (DORA 2025) **[PRIMARY]**; ~41–50% of code in active files is AI‑generated across many orgs; Copilot reached ~20M users / 4.7M paid (Jan 2026). Coding is where adoption is deepest and tooling consolidated fastest.
- **Trust is low.** Stack Overflow 2025: only **~33% trust** AI accuracy, **46% distrust**; **66%** frustrated by "almost right, but not quite"; **45%** say debugging AI code takes *longer* **[PRIMARY]**.
- **The strongest independent study shows a slowdown.** METR RCT: experienced devs were **19% slower** with AI on their own mature repos while *believing* they were 20% faster **[RCT]** — the key counter to vendor "55% faster" claims, scoped to expert open‑ended work with early‑2025 tools.
- **Realistic at‑scale gain:** ~**7.76% median PR throughput** (getDX, 400+ orgs) **[3P]**; JPMorgan CIO cites 10–20% **[3P]**.

## 3. What works

- **Bounded generation:** boilerplate, CRUD services, REST endpoints, SQL, UI components (e.g., CodeMirror extensions), well‑specified functions. Acceptance of suggestions is ~30% — the value is keeping the good third.
- **Multi‑file agentic edits and refactors** on well‑scoped tasks (rename across files, mechanical migrations) — best with strong tests as a safety net.
- **Configuration validation** — point an agent at hierarchical config (Hydra/Helm/Kustomize overrides) to catch conflicting parameters before runtime. High‑value, low‑risk; runs well in a pre‑commit hook.
- **Context‑aware scaffolding** — a model primed on your conventions generates whole modules matching existing patterns (the "scaffold a new service using our internal gRPC framework + logging + metrics + CI YAML" pattern enforces consistency).
- **Frontier benchmarks** keep climbing (Terminal‑Bench 2.1: Codex+GPT‑5.5 83.4%, Claude Code+Opus 4.8 78.9%) but still fail on long, underspecified, or unfamiliar tasks — hence "agent does the work, human reviews each change."

## 4. What doesn't work (yet)

- **The "70% problem" (Addy Osmani, [grounding source ②](#10-grounding--further-reading)).** AI gets you to ~70% fast; the final 30% — edge cases, error handling, security, production integration, polish — *"remains as challenging as ever."* Software quality was *"never primarily limited by coding speed"*; the constraints are understanding requirements, designing maintainable systems, and handling edge cases — exactly what AI does worst. Watch for **"house‑of‑cards code"**: looks complete, lacks edge‑case/type/error handling.
- **Open‑ended/large refactors and greenfield architecture** — exactly where METR's slowdown appeared; no dependency‑graph awareness, cold‑start context.
- **Security:** **45% of AI code introduces OWASP Top 10 vulns** (Veracode, Java worst at 72%) **[VENDOR, large sample]**; package hallucination **19.7%** (5.2% commercial / 21.7% open models) **[RCT/PEER]**; developers feel *more* secure while writing *less* secure code.
- **Persistent debt in the wild ([source ③](#10-grounding--further-reading)).** Large‑scale study of **304,362 AI‑authored commits across 6,275 repos and 5 tools**: **15–29% of every tool's commits introduce at least one issue** (code smells 89%, runtime bugs 6%, security 5%); **24.2% of AI‑introduced issues still survive at HEAD**; AI commits introduce **~2× more security issues than they fix**; the pattern is **consistent across all five tools — switching tools won't fix it.** (Corroborates GitClear: copy‑paste up 8.3%→12.3%, refactoring down to <10%, duplicated blocks ~8×.)
- **The review bottleneck:** generation scaled, review didn't (Faros: more PRs, far longer review times, more bugs/incidents per PR; 31% more PRs merged with *no review*). The constraint moved downstream.

## 5. Tools, models & frameworks

| Category | Options | Notes |
| --- | --- | --- |
| **IDE assistants** | GitHub Copilot, Cursor, Windsurf, Tabnine (air‑gapped), Amazon Q, JetBrains AI+Junie; **OSS/BYOM:** Continue, Cline, Zed | Match tool to workload, not benchmark |
| **CLI agents** | Claude Code, OpenAI Codex CLI (OSS), Gemini CLI (OSS), Aider (OSS), opencode (OSS), Goose (OSS) | For multi‑file refactors/migrations |
| **Async agents** | Devin, **OpenHands** (OSS, self‑hostable, 72.8% SWE‑bench w/ Sonnet 4.5), Jules, Copilot coding agent | Sandbox → PR only; well‑scoped tasks |
| **Models** | Frontier (Claude Opus 4.8/Sonnet 4.6, GPT‑5.5, Gemini 3.1) + **two‑tier routing** to local Qwen3‑Coder/DeepSeek via vLLM for the routine 60% | Reserve frontier for the hard 40% |
| **Context** | Embedding index (Cursor/Continue) or agentic search (Claude Code); repo‑map (Aider) | Large monorepo → index; fast‑changing tree → agentic search |

## 6. Concrete patterns to adopt

1. **Keep the human review gate non‑negotiable** — treat AI output as a junior dev's first PR, keep the good ~30%, reject the rest.
2. **Two‑tier routing** — local/cheap model for completions, config checks, quick edits; frontier model for complex reasoning. The key cost/privacy lever (and a fit for a local‑first stack).
3. **Config‑validation pre‑commit hook** — semantic analysis of Hydra/Helm overrides catches a whole class of runtime errors before push.
4. **Context‑aware scaffolding on your conventions** — enforce architectural consistency, eliminate boilerplate.
5. **Identical quality/security gates for AI and human code** — SAST, dependency/secret scanning, complexity metrics, uniformly applied. The fix is to gate AI, not slow it.
6. **Never run agents with permission‑bypass flags** against secret‑bearing repos (the Nx supply‑chain lesson).
7. **Codify agent permissions as a version‑controlled "Three‑Tier Boundary System" ([source ④](#10-grounding--further-reading)).** Classify every agent action as **Always** (auto‑run — formatting, tests, read‑only queries), **Ask‑First** (propose → human approves — commits, dependency installs, migrations), or **Never** (hard‑blocked — force‑push, production credentials, secret access). Encode it in the agent config (`.claude/settings.json` permissions/hooks, `AGENTS.md`) as *technical enforcement*, not team norms — so it is reviewable, testable, and consistent across the team. A crisper, more adoptable formulation of the autonomy‑threshold idea than an abstract L0–L5 scale.

## 6b. Operating practices for driving a coding agent

*The most actionable how‑to here is Anthropic's Claude Code best‑practices guide ([source ①](#10-grounding--further-reading)); the patterns generalize across agentic tools. Treat the agent like "a very eager junior developer requiring constant supervision" (Osmani).*

1. **Explore → Plan → Implement → Commit.** Separate research/planning from coding to avoid "solving the wrong problem." Use a **plan mode** (read‑only exploration → a reviewed plan → execution). Skip it only when *"you could describe the diff in one sentence."*
2. **Give the agent a check it can run.** A test suite, build exit code, linter, or screenshot‑diff closes the loop so the agent self‑corrects instead of you being the verification loop. *"If you can't verify it, don't ship it."* Escalate the gate as needed: in‑prompt → a goal/stop condition re‑checked each turn → a deterministic CI hook.
3. **Manage context as an information system, not a prompt (Osmani, [source ②](#10-grounding--further-reading)).** Treat the context window like RAM — load‑on‑demand what's relevant, garbage‑collect the rest. Performance degrades as it fills: `/clear` between unrelated tasks; commit progress to git; keep a progress/spec file; delegate file‑heavy investigation to **subagents** that report summaries. When you do load context, include the *right* things: the relevant code files (not just the broken line), **database schemas, full error messages + stack traces** (not "it's broken"), design docs/API specs, an example of the desired output, and explicit constraints/coding standards.
4. **Persistent project memory (`CLAUDE.md` / `AGENTS.md`).** Check a short, pruned conventions file into git — build/test commands, code‑style deltas, repo etiquette, gotchas. Rule of thumb: *"Would removing this line cause a mistake? If not, cut it."* Bloated memory files get ignored.
5. **Writer/Reviewer with fresh context.** Have a *second* agent (or subagent) review the diff in a clean context — *"a fresh model that isn't biased toward code it just wrote."* Prompt it to flag only correctness/requirement gaps (a reviewer told to find problems will invent them → over‑engineering).
6. **Precise prompts beat vague ones.** Scope the file, the scenario, the "done" condition; point at existing patterns to follow; describe the symptom + likely location for bugs. For big features, let the agent *interview you* first and write a spec, then execute it in a fresh session.
7. **Course‑correct early; after two failed corrections, `/clear` and re‑prompt.** A clean session with a better prompt beats a long one polluted with failed approaches.
8. **Scale with guardrails, not abandon.** Parallel sessions / git worktrees and non‑interactive (`-p`) batch runs multiply output — but keep `--allowedTools` scoped and an adversarial review step before "done."

> **The human 30% (Osmani's working patterns):** *AI First Draft* (generate, then manually refactor for modularity + error handling + tests), *Constant Conversation* (fresh chats, tight feedback loops), *Trust but Verify* (mandatory review of all critical paths, automated edge‑case tests, security audits). *"Use AI to accelerate, not replace, your judgment."*

**Decide *what* to delegate — a cost/benefit quadrant ([source ②](#10-grounding--further-reading)).** Not all tasks are worth handing to an agent:

| | Low cost to verify | High cost to verify |
| --- | --- | --- |
| **High benefit** | ✅ *Accept immediately* — boilerplate, test generation, docs, code completion | ⚠️ *Evaluate carefully* — complex refactoring, architecture, perf optimization, security‑critical code |
| **Low benefit** | ◐ *Optional* — formatting, comment generation, variable renaming | ✕ *Avoid* — over‑engineered solutions, unfamiliar patterns in critical paths |

The top‑left is the safe daily 80/20; the top‑right is where the human 30% (and the [test](04_test.md)/[review](04_test.md) gates) must be strongest; the bottom‑right is where AI most often *costs* more than it saves.

## 7. Implementation priorities (80/20)

- **Value/effort:** Medium‑high value, low‑medium effort — but **roll out gated**, not as the headline pilot (already demanded bottom‑up).
- **Sequence:** deploy enterprise‑tier assistants to pilot teams with secret scanning on; pair with [test](04_test.md) generation and gated PR review so coverage and review track generation throughput.
- **Metrics (outcomes, not LoC):** review latency, change‑failure rate, bugs/incidents per PR, rework/churn rate, acceptance‑after‑later‑fixes. Anchor narratives with the METR perception gap.

## 8. Risks & governance

- **Security regression is the headline risk** — uniform gates, treat the agent as an untrusted caller, scan for hallucinated/typosquatted dependencies.
- **Skill erosion for juniors** — pair AI use with "explain before accept" review norms, the **Generation‑Then‑Comprehension** protocol (*generate → review → explain → modify → commit*; make code review probe *understanding*, not just correctness), and **Intentional AI‑Free Skill Zones** (deliberately carve out tasks juniors do *without* AI to build debugging/architecture judgment) ([source ④](#10-grounding--further-reading)). A 2026 vendor briefing cites AI‑assisted juniors at 50% vs 67% control on comprehension quizzes (**−17pp**) **[VENDOR]** — direction consistent with our benchmark, evidence still soft. Watch for **"comprehension debt"** — AI‑generated code that ships and works but that *no one on the team actually understands*, so it can't be safely debugged or evolved later; the practices above are how you avoid accruing it.
- **The senior‑review tax** — budget for it explicitly; it's the largest hidden cost of this phase.
- **Enterprise/zero‑retention tiers only** for proprietary code; never consumer plans. Local models remove the question entirely.

## 9. Key takeaways

1. Coding is **saturated and real on bounded tasks**, but **slows experts on open‑ended work** and **multiplies security findings** — the gains are conditional, not automatic.
2. **The bottleneck moved from typing to review and integration** — scale review (gates, [test](04_test.md) generation) or the throughput gain becomes accelerated debt.
3. **Two‑tier routing + uniform quality gates + a non‑negotiable human review gate** is the durable operating model — keep the keyboard fast and the merge gate disciplined.
4. **Driving the agent well is a learnable skill** — explore‑then‑plan, give it a check to run, manage context, and review with fresh eyes. The teams that win treat the agent as a supervised junior, not a magic code generator.

## 10. Grounding & further reading

*Curated, quality‑reviewed sources behind this phase (full review in [references/sdlc_phases.md](../references/sdlc_phases.md)). The phase's hard *evidence* (METR, DORA, Veracode, GitClear, Faros) lives in the [Business Benchmark](../analysis/01_business_benchmark.md); the sources below add practitioner how‑to and fresh empirical depth.*

① **Anthropic — "Best practices for Claude Code"** — https://code.claude.com/docs/en/best-practices
   *Best concrete how‑to for agentic coding.* Explore→plan→implement→commit, give‑the‑agent‑a‑check verification, context management (`/clear`, progress files, subagents), `CLAUDE.md` discipline, Writer/Reviewer with fresh context, adversarial review subagent, and a "common failure patterns" list. Vendor (Claude Code‑specific) but the patterns generalize; tool‑specific commands are illustrative.

② **Addy Osmani — *Beyond Vibe Coding: A Guide to AI‑Assisted Engineering*** (book) — https://beyond.addy.ie/ · essay: ["The 70% problem"](https://addyo.substack.com/p/the-70-problem-hard-truths-about)
   *Best practitioner realism **and** the most complete how‑to.* The essay gives the framing (70/30; the knowledge paradox — AI helps seniors > juniors; house‑of‑cards code; AI First Draft / Constant Conversation / Trust but Verify). The book adds the systematic mitigations folded into §6b: **context‑as‑information‑systems** (treat the window like RAM — what to load), the **cost/benefit delegation quadrant**, five pillars (plan‑first/mini‑PRD, rich context, visual context, test after *every* change, debug with explicit intent), critique‑driven & role‑based prompting, a quality‑gates checklist, and the senior mindset shift "from coding to curating, from implementation to intent." Its distinctive thesis: *structural discipline over prompting cleverness* — AI as a force multiplier for rigorous engineers, not a shortcut around rigor.

③ **"Debt Behind the AI Boom: A Large‑Scale Empirical Study of AI‑Generated Code in the Wild"** (arXiv, 2026) — https://arxiv.org/abs/2603.28592
   *Fresh, large‑scale empirical grounding.* 304,362 AI‑authored commits, 6,275 repos, 5 tools; 15–29% of commits introduce an issue, 24.2% survive at HEAD, ~2× more security issues introduced than fixed, consistent across tools. Honest limits: no clean human baseline, static‑analysis‑only (misses architectural/runtime debt), popular‑repo selection bias.

④ **LTM — "SDLC AI Radar 2026" (Executive Briefing)** — `../articles/sdlc-ai-executive-slides-final.pdf`
   *Vendor foresight briefing* **[VENDOR]**. Source of the **Three‑Tier Boundary System** (`Always / Ask‑First / Never`), the **Generation‑Then‑Comprehension** learning pattern, **Intentional AI‑Free Skill Zones**, and the −17pp junior‑comprehension figure. Directionally aligned with our benchmark; treat specific numbers as vendor‑sourced.

*Also strong:* Anthropic's ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) and ["Effective harnesses for long‑running agents"](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) (orchestration patterns — see [Technical Architecture](../analysis/02_technical_architecture.md)); the Pragmatic Engineer's ["How AI will change software engineering: hard truths."](https://newsletter.pragmaticengineer.com/p/how-ai-will-change-software-engineering)
