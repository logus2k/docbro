# Phase 3 — Build (Development & Coding)

### How AI is applied to implementation and coding (mid‑2026)

---

## 1. What this phase is for

This is where the specification and design become working code — through inline completion, chat‑assisted coding, multi‑file agentic edits, scaffolding, and the human review that decides what actually ships. It's the most mature and most saturated place for AI in the whole lifecycle: roughly 90% of developers now use AI at work, and on many teams 40–50% of the code in active files is AI‑generated.

Precisely because it's so effective at *generating*, this is also the phase where the productivity‑versus‑quality tension is sharpest. The goal of your plan here is not "write more code faster" — that part is solved — but "**turn that speed into delivered, stable, secure software instead of accelerated debt.**" The way you do that is by keeping the keyboard fast and the merge gate disciplined.

## 2. Where AI genuinely helps (and where it doesn't)

**What works.** AI is reliably useful on *bounded* work: boilerplate, CRUD services, REST endpoints, SQL, well‑specified functions, and UI components. Developers keep roughly the best third of what's suggested, and that third is real value. It handles well‑scoped multi‑file edits — renames, mechanical migrations — competently, especially with strong tests as a safety net. And it's very good at a quiet, high‑value job: **configuration validation**, catching conflicting parameters in hierarchical config before they blow up at runtime.

**Where it doesn't — and this shapes the whole plan.** Three honest limits:

- **The "70% problem."** AI gets you to about 70% of a feature fast; the last 30% — edge cases, error handling, security, production integration, the polish — is as hard as it ever was. Software quality was never mainly limited by typing speed. Watch for **"house‑of‑cards code"**: it looks complete and it compiles, but the error handling and edge cases aren't there.
- **Security regression that isn't improving.** Independent testing finds that ~45% of AI‑generated code introduces an OWASP Top‑10 vulnerability, and models recommend hallucinated (non‑existent) package names about 20% of the time — a live supply‑chain attack surface. Uncomfortably, developers *feel more* secure while writing *less* secure code.
- **The bottleneck moved downstream.** Generation scaled; review didn't. Teams see more PRs, longer review times, and more bugs and incidents per PR — and a meaningful share of PRs merged with no review at all. The constraint is now review and integration, not authoring.

The single most important independent data point to keep in mind: in a controlled trial, experienced developers were **19% slower** with AI on their own mature codebases while *believing* they were 20% faster. The gains are real but conditional — strongest on bounded work, negative on expert open‑ended work.

## 3. The activities — what to actually do

**a) Roll out coding assistants to everyone — but gated, not ungoverned.** Developers already want these tools; the leadership job isn't to push adoption, it's to put the gates in place. Use enterprise/zero‑retention tiers for any proprietary code (never consumer plans), turn on secret scanning, and make clear that AI output is treated like a junior developer's first PR: keep the good part, reject the rest, and never merge unreviewed.

**b) Split work between a cheap/local model and a frontier model (two‑tier routing).** Route the routine ~60% — completions, config checks, quick edits — to a small or local model (e.g., Qwen3‑Coder or DeepSeek served on vLLM), and reserve the frontier model for the hard ~40% (complex reasoning, tricky refactors). This is the biggest single lever for controlling both cost and data residency, and it fits a privacy‑first stack well.

**c) Teach the team to *drive* an agent — it's a learnable skill.** The difference between teams that get value and teams that get chaos is mostly operating practice. Establish these as norms:
- **Explore → Plan → Implement → Commit.** Separate research and planning from coding so the agent doesn't confidently solve the wrong problem. Use a plan mode (read‑only exploration → a reviewed plan → execution); skip it only when you could describe the change in one sentence.
- **Always give the agent a check it can run** — a test suite, a build, a linter, a screenshot diff. That closes the loop so the agent self‑corrects instead of *you* being the verification loop. If you can't verify it, don't ship it.
- **Manage context like memory, not like a prompt.** Load the *right* things (the relevant files, database schemas, full error traces — not "it's broken"), clear context between unrelated tasks, and delegate file‑heavy investigation to sub‑agents that report back summaries.
- **Keep a short project‑conventions file** (`CLAUDE.md` / `AGENTS.md`) in git — build/test commands, code‑style rules, gotchas. Prune it ruthlessly; a bloated one gets ignored.
- **Review with fresh eyes.** Have a *second* agent (or sub‑agent) review the diff in a clean context, prompted to flag only correctness and requirement gaps — a reviewer that isn't biased toward code it just wrote.

**d) Codify what the agent may do as an explicit, version‑controlled policy.** Classify every agent action into three tiers — **Always** (auto‑run: formatting, tests, read‑only queries), **Ask‑First** (propose, then a human approves: commits, dependency installs, migrations), and **Never** (hard‑blocked: force‑push, production credentials, secret access). Encode it in the agent config and `AGENTS.md` as *technical enforcement*, not a team norm, so it's reviewable and consistent for everyone. This "Three‑Tier Boundary System" is a crisper, more adoptable version of an abstract autonomy scale — and a hard rule: **never run agents with permission‑bypass flags against a repo that holds secrets.**

**e) Decide *what* to delegate, not just how.** Not every task is worth handing to an agent. A simple cost/benefit split:

| | Cheap to verify | Expensive to verify |
| --- | --- | --- |
| **High benefit** | ✅ Accept freely — boilerplate, tests, docs, completion | ⚠️ Delegate carefully — complex refactors, architecture, security‑critical code |
| **Low benefit** | ◐ Optional — formatting, renames | ✕ Avoid — over‑engineered solutions, unfamiliar patterns in critical paths |

The top‑left is your daily 80/20; the top‑right is where human review and the test gates must be strongest; the bottom‑right is where AI usually costs more than it saves.

**f) Apply the same quality and security gates to AI code as to human code.** SAST, dependency and secret scanning, complexity checks — uniformly, on every PR, no exceptions for the fact that a model wrote it. The fix for AI's security regression is not to slow AI down; it's to gate it. Treat the agent as an untrusted caller and scan for hallucinated or typosquatted dependencies.

## 4. How to pilot this

Coding assistants roll out in *parallel* with your other pilots (the demand is already there), but they are **not** your headline pilot — the verifiable wins in [Test](04_test.md) and [Release](05_release.md) should lead. Deploy enterprise‑tier assistants to a pilot team with secret scanning on, pair them immediately with AI test generation and gated PR review so that coverage and review capacity grow alongside generation throughput, and measure the real outcomes (below) before widening.

## 5. Guardrails & what to watch for

- **Security regression is the headline risk** — uniform gates, agent‑as‑untrusted‑caller, dependency scanning. Non‑negotiable.
- **Protect junior skill development.** Pair AI use with an "explain before accept" norm; adopt the **Generation‑Then‑Comprehension** habit (generate → review → explain → modify → commit, with code review probing *understanding*, not just correctness); and carve out **intentional AI‑free zones** where juniors solve problems without assistance to build real debugging and architecture judgment. Watch for **"comprehension debt"** — code that ships and works but that nobody on the team actually understands, so it can't be safely debugged or evolved later. (A vendor briefing puts AI‑assisted juniors at 50% vs 67% on comprehension quizzes — the number is soft, the direction matches everything else we see.)
- **Budget the senior‑review tax explicitly.** Review times rise sharply and land disproportionately on your senior engineers; it's the largest hidden cost of this phase and the easiest to under‑fund.
- **Proprietary code, enterprise tiers only** — never consumer plans; prefer zero‑retention endpoints, and remember that local models remove the data‑residency question entirely.

## 6. How you'll know it's working

Measure outcomes, never lines of code (which is trivially inflated and actively misleading here):

- **Review latency** — the new bottleneck; watch it closely.
- **Change‑failure rate** and **bugs/incidents per PR** — the quality signal.
- **Rework / churn rate** — how much AI‑written code gets rewritten within days.
- Anchor any leadership narrative with the perception gap (developers *feel* faster than they measurably are), so decisions weigh measured outcomes over felt speed.

## 7. Tools to reach for

| Category | Options |
| --- | --- |
| **IDE assistants** | GitHub Copilot, Cursor, Windsurf, Tabnine (air‑gapped), Amazon Q, JetBrains AI + Junie; open‑source / bring‑your‑own‑model: Continue, Cline, Zed |
| **CLI agents** (multi‑file refactors, migrations) | Claude Code, OpenAI Codex CLI, Gemini CLI, Aider, opencode, Goose |
| **Async agents** (issue → sandbox → PR only) | Devin, OpenHands (open‑source, self‑hostable), Jules, GitHub Copilot coding agent |
| **Models** | Frontier (Claude Opus 4.8 / Sonnet 4.6, GPT‑5.5, Gemini 3.1) plus two‑tier routing to a local Qwen3‑Coder / DeepSeek via vLLM for the routine work |
| **Context** | Embedding index (Cursor/Continue) or agentic search (Claude Code); repo‑map (Aider) — index for big monorepos, agentic search for fast‑changing trees |

## 8. Evidence & sources

*Reliability tags: [PRIMARY] · [RCT/PEER] · [ANALYST] · [VENDOR] (treat as a ceiling) · [3P].*

- **Adoption:** ~90% of developers use AI at work (DORA 2025) [PRIMARY]; ~41–50% of active‑file code is AI‑generated across many orgs [3P].
- **Trust:** only ~33% of developers trust AI accuracy, 46% distrust it, and 45% say debugging AI code takes *longer* (Stack Overflow 2025) [PRIMARY].
- **The expert slowdown:** experienced devs were 19% slower with AI on their own mature repos while believing they were 20% faster (METR RCT) [RCT/PEER]. Realistic at‑scale gain is single‑digit (~7.76% median PR throughput across 400+ orgs) [3P].
- **Security:** ~45% of AI code carries an OWASP Top‑10 vuln (Veracode) [VENDOR, large sample]; ~19.7% package‑hallucination rate (USENIX 2025) [RCT/PEER].
- **Debt in the wild:** across 304,362 AI‑authored commits, 15–29% of commits introduce an issue and 24.2% survive to HEAD, consistent across five tools [3P].
- **Grounding sources** (full reviews in [references/sdlc_phases.md](../references/sdlc_phases.md)):
  1. **Anthropic — "Best practices for Claude Code"** — the concrete agentic‑coding how‑to behind the operating practices in activity (c). https://code.claude.com/docs/en/best-practices
  2. **Addy Osmani — *Beyond Vibe Coding*** (+ the "70% problem" essay) — the realism and the delegation quadrant. https://beyond.addy.ie/
  3. **"Debt Behind the AI Boom: A Large‑Scale Empirical Study of AI‑Generated Code in the Wild"** (arXiv, 2026) — the persistent‑debt numbers. https://arxiv.org/abs/2603.28592
  4. **LTM — "SDLC AI Radar 2026"** [VENDOR] — the *Three‑Tier Boundary System*, *Generation‑Then‑Comprehension*, *Intentional AI‑Free Skill Zones*, and *comprehension debt*. `../articles/ltm_sdlc_ai_radar_2026.md`

---

*Cross‑references: [Business Benchmark](../analysis/01_business_benchmark.md) · [Technical Architecture](../analysis/02_technical_architecture.md) · [Implementation Planning](../analysis/03_implementation_planning.md) · previous → [Design](02_design.md) · next → [Test & QA](04_test.md).*
