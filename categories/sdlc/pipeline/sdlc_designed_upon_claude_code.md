# An SDLC Pipeline Built on Claude Code

## Thesis

A Claude Code pipeline is not a prompt library. It is a configuration system in which four things — what loads, what is enforced, what is reachable, and what is isolated — are decided *before* any work starts, and then re-decided per task. Every recommendation below exists to make one of those four deterministic. Where determinism is impossible, the design pays for a check instead of trusting a claim.

Five principles govern the whole pipeline:

1. **Placement is not effect.** A file at the right path that was never loaded, and a rule that loaded but was buried, both fail silently. Verify what loaded (`/memory`, `/context`, `claude mcp list`) instead of assuming it.
2. **Advisory vs. deterministic is the primary architectural split.** CLAUDE.md is context, not enforced configuration; to block an action regardless of what the model decides, use a hook ([memory](https://code.claude.com/docs/en/memory), [hooks](https://code.claude.com/docs/en/hooks)).
3. **Everything always-loaded is taxed on every turn.** Context is billed as input tokens each turn; caching reduces the rate but does not zero it ([prompt-caching](https://code.claude.com/docs/en/prompt-caching)). Baseline additions must be justified individually.
4. **Isolation is an output contract, not a mechanism.** A subagent isolates traversal only to the extent that what it returns is small — its return text lands in the main context.
5. **Configuration is written from evidence.** Rules are added for mistakes that happened; rules that were ignored get trimmed and reworded, not stacked.

Every token figure in this document is illustrative arithmetic for sizing decisions, not a measurement — the only documented anchors are that tool definitions can consume a large share of the window (50 tools ≈ 10–20K tokens) and that every token in context is billed as input on every turn, with caching reducing but not eliminating the cost ([costs](https://code.claude.com/docs/en/costs), [prompt-caching](https://code.claude.com/docs/en/prompt-caching)). Measure your own baseline with `/context` before budgeting against any number here.

## The two tracks

| | **Track A — interactive** | **Track B — headless / CI** |
|---|---|---|
| Permission decisions | human, via permission modes | programmatic, via a `PermissionRequest` hook |
| Config source | inherited from user + project scopes | pinned explicitly; MCP locked with `--strict-mcp-config` |
| Recovery from a bad state | `/clear`, `--continue`, restart | fail the run; there is no in-session recovery |
| Destructive actions | explicit skill invocation by a human | explicit, environment-named skill invoked by the script, never model-selected |
| Cost control | `/usage`, status line | budget per run; effort and model pinned by flag |

Every phase below states its Track B delta where one exists.

---

## Phase 1 — Scope wiring and precedence resolution

**Purpose.** Decide where every artifact lives and confirm which copy actually takes effect.

**Entry.** A machine or repo with no `.claude/` layer, or one whose loaded configuration has been assumed rather than observed. Re-enter on: new machine, new repo, org policy change.

**Exit.** `/memory` lists exactly the intended instruction files; `claude mcp list` shows exactly the intended servers; `.claude/agents/` and `.claude/skills/` contain no same-named definitions you did not intend to shadow; `CLAUDE.local.md` and `.claude/settings.local.json` are both in `.gitignore`; `/init` output has been hand-trimmed.

### Mechanisms and why

| Need | Mechanism | Why not the neighbour |
|---|---|---|
| Machine-wide personal defaults | `~/.claude/settings.json` | CLAUDE.md is advisory; settings are read by the harness |
| Team-shared, reviewable project policy | `.claude/settings.json` (committed) | `settings.local.json` is personal and must not be reviewed by the team |
| Personal, per-repo deviations | `.claude/settings.local.json` (ignored) | Putting them in the shared file forces them on the team |
| Operational project knowledge | `./CLAUDE.md` (committed) | settings.json cannot express judgment-shaped guidance |
| Path-specific rules | `.claude/rules/*.md` with `paths:` frontmatter | Root CLAUDE.md loads unconditionally; rules load only for matching files |

**Settings precedence** (highest first): Managed → command-line arguments → `.claude/settings.local.json` → `.claude/settings.json` → `~/.claude/settings.json` ([settings](https://code.claude.com/docs/en/settings)).

**Two corrections to widely-repeated folklore:**

- **Permission rules merge across scopes; they do not replace.** A project `permissions` block *extends* user-level rules rather than wiping them. You do **not** need to re-declare global permissions inside a project. A few security-sensitive settings additionally honour a restrictive value from a scope that otherwise could not override ([settings](https://code.claude.com/docs/en/settings)).
- **CLAUDE.md files are concatenated, not overridden.** All discovered files are concatenated, ordered from the filesystem root down to your working directory, so instructions closest to where you launched Claude are read last ([memory](https://code.claude.com/docs/en/memory)). "More specific wins" is a description of recency, not of a merge algorithm. Two levels that contradict each other both end up in context. Keep levels topically disjoint: global = personal habits, project = stack and conventions, subdirectory = component rules.

**Instruction-file hierarchy** ([memory](https://code.claude.com/docs/en/memory)):

```text
Managed policy   macOS  /Library/Application Support/ClaudeCode/CLAUDE.md
                 Linux/WSL  /etc/claude-code/CLAUDE.md
                 Windows  C:\Program Files\ClaudeCode\CLAUDE.md
User             ~/.claude/CLAUDE.md
Project          ./CLAUDE.md          (or ./.claude/CLAUDE.md — pick one, never both)
Local            ./CLAUDE.local.md    (add to .gitignore yourself)
Subdirectory     packages/api/CLAUDE.md — NOT loaded at launch; included when
                 Claude reads files in that subdirectory
```

### Copy-ready configuration

`~/.claude/settings.json` — deliberately minimal:

```json
{
  "model": "sonnet",
  "effortLevel": "medium",
  "env": {
    "CLAUDE_CODE_SUBAGENT_MODEL": "haiku"
  },
  "permissions": {
    "allow": [
      "Bash(git log *)",
      "Bash(git diff *)",
      "Bash(git status)"
    ]
  }
}
```

`model` is read once at session start and applies on the next restart, not mid-session ([model-config](https://code.claude.com/docs/en/model-config)). `permissions` supports `allow`, `ask`, and `deny`; deny takes precedence over allow ([settings](https://code.claude.com/docs/en/settings)).

Repo hygiene:

```bash
printf '%s\n' 'CLAUDE.local.md' '.claude/settings.local.json' >> .gitignore
git add CLAUDE.md .claude/settings.json .claude/skills .claude/agents .claude/hooks .mcp.json
```

Neither `CLAUDE.local.md` nor `.claude/settings.local.json` is gitignored for you. Running `/init` with `CLAUDE_CODE_NEW_INIT=1` and choosing the personal option adds the `CLAUDE.local.md` entry ([memory](https://code.claude.com/docs/en/memory)).

Path-scoped rules — **the frontmatter key is `paths:`**:

```markdown
---
paths: src/api/**/*.ts
---

# API conventions
- Validate every request body with a Zod schema.
- Use the error shape in src/api/errors.ts; never return raw ORM objects.
```

### Cost

Near-zero at runtime; this phase buys correctness, not tokens. The one real cost is `/init`, which reads the codebase — run it once per project and hand-edit the result.

### Failure modes and guards

| Failure | Guard |
|---|---|
| Rule file never loads (typo in `paths:`, wrong directory) | Run `/memory` and confirm the file is listed; do not assume the harness stays quiet either — a failing hook surfaces as `<hook name> hook error` in the transcript with the detail in debug logging ([hooks](https://code.claude.com/docs/en/hooks)), so check `/debug` before concluding a load failed silently |
| Two project-root CLAUDE.md locations both present | Choose one; delete the other |
| Same-named subagent or skill shadowed by another scope | Inspect `.claude/agents/` and `~/.claude/agents/` directly — as of v2.1.198 `/agents` no longer opens the interactive lister/wizard, it prints a reminder to ask Claude or edit the files ([sub-agents](https://code.claude.com/docs/en/sub-agents)) |
| Personal overrides leak into the team repo | `.gitignore` entries above, verified with `git status` before the first commit |
| `/init` output accepted as-is | Trim to the four admissible content categories (Phase 2) |

### Track B delta

CI inherits nothing it was not given. Provision the repo's `.claude/settings.json` from the checkout, set env vars in the job definition rather than in `~/.claude/settings.json`, and confirm with `claude mcp list` as a pipeline step that the server set is what the config declares.

---

## Phase 2 — Baseline budgeting: what loads before the first prompt

**Purpose.** Cap and *measure* the always-loaded payload — CLAUDE.md hierarchy, imports, rule files, skill names and descriptions, MCP tool names. A correctly-placed rule that is buried is still ignored.

**Entry.** Phase 1 exit holds, and at least one artifact is large enough to compete for attention.

**Exit.** `/context` has been read and the measured baseline accepted deliberately; every CLAUDE.md in the hierarchy is under 200 lines; every retained line survives *"would removing this cause a mistake?"*; procedural workflows have moved to skills and path-specific rules to `.claude/rules/`; every connected server and skill description is individually justified.

### Mechanisms and why

- **`/context`** over guesswork: it visualises current context usage and flags capacity ([commands](https://code.claude.com/docs/en/commands), [context-window](https://code.claude.com/docs/en/context-window)).
- **Skills over CLAUDE.md for procedures**: a skill's body loads only when used, so long reference material costs almost nothing until needed ([skills](https://code.claude.com/docs/en/skills)). This is the only move that genuinely reduces baseline.
- **`.claude/rules/` with `paths:` over root CLAUDE.md**: glob-gated loading, same effect, conditional cost ([memory](https://code.claude.com/docs/en/memory)).
- **`@path` imports** keep the *file* short but still deliver the full imported content into context ([best-practices](https://code.claude.com/docs/en/best-practices)). Use them for genuinely always-needed content only; they satisfy the line count without reducing the token cost.

**The 200-line rule is documented, not folklore:** target under 200 lines per CLAUDE.md file; longer files consume more context and reduce adherence ([memory](https://code.claude.com/docs/en/memory)).

**What earns a line** ([memory](https://code.claude.com/docs/en/memory), [best-practices](https://code.claude.com/docs/en/best-practices)):

| Include | Exclude |
|---|---|
| Build, test, lint commands | Style preferences the model already follows |
| Non-obvious architecture ("all DB access goes through `/src/repositories`") | Language or framework explanations |
| Conventions that break from defaults | Generic good-practice reminders ("write clean code") |
| Gotchas that already cost time | History, rationale, business logic, team metadata |
| Constraints with teeth ("never commit directly to `main`") | Rules for problems never observed |
| What "done" means for this project | |

### Copy-ready configuration

A complete project CLAUDE.md that stays inside budget:

```markdown
# payments-service

## Build and test
- Build: `npm run build`
- Test: `npm test` — run after changing any file under `src/` or `test/`
- Lint: `npm run lint` — must pass before any commit

## Architecture
- All database access goes through `src/repositories`. Controllers never query directly.
- `src/lib/stripe.ts` wraps the Stripe SDK. Do not instantiate Stripe elsewhere.
- `/legacy` is frozen. Do not modify files there.

## Conventions
- Errors: `AppError` from `src/lib/errors.ts`.
- Logging: `src/lib/logger.ts`, never `console.log`.
- The auth module uses a custom JWT helper, not `jsonwebtoken`.

## Done means
- `npm test` green, `npm run build` clean, and every acceptance criterion in the
  linked spec checked off explicitly.

## Constraints
- Never commit directly to `main`.
- Migrations must be reversible.
```

### Cost and how to bound it

| Item | Loaded when | Bounding move |
|---|---|---|
| CLAUDE.md hierarchy + `@` imports | every session, every turn | ≤200 lines per file; delete rather than reword |
| `.claude/rules/*.md` with `paths:` | only when matching files are read | prefer over root CLAUDE.md for anything path-specific |
| Subdirectory CLAUDE.md | when Claude reads that subdirectory | push monorepo conventions down here |
| Skill *names* | always | fewer skills |
| Skill *descriptions* | always, truncated to a dynamic cap of 1% of the context window, 8,000-character fallback, with each entry's combined text capped at 1,536 characters regardless of budget | one sentence each; skills compete for this budget. Raise it with `skillListingBudgetFraction` or the `SLASH_COMMAND_TOOL_CHAR_BUDGET` environment variable, and raise the per-entry cap with `skillListingMaxDescChars` ([skills](https://code.claude.com/docs/en/skills)) |
| MCP tool *names* | always | fewer servers |
| MCP tool *schemas* | deferred by default; loaded on demand via tool search | keep default deferral on ([context-window](https://code.claude.com/docs/en/context-window), [costs](https://code.claude.com/docs/en/costs)) |

### Failure modes and guards

| Failure | Guard |
|---|---|
| Instruction adherence decays as the file grows | Hard 200-line ceiling, enforced at review time like a lint rule |
| `IMPORTANT:`/`YOU MUST` used on every line | Reserve emphasis for the two or three rules that were actually violated |
| Import used to "shorten" a file that still loads everything | Ask whether the content is needed *every* session; if not, it is a skill or a rule file |
| Skill descriptions crowd each other out | Cap the skill count; one job per skill |

### Track B delta

CI sessions are short and single-purpose: give them the smallest CLAUDE.md that still carries build/test commands and constraints, and consider a job-specific working directory so subdirectory files do not load. Measure once with `/context` in an interactive session against the same repo state; you cannot read it mid-pipeline.

---

## Phase 3 — Enforcement wiring: hard stops that bypass the model

**Purpose.** Move every *must* and *must-not* out of prose and into mechanisms that fire regardless of context state. Phase 2 optimises persuasion-per-token; this phase assumes the model will reason around any string-matched rule.

**Entry.** There exists an action whose cost is not recoverable in-session, or an instruction that has already been ignored once.

**Exit.** Every non-negotiable is a hook, an OS/container boundary, a tool whitelist, or `disable-model-invocation: true` — and has been *observed* blocking a real attempt, not merely configured. Deny rules exist only alongside a `PreToolUse` hook. Hook groups are matcher-scoped. Hook scripts read stdin and resolve paths via `$CLAUDE_PROJECT_DIR`.

### Mechanisms and why

Hooks are user-defined shell commands run at fixed lifecycle points, giving deterministic control: certain actions always happen rather than relying on the model to choose ([hooks-guide](https://code.claude.com/docs/en/hooks-guide)). They fire even when the corresponding CLAUDE.md line has been compacted away. The documentation is explicit: to block an action regardless of what Claude decides, use a `PreToolUse` hook rather than an instruction ([memory](https://code.claude.com/docs/en/memory)).

**Deny rules.** `permissions.deny` is documented and deny takes precedence over allow ([settings](https://code.claude.com/docs/en/settings)). Treat it as a *declaration of intent plus a first filter*, never as the only layer: pair every deny rule with a `PreToolUse` hook covering the same action. There is no documented guarantee that a pattern-matched deny cannot be reached by an equivalent command spelling.

**Hook event surface** ([hooks](https://code.claude.com/docs/en/hooks)):

| Event | Fires | Use for |
|---|---|---|
| `PreToolUse` | before a tool call; can deny or rewrite input | protected paths, destructive shell, MCP writes |
| `PostToolUse` | after a tool call succeeds | format, lint, audit |
| `PostToolUseFailure` | after a tool call fails | error capture |
| `PostToolBatch` | after parallel tool calls resolve | batch-level checks |
| `PermissionRequest` | when a permission decision is needed | programmatic approve/deny (Track B) |
| `UserPromptSubmit` | on prompt submit, before processing | inject context; exit 2 rejects the prompt |
| `Stop` | when Claude finishes a turn | tests, build, logging |
| `Notification` | notifications (matchers include `permission_prompt`, `auth_success`) | idle alerts |
| `SessionStart` / `SessionEnd` | session begin/resume, terminate | env injection, audit |
| `InstructionsLoaded` | a CLAUDE.md or `.claude/rules/*.md` loads | observe-only |
| `ConfigChange` | async: a configuration file changes during a session; matchers `user_settings`, `project_settings`, `local_settings`, `policy_settings`, `skills` | notice config drift mid-session |
| `PreCompact` / `PostCompact` | around compaction | see Phase 9 |
| `SubagentStart` / `SubagentStop` / `TaskCreated` / `TaskCompleted` / `TeammateIdle` / `WorktreeCreate` | multi-agent lifecycle | Phase 8 instrumentation |

**Handler types**: `command`, `http`, `mcp_tool`, `prompt`, `agent` ([hooks](https://code.claude.com/docs/en/hooks)). Default to `command` — it is the cheapest and the only one with no per-event model or network latency. `prompt` and `agent` buy semantic judgment at real cost; use them only where a string cannot express the rule.

`/hooks` *displays* the session's current hook configuration; it is not a switch. Enabling or disabling a hook still means editing JSON ([commands](https://code.claude.com/docs/en/commands)).

**Handler-level prefilter.** (What is *not* documented is when the `if` rule is evaluated relative to spawning the handler process, or how it composes with the group-level `matcher` — so do not budget the process spawn away on the strength of an `if` until you have measured it.) A handler object takes an `if` field alongside `type` and `command`, whose value is a permission rule that further filters the group — for example `{"type": "command", "if": "Bash(rm *)", "command": "…/block-rm.sh"}` ([hooks](https://code.claude.com/docs/en/hooks)). Where a guard is a single command shape, express it as `if` on the handler rather than as a `case` arm inside the script: the rule then lives in the config, next to the `matcher`, instead of being buried in shell. Keep the in-script `case` only where one script guards several unrelated patterns, as `guard-bash.sh` below does.

**Exit-code protocol** ([hooks](https://code.claude.com/docs/en/hooks)): `0` = success (stdout parsed for a JSON decision); `2` = blocking error, and the handler's stderr is shown to Claude — for `PreToolUse` it blocks the tool call; anything else = non-blocking error, the action proceeds and the transcript shows `<hook name> hook error`. For `UserPromptSubmit`, exit 2 rejects the prompt.

**Input contract**: command hooks receive the event JSON on **stdin**; HTTP hooks receive it as the POST body. All hooks get `session_id`, `prompt_id`, `transcript_path`, `cwd`, `permission_mode`, `hook_event_name`, and tool events additionally carry `tool_name` — the field matchers filter on ([hooks](https://code.claude.com/docs/en/hooks)). Never read argv or environment for event data.

**MCP tools are matchable**: MCP tool names follow `mcp__<server>__<tool>`, so `PreToolUse` and `PermissionRequest` matchers such as `mcp__.*__write.*` cover external tools with the same machinery as built-ins ([hooks](https://code.claude.com/docs/en/hooks)).

### Copy-ready configuration

`.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/block-protected-paths.sh"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/guard-bash.sh"
          }
        ]
      },
      {
        "matcher": "mcp__.*__(write|create|update|delete).*",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/guard-mcp-write.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "mcp__.*",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/audit-mcp.sh",
            "runInBackground": true
          }
        ]
      }
    ]
  },
  "permissions": {
    "deny": [
      "Read(./.env)",
      "Read(./secrets/**)",
      "Write(./.env)"
    ]
  }
}
```

`.claude/hooks/block-protected-paths.sh` — committed, executable:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT="$(cat)"
TARGET="$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')"

case "$TARGET" in
  ""|*/node_modules/*) exit 0 ;;
  *.env|*/.env|*/.env.*|*/secrets/*|*/credentials*)
    echo "BLOCKED: $TARGET is a protected path. Ask the operator to change it by hand." >&2
    exit 2
    ;;
esac
exit 0
```

`.claude/hooks/guard-bash.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT="$(cat)"
CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')"

case "$CMD" in
  *"rm -rf "*|*"git push --force"*|*"git reset --hard"*|*"DROP TABLE"*|*"TRUNCATE "*)
    echo "BLOCKED: destructive command refused by policy: $CMD" >&2
    exit 2
    ;;
esac
exit 0
```

The exact stdin field name for a write target (`file_path` vs `path`) is version-dependent — the fallback chain above covers both; confirm against the [hooks](https://code.claude.com/docs/en/hooks) reference for your version. `PreToolUse` can also block by emitting a JSON decision with `permissionDecision: "deny"` (and rewrite arguments with `updatedInput`), which yields a cleaner reason string; check the current envelope keys before adopting that form. Exit 2 is the version-stable route.

**Prove the block.** Configuration is not enforcement until observed:

```text
Try to write a file called .env.test in the project root.
```

The turn must fail with your stderr text. If it succeeds, the hook is not wired — check `/hooks`.

### Cost

| Source | Cost | Bound |
|---|---|---|
| Matcher-less `PreToolUse` group | a process spawn before *every* Read/Grep/Glob/Write/Bash | always set `matcher`, and narrow a single-shape guard further with the handler's `if` rule ([hooks](https://code.claude.com/docs/en/hooks)) |
| `http` handler | network latency on every matched event | reserve for genuine remote policy |
| `prompt` / `agent` handler | model tokens per event | reserve for judgments a pattern cannot express |
| Hook stdout | enters context | keep hooks silent on the success path |

### Failure modes and guards

| Failure | Guard |
|---|---|
| Hook configured but never fires | `/hooks` to confirm registration; run the deliberate-violation test above |
| Hook path resolves relative to a launch directory that varies | Always `$CLAUDE_PROJECT_DIR/.claude/hooks/...` |
| Hook script itself has a bug and silently exits 0 | `set -euo pipefail`; keep one hook = one job; pair with a deny rule so a broken hook is not the only layer |
| Chatty hooks eat the window they protect | Success path prints nothing |
| Enforcement attempted purely in CLAUDE.md | Anything phrased "never/always" belongs in a hook or a permission rule |

**Beyond hooks.** For anything genuinely unrecoverable, the boundary belongs below Claude Code: OS file permissions, a container, a dedicated low-privilege user, credentials simply not present in the environment ([sandboxing](https://code.claude.com/docs/en/sandboxing)). Do not point an agentic run at a repository you have not read, and do not leave a long unattended run against anything not rollback-able.

### Track B delta

Headless runs must add a `PermissionRequest` hook (Phase 11) — without one, an unattended session stalls on a dialog nobody will answer. Keep the same `PreToolUse` set: CI is where the destructive spelling is most likely to be attempted and least likely to be watched.

---

## Phase 4 — Integration admission: external reach under least privilege

**Purpose.** Decide which external systems the agent may touch, on which transport, with which credentials, at what scope. Every server is paid for twice: in baseline budget (Phase 2) and in supply-chain exposure.

**Entry.** A task class the agent cannot complete from the filesystem alone — PR/CI status, production stack traces, live schema, current library docs.

**Exit.** Each admitted server has a named owner and auditable source; credentials are not literals in committed files; connection scopes are read-only unless writes are specifically required; privileged writes are covered by a `PostToolUse` audit hook; `claude mcp list` shows nothing that has gone a week unused; CI pins `--mcp-config` with `--strict-mcp-config`.

### Mechanisms and why

**Scopes** ([mcp-quickstart](https://code.claude.com/docs/en/mcp-quickstart)):

| Scope | Stored in | Use for |
|---|---|---|
| `local` (default) | `~/.claude.json`, under the entry for this project | experiments, personal credentials |
| `project` | `.mcp.json` at the project root, committed | servers the whole team needs |
| `user` | `~/.claude.json`, top-level `mcpServers` key | servers you want everywhere |

There is no `~/.claude/mcp.json`; global servers live in `~/.claude.json`.

**Transport.** Four are supported — stdio, HTTP, SSE and WebSocket — and SSE is deprecated in favour of HTTP where an HTTP server exists ([mcp](https://code.claude.com/docs/en/mcp)). Prefer HTTP for remote/hosted servers and stdio for local processes. Hosted services (Sentry, Linear, Notion) run behind OAuth and are added by URL ([mcp-quickstart](https://code.claude.com/docs/en/mcp-quickstart)).

### Copy-ready configuration

```bash
# GitHub — PR review, issue lookup, CI status
claude mcp add --transport http github https://api.githubcopilot.com/mcp/ \
  --header "Authorization: Bearer $GITHUB_PAT"

# Sentry — production stack traces (OAuth)
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp

# Inspect and manage
claude mcp list
claude mcp get github
```

Committed `.mcp.json` for a stdio server, credentials passed by environment:

```json
{
  "mcpServers": {
    "db": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "${DATABASE_URL_READONLY}"
      }
    }
  }
}
```

No filesystem MCP server is named here — the directory-whitelist practice holds whichever implementation you pick, and vetting it is your job. `@modelcontextprotocol/server-postgres` is the MCP server for PostgreSQL and provides read-only access to the database ([@modelcontextprotocol/server-postgres](https://www.npmjs.com/package/@modelcontextprotocol/server-postgres)). `.mcp.json` expands environment variables in place, with `${VAR}` and `${VAR:-default}` as the supported forms — `$VAR` bare is not one of them ([mcp](https://code.claude.com/docs/en/mcp)). Use a role whose grants are `SELECT`-only. Scope any filesystem server to a specific directory, never a home folder.

In-session management ([commands](https://code.claude.com/docs/en/commands)):

```text
/mcp                      open the panel: status, authenticate, reconnect
/mcp disable <server>     stop paying for it without losing the config
/mcp enable <server>
```

Audit hook for privileged writes (registered in Phase 3):

```bash
#!/usr/bin/env bash
set -euo pipefail
INPUT="$(cat)"
printf '%s\t%s\n' "$(date -Iseconds)" \
  "$(printf '%s' "$INPUT" | jq -c '{tool:.hook_event_name, session:.session_id, input:.tool_input}')" \
  >> "${CLAUDE_PROJECT_DIR:-.}/.claude/mcp-audit.log"
```

### Cost

Tool *names* and server instructions load into every session; full schemas stay deferred and load on demand via tool search ([context-window](https://code.claude.com/docs/en/context-window), [costs](https://code.claude.com/docs/en/costs)). Even so, removing servers frees space, and general guidance puts 50 tools in the 10–20K-token range once schemas are loaded ([tool search](https://code.claude.com/docs/en/agent-sdk/tool-search)). Rule: a server that has not been used in a week gets `/mcp disable`.

### Failure modes and guards

| Failure | Guard |
|---|---|
| Tool descriptions are read by the model as instructions | Install only from sources with a clear owner and auditable code; treat third-party servers as untrusted input |
| A server changes its tool descriptions after approval | Re-inspect `claude mcp get <name>` after upgrades; approval prompts show the tool name, not the full metadata the model receives |
| A write-capable connection where read-only would do | Separate credentials per scope; read-only by default |
| Silent connection failure | `/mcp` shows status; a mid-session HTTP or SSE disconnect reads as **pending** while Claude Code reconnects with exponential backoff and as **failed** after five attempts, never as "disconnected" ([mcp](https://code.claude.com/docs/en/mcp)); `claude mcp get <name>` reveals endpoint/transport mismatches |
| Credentials committed | Environment references only; grep the diff before the first commit |

### Track B delta

```bash
claude --mcp-config ./ci/mcp.ci.json --strict-mcp-config ...
```

`--strict-mcp-config` makes the run use only the servers from `--mcp-config`, ignoring all other MCP configuration ([cli-reference](https://code.claude.com/docs/en/cli-reference)). Without it, a CI run inherits whatever drifted into a developer's user settings. The CI server set should be strictly smaller than the interactive one — typically GitHub only.

---

## Phase 5 — Task framing and routing

**Purpose.** Per unit of work, *before any tool call*: clear the window, name the task, point at the spec, choose model and effort deliberately, and choose the execution shape.

**Entry.** A new, unrelated unit of work begins. Carrying debugging context into feature work is active interference, not neutral overhead.

**Exit.** Context cleared (or `--continue` used because it genuinely *is* the same work); model and effort set explicitly and re-checked after any tier switch; execution shape chosen against the rules below; the opening prompt names the spec by path and states whether code may be written yet.

### Mechanisms and why

**Clearing.** `/clear` starts a new conversation with empty context (aliases `/reset`, `/new`); stale context wastes tokens on every subsequent message ([commands](https://code.claude.com/docs/en/commands), [costs](https://code.claude.com/docs/en/costs)). To come back to prior work instead, `claude --continue` (`-c`) loads the most recent conversation ([cli-reference](https://code.claude.com/docs/en/cli-reference)). There is no `/rename`; session continuity is provided by `/resume` and `--continue`, so do not rely on naming a session before clearing it.

**Model and effort are independent axes.**

```bash
claude --model opus --effort high        # architecture, hard debugging
claude --model sonnet --effort medium    # general implementation
claude --model haiku                     # lookups, renames, mechanical greps
claude --model 'opus[1m]'                # whole-codebase session
```

Aliases: `sonnet`, `opus`, `haiku`, `fable`, plus full model names; `[1m]` selects the large window (`opus[1m]`, `sonnet[1m]`, or the suffix on a full name). `opusplan` uses Opus in plan mode then switches to Sonnet for execution — and on subscription tiers where Opus is automatically upgraded to 1M context, `opusplan` receives that upgrade in plan mode as well ([model-config](https://code.claude.com/docs/en/model-config)).

**Effort levels are five, not four, and the default is high, not medium.** Newer models support `low, medium, high, xhigh, max`; older 4.6-family models support `low, medium, high, max`. The default is `high` on every model that supports effort, except Opus 4.7, which defaults to `xhigh` ([model-config](https://code.claude.com/docs/en/model-config)). The practical consequence inverts the usual advice: **you must explicitly step effort *down* for routine work**, not up for hard work. `max` is documented as prone to overthinking with diminishing returns — test before adopting broadly.

Persistence:

```json
{ "effortLevel": "medium" }
```

`effortLevel` in settings accepts `low`, `medium`, `high`, `xhigh` — not `max` ([settings](https://code.claude.com/docs/en/settings)). To pin higher, use the environment:

```bash
export CLAUDE_CODE_EFFORT_LEVEL=max   # overridden by the /effort command
```

The CLI `--effort` accepts `low, medium, high, xhigh, max, ultracode` ([cli-reference](https://code.claude.com/docs/en/cli-reference)). Haiku does not support effort at all; after any switch to Haiku and back, re-check with `/model`, which shows the effort level next to the model name (also visible in the session header, e.g. "with low effort").

For a single hard turn without changing session configuration, include `ultrathink` in the prompt — it requests deeper reasoning for that turn only ([model-config](https://code.claude.com/docs/en/model-config)). Putting it in CLAUDE.md does nothing persistent.

**Execution shape.**

| Shape | Choose when | Because |
|---|---|---|
| Inline | single-file, known shape, few reads | cheapest; no return-contract overhead |
| Skill | the workflow repeats and should be identical each time | a skill's body loads only when used ([skills](https://code.claude.com/docs/en/skills)) |
| Subagent | the work is verbose and the *result* is small | separate context window; returns only a summary ([sub-agents](https://code.claude.com/docs/en/sub-agents)) |
| Agent team | several workers must share discoveries *mid-task* | teammates coordinate directly; subagents cannot ([agent-teams](https://code.claude.com/docs/en/agent-teams)) |

Skills are for reuse; subagents are for cost isolation. Teams are for coordination, and cost roughly 3–4× a single sequential session — do not reach for them until Phase 8's fit criteria pass.

### Copy-ready session opening (Track A)

```text
/clear
```

```text
I'm implementing user notification preferences.
Spec: docs/specs/notifications.md
Start with a plan. Do not write code yet.
Constraints already known: repository layer only, no direct DB access from controllers.
```

Permission modes ([permission-modes](https://code.claude.com/docs/en/permission-modes)): `default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions`. Shift+Tab cycles `default → acceptEdits → plan`; `auto` appears only when your account qualifies, `bypassPermissions` only after starting with certain flags, and `dontAsk` never appears in the cycle. Start planning work with:

```bash
claude --permission-mode plan
```

### Cost

This phase costs a few hundred tokens and saves the largest single category of waste: work done at the wrong tier, and work done on top of an unrelated context. `/usage` (aliased `/cost`) shows session token statistics; the status line can display context usage continuously ([costs](https://code.claude.com/docs/en/costs), [statusline](https://code.claude.com/docs/en/statusline)).

### Failure modes and guards

| Failure | Guard |
|---|---|
| Default effort silently high on trivial work | Set `effortLevel` per project; step down explicitly for mechanical tasks |
| Effort lost after a Haiku detour | `/model` after every tier switch |
| A team spawned for sequential work | Apply Phase 8 fit criteria before spawning |
| Debug context bleeding into feature work | `/clear` at every task boundary |
| Spec pasted into the prompt instead of referenced | Reference by path; the model reads what it needs |

### Track B delta

Nothing is cleared, because nothing is carried: one process per unit of work. Pin model and effort by flag so the run is reproducible regardless of user settings, and pass `--strict-mcp-config`. Do not use `--continue` in CI — a resumed conversation makes the run non-deterministic.

---

## Phase 6 — Bounded exploration: read-only discovery with an output contract

**Purpose.** Acquire understanding without putting the traversal into the window that has to carry the implementation.

**Entry.** The unit of work is discovery rather than change, or plan mode was entered, or the change touches code whose shape is not already known.

**Exit.** Findings exist as a structured, capped summary, not as accumulated file reads. Exploration followed a named path with a stated stop point. Nothing has been written: plan mode is still on, or the subagent has returned.

### Mechanisms and why

**Plan mode**, not a prose instruction: in plan mode Claude reads files, runs shell commands to explore, and writes a plan, **but does not edit your source** ([permission-modes](https://code.claude.com/docs/en/permission-modes)). Note the precision — it is a *no-source-edit* mode, not a no-tool mode; it may still run commands, so Phase 3's Bash guard still matters.

**Subagents**, not inline reading, for wide traversal: each subagent runs in its own context window with its own system prompt, tool access and permissions, and returns only the summary ([sub-agents](https://code.claude.com/docs/en/sub-agents)).

**Two built-ins already cover part of this.** `Explore` is a fast read-only agent for searching and analysing codebases, and `Plan` is the research agent used during plan mode to gather context before a plan is presented, with read-only tools and Write and Edit denied ([sub-agents](https://code.claude.com/docs/en/sub-agents)). Reach for them first; the custom agent below still earns its place, but for a narrower reason than "the built-ins may not exist" — it exists to impose a capped output contract and a cheap model, which the built-ins do not give you.

As of v2.1.198 `Explore` no longer runs on Haiku — it inherits the main conversation's model, capped at Opus on the Claude API ([sub-agents](https://code.claude.com/docs/en/sub-agents)). On an Opus session that makes the built-in expensive for grep-shaped work; define a custom explorer pinned to `model: haiku` when cost matters more than depth.

A returning subagent is not destroyed. The Agent tool result carries an `agentId`, and that instance can be resumed later by passing `resume: <session_id>` with it — except `Explore` and `Plan`, which are one-shot and return no `agentId` ([agent-sdk/subagents](https://code.claude.com/docs/en/agent-sdk/subagents)). What is documented is that a subagent inherits none of the parent's history or tool results; whether siblings can see each other's work is not addressed either way, so keep treating the return string as the only channel.

**Delegation is driven by the `description` field**, together with the task in your request and the current context — not by particular verbs in the prompt ([sub-agents](https://code.claude.com/docs/en/sub-agents)). Write the description as the routing key it is, and invoke by name whenever the invocation is a gate rather than a convenience.

**Scope the exploration.** Unscoped investigation is the named failure pattern: asked to investigate without scoping, Claude reads hundreds of files and fills the context ([best-practices](https://code.claude.com/docs/en/best-practices)).

### Copy-ready configuration

`.claude/agents/codebase-researcher.md`:

```markdown
---
name: codebase-researcher
description: Traces where a behaviour is implemented across the repo and reports the call sites. Use for "find all", "where does X happen", and pre-implementation surveys.
model: haiku
effort: low
tools:
  - Read
  - Grep
  - Glob
---

You survey code and report. You never modify anything.

Return exactly this structure and nothing else:

## Entry points
- `path:line` — one sentence

## Call sites
- `path:line` — one sentence

## Risks
- one sentence each

Hard limits: maximum 20 items total. No code snippets unless a line reference
genuinely cannot describe the problem. If the survey would exceed 20 items,
report the 20 most load-bearing and state what was omitted.
```

A subagent definition also accepts a `memory` key — `user`, `project` or `local` — which turns on persistent memory and cross-session learning, with `memory: user` stored under `~/.claude/agent-memory/<name-of-agent>/`; it is part of auto memory, so disabling auto memory disables it ([sub-agents](https://code.claude.com/docs/en/sub-agents)). Deliberately omit it here: a survey agent must re-derive its findings from the current tree, and a remembered map of a repo that has since moved is worse than no map.

Scoped exploration prompt (Track A):

```text
Check the query in auth/session.ts. If it looks fine, check the Redis client
initialisation in lib/cache.ts. Stop there and report back. Do not read anything else.
```

Delegated exploration with a contract:

```text
Use the codebase-researcher subagent to identify every database call in the auth module.
Return file path, function name and query type. Maximum 20 items.
```

Parallel dispatch is legitimate only when facets are independent — no shared state, no file overlap, no data dependency ([best-practices](https://code.claude.com/docs/en/best-practices)). Subagents cannot spawn subagents, so all orchestration stays with the main session.

### Cost

The savings are entirely a function of the return size. A pass that does 10,000 tokens of internal work and returns 500 tokens saves ~9,500; the same pass returning 8,000 tokens saves 2,000; returning 6,000 tokens makes the delegation approximately pointless. *(These figures are illustrative arithmetic, not measurements.)* Cost control here is therefore: (a) cheap model on the subagent (`model: haiku`, or `CLAUDE_CODE_SUBAGENT_MODEL=haiku` as the default), and (b) a hard item cap in the definition, not only in the prompt.

### Failure modes and guards

| Failure | Guard |
|---|---|
| Subagent returns an essay | Item cap and required fields in the agent file, so every invocation inherits them |
| Exploration silently becomes editing | Stay in plan mode until a plan is approved; give research agents a read-only `tools:` whitelist |
| Parallel agents collide | Dispatch in parallel only for provably independent facets; otherwise chain sequentially |
| Findings are trusted but unsourced | Require `path:line` on every item so any claim can be checked in one read |

### Track B delta

Plan mode is the wrong tool unattended — there is nobody to approve the plan. In CI, express the boundary through subagent tool whitelists instead: a research stage whose agent has `Read`/`Grep`/`Glob` and no write tools, whose structured output is written to a file and passed to the next stage.

---

## Phase 7 — Execution loop: edits under in-loop automation

**Purpose.** Make the changes, with Phase 3's guardrails firing per tool call and formatting/lint automation attached to `PostToolUse` — without letting that automation's output eat the window it was meant to protect.

**Entry.** An approved plan or bounded findings summary exists; plan mode released.

**Exit.** The change is complete against the plan; formatter/lint hooks ran without dominating context; verbose tool output passed through a filter rather than landing raw.

### Mechanisms and why

`PostToolUse` fires after a tool call succeeds, can be matched to specific tools and filtered by file glob, and supports `runInBackground: true` so the hook does not block the next tool call ([hooks-guide](https://code.claude.com/docs/en/hooks-guide)). This is the right layer for formatting because it is unconditional; a CLAUDE.md line asking for formatting is not.

For bulk mechanical work that is genuinely understood, `acceptEdits` (Shift+Tab from default) removes per-edit confirmation. Do not combine it with work that touches anything your `PreToolUse` guards do not cover.

### Copy-ready configuration

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/format-on-write.sh",
            "runInBackground": true
          }
        ]
      }
    ]
  }
}
```

`.claude/hooks/format-on-write.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT="$(cat)"
FILE="$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')"
[ -n "$FILE" ] && [ -f "$FILE" ] || exit 0

case "$FILE" in
  *.ts|*.tsx|*.js|*.jsx) npx --no-install prettier --write "$FILE" >/dev/null 2>&1 || true ;;
  *.py)                  ruff format "$FILE" >/dev/null 2>&1 || true ;;
  *)                     exit 0 ;;
esac
exit 0
```

Note the deliberate silence: the formatter's stdout is discarded. Hook output is fed back into context and therefore costs tokens ([costs](https://code.claude.com/docs/en/costs)).

Filtering verbose command output — the same idea applied to tests:

```bash
#!/usr/bin/env bash
# .claude/hooks/filter-test-output.sh
set -euo pipefail
INPUT="$(cat)"
printf '%s' "$INPUT" \
  | jq -r '.tool_response.stdout // empty' \
  | grep -E '^(FAIL|ERROR|✗|[0-9]+ (failing|failed))' \
  | head -50
```

### Cost

| Source | Bound |
|---|---|
| Formatter output per write | discard it; `runInBackground: true` |
| Slow formatter × many small edits | do not format on write at all — format once between sessions |
| Test/log output | filter to failures only; a large suite's raw output is orders of magnitude larger than its failures |
| Auto-firing skills stacking on hook output | keep auto-invocable skills conservative while `PostToolUse` hooks are active |

### Failure modes and guards

| Failure | Guard |
|---|---|
| Formatting passes silently consume a large share of the window | Measure with `/context` after a long editing session; if formatting is expensive, move it out of the loop |
| `acceptEdits` removes the human decision point a guard assumed | Only use it where `PreToolUse` covers every destructive path; never unattended against non-rollback-able targets |
| Hook failure aborts productive work | `|| true` on non-critical automation; reserve non-zero exits for policy |
| Edits drift from the plan | Re-state the plan's remaining items at each checkpoint |

### Track B delta

Formatting in CI belongs to CI, not to the hook — run it as a pipeline step after the agent finishes. Keep only the filtering hooks in the headless config, since output volume, not developer convenience, is what breaks an unattended run.

---

## Phase 8 — Scale-out coordination (conditional branch of execution)

**Purpose.** Run several full Claude Code instances against one change when the work is genuinely parallel and workers must share discoveries mid-task.

**Entry — all of these, not some:**

- Independent facets, or cross-layer slices with file/directory boundaries agreed upfront.
- No data dependency between units.
- Long enough to justify roughly 3–4× the tokens of a sequential session.
- The experimental flag set persistently in settings, not just in one shell.

Two things people assume here are not documented: that a teammate-to-teammate exchange stays out of the lead's context window, and that a CLAUDE.md section addressed to teammates is acted on by teammates and ignored by solo sessions. The docs confirm only that each teammate has its own window, that the lead's history does not carry over, and that messages are delivered automatically ([agent-teams](https://code.claude.com/docs/en/agent-teams)). Enforce directory boundaries with a hook, not with prose, and do not budget on the context accounting.

**Exit.** Every unit claimed and finished from the shared task list, with no cross-boundary file edits; teammate exchanges stayed between teammates; parallel worktree runs produced one PR per unit; the run was watched closely enough to catch a teammate going off course.

### Mechanisms and why

Agent teams are experimental and disabled by default ([agent-teams](https://code.claude.com/docs/en/agent-teams)):

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "teammateMode": "in-process"
}
```

Both keys go in `~/.claude/settings.json`. `teammateMode` defaults to `in-process` (it was `auto` before v2.1.179); `auto` enables split panes when you are inside tmux or iTerm2 and falls back to in-process otherwise. In the agent panel, arrow keys select a teammate and Enter opens it for direct messaging.

Teams are created in natural language — no config file, no dedicated command ([agent-teams](https://code.claude.com/docs/en/agent-teams)):

```text
I'm adding a payment webhook handler. Create an agent team:
- teammate "api" owns src/api/webhooks/** only
- teammate "schema" owns prisma/ and src/repositories/** only
- teammate "tests" owns test/** only
No teammate edits files outside its directory. Coordinate through the task list.
```

Shared state lives at `~/.claude/tasks/{team-name}/`, persists locally, and is never uploaded. Teammates load the same project context as a regular session — CLAUDE.md, MCP servers, skills — and can reference subagent definitions from any scope, so your `.claude/agents/` roster is available to each of them.

**Teams vs. subagents:** subagents report back to a single decision maker and cannot spawn or message each other; teams coordinate directly. Nesting is not supported in either direction — teammates cannot spawn their own teammates.

For codebase-wide mechanical migrations, prefer the bundled `/batch` skill over hand-rolled teams: it researches, decomposes the work into 5–30 units, and spawns background subagents in git worktrees ([commands](https://code.claude.com/docs/en/commands)).

CLAUDE.md addition (advisory — see Tensions):

```markdown
## Agent team notes
When running as a teammate:
- Check the shared task list before starting any new work.
- Claim only tasks in your assigned scope.
- Do not modify files outside your assigned directory.
- Message the owning teammate directly when a discovery affects their scope.
```

### Cost

Roughly 3–4× a sequential session, higher with plan mode. CLAUDE.md is loaded once per teammate, so the baseline from Phase 2 is multiplied by team size — a fat CLAUDE.md is much more expensive here than in solo work. Instrument with `SubagentStart`/`SubagentStop`/`TaskCreated`/`TaskCompleted`/`TeammateIdle` hooks if you need a record of what each unit cost.

### Failure modes and guards

| Failure | Guard |
|---|---|
| Two teammates edit the same file | Directory ownership stated at spawn time *and* enforced by a `PreToolUse` hook that rejects writes outside the teammate's directory — the CLAUDE.md section alone is advisory |
| A teammate drifts off course unnoticed | Split-pane mode (`teammateMode: "auto"` inside tmux) or a `TeammateIdle`/`SubagentStop` hook that notifies |
| Team spawned for sequential work | Re-read the entry criteria; if any fails, use subagents |
| Coordination state itself becomes the cost | Small teams (2–4), short-lived, one merge target per unit |

### Track B delta

Do not run experimental agent teams unattended. In CI, use `/batch`-style decomposition with one worktree and one PR per unit, so each unit is independently reviewable and revertable, and so a failure is scoped to one PR rather than one shared branch.

---

## Phase 9 — Session-health triage (continuous overlay on 6–8)

**Purpose.** Detect and respond to degradation of the working session itself, and to failures of the harness rather than of the code.

**Entry — any signal:** answers hedging on details handled confidently earlier; earlier decisions misremembered or referenced inconsistently; usage climbing toward the compaction threshold; a skill that never fired; a hook that did nothing; a server showing as failed.

**Exit.** The escalating intervention was applied and the symptom is gone; harness failures were diagnosed at the mechanism, not worked around by re-prompting.

### The escalation ladder (cheapest first)

| Step | Command | Use when | Loses |
|---|---|---|---|
| 1 | `/context` | any degradation signal, before doing anything else | nothing — it only measures, and it is the documented first rung ([commands](https://code.claude.com/docs/en/commands)) |
| 2 | `/clear` | genuine task boundary | everything; that is the point |
| 3 | `claude --continue` | same work, new sitting | nothing, but carries any rot forward |
| 4 | Ask for a summary in-session, then `/compact <focus>` | mid-task, before a phase change | tool-call detail, exact file contents, precise constraint wording |
| 5 | Close and restart from a written spec | genuine degradation | the transcript; keeps the decisions |

The documented escalation runs `/context` → `/compact` → `/clear` → `/resume`: measure with optimisation suggestions first, summarise without clearing next, start fresh third, and return to a cleared conversation last ([commands](https://code.claude.com/docs/en/commands)). The ordering above differs only in preferring a clean boundary over a lossy summary once the work has genuinely changed.

```text
Summarise what we have established about the auth module: decisions made,
constraints I stated, files already changed. Then stop.
```

```text
/compact Focus on code changes made, test results, and the constraints I stated.
```

Standing preferences can be declared once in CLAUDE.md, but compaction is not obliged to honour them:

```markdown
## Compact instructions
When compacting, preserve: code changes made this session, error patterns found,
and any explicit constraint I stated about this codebase.
```

Monitoring: `/context` for the breakdown, `/usage` (`/cost`) for session token statistics, the status line for continuous display ([costs](https://code.claude.com/docs/en/costs), [context-window](https://code.claude.com/docs/en/context-window)).

Auto-compaction fires at roughly 85% of window capacity, and the trigger point is tunable: `CLAUDE_CODE_AUTOCOMPACT_PCT_OVERRIDE` accepts 1–100, with lower values compacting earlier ([env-vars](https://code.claude.com/docs/en/env-vars)). Treat that percentage as the deadline the ladder above exists to beat — an intervention you choose is always cheaper than one the harness performs for you.

### Harness triage — diagnose at the mechanism

| Symptom | First check | Interpretation |
|---|---|---|
| Skill never fires | invoke it directly as `/skill-name` | works manually → the `description` is the fault, not the body |
| Hook does nothing | `/hooks`, then run the deliberate violation | not listed → registration; listed but passive → script bug |
| Rule never applied | `/memory` | not listed → wrong path or wrong frontmatter key (`paths:`) |
| Subagent not selected | invoke it by name in the prompt | works → description matching; fails → definition |
| MCP server failing | `/mcp`, then `claude mcp get <name>` | endpoint/transport mismatch is the common cause |
| Something inexplicable | `/debug`, then `/doctor` | enables debug logging for the session and reads the log |

### Failure modes and guards

| Failure | Guard |
|---|---|
| Re-prompting around a broken mechanism | Any repeated instruction is a mechanism failure until proven otherwise — run the table above |
| Compaction eats the constraint that mattered | Write constraints into the spec file, not only into the conversation |
| Degradation mistaken for model incompetence | Check `/context` before concluding anything about quality |

### Track B delta

There is no triage inside a headless run. Instead: keep every run short and single-purpose, and treat "the run needed compaction" as a signal that the unit of work was too large. Capture the transcript path from the hook payload at `SessionEnd` so a failed run can be inspected afterwards.

---

## Phase 10 — Verification gate: independent check, not self-report

**Purpose.** Establish correctness by a mechanism the implementing context did not control. The implementer's report that it is done is not evidence.

**Entry.** Implementation reports complete.

**Exit.** Tests and build ran from the `Stop` hook, not from the model's initiative; a tool-restricted reviewer invoked **by name** returned a capped structured finding list over the changed files only; the spec's acceptance criteria were walked item by item; every finding is fixed or explicitly recorded as not done.

### Mechanisms and why

`Stop` fires when Claude finishes a turn and is the documented place to run tests or post-completion work ([hooks](https://code.claude.com/docs/en/hooks), [hooks-guide](https://code.claude.com/docs/en/hooks-guide)). Attaching verification here removes the model's discretion over whether verification happens. Give it a check it can run — tests, a build, a screenshot to compare — because that is the difference between a session you watch and one you can walk away from; passing tests alone do not close the trust-then-verify gap ([best-practices](https://code.claude.com/docs/en/best-practices)).

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/verify.sh"
          }
        ]
      }
    ]
  }
}
```

```bash
#!/usr/bin/env bash
# .claude/hooks/verify.sh — runs on every turn end; must stay fast and quiet
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}"

git diff --quiet && git diff --cached --quiet && exit 0   # nothing changed, nothing to verify

OUT="$(npm run -s build 2>&1)"; BUILD=$?
if [ $BUILD -ne 0 ]; then
  { echo "BUILD FAILED"; printf '%s\n' "$OUT" | tail -30; } >&2
  exit 2
fi

OUT="$(npm test --silent 2>&1)"; TESTS=$?
if [ $TESTS -ne 0 ]; then
  { echo "TESTS FAILED"; printf '%s\n' "$OUT" | grep -E '^(FAIL|ERROR|✗)' | head -30; } >&2
  exit 2
fi
exit 0
```

Exit 2 is a blocking error, and the hook's stderr is shown to Claude rather than merely logged ([hooks](https://code.claude.com/docs/en/hooks)) — which is why the failure text above is written to be read by the thing that must fix it.

Reviewer subagent — `.claude/agents/code-reviewer.md`:

```markdown
---
name: code-reviewer
description: Reviews changed code for security issues, logic bugs and performance problems. Use after writing or modifying code, before opening a PR.
model: sonnet
effort: high
tools:
  - Read
  - Grep
  - Glob
  - Bash
disallowedTools:
  - Write
  - Edit
---

You review only the files listed in the invoking prompt. You never modify anything.
Style and formatting are out of scope — the formatter owns those.

Return at most 10 findings, most severe first, each as:
- SEVERITY (high|medium|low) — `path:line` — one sentence on the defect
  and one sentence on the concrete fix.
No code snippets unless the defect cannot be described without one.
If there are no substantive findings, say exactly: "No substantive findings."
```

Invoke it **by name** — explicit invocation bypasses description matching:

```text
Use the code-reviewer agent on the files changed in this branch (git diff --name-only main...HEAD).
Maximum 10 findings.
```

Then the half no hook can perform:

```text
Walk docs/specs/notifications.md acceptance criteria one at a time.
For each: quote the criterion, state met / not met, and cite the file:line or the
test name that demonstrates it. Do not summarise. Do not skip criteria.
```

For a quality (not bug-hunting) pass, `/code-review --fix` is the bug-finding review; `/simplify` is now a cleanup-only review that applies fixes without hunting for bugs ([code-review](https://code.claude.com/docs/en/code-review)).

### Cost

The reviewer's cost is bounded by the finding cap and by scoping it to changed files. The `Stop` hook's cost is bounded by the early exit on a clean tree and by filtering output to failures — an unfiltered failing suite is the single largest accidental context expense in this phase.

### Failure modes and guards

| Failure | Guard |
|---|---|
| Green suite treated as done | Acceptance-criteria walk is a separate, mandatory step |
| Reviewer edits the code it is reviewing | `disallowedTools: [Write, Edit]` |
| Reviewer never invoked because a description did not match | Always invoke by name for gates |
| Stop hook runs on every trivial turn | Early exit when the working tree is unchanged |
| Findings quietly dropped | Each finding ends as fixed, or written down as "NOT done: …" |

### Track B delta

The same `Stop` hook is the CI gate; its exit code is the run's verdict. The acceptance-criteria walk should write its result to a file that the pipeline archives, so the judgment half of verification leaves an artifact instead of living in a transcript nobody reads.

---

## Phase 11 — Irreversible handoff

**Purpose.** Cross the line where actions escape the session — deploys, migrations, PRs, unattended runs. This phase inverts every earlier default.

**Entry.** Verification passed, and the next action writes outside the working tree or runs without a human watching.

**Exit.** Every destructive workflow is explicit-only and environment-named; unattended runs decide permissions through a `PermissionRequest` hook; `bypassPermissions` is granted to one specifically trusted agent, never globally; CI pins its MCP set strictly; the target is rollback-able or the run is attended; a session/audit trail exists.

### The inversions

| Earlier default | Here |
|---|---|
| Skills auto-invoke when relevant | `disable-model-invocation: true` |
| One skill name per job | one name per *environment* |
| Permission dialogs answered by a human | answered by a hook |
| Config inherited from user scope | pinned by flag |
| Mistakes recovered with `/clear` | nothing is recoverable |

### Copy-ready configuration

Name destructive skills per environment rather than relying on override. Precedence runs enterprise > personal > project, and a skill at any level also overrides a bundled skill of the same name; only plugin skills are namespaced (`plugin-name:skill-name`) and so cannot collide ([skills](https://code.claude.com/docs/en/skills)). A project-level `deploy` will not win against a personal one.

`.claude/skills/deploy-prod/SKILL.md`:

```markdown
---
name: deploy-prod
description: Deploys payments-service to PRODUCTION. Human-invoked only.
disable-model-invocation: true
---

Preconditions — abort and report if any fails:
1. `git rev-parse --abbrev-ref HEAD` is `main`.
2. Working tree clean: `git status --porcelain` is empty.
3. Build and tests green in this session.
4. The migration in this release is reversible, or there is no migration.

Steps:
1. Print the diff of what will ship: `git log --oneline origin/prod..HEAD`.
2. Ask the operator to confirm the release tag before proceeding.
3. Run `./scripts/deploy.sh prod`.
4. Poll health for 5 minutes: `./scripts/health.sh prod`.
5. On any non-200, run `./scripts/rollback.sh prod` and report.
```

`disable-model-invocation: true` means only you can invoke the skill — the documented use is exactly workflows with side effects, like `/commit`, `/deploy`, `/send-slack-message` ([skills](https://code.claude.com/docs/en/skills)). Keep `deploy-staging` and `deploy-prod` as *distinct names*; do not rely on one scope overriding another to change which environment a `/deploy` hits.

Unattended permission policy — `PermissionRequest` hook:

```json
{
  "hooks": {
    "PermissionRequest": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/ci-permissions.sh"
          }
        ]
      }
    ]
  }
}
```

```bash
#!/usr/bin/env bash
# Deny by default. Approve only an explicit, small allowlist.
set -euo pipefail
INPUT="$(cat)"
TOOL="$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')"
CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')"

case "$TOOL" in
  Read|Grep|Glob) exit 0 ;;
  Bash)
    case "$CMD" in
      "npm test"*|"npm run build"*|"npm run lint"*|"git diff"*|"git log"*) exit 0 ;;
    esac
    ;;
esac
echo "DENIED in unattended mode: $TOOL ${CMD:-}" >&2
exit 2
```

The exit-code form above is the version-stable minimum. For a richer response than allow/deny, emit the documented decision envelope instead — the wrapper keys are `hookSpecificOutput.hookEventName`, `hookSpecificOutput.permissionDecision` and `hookSpecificOutput.permissionDecisionReason` ([hooks](https://code.claude.com/docs/en/hooks)):

```bash
jq -n '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: "Destructive command blocked by hook"
  }
}'
```

**Do not gate unattended behaviour on `$CLAUDE_CODE_REMOTE`.** It is set to `true` only when Claude Code runs as a *cloud* session, and is a way to detect that specific case ([env-vars](https://code.claude.com/docs/en/env-vars)) — a local headless run (`claude -p` in CI on your own runner) does not set it. Detect your own CI with the runner's own variable (`CI`, `JENKINS_URL`, `GITHUB_ACTIONS`) instead.

Trusted deploy agent, and only that agent:

```markdown
---
name: release-runner
description: Executes the release script for an already-approved release. Invoked explicitly by the release pipeline only.
model: sonnet
permissionMode: bypassPermissions
tools:
  - Bash
  - Read
---
Run only ./scripts/deploy.sh with the environment given in the prompt.
Do not modify files. Report the deploy id and the health-check result.
```

`permissionMode: bypassPermissions` on a subagent skips the normal per-tool-call permission rules ([sub-agents](https://code.claude.com/docs/en/sub-agents)). Grant it to exactly one narrow agent with a two-tool whitelist, never to the main session.

CI invocation:

```bash
claude \
  --model sonnet \
  --effort medium \
  --mcp-config ./ci/mcp.ci.json \
  --strict-mcp-config
```

Confirm your version's non-interactive entry flag with `claude --help` before wiring the job; the flags above are the ones this design depends on being pinned ([cli-reference](https://code.claude.com/docs/en/cli-reference)).

Post-release watch, in an attended session:

```text
/loop 60s Check ./scripts/health.sh prod and report only status changes.
```

`/loop [interval] [prompt]` runs only while the session is open; omit the interval to let Claude self-pace ([commands](https://code.claude.com/docs/en/commands)).

Audit trail — `SessionStart`/`SessionEnd` hooks writing `session_id`, `cwd`, `permission_mode` and the transcript path from the stdin payload to a log the pipeline archives.

### Failure modes and guards

| Failure | Guard |
|---|---|
| A deploy skill auto-fires mid-conversation | `disable-model-invocation: true` on every side-effecting skill |
| Wrong environment deployed because a name was shadowed | Environment-distinct names; never rely on override precedence |
| Unattended run stalls on a dialog | `PermissionRequest` hook present and tested |
| Unattended run approves something new | Deny-by-default allowlist, as above |
| CI inherits a developer's MCP servers | `--strict-mcp-config` |
| Nothing to roll back to | Release only rollback-able targets unattended; everything else is attended |

### Track A delta

Interactively, the `PermissionRequest` hook should be absent or permissive — the human *is* the policy. What stays identical across tracks: `disable-model-invocation`, environment-distinct names, and the rollback precondition.

---

## Phase 12 — Compounding maintenance: corrections back into the layers

**Purpose.** Convert observed failures into configuration, and delete configuration that has decayed. This is the only phase whose input is evidence rather than intent, and the only one whose default operation is **subtraction**.

**Entry.** The same correction has been typed more than twice in one project; or a rule that exists was violated; or a debugging session just ended; or a periodic audit is due.

**Exit.** The correction is captured and reconciled: the hierarchy re-read like code, stale lines cut, the ignored rule reworded or promoted out of prose into a hook, procedural detail moved into a skill, unused MCP servers dropped; the file is back under 200 lines; `/memory` confirms what now loads; rules for problems never observed have been removed.

### The decision table — where a correction goes

| The correction is… | Destination | Reason |
|---|---|---|
| A fact about this project (command, path, boundary) | `CLAUDE.md` | needed every session, cheap, judgment-shaped |
| Specific to one area of the tree | `.claude/rules/<area>.md` with `paths:` | conditional loading |
| A multi-step procedure | a skill | body loads only when used |
| Something that must *never* happen | `PreToolUse` hook (+ a deny rule) | advisory layers are not enforcement |
| Something that must *always* happen after edits | `PostToolUse` / `Stop` hook | survives compaction |
| A one-off | nowhere | do not pay baseline cost for it |

### The subtractive protocol

When a rule that exists was ignored, **do not add a rule**. In order:

1. Measure: `/context`, and count lines in every CLAUDE.md in the hierarchy.
2. Cut anything that fails *"would removing this cause a mistake?"*.
3. Cut anything describing behaviour the model already gets right unprompted.
4. Reword the ignored rule to be concrete and singular; if it is a hard stop, promote it to a hook and delete the prose.
5. Re-verify with `/memory`, and re-run the scenario that failed.

Audit sweep, quarterly or after any large feature:

```bash
wc -l CLAUDE.md .claude/rules/*.md ~/.claude/CLAUDE.md
claude mcp list                 # drop or /mcp disable anything unused for a week
ls .claude/skills .claude/agents # one job per skill; distinct names per environment
git log --oneline -- CLAUDE.md   # rules that never changed after a failure are suspects
```

### Failure modes and guards

| Failure | Guard |
|---|---|
| Rules stacked onto rules | The subtractive protocol above; a net-line-count check at review |
| Speculative rules for unobserved problems | The "corrected more than twice" threshold is the entry condition |
| Global and project files drifting into contradiction | Remember they are concatenated, not resolved — keep the levels topically disjoint |
| Skills accumulating and crowding the description budget | Delete unused skills; they cost baseline even when never invoked |
| A stale CLAUDE.md line contradicting current code | Treat CLAUDE.md as code in review: any PR that changes a documented command must change the line |

### Track B delta

CI has no `#`-style mid-session capture and no human to notice a repeated correction. Instead, mine the artifacts: failed-run transcripts and the `PreToolUse` denial log are the evidence stream. A denial that fires repeatedly for a legitimate action is a signal to fix the config, not the agent.

---

## Tensions

Each of these is a real trade in this design. The decision rule matters more than the trade.

**1. Determinism costs the context that makes instructions work.**
Hooks fire regardless of context state — but their output re-enters context, and a matcher-less `PreToolUse` group spawns a process before every Read, Grep, Glob, Write and Bash. *Decide by recoverability:* if the action is not recoverable in-session, pay for the hook and make it silent. If it is recoverable, a CLAUDE.md line is the cheaper instrument.

**2. Deny rules are documented but must never stand alone.**
`permissions.deny` exists and takes precedence over allow. What is *not* documented is any guarantee that a pattern cannot be reached by an equivalent spelling of the same command. *Decide by asymmetry:* deny rules cost nothing, so keep them, but never let one be the only layer between the agent and something unrecoverable. Below both sits OS/container isolation, which is the only layer that does not depend on string matching.

**3. Imports satisfy the 200-line metric while defeating its purpose.**
`@path` keeps the file short; the imported content still lands in context. Only skills (body loads on use) and `paths:`-scoped rules (glob-gated) actually reduce the baseline. *Decide by frequency:* content needed every session may be imported; anything else moves to a skill or a rule file.

**4. Subagent isolation is an output contract, not a property of the mechanism.**
A verbose return negates the isolation entirely, and a general-purpose subagent inherits the parent's model, so it isolates context without reducing the rate. *Decide by ratio:* delegate when expected internal work is at least ~10× the size of the contracted return. If you cannot state the return contract in two lines, do the work inline.

**5. Big windows enable exactly the behaviour the context discipline forbids.**
`opus[1m]` invites loading a whole codebase; adherence still decays with volume, and an 800K-token session costs far more than a 100K one. *Decide by task:* use the large window for genuinely global questions (architecture, migration surveys) and for nothing else. Never use it as a substitute for scoping.

**6. Effort is pinned in three places that disagree.**
`effortLevel` in settings takes `low|medium|high|xhigh` but not `max`; `CLAUDE_CODE_EFFORT_LEVEL` pins `max` but is overridden by `/effort`; `ultrathink` applies to one turn; Haiku ignores effort entirely, and the default is `high` (or `xhigh` on Opus 4.7), not medium. *Decide by locus:* pin the project's normal level in `settings.json`, use `--effort` for a deliberate session, `ultrathink` for a single hard turn, and re-check `/model` after any tier switch. Do not encode effort in more than two places.

**7. Skill auto-invocation is unreliable and every fix costs baseline.**
Routing depends on the `description`, which competes with every other skill inside a budget of 1% of the window. Patching it with a CLAUDE.md protocol line ("use `/pr-review` before every PR") pays permanent baseline cost to repair a routing mechanism. *Decide by consequence:* if a missed invocation is merely inconvenient, accept the miss. If it is a gate, do not rely on routing at all — invoke by name, or put the check in a `Stop` hook.

**8. Safety on deploy is bought by moving the gate to the least reliable layer.**
`disable-model-invocation: true` makes destructive skills safe by removing automation — which relocates the trigger to human memory, and the usual mitigation for that is a CLAUDE.md line, i.e. the advisory layer. *Decide by track:* interactively, accept it and add the reminder. Unattended, do not rely on memory at all — the pipeline script invokes the named skill, and the `PermissionRequest` hook denies everything else by default.

**9. Precedence has no single mental model.**
Settings follow a documented priority order, permission *rules* merge across scopes, and CLAUDE.md files are concatenated root-to-cwd rather than overridden. Same-named artifacts across scopes are the reliable source of surprise. *Decide by naming:* make names distinct across scopes and stop reasoning about precedence — it is the only strategy that is correct under all three models.

**10. `/clear` is the cheapest hygiene intervention and the most lossy.**
Wiping destroys accumulated understanding, and there is no `/rename` to make the wiped session findable by a chosen label. *Decide by boundary:* clear only at genuine task boundaries; within one task, prefer a manual summary, then `/compact` with focus, then a restart from a written spec — which is why the spec must be a file, not a paragraph in the transcript.

**11. Compaction mitigations inherit the unreliability they are meant to fix.**
Standing compact instructions are not guaranteed to be honoured, and compaction is lossy precisely for tool-call detail, exact file contents and exact constraint wording. *Decide by durability:* anything you cannot afford to lose belongs in a file on disk — the spec, the plan, the constraint list — not in the conversation.

**12. Evidence-driven configuration underspecifies new projects on purpose.**
The "corrected more than twice" threshold means a fresh repo pays for its own configuration in real mistakes. *Decide by cost of the mistake:* seed only the categories that are cheap and unambiguous (build/test commands, frozen directories, hard constraints), and let everything else be earned. Do not pre-write a style guide.

**13. `acceptEdits` removes the human decision point that guards assume.**
Speed on mechanical work pulls directly against the human-in-the-loop assumption behind confirm-style guards. *Decide by coverage:* auto-accept only where a `PreToolUse` hook already covers every destructive path in scope, and never for an unattended run against a non-rollback-able target.

**14. Agent teams save the lead's context and multiply everything else.**
3–4× tokens, CLAUDE.md loaded once per teammate, and the file-boundary guarantee resting on advisory prose. *Decide by enforceability:* only spawn a team if you are willing to enforce the boundaries with a hook. If the boundary is prose-only, use subagents and accept the sequential cost.

**15. Every MCP server is reach paid for on every turn, plus supply-chain exposure.**
Deferred schemas relieve most of the token cost, but names and instructions stay resident, and the approval prompt shows a tool name rather than the metadata the model actually receives. *Decide by task class:* admit a server only when a named task class is impossible without it, and disable it the week it stops being used.

**16. Verification's judgment half has no enforcement mechanism.**
The `Stop` hook can prove the build and tests; nothing can prove that acceptance criteria were walked. *Decide by artifact:* make the criteria walk produce a written, quotable output that a human or a later pipeline step can check — turning an unenforceable norm into a reviewable file.

---

## Reference: phase → mechanism → artefact

| # | Phase | Primary mechanism | Config artefact | Verify with |
|---|---|---|---|---|
| 1 | Scope wiring | settings scopes, CLAUDE.md hierarchy | `~/.claude/settings.json`, `.claude/settings.json`, `.claude/settings.local.json`, `CLAUDE.md`, `.gitignore` | `/memory`, `claude mcp list` |
| 2 | Baseline budgeting | CLAUDE.md, `.claude/rules/`, skills | `CLAUDE.md` (<200 lines), `.claude/rules/*.md` with `paths:` | `/context` |
| 3 | Enforcement | hooks, permissions, OS isolation | `.claude/settings.json` `hooks`, `.claude/hooks/*.sh`, `permissions.deny` | `/hooks` + a deliberate violation |
| 4 | Integration admission | MCP | `.mcp.json`, `~/.claude.json`, `PostToolUse` audit hook | `claude mcp list`, `claude mcp get` |
| 5 | Task framing | CLI flags, permission modes | `--model`, `--effort`, `--permission-mode plan`, `effortLevel` | `/model`, `/usage` |
| 6 | Bounded exploration | plan mode, research subagent | `.claude/agents/codebase-researcher.md` | plan returned; no writes |
| 7 | Execution loop | `PostToolUse` hooks | `.claude/hooks/format-on-write.sh`, filter hooks | `/context` after the run |
| 8 | Scale-out | agent teams, `/batch` | `~/.claude/settings.json` (`env`, `teammateMode`), `~/.claude/tasks/{team}/` | agent panel; `TaskCompleted` hook |
| 9 | Session health | `/clear`, `/compact`, `/debug`, `/hooks`, `/mcp` | `## Compact instructions` in CLAUDE.md | `/context`, `/usage` |
| 10 | Verification gate | `Stop` hook, reviewer subagent | `.claude/hooks/verify.sh`, `.claude/agents/code-reviewer.md` | non-zero exit blocks; criteria walk output |
| 11 | Irreversible handoff | explicit-only skills, `PermissionRequest` | `.claude/skills/deploy-prod/SKILL.md`, `.claude/hooks/ci-permissions.sh`, `--strict-mcp-config` | denial log; audit log |
| 12 | Maintenance | subtraction across layers 1–4 | diffs to `CLAUDE.md`, rules, skills, `.mcp.json` | `/memory`, line counts |
