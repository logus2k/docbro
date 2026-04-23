# 11. AI Assistant + MCP
## 11.1 Concept primer

noted's AI assistant is a chat-style interface wired to two backends (local Gemma 4 via an OpenAI-compatible agent server, and Claude Sonnet/Opus/Haiku 4.x via the Anthropic API) with shared context plumbing, a tool-calling loop, a skills registry, and a write-confirmation pattern. The assistant is not a general-purpose chatbot - it is a developer copilot anchored to the state of the user's notebook, files, MLflow runs, and Hydra configuration.

Five design choices deserve surfacing:

1. **Both backends, always.** The same conversation, skills, tools, and context system work against local Gemma or Claude. No "Claude-only" paths. This is a load-bearing rule (memory documents it explicitly) because it keeps the assistant usable offline and keeps the cost curve bounded.
2. **Dynamic context injection.** The backend does not blindly send the whole notebook or every project file. A context router classifies the user's question against domain keywords (mlflow, airflow, hydra, files, ...) and injects only the relevant context blocks and skills. For Claude this saves ~2000 tokens per turn.
3. **Skills as inline micro-docs.** Each skill is a small markdown file with YAML frontmatter. The skill registry loads them at startup, and the context router auto-injects the ones whose triggers match the current user session. The LLM sees skills as part of its system prompt, not as retrieved documents - they are *always* considered when they match.
4. **Write confirmations.** Any tool that modifies state (cell edits, file writes) is gated by an explicit confirmation step. The LLM emits a "pending_action" event; the user sees a diff and clicks Approve or Reject. No silent mutation.
5. **MCP as the tool surface.** The assistant's ~24 tools are defined in MCP (Model Context Protocol) JSON Schema format. This makes them portable to external MCP clients and consistent whether the assistant calls them directly or a subagent does.

## 11.2 Frontend: ChatPanel + ChatService

`frontend/js/ChatPanel.js` is the chat UI. It lives in the right panel, undockable. Responsibilities:

- Render the conversation as message bubbles (user vs assistant) with markdown, hljs syntax highlighting, KaTeX, and copy buttons.
- Show a model selector dropdown populated by `/api/llm/health`.
- Offer a "think" checkbox (enables extended-thinking mode for Claude) and a debug checkbox (opens the debug log panel).
- Render streaming tokens as they arrive, with a typing indicator.
- Show tool-call badges inline as the LLM invokes them.
- Show skill badges when auto-injected skills are surfaced.
- Render the user's token-usage meter (input/output/budget %).

Key methods:

- `startStreamingMessage()` - creates the container for the next assistant response.
- `appendToken(token)` - appends a token and re-renders markdown on each update.
- `finalizeStreamingMessage(thinking)` - applies syntax highlighting and inserts the collapsible "reasoning" section if Claude's thinking was included.
- `appendToolBadge(toolInfo)` - shows tool name + args.
- `updateTokenUsage(usage)` - updates the footer meter.

`frontend/js/ChatService.js` is the streaming client:

- On init, checks `/api/llm/health`, loads chat history, wires STT/TTS services.
- On send, POSTs to `/api/llm/chat` as SSE. Each `data:` line is parsed as JSON; tokens are appended; tool calls, pending actions, and skill badges are dispatched to ChatPanel.
- Parses `<think>`, `<voice>`, `<tool_call>` tags from the local LLM's output via a `ThinkingParser` (the local model is prompted to emit these explicitly; Claude uses native tool-use content blocks).
- Sends `pending_action` confirmations back to `/api/llm/confirm` when the user approves.

## 11.3 Backend: `/api/llm/*` router

`backend/app/routers/llm.py` exposes the HTTP surface:

- `POST /api/llm/chat` (line 202) - streaming SSE endpoint. Assembles context, runs the tool loop, streams tokens.
- `POST /api/llm/confirm` (line 689) - approves or rejects a pending write tool; resumes the stream with the result.
- `GET /api/llm/health` (line 864) - returns the list of available models plus the active one.
- `POST /api/llm/model` (line 854) - switch active model. If the target starts with `claude-`, `NOTED_TERMINAL_SECRET` must be provided (line 857) - the gate for API-cost models.
- `GET/DELETE /api/llm/history/{client_id}/{project_id}` - per-project chat history.
- `POST /api/llm/complete` - single-turn code completion (used by the autocomplete integration).
- `GET/POST /api/llm/skills` - list / retrieve skill metadata and content.
- `POST/GET/DELETE /api/llm/debug` - toggle debug logging and retrieve events.

## 11.4 The manager stack

`backend/app/managers/` contains the LLM layer split across six files.

### 11.4.1 `llm_router.py` (120 lines)

`LLMRouter` is the backend switch. `_is_anthropic(model_id)` checks for the `claude-` prefix. `_active_manager()` returns the right manager. Public methods (`chat_stream`, `chat`, `complete`, `health`) delegate to the active manager with a thin adapter.

`health()` queries both backends and returns a merged model list so the frontend dropdown shows every option that is currently reachable.

### 11.4.2 `anthropic_llm_manager.py` (250+ lines)

Implements the Anthropic Messages API. `ANTHROPIC_MODELS` (line 40) lists the three active IDs: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001`.

`chat_stream()` (line 156) POSTs to `https://api.anthropic.com/v1/messages` via `aiohttp`, streams the SSE response, and normalizes each chunk into the common event shape: `{"choices": [{"delta": {"content": "..."}}]}` for text, `{"tool_call": {...}}` for native tool-use content blocks.

Extended thinking is turned on by setting `thinking={"budget_tokens": 8000}` with `temperature=1.0` when the user checks the Think box (lines 138-142).

Message normalization (lines 98-120) merges consecutive same-role messages because the Messages API rejects repeated user or assistant turns.

### 11.4.3 `llm_manager.py` (120 lines)

Implements the local Gemma 4 path via an OpenAI-compatible agent server (`http://agent_server:7701` by default). `chat_stream()` POSTs to `/v1/chat/completions` with `stream=True` and yields chunks.

A single-quirk detail (lines 51-52): the Gemma 4 model occasionally hallucinates tool-call results; the fix is to set a stop token on `<tool_call|>` which the prompt template uses to separate tool emission from continued generation.

### 11.4.4 `llm_context.py` (200+ lines)

`build_context_message(ctx, managers)` (line 42) is the context assembly. It builds a single user-role message holding every relevant context block, returned as `(message_dict, skill_names)`.

Context blocks it may include:

- `_notebook_block` - current notebook state: cells, indices, a selection of outputs.
- `_file_block` - in-memory editor state of any open file (up to 20 k chars).
- `_run_block` - the active MLflow run's metrics/params, or a summary of the active experiment.
- `_config_block` - the resolved Hydra config.

Skill injection (lines 92-99) asks the SkillRegistry for static skills whose triggers match the current context (e.g. `notebook_cell_selected`, `mlflow_run_in_context`, `hydra_config_in_context`). These are prepended to the system prompt automatically.

### 11.4.5 `llm_tools.py` (150+ lines)

`TOOL_DESCRIPTIONS` (line 20) lists the ~24 tools the assistant can call: MLflow queries, Airflow DAG inspection, DVC hash lookups, file ops, Hydra config queries, notebook mutations, skill retrieval, subagent invocation, write ops, lint diagnostics, web fetch.

`parse_tool_call(text)` (line 114) extracts `<tool_call>{...}</tool_call>` from text (local-LLM path). For Claude, native `tool_use` content blocks are parsed by `anthropic_llm_manager.py` directly.

`is_write_tool(tool)` is the predicate that gates confirmation: tools like `update_cell`, `insert_cell`, `update_file`, `create_file`, `fix_lint_issues` are write tools and trigger the pending-action flow. Read tools execute immediately.

`execute_tool()` dispatches to the right manager - notebook_mgr, mlflow_mgr, dvc_mgr, etc. - and returns a result string that the LLM sees as the tool's response.

### 11.4.6 `llm_agents.py` (150+ lines)

`AgentRegistry` (line 44) loads `AGENT.md` files from `.noted/agents/`. Each defines a subagent: name, description, model (default Haiku 4.5 for speed), tools (restricted set), max_tokens, system_prompt.

`run_subagent(task, agent_name, managers)` (line 115) runs a subagent as a fresh conversation with no parent history, a tool loop limited to `MAX_AGENT_ROUNDS=4`, and a compact summary as the return value. Always uses Anthropic (because subagents are typically small, fast, parallel tasks).

### 11.4.7 `llm_memory.py` and `llm_debug.py`

`llm_memory.py` - per-client per-project history. `append(key, role, content)`, `get_messages_for_llm(key)`, and `get_compaction_input()` for auto-summarization of old history.

`llm_debug.py` - a ring buffer of timestamped debug events (api, tool, skill, file, llm, context). When the user toggles debug in the UI, events are emitted to the frontend's debug panel. Critical for diagnosing why a specific tool call fired or a specific skill was injected.

## 11.5 Skills system

`data/skills/` holds the skill library. Each skill is a folder containing `SKILL.md`:

```yaml
---
name: mlflow-run-interpretation
description: Explain what a run's metrics and tags mean in context.
triggers: [notebook_cell_selected, mlflow_run_in_context]
priority: 2
max_tokens: 500
---
# When a run's metrics are shown ...

...skill content as markdown...
```

`backend/app/managers/llm_skills.py` parses the YAML frontmatter (`SkillRegistry.load_skills`, line 41) and exposes `get_skill(name)` for explicit retrieval and `get_static_skills(conditions)` for auto-injection.

Skill categories currently populated (~42 total):

- **Airflow (8)**: dag-creation, dag-overview, performance, scheduling, sweep-strategy, task-debugging, task-dependencies, trigger-config.
- **DVC (5)**: best-practices, checkout, lineage, sync-debugging, tracking, versioning.
- **Evidently (3)**: data-quality, drift-detection, monitoring.
- **Hydra (5)**: composition, groups, pipeline-integration, setup, sweep-design, templates.
- **MLflow (10)**: artifacts, hyperparameter-analysis, model-registration, reporting, run-comparison, run-debugging, run-interpretation, serving, snapshots, training-curves.
- **noted core (5)**: auto-instrumentation, coding-conventions, lineage, notebook-resolution, platform-overview.
- **General (2+)**: ml-workflow-guidance, python-linting, web-fetch, noted-troubleshooting.

Skills can reference each other via `references/` subfolders. The registry exposes `get_skill_reference(skill_name, ref_path)` for following cross-references.

## 11.6 MCP tool surface

`backend/app/mcp/tools.py` (200+ lines) defines the tools in MCP JSON Schema format. The same definitions are consumed by the assistant's tool loop *and* (via MCP) by external MCP clients that connect to noted.

Read-tier tools (auto-execute, no confirmation):

- MLflow: `get_experiment_runs`, `get_run_details`, `compare_runs`.
- Airflow: `list_dags`, `get_dag_status`, `get_task_log`.
- DVC: `get_dvc_*` (status, file history).
- Files: `get_file_contents`, `list_files`, `search_files`.
- Hydra: `get_hydra_config`.
- Notebook: `get_notebook_cells`, `scroll_to_cell`.
- Knowledge: `query_knowledge_graph`, `get_skill`.
- Agents: `run_agent`.
- Web: `fetch_url`.

Write-tier tools (confirmation required):

- `update_cell`, `insert_cell`, `batch_update_cells` - notebook mutations.
- `update_file`, `create_file` - file writes.
- `get_lint_diagnostics`, `fix_lint_issues` - lint-driven edits.

## 11.7 Dynamic context router

`backend/app/mcp/context_router.py` is the budget manager. When the user sends a message, `classify_domains(message, context)` (line 118) scores the message against per-domain keyword lists (mlflow, airflow, dvc, files, hydra, notebook, linting, knowledge, skills, web).

`select_tools(message, context, all_tools)` returns the filtered tool list: tools that match the classified domains plus an always-included set (`get_file_contents`, `get_notebook_cells`, `scroll_to_cell`). This is the save-2000-tokens-per-turn optimization for Claude; the local LLM ignores it and gets all tools because its context is larger relative to the tool definitions.

## 11.8 Web fetch via Camoufox

`backend/app/managers/web_fetch_manager.py` (144 lines) wraps Camoufox (an anti-detect Firefox) as a singleton browser instance.

`_ensure_browser()` (line 45) recycles the browser after 50 requests or 1 hour to avoid memory bloat and to reset tracking state. `fetch_url(url)` (line 105) is the async wrapper - renders the page, waits for `domcontentloaded`, extracts HTML, strips to text, truncates to ~10 k chars. Falls back to plain `httpx` if Camoufox is unavailable.

The LLM reaches this via the `fetch_url` tool. When invoked, the tool result (text + URL) is injected back into the conversation so the LLM can cite it in its answer.

## 11.9 Debug assistant

Distinct from `llm_debug.py` (which is the debug log), there is a `llm_debug.py` for per-cell guided debugging. It ties a cell execution's traceback + surrounding context into a targeted prompt that tries to localize the bug and propose a fix. Because it is a write-gated flow (any proposed fix is a `update_cell` tool call), it goes through the pending-action confirmation just like manual edits.

## 11.10 Operations

### Add a new skill

1. Create `data/skills/my-new-skill/SKILL.md` with YAML frontmatter.
2. Pick triggers from the existing condition vocabulary (notebook_cell_selected, mlflow_run_in_context, hydra_config_in_context, etc.).
3. Save. The SkillRegistry reloads on next backend restart. For hot reload, `POST /api/llm/skills/reload` (if exposed - otherwise restart).
4. Verify auto-injection by triggering the matching context and checking the skill badges in the ChatPanel.

### Add a new tool

1. Define the tool in `backend/app/mcp/tools.py` with MCP JSON Schema.
2. Add an `if tool_name == "..."` branch in `llm_tools.execute_tool()` that delegates to a manager method.
3. Decide read-tier vs write-tier (`is_write_tool(tool)`).
4. If write-tier, the pending-action flow is automatic.
5. Update the context_router's domain classifier if the new tool should be context-scoped.

### Switch from local to Claude

1. Ensure `ANTHROPIC_API_KEY` env var is set for the backend container.
2. `POST /api/llm/model` with `{"model": "claude-sonnet-4-6", "secret": "..."}`.
3. The frontend dropdown reflects the new active model on the next `health` refresh.

### Investigate why a skill was not injected

1. Enable the debug panel (`POST /api/llm/debug { "enabled": true }`).
2. Send the triggering message.
3. In the debug panel, look for `skill` category events. Each line shows the skill name and the trigger conditions that were satisfied.
4. Cross-reference with `SkillRegistry`'s loaded skills (`GET /api/llm/skills`).

### Inspect a tool call in flight

1. With debug enabled, send a message that should invoke a tool.
2. Watch for `tool` category events: `tool_call_start`, `tool_result`, timings.
3. If the tool raised, the traceback is in the `tool_result` event.

## 11.11 Discussion-ready talking points

**Q: Why wire both local and Claude instead of picking one?**
A: Because the two have different strengths and different costs. Local Gemma 4 is free and offline; Claude is more capable and pay-per-token. The ability to swap mid-conversation lets users iterate quickly on local for cheap turns and escalate to Claude for difficult ones. Forcing one path would either price out small experiments or cap the ceiling on complex reasoning.

**Q: Why is the context router a separate layer?**
A: Because context shape differs between backends. Claude has a 200k context window and can accept large tool definitions, but tokens are billable. Gemma's context is tighter, but tokens are free. The router lets each backend see the right shape: Claude gets a filtered tool list (save money); Gemma gets the full tool list (use the context you have). Bolting context decisions into each manager would duplicate logic.

**Q: Why are skills a file-based registry instead of a database?**
A: Same reason the project registry is file-based (Chapter 7.8). Skills are co-located with the project's state, reviewable in git, and editable with the same tools that edit code. A database would require a separate admin UI. Scaling to thousands of skills would eventually warrant indexing; at 42 skills, plain Markdown is the right shape.

**Q: Why do write tools require confirmation?**
A: Because the assistant will sometimes propose changes that are wrong, plausibly wrong, or right-but-surprising. Unconditional auto-apply would create a class of bugs where the user does not know what changed until something breaks later. Pending-action + diff keeps the user in the loop without forcing them to hand-write every change the LLM suggests.

**Q: How does the MCP tool layer help external integrations?**
A: Because MCP is a portable spec. The same tool definitions that power noted's own assistant can be served to an external MCP client (e.g. a VS Code extension with MCP support). The plumbing is already there; exposing it is primarily an auth and transport concern. This is why the tool definitions live in `backend/app/mcp/tools.py` rather than in `llm_tools.py`.

**Q: Why is `run_agent` a tool rather than a direct backend feature?**
A: Because the assistant chooses when to delegate. Subagents are useful for tasks that are independent, parallelizable, or that benefit from a fresh context (e.g. "research how to do X" vs. "refactor this notebook"). Letting the LLM decide to delegate gives better outcomes than hardcoding the decision into the backend. Agents registered in `.noted/agents/` act as the specialization vocabulary.

**Q: How does the conversation memory handle long sessions?**
A: `llm_memory.py` has a `get_compaction_input()` method that returns old messages when history exceeds a threshold. The assistant is prompted to compact them into a summary, which replaces the old messages in storage. This is the same pattern most chatbots use; noted's version is per-client-per-project so the compaction scope is natural.

**Q: What is the failure mode when the agent_server is down?**
A: The local backend's health check fails, the model dropdown shows only Claude options (if the key is set) or nothing, and any attempted local call returns a clear error. No silent fallback; the user has to pick a working model explicitly.
