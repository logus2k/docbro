# 13. Multi-Language Runtime

## 13.1 Concept primer

noted supports multiple programming languages via a **strategy pattern** applied consistently across three concerns: the LSP server (language intelligence: completion, diagnostics, goto-def), the DAP adapter (debugging: breakpoints, stepping, variable inspection), and the package manager (install / remove / list libraries). A new language is added by authoring three strategy classes plus a `runtime.json` manifest; the rest of the backend routes to the right strategy by looking up the runtime's `language_id`.

Four design properties are worth naming:

1. **Per-language kernel commands.** Each runtime has a `kernel_cmd` in its `runtime.json`. The backend's `KernelManagerService` (Chapter 7.5) templates that command with the project path and starts the kernel. Python uses ipykernel; R uses ark; JavaScript uses ijskernel. The kernel protocol (ZMQ + Jupyter messaging) is the same across all three.
2. **Per-language LSP servers.** CodeMirror in every cell connects to a WebSocket LSP proxy that forwards JSON-RPC to the language's LSP server. Python uses ruff (lint) + jedi (completion). JavaScript uses biome + typescript-language-server. R uses the `languageserver` R package.
3. **Per-language DAP adapters.** Python via debugpy over the Jupyter control channel. JavaScript via vscode-js-debug over TCP. R DAP is deferred (Section 13.4).
4. **Per-language package managers.** Python = pip or uv; JavaScript = pnpm; R = renv. Each is implemented as a subclass of `BasePackageManager` with a common `install_stream()` interface so the UI renders identical progress output regardless of language.

The runtime metadata is file-based. `data/runtimes/{language}/{version}/runtime.json` declares a language version; `data/environments/{language}/{version}/{env_name}` holds per-env state (installed packages, launcher scripts). This keeps the runtime registry inspectable and editable without a database.

## 13.2 Language strategies

`backend/app/managers/language_strategies.py` (530 lines) defines the base class and three concrete implementations.

`BaseLanguageStrategy` (the abstract class) has four responsibilities:

- Expose the list of LSP server configs: `[{server_type, command, args}, ...]`.
- `setup_debug(session)` - start a DAP transport against the running kernel.
- `wrap_code(code, cell_id)` - prepare code for execution (e.g. inject filename metadata for debug lookups).
- `enrich_diagnostic(diag)` - transform language-server-raw diagnostics into noted's display shape.

The registry at the bottom of the file (line 516) is a dict mapping `kernel_language` to the strategy instance: `"python" -> PythonStrategy()`, `"javascript" -> JavaScriptStrategy()`, `"r" -> RStrategy()`.

### 13.2.1 PythonStrategy (lines 84-308)

- LSP servers: `ruff server --preview` (linting, formatting) and `jedi-language-server` (completion, hover, goto). Both run in the project's venv.
- DAP: creates a `BlockingKernelClient` on the Jupyter control channel, wraps it in `ZMQDebugTransport`. debugpy runs inside the kernel; the control channel tunnels DAP messages.
- `wrap_code()` injects filename/line metadata on every cell so debugpy's `setBreakpoints` can map breakpoints to the right source.

### 13.2.2 JavaScriptStrategy (lines 310-476)

- LSP servers: `biome lsp-proxy` (linting, formatting) and `typescript-language-server --stdio` (completion, hover, refactor).
- DAP: launches vscode-js-debug, which opens a TCP port on the V8 Inspector protocol. `TCPDebugTransport` relays DAP JSON-RPC over that port.
- `build_debug_all_script()` processes cell markers (`// %%`) into boundary markers so a "Run All" with breakpoints can map each breakpoint to the right cell.

### 13.2.3 RStrategy (lines 478-512)

- LSP server: `R --slave --no-save -e "languageserver::run()"`. Single server providing completion, hover, formatting. Diagnostics from `lintr` are enriched into the `message + label` shape (lines 78-88) so the UI can render rule codes cleanly.
- DAP: stubbed. `setup_debug()` raises `NotImplementedError`. Reason discussed in Section 13.4.
- `wrap_code()` only adds line padding; no debugpy-like instrumentation is available.

## 13.3 Python runtime

Python 3.12 is the primary runtime. The `runtime.json` (at `data/runtimes/python/3.12/runtime.json`) declares:

- `executable`: `python3.12`
- `env_create_cmd`: `{executable} -m venv {env_path}`
- `kernel_cmd`: `{env_path}/bin/python -m ipykernel_launcher -f {connection_file}`
- `kernel_language`: `python`
- `env_post_create_cmds`: install `ipykernel`, `mlflow`, `hydra-core` after venv creation.

**Package manager** is `pip` or `uv` (`pip_manager.py`, 151 lines). The default is `uv pip install --python {env_path}/bin/python` for speed; legacy `pip` is kept as a fallback. `install_stream()` runs the install in a PTY, parses the output line-by-line, and forwards progress events over the WebSocket.

**LSP routing** uses jedi for navigation (completion, hover, goto-definition) and ruff for diagnostics. The LSP proxy rewrites virtual URIs (cell references) to real file URIs before handing them to jedi, then rewrites them back in the response (`python_strategy.py:98-99`).

**DAP routing**: debugpy listens inside the kernel. The `dap_manager.py` layer opens a TCP socket to debugpy via the Jupyter control channel; breakpoints, step events, and variable inspections flow through that tunnel.

## 13.4 R runtime

R is Phase 2-complete: kernel and LSP work; DAP is deferred. Six versions are supported: **3.6.3, 4.0.5, 4.2.3, 4.3.3, 4.4.2, 4.5.1**. Each has its own `runtime.json` at `data/runtimes/r/{version}/runtime.json`.

### 13.4.1 The ark vs IRkernel split

R notebooks in noted use **ark** (`/usr/local/bin/ark`) as the kernel, not the more commonly known IRkernel. The reasons:

- ark is Positron's native R notebook kernel - a Rust binary that wraps R via `system()` calls. It is faster, has better session management, and integrates with Positron's graphics display pipeline.
- IRkernel was considered and rejected because its package discovery and startup profile was slower, and because ark's output channel semantics line up more cleanly with noted's IOPub dispatcher.

LSP is a separate process: the `languageserver` R package runs via `R --slave --no-save -e "languageserver::run()"`. ark is used only for execution; LSP is independent.

R versions 3.6.3 and 4.0.5 do *not* have `languageserver` available on CRAN. `lsp_manager.py:169-178` checks the binary at startup and falls back to "kernel-only mode" for those versions - notebooks still execute but without LSP features.

### 13.4.2 Why R DAP is deferred

T-5.R6 (R Debug) was planned but deferred. Three blocking issues:

1. **ark does not expose DAP outside Positron.** Positron's DAP integration lives in Positron's Rust-side IPC layer, not in ark itself. Extracting it would require either contributing upstream or re-implementing DAP on top of ark's existing protocol.
2. **vscDebugger reverse-protocol.** The only standalone R DAP implementation is vscDebugger, which uses a reverse `startDebugging` pattern requiring child session spawning for the R evaluator subprocess. Wiring this through noted's proxy is non-trivial.
3. **Protocol translation.** Even if the above were solved, ark's internal debug messages and vscDebugger's DAP output use different schemas. Translating between them requires writing a compatibility layer that has to stay in sync with upstream changes.

The trade-off: users debug R by inserting `print()` or `browser()` statements. Full DAP support is queued pending either a Positron ark contribution or a vscDebugger release that ships a standalone DAP server.

### 13.4.3 R environment setup

Each R env has an auto-generated `bin/Rscript` launcher (`env_manager.py` post-create step) that injects:

- `R_HOME`, `LD_LIBRARY_PATH` - point at the correct R installation.
- `RENV_PATHS_*` - point at the env's renv library.
- `RENV_CONFIG_SYNCHRONIZED_CHECK=FALSE` - disables renv's startup check that would otherwise slow down every kernel launch.

`renv_manager.py` (170 lines) implements the R package manager: `list_packages()` reads `renv/library` DESCRIPTION files; `install_packages()` spawns `Rscript -e "renv::install('pkg')"` with line-buffered output; `remove_packages()` calls `renv::remove()` followed by `renv::snapshot()` to pin the change.

## 13.5 JavaScript runtime

JavaScript uses **ijskernel** (an npm-installed kernel) as the execution engine.

Runtime manifest (`runtime.json`):
- `kernel_cmd`: `{env_path}/node_modules/.bin/ijskernel --protocol=5.1 {connection_file}`
- `env_create_cmd`: `{executable} -m init` (pnpm init for the project env).
- `package_manager`: pnpm.

**LSP**: biome for linting and formatting; typescript-language-server for completion and navigation. Both are installed into the project's pnpm workspace so their versions track the user's own typescript/biome versions.

**DAP**: vscode-js-debug listens on a TCP port once the kernel is put into debug mode. `TCPDebugTransport` in `javascript_strategy.py:363-393` relays DAP messages over that port. `build_debug_all_script()` rewrites cell markers into file-line mappings so breakpoints set in one cell hit correctly during a Run All execution.

**Package manager** is pnpm (`pnpm_manager.py`, 146 lines). The normalization step (lines 41-49) converts pnpm's `{dependencies: {name: {version}}}` dict layout into the flat `[{name, version}]` shape that noted's UI expects.

## 13.6 Package manager strategy

`backend/app/managers/package_managers/` holds the per-language implementations plus a base class.

`base.py` defines:

- `PmContext` - a dataclass carrying the runtime spec, env path, a template resolver, and process registration callbacks (for cancellation).
- `BasePackageManager` - abstract class with `list_packages()`, `install_packages()`, `install_stream()`, `remove_packages()`.

Concrete subclasses:

- `pip_manager.py` - Python. Uses `uv pip` by default with `pip` fallback. PTY-based output streaming.
- `pnpm_manager.py` - JavaScript. PTY-based.
- `renv_manager.py` - R. Uses `Rscript -e "renv::install(...)"` with stdout readline (no PTY; R's output is line-buffered enough to avoid TTY detection).

`EnvironmentManager.install_packages()` (`env_manager.py:599`) dispatches by the runtime's language field. A new language's package manager is registered via the same lookup.

## 13.7 Environment management

`backend/app/managers/env_manager.py` (600+ lines) is the top-level env owner.

Key subsystems:

- **RuntimeRegistry** (lines 12-75) - scans `data/runtimes/` at startup, loads every `runtime.json`, validates required fields, exposes `get_runtime(language, version)`.
- **EnvironmentManager** (lines 77-600+) - per-runtime env lifecycle. Discovers envs via recursive scan of `data/environments/{lang}/{ver}/{env}/`. Creates envs via the runtime's `env_create_cmd` + `env_post_create_cmds`. Generates per-env launcher scripts (e.g. R's `bin/Rscript`).

The flat-to-hierarchical migration (lines 88-108) handles older installs where envs lived at `data/environments/{name}` without language/version subdirs. On startup, orphaned flat envs are moved under `python/3.12/`.

**Venv repair** (lines 140-245) runs on every backend startup for Python venvs. Symlink targets and shebang lines are fixed up in case the Python binary moved (common after container rebuilds). This avoids forcing users to recreate venvs after image changes.

`backend/app/managers/venv_manager.py` (106 lines) is a thin legacy wrapper over `EnvironmentManager` that preserves the older flat-name API for any code that still uses it.

## 13.8 LSP proxy

`backend/app/routers/lsp.py` (150+ lines) is the WebSocket endpoint. One connection per client per server. The proxy:

1. Accepts the connection with a `server_type` query parameter (`jedi`, `ruff`, `biome`, `typescript`, `r`).
2. Resolves the project, env, and runtime via the ProjectRegistry.
3. Asks the language strategy for the `(command, args)` to launch the server.
4. Spawns the LSP server subprocess with that env's activated PATH.
5. Relays JSON-RPC messages between browser WebSocket and server stdio (Content-Length framing on both sides).

`LSPProxyManager` (`lsp_manager.py:131-200+`) caches server subprocesses per `(project_id, env_name, server_type)` tuple so rapid reconnections reuse the same process.

Diagnostic enrichment (`lsp.py:30-49`) calls the strategy's `enrich_diagnostic()`. For Python/ruff, the output is `"rule-code|category|message"` with `\x1f` separators, letting CodeMirror render rule codes as clickable pills.

Frontend side: `frontend/js/CellEditor.js` connects via `codemirror-languageserver`, which speaks LSP over the WebSocket and populates the gutter, the hover tooltip, the autocomplete menu, etc.

## 13.9 DAP proxy

`backend/app/routers/dap.py` is the DAP WebSocket endpoint. Two transport types, both DAP over Content-Length framing:

- **Python: ZMQ tunnel.** `ControlChannelDispatcher` (`dap.py:35-104`) multiplexes DAP requests/replies over the Jupyter control channel. A single-reader pattern routes responses by msg_id because the Jupyter control channel does not support multiple concurrent receivers.
- **JavaScript: TCP.** `DAPProxyManager` (`dap_manager.py:141-188`) opens a TCP connection to vscode-js-debug's V8 Inspector port.

Cell-to-file mapping:

- Python pre-processes the `setBreakpoints` request: it calls the `dumpCell` kernel command to create a temp file for the cell (language_strategies.py:211-223), then maps the breakpoint file URI to the temp file.
- JavaScript writes each cell's code to `/tmp/noted_js_cell_{hash}.js` at execute time, and breakpoints set in those files just work.

Debug session lifecycle:

1. `setup_debug(session)` creates the transport.
2. `handle_handshake(session)` sends `initialize`, `attach`, and `configurationDone`.
3. Relay loop shuttles messages between WebSocket and transport until disconnect.
4. Teardown calls `disconnect` on the adapter and closes the transport.

## 13.10 Discussion-ready talking points

**Q: Why a strategy pattern instead of per-language branches in the router?**
A: Because each concern (LSP, DAP, PM) has its own dispatch shape and adding a language means updating N routers. The strategy pattern centralizes per-language knowledge in one class, makes language additions local (write a new strategy, register it), and keeps routers ignorant of which language they are serving.

**Q: Why is ark preferred over IRkernel for R?**
A: Because ark is purpose-built for notebook-style execution (originally by Posit for Positron) and has better graphics handling, faster startup, and cleaner output channel semantics. IRkernel is fine but older; the trade-off was worth the dependency on a Rust binary that needs to be in the image.

**Q: What is the cost of not having R DAP?**
A: Moderate. Users debug R by `browser()` statements or `print()` inspection. For small notebooks this is adequate; for complex R codebases it is a visible gap. The deferral is pragmatic - the integration work does not fit in the Tutorial 3 timeline, and R users in the immediate cohort have not flagged it as blocking.

**Q: Why separate runtime.json files per version instead of a single multi-version file?**
A: Because each version may have different kernel commands, different post-create steps, different LSP availability. Keeping them separate means adding a new version is copy-paste + edit, no schema changes. It also lets individual versions be removed without affecting others.

**Q: Why is venv repair idempotent and automatic?**
A: Because container rebuilds are routine. Every rebuild of the main noted image moves `/usr/local/bin/python3.12` to a new inode; Python venv symlinks encode the old inode. Without automatic repair, every rebuild would force users to recreate every Python env, which is noisy. The repair path is O(number of envs) on startup and is negligible in practice.

**Q: Why use PTY for pip/pnpm streaming but not R?**
A: Because pip and pnpm detect whether stdout is a TTY and change their output shape accordingly. Without a PTY, their output is flat and hard to parse; with a PTY, they emit progress bars and richer output. R's renv uses line-buffered stdout and does not care about TTY, so a simpler readline loop suffices.

**Q: How does the LSP proxy handle multiple clients on the same file?**
A: Each WebSocket connection gets its own server subprocess. The LSP protocol is inherently per-client (the server maintains per-document state for each client), so sharing across connections would require conflating their document states. The subprocess cost is acceptable for a small-team, single-user deployment; at scale, a shared server with client-ID-prefixed documents would be the next step.

**Q: What about adding Rust, Go, Julia support?**
A: Each would need (1) a kernel (there are community Jupyter kernels for all three), (2) LSP integration (rust-analyzer, gopls, LanguageServer.jl), (3) DAP integration (lldb-dap for Rust, delve for Go, JuliaInterpreter for Julia), (4) a package manager adapter (cargo, go mod, Pkg.jl). The plumbing is there; each language adds a few hundred lines of strategy code plus a runtime.json.
