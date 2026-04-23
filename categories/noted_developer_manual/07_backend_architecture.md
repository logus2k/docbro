# 7. Backend Architecture
## 7.1 Concept primer

noted's backend is a **FastAPI + socket.io + jupyter_client** stack running a single Python process. FastAPI serves REST endpoints over standard HTTP; socket.io handles bidirectional events over WebSocket; jupyter_client manages Python (and R) kernels via ZMQ. The three layers are composed inside a single ASGI app (`socketio.ASGIApp` wrapping the FastAPI app) so the whole service runs from one `uvicorn` command.

The architectural style is **manager-oriented**. Routing is thin - each router file has 50-200 lines and mostly delegates to a `manager` class under `backend/app/managers/`. The managers own state: the kernel sessions, the notebook files, the Hydra cache, the MLflow tracking URI, the DVC file catalog, the git working-tree knowledge. Routes are stateless functions that call into managers.

This separation is what makes the codebase readable at scale. There are 45 manager files, each ~100-500 lines, each owning one responsibility. No central ServiceLocator, no dependency injection framework - managers are singletons held on the FastAPI app object and reached via module-level helpers.

## 7.2 FastAPI app setup

`backend/app/main.py` is the entry module. It imports every router, instantiates the socket.io server, wires startup/shutdown hooks, and mounts static files.

Lines 30-34 create the socket.io server:

```python
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    max_http_buffer_size=100 * 1024 * 1024,  # 100 MB
)
```

The 100 MB buffer is load-bearing: cell outputs can carry large figures (matplotlib PNGs embedded as base64), chart renderings, or streamed log output. A smaller buffer silently drops messages.

Line 128 instantiates the FastAPI app with a lifespan context manager (lines 86-125) that:

- Starts the `KernelManagerService` background tasks.
- Pre-warms MLflow's tracking URI resolver.
- Initializes the MCP session manager (for the AI assistant's tools).
- Creates the `Examples/Welcome.ipynb` notebook if none exists.

Line 1257 wraps the FastAPI app as ASGI:

```python
app = socketio.ASGIApp(sio, other_asgi_app=app)
```

This lets `uvicorn app.main:app` serve both HTTP (FastAPI) and WebSocket (socket.io) from a single process.

**Middleware.** Deliberately thin. Socket.io handles CORS internally. There is no JWT/OAuth middleware; secrets are validated at event or endpoint level (Section 7.9). Line 1202 mounts the frontend at `/static` and `/wallpapers`.

## 7.3 Routers

`backend/app/routers/` contains 22 files, each an `APIRouter` mounted under `/api` in main.py lines 129-154. One-line summaries:

| Router | Responsibility |
|---|---|
| `notebooks` | Notebook CRUD, project listing, cell-level operations |
| `venvs` | Python virtualenv management per project |
| `documents` | Knowledge Base documents catalog and file serving |
| `git` | Git status, branches, commits, diffs for project trees |
| `files` | Generic filesystem browse/upload (500 MB limit) |
| `dvc` | DVC status, file history, checkout |
| `minio` | MinIO bucket/object ops |
| `projects` | Project metadata and registration |
| `mlflow` | MLflow experiments/runs/registered models/Logged Models |
| `export` | Notebook export formats (HTML, PDF, .py) |
| `hydra` | Config schema, compose, experiments, runs, load-bundle |
| `airflow` | DAG listing, run triggers, status, logs |
| `snapshots` | Per-notebook local snapshot versioning |
| `registry` | Project registry + noted-wide metadata |
| `serving` | Model serving lifecycle (load, unload, health, predict) |
| `reports` | Generated report documents |
| `graph_proxy` | Knowledge Graph service proxy |
| `llm` | AI assistant API (uses `NOTED_TERMINAL_SECRET` auth) |
| `lsp` | LSP proxy (completion, diagnostics, goto-def) |
| `dap` | DAP proxy (debug adapter protocol) |
| `evidently` | Evidently health endpoints (projects, data-health, drift-status) |
| `file_debug` | Single-file debugger |

Every router follows the same shape: a `router = APIRouter(prefix="/api/X", tags=["X"])`, a handful of endpoint functions that parse inputs, call a manager method, and return a Pydantic model or a dict. Thin by design.

## 7.4 Socket.io server

The socket.io event surface is the bidirectional contract between frontend and backend. main.py is the handler file; it registers one `@sio.on` handler per event, each delegating to managers.

### 7.4.1 Events the backend consumes

- **Connection:** `connect` (line 187), `disconnect` (line 192, schedules 15-second cleanup to tolerate brief network drops).
- **Notebook lifecycle:** `notebook:open` (line 224, joins a room named after the notebook, loads state, returns kernel status), `notebook:close` (line 675), `notebook:save` (line 692), `notebook:relint` (line 717).
- **Collaborative editing:** `cell:lock`/`cell:unlock` (lines 745/758), `cell:update`/`cell:add`/`cell:delete`/`cell:move` (lines 767-800). Notebook mutations broadcast to the room.
- **Execution:** `cell:execute` (line 809), `run:execute` (line 831).
- **Kernel control:** `kernel:start`/`kernel:stop`/`kernel:restart`/`kernel:interrupt` (lines 928-1086), `heartbeat` (line 1094).
- **Terminal:** `terminal:auth` (line 1106), `terminal:start` (line 1117), `terminal:input` (line 1152), `terminal:resize` (line 1170), `terminal:kill` (line 1190).

### 7.4.2 Events the backend emits

- **State:** `notebook:state`, `notebook:saved`, `kernel:status`.
- **Cells:** `cell:updated`, `cell:added`, `cell:deleted`, `cell:moved`, `cell:lock_changed`, `cell:execute_start`, `cell:execute_complete`, `cell:output`, `cell:diagnostics`.
- **Runs:** `run:started`, `run:complete`, `metrics:update` (from the MLflow monkey-patch).
- **Pipeline:** `pipeline:status`, `pipeline:task_status` (Airflow DAG progress).
- **Terminal:** `terminal:output`, `terminal:exit`, `terminal:auth_ok`, `terminal:auth_failed`.
- **Errors:** `error` with codes like `NO_KERNEL`, `NOT_FOUND`, `LOCKED`.

### 7.4.3 Routing IOPub messages to cell handlers

The interesting flow is the cell output dispatch. Here is the chain:

1. A notebook cell is executed via `cell:execute`. The backend calls `ExecutionBridge.execute_cell()`.
2. That method sends an `execute_request` ZMQ message to the kernel via `kc.execute(code)` and records the `msg_id` in `_pending[session_id][msg_id] = handler`.
3. Meanwhile, an `_iopub_loop()` task (execution_bridge.py:485) polls `kc.get_iopub_msg()` continuously.
4. Each IOPub message has a `parent_header.msg_id` - the id of the request that caused it. `_dispatch_iopub_msg()` (line 525) looks up the handler by that id.
5. The handler processes the message by type and emits a `cell:output` event to the notebook's socket.io room.

This is what makes the frontend's live output rendering work. Kernels push on their own schedule; the backend multiplexes the push stream into per-cell event streams that the frontend subscribes to.

## 7.5 KernelManager

`backend/app/managers/kernel_manager.py` is the kernel lifecycle owner. Two classes:

- `KernelSession` (lines 13-42): a dataclass holding `session_id`, `kernel_manager` (from `jupyter_client`), `kernel_cmd`, `language`, `project_id`, `notebook_path`, `client_sid`, `status`, debug state, and a `_cached_client`.
- `KernelManagerService` (lines 45-547): manages all sessions.

### 7.5.1 Starting a kernel

`start_kernel()` (line 67) spawns a new kernel process:

1. Resolve the project path via ProjectRegistry.
2. Build the environment (PYTHONPATH, GPU libs, seed env vars).
3. Pick the kernel command based on the notebook's language metadata.
4. Call `JupyterKernelManager.start_kernel(cwd=project_root, env=env)`.
5. Create and cache a client (line 170): `kc = km.client()`.

**Why cache the client?** This is the memory-documented ZMQ identity gotcha. `km.client()` creates a new ZMQ connection with a fresh identity each time. Calling it twice in the same session produces two clients that both register with the kernel but only one can own the shell channel at a time. The race is silent - the second client's messages are delivered intermittently depending on which client the kernel's round-robin picked. The fix: eagerly create one client per session on kernel start, cache it on the session object, and always return the cached instance.

`get_kernel_client()` (line 480) implements the pattern: fast-path returns `session._cached_client` if its channels are running; slow-path creates a new client under an async lock to prevent concurrent creation during recovery.

### 7.5.2 Stopping, restarting, heartbeat

- `stop_kernel()` (line 405): kill the process, clean up channels.
- `restart_kernel()` (line 428): reuse the session_id but restart the process; the `_cached_client` is refreshed.
- `heartbeat()` (line 511): update `last_heartbeat` on a session. Used by an idle-timeout reaper that stops kernels after N minutes of inactivity.
- `init_debugpy()` (line 209): enables debugpy on demand and captures the listen port for DAP proxying.

## 7.6 ExecutionBridge

`backend/app/managers/execution_bridge.py` bridges socket.io events with the kernel's ZMQ channels. Its public surface is small; most of the file is dispatch logic.

Public methods:

- `execute_cell(session_id, cell_id, code, ...)` (line 88) - single-cell execution. Injects the Hydra prelude if the notebook has `hydra_config` (Chapter 1.3.2), wraps JavaScript cells in IIFE for re-runnability, returns when the kernel's `execute_reply` arrives.
- `execute_run(session_id, project_id, cells, ...)` (line 277) - Run Manager path. Silently injects `get_run_start_code()` before cells, then executes each in sequence, then injects `get_run_end_code()`. All cells share one MLflow run.
- `stop_iopub_listener(session_id)` (line 680) - stops the async task for a session; called on notebook close.

The `_iopub_loop` polls the ZMQ channel in an executor thread (because `get_iopub_msg` is blocking), wraps each message into an async callback, and delivers it to `_dispatch_iopub_msg`. That dispatcher is the single place that interprets IOPub message types (`stream`, `display_data`, `error`, `execute_input`, `execute_reply`) and translates them into `cell:output`, `cell:execute_complete`, or `metrics:update` socket.io emissions.

## 7.7 NotebookManager

`backend/app/managers/notebook_manager.py:11` is `NotebookManager`. It is the filesystem-facing interface for `.ipynb` files.

Key methods:

- `get_notebook(project_id, notebook_name)` - loads from disk with path-traversal validation at line 20.
- `update_notebook(...)` - persists a dict to disk via `json.dump`.
- `create_notebook(...)` / `delete_notebook(...)` - file ops.
- `create_project(...)` / `list_projects(...)` - directory ops.
- `ensure_welcome_notebook()` (line 62) - bootstraps `Examples/Welcome.ipynb` at startup.
- `_notebook_path(project_id, notebook_name)` (line 17) - resolves to an absolute path via ProjectRegistry, rejects traversal.

Cells are not a separate entity in the backend - they are Python dicts inside the notebook's JSON. Mutation methods (`add_cell`, `update_cell`, `delete_cell`) modify the in-memory dict and persist the full notebook. Cell-level versioning is not supported; the unit of versioning is the notebook itself, via `snapshots` router + `snapshot_manager.py`.

## 7.8 ProjectRegistry

`backend/app/managers/project_registry.py` is the project discovery layer. Two sources:

1. **Internal projects** - subdirectories of `data/projects/`.
2. **Mounted projects** - external paths referenced by the user via `data/NOTED.md` frontmatter.

`data/NOTED.md` uses YAML frontmatter:

```yaml
---
mounts:
  - name: "jena_weather"
    host_path: "/mnt/data/jena_weather"
  - name: "jena_client"
    host_path: "/mnt/data/jena_client"
---
```

Line 79 parses this frontmatter on each backend startup. Each mount becomes a project with `source=mount`, `path=<resolved-mount-path>`, `host_path=<absolute-host-path>`.

Key methods:

- `resolve(project_id)` (line 90) - project_id -> filesystem path. Strips the legacy `__mount__:` prefix on line 146 so older metadata keeps working.
- `list_projects()` (line 121) - all projects with metadata.
- `is_internal()` / `is_mount()` (lines 128 / 134) - type checks used by routers.

The mount resolution is performed at app startup by `docker-compose.mounts.yml` (auto-generated from `NOTED.md`) which Docker applies as additional bind mounts. The frontend's project list and the container's mount list agree because both derive from the same YAML source.

## 7.9 Auth and secrets

Auth is thin and event-scoped:

- **`NOTED_TERMINAL_SECRET`** (env var, lines 1106-1130 of main.py). If set, socket.io `terminal:auth` events require a matching value. If not set, terminal auth succeeds silently (dev mode). The same secret is used by `routers/llm.py` for its endpoints.
- **`ANTHROPIC_API_KEY`** (env var, loaded at `anthropic_llm_manager.py:1`). If set, Claude models are selectable in the AI assistant. If not set, only the local Gemma model works.

There is no JWT, OAuth, or session cookie layer. The default deployment assumes a single-user, local-network context. Multi-user / multi-tenant deployments require fronting the app with a reverse proxy that handles auth (e.g. oauth2-proxy + nginx).

## 7.10 Testing

`tests/` exists and is organized as:

- `tests/api/` - FastAPI endpoint tests numbered `test_01_setup.py` through `test_31_nice_to_have.py`.
- `tests/e2e/` - Playwright browser automation tests.
- `tests/kernel_tests/` - kernel-specific tests.
- `tests/conftest.py` (489 lines) - shared pytest fixtures: FastAPI test client, temporary project/notebook fixtures, kernel session management, socket.io connection helpers.

`tests/pytest.ini` configures `testpaths = api e2e`, markers `api / e2e / socketio / slow`, 120-second timeout, `asyncio_mode = auto`.

A `Dockerfile` + `docker-compose.test.yml` exist for CI-style containerized runs. The test pyramid is backend-heavy; frontend E2E coverage is minimal pending post-demo investment.

## 7.11 Discussion-ready talking points

**Q: Why a single ASGI app instead of separate FastAPI and socket.io processes?**
A: Because both layers need the same in-process state: the KernelManagerService, the cell output dispatch tables, the Hydra cache, the manager singletons. Splitting them would require IPC and a shared state backend (Redis?) that neither layer currently uses. The single-process design is fine for noted's scale (one user, one host); scaling out would first require extracting the kernel layer to a dedicated service.

**Q: Why are managers singletons on the app object instead of DI'd into routers?**
A: Because the dependency graph is shallow and stable. Every manager is initialized once at startup; every router has access to every manager via `from app.managers.xxx_manager import xxx_mgr`. DI would add a dependency-scope concept (request / session / app) that noted does not need. The cost is that tests have to mock at the module level rather than override a provider; the benefit is that the code is simpler to read.

**Q: Why does the backend poll the IOPub channel in a thread instead of using asyncio natively?**
A: Because `jupyter_client`'s IOPub receive is blocking and does not expose an awaitable. Running it in `loop.run_in_executor(...)` keeps the main event loop responsive. The alternative (a fully async jupyter_client) does not exist as a maintained library. This is the kind of integration trade-off that justifies keeping the ExecutionBridge layer thin and focused on the channel bridge.

**Q: What is the unit of failure isolation?**
A: The kernel session. Each notebook has its own kernel process; a crash in one does not affect others. ExecutionBridge and KernelManagerService hold per-session state in dicts keyed by session_id. The backend process itself is a shared dependency - a crash there takes down all sessions. Restart resilience is owned by the compose `restart: unless-stopped` policy, not by the backend itself.

**Q: Why is the project registry derived from `NOTED.md` rather than the backend's database?**
A: Because there is no backend database. noted is a filesystem-first tool - the working directory is the source of truth for projects, notebooks, configs, and data. `NOTED.md` sits alongside other project files and is human-editable. A database would move the source of truth out of the filesystem and make `git diff` less informative. The trade-off: scaling to hundreds of projects with complex metadata would eventually want a database; at noted's current scale, plain YAML is the right shape.

**Q: How does the backend tolerate a network drop that kills a socket.io connection?**
A: `disconnect` schedules a 15-second grace cleanup (main.py:192). If the same client reconnects within that window, the cleanup is cancelled and the session resumes. If not, locks are released, rooms are cleaned, and the client is considered gone. Kernel sessions continue to run - they are not tied to socket.io lifetime - so a reconnecting user rejoins their running kernel with all state intact.

**Q: What about scaling beyond a single machine?**
A: Not supported as-is. The kernel processes are local, the socket.io rooms are in-process, the ProjectRegistry reads from a single filesystem. Horizontal scaling would require extracting kernels to a dedicated service (the long-term Serving refactor Phase 0b is a first step in that direction), routing socket.io through Redis, and treating projects as a networked resource. None of this is on the Apr 21 demo path; it is post-demo infrastructure work.
