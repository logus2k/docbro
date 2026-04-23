# 8. Notebook Execution Flow
## 8.1 Concept primer

Notebook execution in noted is the path a user's `Shift+Enter` press takes from a browser keystroke to a rendered output in the cell. This module traces the path end-to-end. Prior modules covered each stopping point in isolation: Chapter 1 described the Hydra injection, Chapter 2 described the MLflow prelude, Chapter 6 described the frontend's event dispatch, Chapter 7 described the backend's routers and managers. This module is the glue - a time-ordered walk through what actually happens when a cell executes.

There are two distinct execution paths in noted:

1. **Cell execution** - single cell, via `cell:execute` socket event. Triggered by `Shift+Enter`, the Play button on the cell toolbar, or a programmatic call from the AI assistant. This path does **not** open an MLflow run and does **not** install the metrics monkey-patch by default.
2. **Run execution** - a sequence of cells wrapped in an MLflow run, via `run:execute` socket event. Triggered by the Run Manager's Run button. This path installs the full prelude (Hydra injection, MLflow run-start, metrics patch, DVC hash logging) and runs all specified cells inside a single active run context.

Both paths converge on the same ZMQ `execute_request` call into the kernel. The divergence is only in what gets injected before the user's code runs.

## 8.2 Cell execution path

### Step 1 - Frontend keydown

The user focuses a CodeMirror cell editor and presses `Shift+Enter`. The `NotebookEditor` class (`frontend/js/NotebookEditor.js`) has a keydown listener on the editor container that intercepts this combination, captures the cell's current code, and calls `app._kernelClient.executeCell(cellId, code, opts)`.

### Step 2 - Socket emit

`KernelClient.executeCell()` (`frontend/js/KernelClient.js`) emits a `cell:execute` event with payload:

```javascript
{
  notebook_id: <current notebook>,
  session_id: <current kernel session>,
  cell_id: <cell being executed>,
  code: <cell content as string>,
  hydra_config: <null or {notebook_uid, baseline_source, group_selections, overrides}>,
  debug: <false or true>
}
```

`hydra_config` is present when the notebook has a Hydra baseline; its composition identity (Chapter 1.3.3) is the *only* place the frontend needs to assert the Hydra state per cell execute - the backend re-composes from scratch on every request.

### Step 3 - Backend handler

`backend/app/main.py:809` is the handler. It:

1. Looks up the kernel session by `session_id`. If missing, emits `error: NO_KERNEL` and returns.
2. Updates the session's `last_heartbeat`.
3. Calls `ExecutionBridge.execute_cell(session_id, cell_id, code, hydra_config=..., debug=...)`.

### Step 4 - Hydra injection (if present)

Inside `ExecutionBridge.execute_cell()` (`backend/app/managers/execution_bridge.py:88`), if `hydra_config` is not None, `_build_hydra_injection()` is called. It:

1. Calls `HydraManager.compose_from_source(...)` to produce the resolved config.
2. Serializes it to JSON.
3. Constructs the Python prelude string (`cfg = OmegaConf.create({...})`, `__noted_hydra_hash__ = 'sha256:...'`).
4. Executes it silently via `_execute_silent()` (`execution_bridge.py:411`), which sends a shell request and waits for the reply without surfacing any output.

The prelude returns before the user's cell code runs. By the time the cell body begins, `cfg` is available in the kernel's global namespace.

### Step 5 - User code execution

The user's cell code is sent to the kernel via `kc.execute(code, ...)`. This returns a `msg_id`. The ExecutionBridge records `_pending[session_id][msg_id] = handler` where `handler` is an object that accumulates output, tracks the `execute_reply`, and knows the cell_id to emit events against.

The kernel now begins executing. ZMQ IOPub messages flow back: `execute_input` (the code that was actually executed), `stream` (stdout/stderr), `display_data` (rich outputs), `error` (tracebacks), and finally `execute_reply` on the shell channel.

### Step 6 - IOPub dispatch

`_iopub_loop()` (`execution_bridge.py:485`) polls `kc.get_iopub_msg()` in an executor thread. Each received message is handed to `_dispatch_iopub_msg` (line 525), which:

1. Extracts `parent_header.msg_id` from the message.
2. Looks up the handler in `_pending[session_id][msg_id]`.
3. Switches on message type:
   - `execute_input` -> extract `execution_count`, forward to handler.
   - `stream` -> emit `cell:output` event to the notebook room with `{type: 'stream', text, cell_id}`.
   - `display_data` -> check for custom mime types (`application/x-noted-metric`, `application/x-noted-run-start` - see Chapter 2.3.2 and 2.3.3), otherwise emit `cell:output` with `{type: 'display_data', data, cell_id}`.
   - `error` -> emit `cell:output` with `{type: 'error', traceback, ename, evalue, cell_id}`.

### Step 7 - Completion

When the shell channel's `execute_reply` arrives, the handler's `done` event is set. The handler's accumulator now has the final `execution_count`. The bridge emits `cell:execute_complete` to the room and removes the handler from `_pending[session_id]`.

### Step 8 - Frontend render

The frontend's NotebookEditor subscribed to `cell:output` and `cell:execute_complete`. As events stream in:

- `cell:output` events append to the cell's output area in DOM order. The renderer picks the representation based on mime type: text for `stream`, `<img>` for `image/png`, `<div>` with marked for `text/markdown`, KaTeX for LaTeX, echarts for chart JSON.
- `cell:execute_complete` sets the execution counter in the cell gutter (`In [N]`) and transitions the cell state from `running` to `idle`.

## 8.3 Run execution path

The Run Manager path is structurally similar but has a longer prelude and executes multiple cells inside a shared MLflow run context.

### Step 1 - Frontend Run click

The user opens the Run Manager panel (right sidebar), picks a set of cells (or accepts the default of all code cells), optionally overrides Hydra inputs, and clicks Run. `RunManagerPanel.js` emits `run:execute` with payload:

```javascript
{
  notebook_id, session_id,
  cells: [{cell_id, code}, ...],
  hydra_config: {...},                  // same shape as cell:execute
  experiment_name: 'Jena Weather',
  run_name: 'gru_baseline_jena_2012',
  dataset_hashes: {...}                 // ignored for Hydra-using notebooks
}
```

### Step 2 - Backend handler

`backend/app/main.py:831` dispatches to `ExecutionBridge.execute_run(session_id, project_id, cells, ...)` (`execution_bridge.py:277`).

### Step 3 - Resolve dataset hashes

Before touching the kernel, main.py (line 673-749, `on_run_execute`) resolves `dataset_hashes`:

- If `hydra_config` is present, compose the cfg, read `cfg.data.file`, look it up in `dvc_mgr.status()`, and use `{cfg.data.file: hash}` as the single-entry dataset hashes dict. The frontend's `datasets[]` is ignored.
- If no `hydra_config`, fall back to the frontend's `datasets[]` as a list of file paths, resolved via the same DVC lookup.

This is Chapter 3.3.3's logic - Composer is the single source of truth for Hydra-using notebooks.

### Step 4 - Prelude injection

`ExecutionBridge.execute_run()` calls `AutoInstrumentation.get_run_start_code(experiment_name, run_name, dataset_hashes=..., hydra_hash=...)` and silently executes the result.

The prelude is a single Python blob composed of:

- `METRICS_HOOK_CODE` - monkey-patches `mlflow.log_metric`, `mlflow.log_metrics`, and `mlflow.start_run` to emit display_data with custom mime types.
- `RUN_START_CODE` - calls `mlflow.set_experiment(experiment_name)` and `mlflow.start_run(run_name=run_name)`, stores the run handle as `run`.
- `_get_dataset_logging_code(dataset_hashes)` - calls `mlflow.log_param("dvc_data_hash", hash)` and `mlflow.set_tag("dvc.data_file", path)` for each entry.
- `_get_hydra_logging_code(hydra_hash)` - records `noted.hydra_config_hash` on the active run.

After the prelude runs, the kernel has:

- `mlflow.log_metric` replaced with the live-streaming wrapper.
- `mlflow.start_run` replaced with the wrapper that emits the run-start hook.
- A running MLflow run with experiment, name, data hash tag, config hash tag set.
- `run` as a module-level variable holding the run handle.

### Step 5 - Hydra injection (same as cell path)

The Hydra injection runs after the prelude so `cfg` is available before cell 11 (the seed cell, Chapter 1.2) runs.

### Step 6 - Cell-by-cell execution

For each cell in the list, `execute_run()` executes the cell code via the same `kc.execute(code, ...)` mechanism used by the cell path. Each cell produces its own `cell:output` events and its own `cell:execute_complete`. The user sees them arrive in sequence in the Run Manager's execution log.

If a cell raises, the loop is *not* aborted by default - the next cell runs too. This matches Jupyter's semantics and lets the user see all output even if an early cell had a warning. The Run Manager UI surfaces each cell's status (idle/running/success/error) so execution progress is clear even across failures.

### Step 7 - MLflow run-start event handler

One of the display_data messages emitted early in the run is `application/x-noted-run-start` (Chapter 2.3.3). The IOPub dispatcher picks this up, extracts the run_id, and fires `_log_hydra_bundle_for_run(run_id)` in a background thread. That thread:

1. Re-composes the Hydra config from the current `hydra_selections`.
2. Calls `HydraManager.assemble_bundle_from_source(...)` to build the `hydra/` artifact tree.
3. Writes to a tempdir, uploads via `client.log_artifacts(run_id, tmpdir, "hydra")`.
4. Tags the run with `noted.hydra_config_hash`, `noted.project_id`, `noted.git_commit`, `mlflow.source.git.branch`.

The upload is fire-and-forget - failures are logged but do not affect the running execution.

### Step 8 - Metrics streaming

As the user's code calls `mlflow.log_metric(...)` (usually from a Keras `on_epoch_end` callback, Chapter 2.3.2), each call emits `application/x-noted-metric` display_data. The IOPub dispatcher intercepts these, suppresses them from cell output, and emits `metrics:update` socket events. The frontend's live metrics chart updates in real time.

### Step 9 - Run completion

After the last cell, `execute_run()` injects `RUN_END_CODE` which calls `mlflow.end_run()`. The MLflow run is now closed - its metrics, params, and tags are frozen. The bridge emits `run:complete` with the run_id. The frontend's Run Manager shows the final state and links to the run in the MLflow view.

## 8.4 Execution contention: the collaborative editing case

noted supports multiple clients connected to the same notebook via socket.io rooms. Execution is kernel-scoped, not client-scoped - all clients in the room share one kernel. This means:

- A cell execution triggered by client A is visible to client B in real time. The `cell:execute_start` and `cell:output` events broadcast to the whole room.
- Concurrent edits to the same cell are prevented by per-cell locks (`cell:lock` / `cell:unlock`, `main.py:745`). A client acquires a lock before editing, releases it on blur. Attempted edits on a locked cell are rejected.
- Concurrent execution requests to the same cell are serialized by the kernel - the second `execute_request` queues until the first finishes. Visually, the second client sees their cell go from `idle` to `queued` to `running`.

## 8.5 Interruption and cancellation

A user can interrupt a running cell via the stop button on the cell toolbar or the `kernel:interrupt` menu action. This sends a SIGINT to the kernel process via jupyter_client's `km.interrupt_kernel()`. Python's normal signal-handling raises `KeyboardInterrupt` in the running cell code.

Important caveats:

- Interrupting a C-extension call (numpy operations, TensorFlow training) does not always work immediately because the C code may not check for Python signals. Interrupt is advisory; kernel restart is the hammer.
- Interrupt during `_execute_silent` prelude code is not supported - the prelude is expected to be short and side-effect-free enough that users never need to interrupt it.
- For long-running trainings, pressing interrupt usually takes effect at the next epoch boundary (when Keras's callbacks have a chance to check for signals). Patience; then restart if needed.

## 8.6 Error paths and the `NO_KERNEL` story

Several failure modes are handled explicitly:

- **No kernel yet.** If the user tries to execute before starting a kernel, the backend emits `error: NO_KERNEL` and the frontend shows a notification with a "Start Kernel" button.
- **Kernel crashed.** If the kernel process dies during execution, the IOPub channel closes and `_iopub_loop` catches the exception. Pending handlers are notified with an error. The session state flips to `dead`; the frontend shows the kernel-dead banner with a restart button.
- **Socket disconnected mid-execution.** The kernel continues running. The handler is still in `_pending`. When the client reconnects and rejoins the room, a state-refresh query can fetch the in-progress cell's current output. (Not fully implemented at v0.1 - reconnect after a long execute may lose the streaming log, though the final `cell:output` event is still persisted to the notebook file on save.)
- **Hydra compose failure.** If `_build_hydra_injection` fails (e.g. malformed YAML in the config tree), the injection is skipped, a warning is logged, and the cell runs without `cfg` - the user sees a `NameError` on the first `cfg.X.Y` access. The trade-off between hard-failing the cell vs. running it is biased toward letting the user see the error in their own code rather than a backend abstraction.

## 8.7 The Run All path: where it differs

"Run All" is a third path, technically. It is a frontend-only loop that issues `cell:execute` events one at a time for every code cell. The backend sees a series of single-cell executions.

Critical difference from Run execution: **Run All does not install the MLflow prelude**. Consequences:

- `mlflow.log_metric` calls in the user's code go through the unpatched function. They are logged to MLflow if a run is active, but no `metrics:update` socket events fire - the live metrics panel stays empty.
- There is no active run when cell 11 (seeding) runs. Cell 116 opens a new run inside its `with mlflow.start_run():` block; this run has no `noted.*` tags and no `hydra/` bundle archived against it because the run-start hook was never installed.

This is the known foot-gun documented as "Run All vs Run Manager" in Chapter 5.5. The pragmatic guidance is "use Run Manager for anything you want tracked"; the engineering fix is to have `cell:execute` check whether the next cell is likely to start a run, and pre-install the metrics patch.

## 8.8 Debug execution

`Ctrl+Shift+Enter` on a cell triggers a debug execution instead. The flow:

1. `cell:execute` is emitted with `debug: true`.
2. `ExecutionBridge.execute_cell()` checks the flag and calls `_inject_debug_bootstrap()` which ensures `debugpy` is listening (via `KernelManagerService.init_debugpy()`).
3. The cell code is wrapped in a pre-bounce `debugpy.wait_for_client()` if no DAP client is attached yet.
4. The frontend's DAP client connects through noted's DAP proxy (`backend/app/routers/dap.py`) to debugpy.
5. Once attached, breakpoints set in the cell are honored; the user can step via F10/F11, continue via F5, stop via Shift+F5.

Debug is per-cell; exiting a debug session does not leave the kernel in debug mode. The next non-debug execution runs normally.

## 8.9 Discussion-ready talking points

**Q: Why are cell and run execution two paths instead of one?**
A: Because the prelude machinery (MLflow start_run, metrics patch, dataset hash logging, bundle archival) is expensive to install and pollutes the kernel's namespace. Single-cell debugging or exploration does not need any of that. The split lets users do quick iteration with minimal overhead and opt into the full tracking surface only when they are running a real experiment. The cost is that Run All produces "tracked" output that is less rich than Run Manager output - which is the foot-gun Chapter 5.5 documents.

**Q: Why is the prelude executed silently instead of appearing as a cell?**
A: Because the notebook is meant to be the user's unit of authorship. A visible prelude cell - even one that says "# noted will replace this with your config" - is a surface the user would be tempted to edit, and breaking it would break the run. Silent injection keeps the notebook identical across users, across versions, and across hosts. The cost (documented as "tech-debt invisible preludes" in memory) is that the notebook is not portable outside noted without manual compose code.

**Q: Why does `_iopub_loop` run in a thread instead of using asyncio natively?**
A: Because `jupyter_client.KernelClient.get_iopub_msg()` is a blocking call with no async variant. Offloading it to `loop.run_in_executor(...)` keeps the event loop free to handle other socket events, router requests, and heartbeats. An async-native jupyter_client would be nice; none exists, and writing one is a larger yak-shave than it is worth.

**Q: What prevents two cells from interleaving their output if executed rapidly?**
A: The kernel's shell channel processes one `execute_request` at a time - the second queues. IOPub messages carry the `parent_header.msg_id` of the request that caused them, so the bridge's dispatcher routes each message to its correct cell handler regardless of interleaving order at the ZMQ layer. The frontend sees outputs in cell-specific streams.

**Q: How does the backend know when a run is "done" vs "still writing the last metric"?**
A: The shell channel's `execute_reply` for the final cell in the run is the marker. Once that fires for the injected `RUN_END_CODE`, `execute_run()` considers the run complete and emits `run:complete`. The MLflow server may still be flushing artifacts to disk at that point, but the run record itself is closed.

**Q: Can the backend replay a past execution?**
A: No. Replay is a Hydra-mediated workflow: load the archived bundle into the Composer (Section 5.1), click Run, observe the new run. There is no "re-run cell N with these exact inputs" primitive because cell N's inputs depend on its preceding cells' side effects, which are not captured in noted's state model. If true replay is ever needed, a full-notebook determinism harness would be required.

**Q: What is the `notebook:save` contract?**
A: On save, the current notebook state - cells, cell outputs, cell metadata, notebook-level metadata - is serialized to the `.ipynb` file on disk. Metadata includes `hydra_selections`, `hydra_baseline_source`, and `notebook_uid` (Chapter 1.3.3). Cell outputs are saved, so reopening a notebook shows the last execution's output without re-running. This is standard Jupyter behavior; noted adds no twist.
