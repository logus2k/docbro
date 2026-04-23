# 2. MLflow
## 2.1 Concept primer

MLflow is the *accounting system* of an MLOps stack. Where Hydra answers "what was this run configured to do?", MLflow answers "what did this run actually do, and what artifact did it produce?". The framework has four primary concepts that noted uses:

1. **Tracking server and runs.** An MLflow tracking server stores per-run records: a UUID, a start/end time, parameters (immutable once logged), metrics (time-series of scalar values), tags (mutable key/value strings), and artifacts (any file). Runs live inside experiments, which are named buckets. Every training cell execution in noted produces one run if it entered a `start_run()` context, or zero runs if it did not.
2. **Model registry.** Separate from runs, the registry is a named catalog of model versions. `mlflow.register_model(runs:/<run_id>/model, "Jena Weather Forecaster")` takes the model artifact produced by a run and exposes it as version N of a registered name. Versions are immutable once created; what moves between them is *alias pointers* (`@champion`, `@challenger`, `@staging`).
3. **Logged Models (MLflow 3.x).** A new first-class entity introduced in MLflow 3.x. A Logged Model is the model artifact stored *with* its own ID, its own tags, its own lineage - distinct from the run that produced it. In the old model, the run owned the artifact and the registry owned versions; in the new model, the Logged Model is a third entity that can be referenced independently. noted's Logged Models view in the Explorer reflects this: each Logged Model appears as a subtree under its producing run with MLmodel, conda.yaml, python_env.yaml, requirements.txt, and any supporting files visible.
4. **Signatures and flavors.** Every model artifact carries a *signature* (typed input/output schema: tensors with dtype and shape, or columnar specs) and one or more *flavors* (e.g. `tensorflow`, `pyfunc`, `sklearn`). The signature is what the serving client uses to validate a request payload before forwarding it to the model. The flavor tells `mlflow.pyfunc.load_model` how to reconstruct the model in memory.

**What MLflow does not do.** It does not compose configurations (that is Hydra). It does not orchestrate (that is Airflow). It does not version data (that is DVC). In noted, MLflow is the ledger; everything else writes into it.

**Why aliases matter.** `@champion` decouples *which version is in production* from *what its version number is*. A rollback from v7 to v6 is `client.set_registered_model_alias(name, "champion", 6)` - the alias hops, the serving client (pointing at `@champion` by default) picks up the change on its next health refresh, and there is no redeploy. Version numbers are stable historical identifiers; aliases are movable pointers. This is the single most important MLOps idiom to internalize.

## 2.2 Where MLflow lives in the notebook

The Tutorial 3 notebook imports MLflow in **cell 116** and uses it in exactly two places: the training run (cell 116) and the promotion step (cell 117). Everything else happens automatically, either in noted's Run Manager prelude or in the `register_and_promote` helper.

**Cell 94** - training. No direct MLflow calls. `train_model()` is called with `es_cfg` and `lr_cfg` from Hydra. The Keras `on_epoch_end` callback reaches `mlflow.log_metric(...)` through the monkey-patched wrapper that noted installs at prelude time. If no run is active the log call is a no-op; if one is active, the metric streams to both the tracking server and to the live metrics panel via the `application/x-noted-metric` mime type.

**Cell 116** - tracking and logging. After training, the user's code calls:

```python
mlflow.set_experiment("Jena Weather")
with mlflow.start_run() as run:
    mlflow.log_params(...)         # hyperparameters from cfg
    mlflow.log_metrics(...)         # final test metrics
    mlflow.set_tag("task", "forecasting")
    mlflow.tensorflow.log_model(model, artifact_path="model",
                                signature=infer_signature(...))
```

However: because noted's Run Manager has already opened a run via the prelude (Section 2.3.1), this `start_run()` call will either *re-enter* the active run or open a nested run depending on MLflow's behavior. In practice, cell 116 is shaped to *continue* the Run Manager's run rather than open a new one. The `if run is not None:` guard at the top of cell 117 is what keeps this defensive: if for any reason there is no active run (e.g. the user executed cell 116 outside a Run Manager context), promotion is skipped rather than crashing.

**Cell 117** - promotion.

```python
if run is not None:
    result = register_and_promote(run_id=run.info.run_id,
                                  model_name="Jena Weather Forecaster",
                                  new_mae=test_mae)
```

`register_and_promote` lives at `/home/logus/env/iscte/jena_weather/src/evaluation/promote.py:29`. It:

1. Reads the current `@champion` version's test MAE via `MlflowClient.get_model_version_by_alias(name, "champion")`.
2. Calls `mlflow.register_model(f"runs:/{run_id}/model", model_name)` to produce version N+1.
3. Compares `new_mae < champion_mae`; if better, calls `client.set_registered_model_alias(name, "champion", new_version)`.
4. Returns `{promoted: bool, improvement_pct: float, new_version: int}`.

This is the whole promotion pipeline. Every other MLflow side-effect (tags, parameters, metrics, artifacts, bundle archival) happens earlier, either in the Run Manager prelude or as monkey-patched wrappers.

## 2.3 How noted bridges to MLflow

noted's MLflow integration is a tracking layer wrapped around a user-written notebook. The user writes MLflow calls as if they were in a bare script; noted's prelude transparently installs the run context, the metric streaming, and the lineage tags.

### 2.3.1 The Run Manager prelude

`backend/app/managers/auto_instrumentation.py` holds three injected code blobs:

- `RUN_START_CODE` (lines 15-26) - opens a run with `mlflow.set_experiment(experiment_name)` and `mlflow.start_run(run_name=run_name)`, sets the `instrumentation=experiments` tag, stores the run handle in the kernel as `run`.
- `RUN_END_CODE` (lines 28-36) - closes the run with `mlflow.end_run()` after the last cell.
- `METRICS_HOOK_CODE` (lines 44-124) - monkey-patches `mlflow.log_metric`, `mlflow.log_metrics`, and `mlflow.start_run` so that each call additionally emits a noted-specific IPython display message.

`get_run_start_code()` (line 134) assembles the three blobs plus optional dataset-logging code (DVC hashes) and returns them as a single Python string. `ExecutionBridge.execute_run()` (line 277 of `execution_bridge.py`) silently executes that string before any cell runs, then executes `get_run_end_code()` after the last cell completes.

The variable `run` lives in the kernel namespace from prelude time onward; cell 117's `if run is not None:` is therefore a cross-cell check whose truth is set up by the prelude. This pattern is what the memory flags as "tech-debt invisible preludes" - it works, but it makes the notebook non-portable.

### 2.3.2 Live metrics via `application/x-noted-metric`

Inside `METRICS_HOOK_CODE` (around line 58), `mlflow.log_metric` and `mlflow.log_metrics` are replaced with wrappers that call through to the real function *and* emit an IPython display with a custom mime type:

```python
IPython.display.display({
    'application/x-noted-metric': json.dumps({
        'run_id': run_id, 'key': key, 'value': value,
        'step': step, 'timestamp': timestamp
    })
}, raw=True)
```

The backend's IOPub dispatcher (`execution_bridge.py:582-594`) watches for this mime type. On receipt it:

1. Suppresses the display from the cell output (so the notebook cell does not print a giant JSON blob).
2. Parses the JSON.
3. Emits a `metrics:update` socket.io event to the frontend.

The frontend's live metrics panel subscribes to this event and updates the chart in real time, *during training*, with no polling and no hooks the user has to write. Every `log_metric` call in any library (Keras callback, custom logger, `model.fit` instrumentation) surfaces live because the patch is on MLflow itself.

### 2.3.3 Run-start hook via `application/x-noted-run-start`

The same metrics-hook patch wraps `mlflow.start_run` (around line 89) so that, on successful run creation, it emits:

```python
IPython.display.display({
    'application/x-noted-run-start': json.dumps({
        'run_id': run_id, 'timestamp': timestamp
    })
}, raw=True)
```

`ExecutionBridge._dispatch_iopub_msg` (line 599-613) watches for this mime type and, on first occurrence per session, fires `_log_hydra_bundle_for_run(run_id)` in a background thread. This is the handoff from MLflow-aware code to Hydra-aware code: the run must exist before the bundle can be uploaded against it, so the bundle-upload side-effect is triggered by the run-start event.

### 2.3.4 Tag injection and git lineage

When `_log_hydra_bundle_for_run` runs, it also injects lineage tags on the run (lines 775-850):

- `noted.hydra_config_hash` - SHA256 of the resolved config (the primary key for "same config").
- `noted.project_id` - the noted project id (used by the Composer's Time Machine to filter runs that belong to this notebook's project).
- `noted.git_commit` - current commit SHA of the project directory, resolved via subprocess.
- `mlflow.source.git.commit` / `mlflow.source.git.branch` - the standard MLflow tags, populated the same way.

The git tags are best-effort: if the project is not a git repo, they are omitted silently. This is the one place where "silently" is acceptable because the absence of git info is diagnosable from the tag list - a missing tag is a visible null, not a masked failure.

### 2.3.5 `target_mean` / `target_std` for serving

During training, the notebook computes the target's scaler statistics (mean and standard deviation of the regression target on the train split). Two of the most load-bearing logged params:

```python
mlflow.log_param("target_mean", float(train_target.mean()))
mlflow.log_param("target_std",  float(train_target.std()))
```

These are not training hyperparameters; they are *inference-time* stats that `jena_client` reads back to inverse-transform the model's scaled predictions into human-readable Kelvin. Without them, the downstream client would return scaled numbers that nobody could interpret. Logging them on the run ties the inverse-transform to the exact training run and its champion version.

### 2.3.6 Model Registry, Deploy / Unload / Try It

The Registry panel (`frontend/js/panels/explorer/ExplorerRegistryViews.js`) renders:

- A **Models tree** fetched from `GET /api/registry/models`. Each registered model is a tree node; its children are versions, each with its alias labels rendered inline (`v7 @champion`, `v6`, `v5`, etc.).
- A **Model detail view** (line 118-232) showing the version table, signature parsed from MLmodel YAML, flavors, and alias assignment buttons.
- A **Version detail view** (line 234-358) with three action buttons: Deploy, Unload, Try It.

**Deploy** (line 466+) instantiates `ModelDeployer` and posts to `/api/serving/load` as NDJSON streaming. Phases stream in: `starting`, `downloading_artifact`, `loading_model`, `ready`, `failed`. The button flips to "Unload" on success. State polling via `/api/serving/health` keeps the button state coherent across sessions and users.

**Unload** posts to `/api/serving/unload` and releases VRAM via the model loader's explicit `del model; gc.collect()` sequence (Step 1 refactor on 2026-04-15 neutered the runtime-install path that caused stale C-extension crashes).

**Try It** opens the `ExplorerServingViews.showTryItPanel(...)` panel: a form rendered from the model's signature, with each tensor-spec field auto-populated from a sample row of the train split. Submit hits the serving endpoint and the response appears inline.

### 2.3.7 Logged Models (MLflow 3.x)

The Logged Models view (`ExplorerMlflowViews.js:595-815`) is a nested tree under each run. Backend endpoints:

- `GET /api/mlflow/runs/{run_id}/logged_models` (`mlflow.py:59`) - `MlflowManager.list_logged_models_for_run` (`mlflow_manager.py:224`) scans the experiment's `models/` directory, picks MLmodel files whose `run_id` matches, and returns a flat artifact tree for each Logged Model it finds.
- `GET /api/mlflow/logged_models/{experiment_id}/{model_id}/download?path=...` (`mlflow.py:76`) - streams a single file from a Logged Model's artifact root via MLflow's artifact proxy.

In the frontend, each file node (`MLmodel`, `conda.yaml`, `python_env.yaml`, `requirements.txt`, `model.keras`, etc.) opens a detail pane with the file contents highlighted by hljs. Binary files (`.keras`, `.npy`) render a placeholder instead of attempting to syntax-highlight.

## 2.4 Operations

### What `register_and_promote` does

The full sequence for a single promotion:

1. Read `@champion` version if one exists: `client.get_model_version_by_alias(name, "champion")`.
2. Read its `test_mae` metric via `client.get_run(version.run_id).data.metrics["test_mae"]`.
3. Compare against the new run's `test_mae`.
4. Register the new artifact: `mlflow.register_model(f"runs:/{run_id}/model", name)` - returns a `ModelVersion` with a version number N+1.
5. If `new_mae < champion_mae`: `client.set_registered_model_alias(name, "champion", new_version.version)`.
6. Return `{promoted: bool, improvement_pct: float, new_version: int}`.

Every step is idempotent by design: re-running against the same run_id produces the same new version number if called within a short window (MLflow coalesces), and alias assignment overwrites prior assignments.

### Inspect a model's signature and parameters

From the Registry panel:
1. Click a model, then a version. The Version detail pane shows the signature parsed from MLmodel YAML: each input tensor's name, dtype, and shape; each output tensor's name, dtype, and shape.
2. Click "Logged Model" in the same pane to open the Logged Model subtree. MLmodel contents render with hljs syntax highlighting.
3. The params table shows every key logged via `log_param`, including `target_mean` and `target_std` for Tutorial 3.

From MLflow's own UI (at `:5000`), the same data is available but without the noted-specific rendering.

### How `@champion` drives serving

The serving container (`client/app/model_loader.py`) resolves `@champion` on every deploy:

1. Client POSTs `/api/serving/load` with `{name, version}` or `{name, alias: "champion"}`.
2. `ModelLoader.load_by_alias(name, alias)` calls the MLflow tracking API's `/registered-models/{name}/alias/{alias}` endpoint.
3. The response gives the current version number the alias points at.
4. `ModelLoader.load_by_version(name, version)` downloads the Logged Model artifact tree and calls `mlflow.pyfunc.load_model(...)`.

Rolling back is a single alias-hop on the MLflow server - no redeploy required if the client is configured to re-resolve the alias on health-check intervals.

## 2.5 Discussion-ready talking points

**Q: Why is Run Manager-only tracking a deliberate design?**
A: Because every notebook-surfaced MLflow call is a user-visible side-effect, and the user does not want to write `start_run() / end_run()` boilerplate in every cell. The Run Manager prelude owns the run lifecycle; the notebook only has to call `log_params`, `log_metrics`, and `tensorflow.log_model`. The `if run is not None:` guard in cell 117 is the only defensive concession - it lets the same notebook survive being run outside noted (in which case promotion is a no-op).

**Q: How does the live metrics panel know about Keras callbacks?**
A: It does not know about Keras. It knows about `mlflow.log_metric`. The monkey-patch at prelude time wraps `log_metric` itself, so any source that calls it (Keras callback, a custom `on_epoch_end`, a manual log statement) emits a `application/x-noted-metric` display which the IOPub dispatcher forwards to the frontend over socket.io. The patch is library-agnostic; it works for sklearn, xgboost, or plain-Python training loops as long as they call MLflow.

**Q: MLflow 2.x model artifacts vs MLflow 3.x Logged Models - why do both views exist in noted?**
A: Backward compatibility. A run produced against MLflow 2.x has its model artifact at `{run_id}/artifacts/model/` inside the run. A run produced against MLflow 3.x additionally has a Logged Model entity at `{experiment_id}/models/{model_id}/` with its own identity. noted's Registry view reads from the registered-model API (works for both). The Logged Models view (`ExplorerMlflowViews.js:595`) is a 3.x-only surface that shows the independent Logged Model entity, which is useful for inspecting artifacts that may not have been registered at all.

**Q: How does `@champion` decouple deployment from version numbers?**
A: Version numbers are stable historical identifiers - v7 was produced at time T with these metrics, and that fact never changes. `@champion` is a movable pointer that the serving client resolves on every load (or on a health-check cadence). Rolling back from v7 to v6 is a one-line alias reassignment on the MLflow server; the serving client picks it up at its next resolve. Redeploy is not required as long as the client is alias-aware. The alternative - hard-coding a version number in the serving config - forces a redeploy for every rollback or promotion.

**Q: What prevents two promotion attempts from racing?**
A: Nothing at the MLflow layer - `set_registered_model_alias` is last-writer-wins. In practice, noted's promotion is single-user from a notebook cell, and the DAG's promotion task is gated by the training task succeeding. If a multi-writer scenario became a concern, optimistic concurrency on the champion's run_id tag would be the minimal add.

**Q: Why is `target_mean` / `target_std` logged as a param rather than as a file?**
A: Because params are trivially readable via the run's API without downloading any artifact. A file would require an additional round-trip and artifact-path knowledge. The two scalars are tiny, immutable once logged, and accessed by `jena_client` on every prediction to invert the scaling. Params are the right shape for that access pattern.

**Q: What happens when an MLflow experiment is deleted out from under noted?**
A: A soft-deleted experiment is MLflow's foot-gun: `get_experiment_by_name(...)` returns the deleted experiment, `start_run` fails cryptically, and the user sees a confusing error with no recovery path in the UI. The known post-demo backlog item is to detect this in `RUN_START_CODE`, surface it as an explicit frontend notification, and offer restore/purge actions. Until that lands, the workaround is to purge the deleted experiment from the MLflow UI and let noted recreate it.

**Q: Why does noted use MLflow's artifact proxy for Logged Model downloads instead of a direct file URL?**
A: Because the MLflow server is the authoritative resolver of artifact URIs, which may point at MinIO, local disk, or any artifact store. Going through the proxy means noted's frontend never has to know where artifacts physically live - the proxy translates `experiment_id/model_id/path` into whatever the backing store requires. This also gives a single place to add auth in the future.
