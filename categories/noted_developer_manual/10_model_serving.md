# 10. Model Serving
## 10.1 Concept primer

noted-serving is the inference container that turns a registered MLflow model into an HTTP endpoint. Its responsibilities are narrow: load the artifact, validate the request, run the prediction, return the response. The surrounding complexity - tracking which version to load, caching artifacts, managing VRAM, streaming deploy progress to the UI - is what the module is actually about.

Three ideas explain most of the design:

1. **Loader + FastAPI in one process.** The container is a single uvicorn process hosting a FastAPI app plus a `ModelLoader` singleton. All state is in-process memory. This is the Phase 0a design; it has a specific failure mode (stale C-extension imports after reloading) documented in Section 10.8 and a queued Phase 0b refactor that moves loading to a worker subprocess.
2. **NDJSON streaming for observability.** A model load can take 10-60 seconds depending on artifact size. Streaming per-phase progress (`resolving`, `downloading`, `loading_model`, `ready`) lets the frontend show what the backend is doing instead of a spinner that outlives the user's patience. This is the `DeployEventStream` pattern.
3. **Alias-driven deployment.** The serving container resolves `@champion` on every load request by querying MLflow. Clients do not see version numbers unless they specifically ask for them. Rolling back is an alias hop (Chapter 2.5), and the serving layer picks it up on the next deploy.

The external `jena_client` demo app is a separate project outside noted - a FastAPI + socket.io demo UI that proxies to noted-serving. It exists to prove the serving contract is usable from a standalone application, not just from noted's own Try It panel.

## 10.2 The serving container

`client/Dockerfile` (18 lines):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY app/ app/
EXPOSE 5522
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5522"]
```

The image installs `uv` (a fast pip replacement from Astral) and uses it to install the requirements. Removing the `--no-cache` flag on 2026-04-15 was the fix that restored proper Docker layer caching - previously every rebuild re-downloaded 2 GB of CUDA wheels because uv was downloading fresh on every build.

`client/requirements.txt` lists the baseline dependencies:

- `fastapi`, `uvicorn`, `httpx` - the HTTP layer.
- `mlflow` - artifact resolution and pyfunc loading.
- `numpy`, `pandas`, `scikit-learn` - common data handling.
- `tensorflow[and-cuda]`, `torch`, `pytorch-lightning` - major ML frameworks.
- `xgboost`, `lightgbm` - tree ensemble frameworks.
- `boto3` - S3/MinIO access for artifact download.

The image is deliberately a *superset*. Phase 0a's working-by-baseline approach assumes that any model the user promotes will be loadable without additional installs - the superset covers TF, Torch, sklearn, XGBoost, LightGBM. For a model with exotic dependencies, Phase 0b (Section 10.8) is the correct escape hatch.

`client/app/main.py` is the FastAPI app (136 lines). Endpoints:

- `GET /health` - return the loader's status dict.
- `POST /load` - stream NDJSON progress, end with `ready` or `error`.
- `POST /unload` - free the loaded model.
- `GET /schema` - return the cached input/output schema.
- `POST /predict` - run inference on the request payload.

CORS is permissive (`*`) because the container runs inside the compose network and is reached only through the backend's proxy (`backend/app/routers/serving.py`). Exposing it directly would require tightening CORS.

## 10.3 `ModelLoader` and the RLock

`client/app/model_loader.py` is the state owner. The class is ~430 lines and holds:

- `_lock` - an `RLock` serializing load/unload operations.
- `_status` - one of `idle`, `loading`, `ready`, `error`.
- `_model`, `_model_info` - the loaded pyfunc model and metadata.
- `_phase`, `_phase_detail`, `_phase_callback` - for streaming deploy events.

Key methods:

### `load(model_name, version=None, alias=None)` (line 67)

1. Acquire the RLock.
2. If the same model+version is already loaded, return immediately (idempotent).
3. Set status to `loading`, clear previous model.
4. Delegate to `_load_inner(...)`.

### `_load_inner(...)` (line 104)

1. Resolve version from alias if provided (query MLflow `registered-models/{name}/alias/{alias}` API).
2. Emit `resolving` phase with the resolved version string.
3. Download the model artifact. Three resolution strategies are tried in order:
   - Read `model_uri` tag from the registered model version.
   - Scan the experiment's `models/` directory for a matching MLmodel (MLflow 3.x Logged Models).
   - Fall back to the legacy `runs:/<run_id>/model` URI.
4. Emit `downloading` phase with byte count updates.
5. Call `mlflow.pyfunc.load_model(local_path)`.
6. Emit `loading_model` phase with framework detection progress.
7. Extract signature, flavors, framework name, parameter count, artifact size.
8. Cache the schema (via `schema_builder.build_schema`).
9. Emit `ready` phase with the full health payload.

### The neutered `_install_model_deps()` (line 194-201)

Before 2026-04-15, the loader had a `_install_model_deps()` method that read the model's `requirements.txt` from the artifact and called `uv pip install` at runtime to pin the exact versions the model was trained against. This caused a subtle failure mode: installing numpy 2.x over an already-loaded numpy 1.x left TensorFlow holding stale C-extension pointers. Subsequent predictions would segfault or silently return garbage.

The fix: neuter `_install_model_deps()` to a no-op. Rely on the baseline image's superset and on MLflow's warning-mode loading (which tolerates version mismatches with a log message rather than refusing to load). The trade-off: for models with pins outside the baseline, loading will fail at import time rather than install-and-corrupt. Phase 0b solves this structurally by running the load in a fresh process each time.

### `unload()` (line 328)

1. Non-blocking acquire of RLock (refuse if a load is in progress).
2. Null the model and model_info; set status to `idle`.
3. `gc.collect()` to force Python GC.
4. Framework-specific cleanup:
   - `tf.keras.backend.clear_session()` for TensorFlow.
   - `torch.cuda.empty_cache()` for PyTorch.
   - `jax.clear_caches()` for JAX.
5. Release lock.

This is best-effort. In-process VRAM cleanup is never fully reliable - CUDA keeps a driver-level context per process, and frameworks maintain internal pools. Phase 0b's process-exit approach is the only clean guarantee; Phase 0a accepts the imperfection as a demo trade-off.

### `get_health()` (line 403)

Returns the current status dict *without acquiring the lock*. Must stay responsive while a load is in progress.

## 10.4 `DeployEventStream` and the NDJSON contract

`client/app/deploy_stream.py` (130 lines) bridges the synchronous `ModelLoader.load()` call to a FastAPI `StreamingResponse`.

The pattern:

1. Register a phase callback on the loader (`loader.set_phase_callback(my_callback)`).
2. Run the load in an executor thread via `loop.run_in_executor(None, loader.load, ...)`.
3. The loader's internal `_set_phase(phase, detail)` calls fire the callback on each state transition.
4. The callback puts an event onto an asyncio queue.
5. The `StreamingResponse` yields each queued event as a JSON line followed by `\n`.

Event shape:

```
{"phase": "resolving", "detail": "version 7 (alias=champion)"}
{"phase": "downloading", "detail": "45 MB / 120 MB"}
{"phase": "loading_model", "detail": "loading tensorflow flavor"}
{"phase": "ready", "result": { ...health payload... }}
```

On failure:

```
{"phase": "error", "error": "Could not load model: ..."}
```

The frontend's `ModelDeployer` reads these events with a native ReadableStream + TextDecoder (no polling). The exact bytes are the contract.

## 10.5 Backend proxy and frontend Deploy

`backend/app/routers/serving.py` (138 lines) proxies every serving endpoint through noted's backend:

- `GET /api/serving/health` -> forwards to client `/health`.
- `POST /api/serving/load` -> forwards NDJSON stream. Uses `httpx.AsyncClient` with `read=600s` timeout and streams line-by-line.
- `POST /api/serving/unload` -> forwards POST.
- `GET /api/serving/schema` -> forwards.
- `POST /api/serving/predict` -> forwards.

The proxy exists so the frontend never has to know the serving container's address. CORS is handled once at the noted backend; secrets (if ever added) are enforced once. The trade-off is one extra hop per request; for model inference that is in the seconds range, the milliseconds of proxy overhead are invisible.

`frontend/js/ModelDeployer.js` (~158 lines) is the client-side streaming consumer. Key method `_readStream()` (line 112) uses `response.body.getReader()` + `TextDecoder` to buffer partial lines, parse JSON per newline, and dispatch `onPhase(phase, detail)` or the terminal `onReady(result)` / `onError(msg)` callback.

`frontend/js/panels/explorer/ExplorerServingViews.js` is the UI integration. `showTryItPanel(modelName, version)` (line 27) opens a jsPanel, polls `/health` to confirm the right model is loaded, and calls `_buildInputForm()` (line 117) to render a form from the schema. Each field is typed from the signature (float / int / text) or falls back to a JSON textarea for complex shapes. A "Sample" button auto-populates values from `schema.example_input` or randomly-generated values per-type.

## 10.6 Logged Model artifact proxy

Chapter 2.3.7 described the Logged Models view; the serving path uses the same backend endpoints to discover and download artifacts. For completeness:

- `GET /api/mlflow/runs/{run_id}/logged_models` (`mlflow.py:59`) lists Logged Model entities linked to the run.
- `GET /api/mlflow/logged_models/{experiment_id}/{model_id}/download?path=X` (`mlflow.py:76`) streams a single file via MLflow's artifact proxy with directory-traversal validation.

The ModelLoader's three-strategy artifact resolution uses these endpoints indirectly - `mlflow.pyfunc.load_model("runs:/<run_id>/model")` goes through MLflow's Python SDK, which resolves to the same artifact paths the proxy exposes.

## 10.7 `jena_client` external demo

`/home/logus/env/iscte/jena_client/` is a separate project outside noted. Its purpose is to demonstrate that a standalone application can consume noted-serving's predictions through stable HTTP contracts.

Structure (`web/backend/server.py`, 150 lines):

- FastAPI + socket.io, serves a static frontend from `web/frontend/`.
- Proxies `/api/health`, `/api/schema`, `/api/predict` to `http://noted-serving:5522` via httpx.
- Queries `http://mlflow:5000` for model lists, version metadata, and **crucially** run parameters including `target_mean` and `target_std` (the scaler stats logged at notebook cell 116, Chapter 2.3.5).

The frontend is three dropdowns (project, model name, version with `@champion` default), a form for input features, and a result display. After receiving a scaled prediction from the serving endpoint, jena_client applies the inverse transform `y_real = y_scaled * target_std + target_mean` and displays the result in real units (degrees Celsius).

This is the proof that noted's training lineage survives the serving boundary. A future client in a different language (Go, TypeScript, Rust) would need to replicate the same three HTTP calls and the same inverse-transform math - nothing more.

## 10.8 Phase 0a vs Phase 0b

**Phase 0a (current, shipped 2026-04-15).** Single-process serving. Baseline image has a superset of frameworks. `_install_model_deps` is neutered. Models with pins outside the baseline will fail at load time with a clear error. VRAM cleanup is best-effort. Fast enough for demo scale.

**Phase 0b (deferred, designed but unshipped).** Worker subprocess architecture. Plan in `documents/serving_worker/serving_worker_plan.md` (467 lines).

Key properties of Phase 0b:

- Each Deploy spawns a fresh Python interpreter via `asyncio.create_subprocess_exec()`.
- Worker does `uv pip install` from the model's `requirements.txt` against a clean import state.
- Worker loads the model, exposes a mini-FastAPI on a localhost port, streams NDJSON to its stdout.
- Control plane (the original uvicorn process) proxies to the worker. Control plane never imports ML libraries, so it stays responsive.
- VRAM release guaranteed via process exit.
- Stale-import bug impossible because each Deploy is a fresh process.

Three optional layers on top:

- Layer 1 - uv cache volume shared across workers (faster installs).
- Layer 2 - per-model venvs with hash-based lookup (skip install if a matching venv exists).
- Layer 3 - worker pool with same-hash in-place model switch (swap models within a venv).

Estimated effort: 9-15 hours. Deferred because Phase 0a meets the Apr 21 demo needs. Revisited post-demo.

## 10.9 Operations

### Deploy a model

1. Open the Registry view.
2. Select the model, then a version.
3. Click Deploy. The button streams phases in place: `resolving` -> `downloading` -> `loading_model` -> `ready`.
4. On `ready`, the button flips to Unload. The Try It button becomes enabled.

### Unload a model

1. Click Unload on the deployed version's card. The loader releases the model and frees (best-effort) VRAM.
2. The Deploy button on that version becomes available again.
3. Refusal (e.g. a concurrent load in progress) shows an inline error; the button re-enables when safe.

### Try a model

1. With a model deployed, click Try It.
2. A jsPanel opens with a form built from the model's signature. Each field is typed.
3. Click Sample to auto-populate, or enter values manually.
4. Click Predict. The request goes to `/api/serving/predict`, the response renders inline as a table, line chart, or scalar depending on output shape.

### Debug a failing load

1. Check `/api/serving/health` - the `error` field has the exception message.
2. Check the noted-serving container logs: `docker logs noted-serving --tail 200`.
3. Common causes: artifact download fails (check MinIO), model signature cannot be parsed (check MLmodel file), framework version mismatch (a runtime-install path existed to fix this but was removed; Phase 0b is the proper fix).

### Inspect the Logged Model artifacts

1. Open the run in the MLflow view or via the Registry.
2. Navigate to the Logged Models subtree.
3. Open `MLmodel`, `conda.yaml`, `python_env.yaml`, `requirements.txt` - all render with hljs syntax highlighting.

## 10.10 Discussion-ready talking points

**Q: Why does the serving container use MLflow's pyfunc instead of a framework-specific load?**
A: Because pyfunc is the common interface that every flavor (tensorflow, pytorch, sklearn) implements. Loading via pyfunc lets the container be framework-agnostic at the load call site. Framework-specific operations (cleanup, parameter counting) branch on the detected flavor after the fact. The alternative - a big switch at load time - would duplicate logic per framework and make adding a new flavor a serving-side change.

**Q: Why NDJSON streaming instead of server-sent events or WebSockets?**
A: Because NDJSON over a plain HTTP POST response is the simplest contract that gives streaming progress. SSE would work but requires a specific content type and a different client API. WebSockets would require a separate connection setup and ping/pong management. The Deploy stream is a one-shot linear sequence - it is born, streams, dies. HTTP+NDJSON matches that shape perfectly.

**Q: Why is `get_health()` lock-free?**
A: Because health queries must respond during a load. If `get_health` acquired the same lock as `load`, a 45-second load would make the frontend's health polling hang, which would make the UI appear frozen. The lock-free read is safe because the fields it reads are atomic - Python's dict assignment is protected by the GIL for individual keys.

**Q: What prevents two Deploys from racing?**
A: The RLock in `ModelLoader.load()`. The second Deploy waits for the first to finish or fail. The frontend's Deploy button is disabled during the stream, so the user cannot fire two from the same tab; concurrent Deploys from different tabs would see the second one queue.

**Q: Why is the baseline image a superset instead of a minimal image?**
A: Because the cost of a missing package at Deploy time is user-visible (failed load with an import error), and the cost of an extra 2 GB in the image is only disk space. For demo-scale deployment the trade favors breadth. For production at scale, Phase 0b's per-model venvs is the right inversion.

**Q: How is the `@champion` alias resolved?**
A: The loader queries MLflow's `registered-models/{name}/alias/{alias}` endpoint, which returns the current version number the alias points at. Version number is then used for the artifact download. If the alias is reassigned between load and predict, subsequent predicts continue to hit the loaded version - there is no live-rebind. Rebinding requires Unload + Deploy; a future feature could add hot-swap.

**Q: What stops the serving container from being called directly from outside noted?**
A: Nothing - port 5522 is exposed on the host by `docker-compose.yml`. The intent is the noted backend's proxy. Direct access works for debugging but bypasses whatever auth the backend might add in the future. Production should either remove the port binding or front the container with an auth proxy.

**Q: Why is jena_client outside noted instead of integrated?**
A: Because it is proof-of-concept for the serving contract from an external client's perspective. If it were part of noted, it would be tempting to cheat - to call internal APIs, to reuse backend code, to share state. Keeping it in a separate repo forces the contract to be honest: three HTTP endpoints (`/health`, `/schema`, `/predict`) plus an MLflow API call to fetch scaler stats. That is the complete surface area a third-party integration needs.
