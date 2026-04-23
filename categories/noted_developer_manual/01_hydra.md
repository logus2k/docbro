# 1. Hydra
## 1.1 Concept primer

Hydra is a configuration framework for Python applications. Its value for MLOps is that it separates *what* a run depends on (the resolved configuration) from *how* that configuration was assembled (a stack of YAML fragments and CLI-style overrides). The framework is deceptively small - three ideas do most of the work:

1. **The defaults list.** A top-level `config.yaml` declares a `defaults:` list that points at *groups*. Each group is a directory, each option inside it a YAML file. `data: jena_full_dataset` selects `config/data/jena_full_dataset.yaml`. At compose time Hydra recursively merges each selected group's contents into a single tree. Changing `data: jena_2012_dataset` swaps one YAML file, the merge result changes, and nothing else in the calling code has to move.
2. **Group composition as a first-class operation.** Composition is not string substitution: groups can override each other, can set `_self_` to anchor their position, and can be swapped entirely at invocation time with `+group=option` or `group=option` syntax. The same `config.yaml` can produce radically different resolved configs just by moving options in the defaults list.
3. **Overrides as fine-grained edits.** After the defaults are resolved, Hydra applies a list of `key.path=value` overrides. `training.epochs=10` rewrites one leaf without touching anything around it. Overrides are ephemeral unless persisted - they live in the invocation, not in the YAML.

The resolved configuration is an `OmegaConf.DictConfig` - a nested dict with attribute access (`cfg.data.file`), type coercion, interpolation (`${other.path}`), and strict mode that catches typos at access time instead of at `.get()` time. When noted injects `cfg` into a running kernel it uses the same `OmegaConf.create(...)` call, so a notebook cell reading `cfg.training.epochs` cannot tell whether Hydra was invoked from the CLI or composed inside the backend and shipped over ZMQ.

**Why "templates not config" matters.** A critical mental model: the YAML files on disk are not the configuration. They are templates that *produce* a configuration when composed with a particular defaults list and override set. Two runs that read the same `config.yaml` can produce entirely different resolved configs if their defaults list picked different groups or their overrides differ. This is the whole reason Hydra exists and the reason every reproducibility story in noted ends with "...and then we hash the resolved config, not the templates."

**Hashing as a reproducibility primitive.** noted computes `sha256(resolved_yaml)` at compose time and stores it under `__noted_hydra_hash__` in the kernel and under `noted.hydra_config_hash` as an MLflow tag. Two runs with byte-identical hashes are guaranteed to have seen byte-identical configs, regardless of which templates were edited between them. Two runs with different hashes cannot be casually compared - the hash is the proof-of-identity for a configuration, and the Composer's baseline badge (Section 1.3) is the UI affordance that surfaces this fact to users who would otherwise not know their config had drifted.

**What Hydra does not do.** It does not orchestrate, track, or version. It has opinions about configuration and no opinions about anything else. In noted, Hydra is the gatekeeper for "what was this run told to do?"; MLflow is the gatekeeper for "what did this run actually do?"; and the two meet at `noted.hydra_config_hash` on each MLflow run. Keep these lanes straight and the rest of the integration stops being confusing.

## 1.2 Where Hydra lives in the notebook

The Tutorial 3 notebook (`notebooks/emi_tutorial3_jena_weather.ipynb` in the `jena_weather` project) is the worked example. It never imports `hydra` directly - `cfg` is injected by noted before any cell runs. Every cell that reads from `cfg` is therefore a consumer of the composed configuration, and every such access can be traced back to a specific YAML group or a specific override input in the Composer.

The notebook's configuration tree at v0.1 of this manual lives at `config/` in the project root:

```
config/
  config.yaml                    <- defaults list + inlined training block
  data/
    jena_2012_dataset.yaml      <- one-year subset (DVC md5 d3956bd0...)
    jena_full_dataset.yaml      <- full 2009-2016
  model/
    gru_baseline.yaml
    gru_evolutionary.yaml
  scaler/
    minmax.yaml
    robust.yaml
    standard.yaml
```

`config.yaml` declares `defaults: [data: jena_full_dataset, model: gru_baseline, scaler: standard]` and inlines the entire `training` subtree (epochs, batch_size, learning_rate, clipnorm, early_stopping.*, lr_reduction.*). There is no `training/` group directory - training was inlined on 2026-04-13 so every training knob surfaces as an override input in the Composer without forcing users to pick a "training preset" they don't care about.

**Cell-by-cell consumer map:**

| Cell | `cfg` accesses | Purpose |
|------|----------------|---------|
| 11 | `cfg.seed` | Seeds numpy, tf, random; replaces the old hardcoded `SEED = 42` |
| 16 | `cfg.data.file` | Resolves `DATASET_PATH = PROJECT_ROOT / cfg.data.file` - this is the load-bearing line that makes Composer data-switching actually switch the CSV |
| 41 | `cfg.data.features` | Column subset for training |
| 44 | `cfg.data.target` | Regression target name |
| 59 | `cfg.data.split.train`, `cfg.data.split.val` | Time-ordered split ratios |
| 63 | `cfg.scaler.name` | Selects scaler class from the scaler group |
| 70 | `cfg.data.lookback`, `cfg.data.horizon` | Window dimensions for the sliding-window generator |
| 91 | `cfg.model.*`, `cfg.training.*`, `cfg.scaler.name` | `build_model_from_cfg()` - entire model factory reads from cfg |
| 94 | `cfg.training.early_stopping`, `cfg.training.lr_reduction` | Passed to `train_model(...)` as `es_cfg=...`, `lr_cfg=...` |
| 95 | `cfg.training.epochs`, `cfg.training.lr_reduction.factor`, `cfg.training.early_stopping.patience` | Markdown cell referencing active values (rendered at execution time) |
| 116 | (reads none directly; logs run params) | MLflow `log_params` pulls from `cfg` indirectly |

Cell 11 and cell 16 are the two most important. Before their 2026-04-13 revisions the notebook was not actually cfg-driven despite looking like it: `SEED = 42` was hardcoded and `DATASET_PATH` pointed at a literal filename. Composer selections influenced training hyperparameters but not the two things users would most obviously want to control - seed and dataset. This is the kind of subtle regression that reproducibility primitives are supposed to catch, and the reason noted's hash-badge + MLflow tag design matters in practice.

## 1.3 How noted bridges to Hydra

noted's Hydra integration is a four-layer stack: a **backend composition engine**, a **kernel injection pipeline**, a **notebook metadata contract**, and a **Composer UI** with time-travel support. Nothing in Hydra itself is modified - noted composes by calling OmegaConf and by walking YAML files directly rather than using `hydra.main`, because the framework's decorator model assumes a CLI entry point and noted's entry point is a notebook kernel.

### 1.3.1 The composition engine: `HydraManager` + `HydraSource` + `HydraCache`

**HydraManager** (`backend/app/managers/hydra_manager.py:23`) is the orchestration layer. Its core methods:

- `get_schema_from_source(source)` [hydra_manager.py:147] - walks a source tree, discovers groups and options, returns `{groups, schema, defaults, baseline_source}`. The schema enumerates every leaf in the resolved config.yaml so the Composer knows which inputs to render as override fields.
- `compose_from_source(source, group_selections, overrides)` [hydra_manager.py:318] - merges `config.yaml` with the selected group files, applies overrides, and returns `{resolved, yaml, hash, sources}`. The hash is `sha256(yaml)` and becomes the primary key for comparing two runs.
- `assemble_bundle_from_source(source, ...)` [hydra_manager.py:57] - produces the per-run archive: a flat `dict[str, bytes]` containing the full config/ tree plus `selections.json` (the group choices and override values that produced this resolved config) plus `resolved.yaml` (the composed output). This is what gets uploaded to MLflow.

**HydraSource** (`backend/app/managers/hydra_source.py:27`) is an abstraction over "where does the config tree live?". Two concrete implementations:

- `LocalSource` [hydra_source.py:56] reads from the project's on-disk `config/` directory. Used when the baseline is `project://config/`.
- `MlflowSource` [hydra_source.py:113] reads from a past run's archived bundle. Used when the baseline is `mlflow://<run_id>`. It lazily fetches the bundle via `HydraCache` the first time a read is requested.

Both implement the same `exists()`, `read_text()`, `walk()` contract (file-system shaped), so `HydraManager` is source-agnostic. This is what makes Time Machine possible: the exact same composition code path resolves a config whether it comes from the working tree or from MLflow.

**HydraCache** (`backend/app/managers/hydra_cache.py:27`) is an in-memory LRU keyed by `(notebook_uid, run_id)` with `MAX_ENTRIES=500` and FIFO eviction. The `fetch_from_mlflow()` method [hydra_cache.py:64] downloads the `hydra/` artifact tree from a run on first access, flattens it into a `dict[str, bytes]`, and returns it. Subsequent loads of the same run are cache hits. The cache is ephemeral - it has no disk backing and is rebuilt on restart - which is deliberate: the archived bundles are the ground truth in MLflow, the cache is a performance optimization.

### 1.3.2 Kernel injection: `_build_hydra_injection`

Before any cell executes, noted runs a short Python prelude in the kernel to make `cfg` and its hash available as module-level variables. `ExecutionBridge._build_hydra_injection()` (`backend/app/managers/execution_bridge.py:877`) is called both on single-cell execute (line 149-164) and on run-start (line 327-332).

The injected code (simplified):

```python
import json as _json
__noted_hydra_config__ = _json.loads('<resolved_json>')
__noted_hydra_hash__ = 'sha256:<hash>'
try:
    from omegaconf import OmegaConf as _OC
    cfg = _OC.create(__noted_hydra_config__)
except Exception:
    cfg = __noted_hydra_config__  # dict fallback
```

The `hydra_config` dict passed into this call carries four fields that together encode the run's configuration identity: `notebook_uid`, `baseline_source`, `group_selections`, `overrides`. The backend composes the resolved tree on the fly for every injection - it does not trust a cached resolved value because overrides can change per invocation.

Execution of this prelude is via `_execute_silent()` (`execution_bridge.py:411`), which waits for the kernel's shell reply before returning control. The cell never sees the prelude output; only `cfg` and `__noted_hydra_hash__` remain in the user namespace.

This is the tech-debt-flagged "invisible prelude" the notebook depends on. It makes the notebook fail to run outside noted unless the user writes their own `cfg = compose(...)` cell. That trade-off is tracked in the backlog for post-demo cockpit design.

### 1.3.3 Notebook metadata contract

Two keys are persisted under `notebook.metadata.noted`:

- `notebook_uid` - a UUID generated lazily on the frontend the first time the user clicks Apply in the Composer (`frontend/js/NotebookEditor.js:2333`). Stable for the life of the notebook. This is what the HydraCache uses as part of its cache key, and what the backend uses to detect "is this notebook Hydra-using?".
- `hydra_selections` - a nested `{group_selections: {data: "...", model: "..."}, overrides: {"seed": 42, "training.epochs": 10}}` dict persisted on every Apply (`NotebookEditor.js:2353-2362`).
- `hydra_baseline_source` - either `project://config/` (the default) or `mlflow://<run_id>` (set when the user switches to Experiment Run mode and applies a past run as the baseline).

Writing happens in the frontend; persistence goes through `backend/app/routers/notebooks.py:101` which calls `NotebookManager.update_notebook()` to serialize the `.ipynb` file. The round-trip is sync per Apply click, not batched.

### 1.3.4 Per-run bundle archival

When a run begins executing, the Run Manager prelude emits a `display_data` message with the mime type `application/x-noted-run-start`. `ExecutionBridge` watches for this (line 599-613) and, on first detection, calls `_log_hydra_bundle_for_run()` (`execution_bridge.py:695`) in a fire-and-forget thread.

That function:

1. Re-composes the config from the notebook's current `hydra_selections` and baseline source.
2. Calls `HydraManager.assemble_bundle_from_source(...)` to produce the flat bundle.
3. Writes the bundle to a tempdir.
4. Uploads via `client.log_artifacts(run_id, tmpdir, artifact_path="hydra")`.
5. Tags the run with `noted.hydra_config_hash=<sha256>`, `noted.project_id=<id>`, and (if in a git repo) `noted.git_commit` + `mlflow.source.git.branch`.

Failures are logged and ignored - bundle archival is additive metadata, not a correctness prerequisite. Deduplication per-session prevents double-logging if the prelude fires twice.

The archived tree on MLflow looks like:

```
hydra/
  config.yaml
  data/jena_2012_dataset.yaml
  data/jena_full_dataset.yaml
  model/gru_baseline.yaml
  ...
  selections.json          <- {group_selections, overrides}
  resolved.yaml            <- composed output
```

This is what `MlflowSource` reads back when a user selects that run in Time Machine.

### 1.3.5 Composer panel and Time Machine

The Composer (`frontend/js/panels/explorer/ExplorerHydraViews.js`) is a jsPanel tab with:

- A **mode toggle** between Local (read from `project://config/`) and Experiment Run (read from `mlflow://<run_id>`).
- Three **group dropdowns** (data/model/scaler) populated from the active schema.
- About **ten override inputs** for `seed` and the inlined `training.*` keys.
- An **Apply button** that calls `NotebookEditor.setHydraSelections(...)` which in turn persists metadata and triggers schema refresh + badge recomputation.

The four Composer-to-backend endpoints live at `backend/app/routers/hydra.py`:

- `GET /api/hydra/experiments/{project_id}` [line 182] - lists experiments that have at least one run tagged with the project id.
- `GET /api/hydra/runs/{project_id}/{experiment_id}` [line 233] - lists runs that have a `hydra/` artifact bundle.
- `POST /api/hydra/compose-mlflow` [line 277] - compose using an MLflow run as baseline plus user tweaks.
- `POST /api/hydra/load-bundle` [line 307] - fetch a past run's bundle into the cache and return the selections/overrides that produced it (so the Composer can pre-populate).

### 1.3.6 Baseline badge state machine

The badge in the notebook bar (`NotebookEditor.js:2106`) shows one of three labels paired with one of three state dots:

- Label: `BASELINE` (gray) when `hydra_baseline_source` is `project://config/`.
- Label: `RUN xxxxxx` (purple, short hash) when it is `mlflow://<run_id>`.
- Dot: green check when current selections match defaults (Local) or match the archived bundle (MLflow). Orange exclamation when they differ (drift). Red X when the baseline source is unreachable (e.g. MLflow run deleted).

Drift is computed by `_computeBaselineBadgeState()` (`NotebookEditor.js:2166`) and `_selectionsEqual()` (line 2250). The tooltip includes a `Drift:` section listing the exact keys that differ - this was added after a real incident where a user had legacy flat-format metadata with stale group names, and the badge was going orange with no diagnostic surface.

`_refreshActiveSchema()` (line 2399) fetches `/api/hydra/schema/{project_id}` against the current baseline source and is called after every Apply to keep the schema used for drift comparison in sync with the baseline.

## 1.4 Operations

### Add a new group

1. Create the directory under `config/`, e.g. `config/optimizer/`.
2. Add one YAML option file per choice, e.g. `config/optimizer/adam.yaml`, `config/optimizer/adamw.yaml`.
3. Add `optimizer: adam` to the `defaults:` list in `config.yaml`.
4. Reopen the Composer in noted - the new group appears as a dropdown automatically. `HydraManager.get_schema_from_source()` rediscovers groups on every render.

### Add a new override input

Anything that is a leaf in `config.yaml` (not referenced as a group) automatically surfaces as an override input. If training is inlined (as it is in Tutorial 3), adding `training.optimizer_beta_1: 0.9` to `config.yaml` produces a new input in the Composer on next render. If training were a group, you would need to edit each training option file.

There is a known gap: group-file leaves are not currently exposed as override inputs. If a leaf only lives in `data/jena_full_dataset.yaml` and not in `config.yaml`, it cannot be overridden via the Composer UI. The workaround for now is to inline the value in `config.yaml`. Extending `_extract_schema` to walk into selected group files is queued for post-demo work.

### Debug a hash mismatch

Symptom: two runs that should be "the same" produce different `noted.hydra_config_hash` tags, or the badge shows orange after you thought you only changed a defaults list alignment.

1. Open both runs in MLflow and download the `hydra/selections.json` artifact from each. A diff of selections reveals any change in group selection or override values.
2. If selections match, diff `hydra/resolved.yaml`. Any byte difference changes the hash. Whitespace and key ordering are preserved by OmegaConf's dumper and do count as differences if templates were edited in between.
3. If resolved YAMLs match, suspect different Hydra library versions (unlikely in noted's pinned stack) or a silent edit to a template file that happened to compose to the same leaves but a different byte layout.

The badge's drift tooltip (`_computeBaselineBadgeState`) lists keys that differ from the baseline, which is usually enough to find the culprit without opening MLflow.

### Load a past run's config back into the notebook

1. Open the Composer.
2. Switch to Experiment Run mode.
3. Pick the experiment and run from the two dropdowns.
4. Click Apply. The backend fetches the run's `hydra/` bundle via `load-bundle`, validates composition produces the same hash, and pre-populates the Composer with the archived selections.
5. The notebook's `hydra_baseline_source` is now `mlflow://<run_id>` and the badge reads `RUN xxxxxx` with a green dot.

From this state, any additional Composer tweak is a *delta* against the archived run. On next execute, the run's bundle is re-archived under the new run id, so lineage is preserved at each generation.

## 1.5 Discussion-ready talking points

**Q: Why compose configs inside noted instead of calling `hydra.main`?**
A: `hydra.main` assumes a script entry point with CLI arguments. Notebooks have neither. noted composes via `OmegaConf.create(...)` on the backend and ships the resolved tree to the kernel as an injected variable. This keeps notebook code free of Hydra imports and lets the same composition engine run against either a local source or an MLflow-archived bundle via the `HydraSource` abstraction.

**Q: Why is "templates not config" a load-bearing idea?**
A: Because the YAML files on disk do not uniquely determine the resolved configuration - the defaults list and the overrides do. Two runs reading the same templates can produce different configs. The `sha256(resolved_yaml)` hash is the only proof-of-identity that survives template edits. The baseline badge surfaces this to users who would otherwise not realize their config drifted.

**Q: What exactly does the config hash buy at compare time?**
A: Byte-identical `noted.hydra_config_hash` between two runs is a guarantee that they saw identical resolved configurations. It does not guarantee identical metrics (GPU nondeterminism, floating-point ordering, dataset hash can still differ) but it rules out configuration as the source of any observed divergence. In the Compare panel, two runs with matching hashes and differing metrics point at data or seed; two runs with differing hashes point at the config diff as the first place to look.

**Q: What happens when an archived bundle is missing or partial?**
A: The original symptom - before the `MlflowSource.walk()` fix on 2026-04-12 - was that bundles produced against an MLflow baseline contained only `config/config.yaml` with no group files. The fix made `walk()` correctly recurse into subdirectories. Old pre-fix bundles remain permanently incomplete and are detected by `load-bundle` validation: composition against a partial source fails the hash check. The Composer surfaces a red X on the badge and refuses to apply.

**Q: Why is training inlined in `config.yaml` instead of being a group?**
A: Because every training knob is something users reasonably want to sweep independently. Making `training` a group forces users to pick a "training preset" as a whole, which is a bad abstraction for hyperparameter exploration. Inlining surfaces each knob as its own Composer input. The same reasoning could apply to future groups that currently only have one meaningful option.

**Q: What is the difference between a Composer override and editing a YAML file?**
A: An override is ephemeral - it lives in the notebook's `hydra_selections` metadata and produces a distinct resolved hash. Editing a YAML changes the template, which changes what every notebook composing against that template will resolve to. The convention is: exploratory sweeps use overrides (no git diff), decisions that should stick use YAML edits (committed to the repo). The badge treats both the same way - it only cares about the hash.

**Q: Why is HydraCache in-memory with no disk backing?**
A: Because MLflow is already the durable store. The cache is a per-session speedup for repeatedly loading the same run in the Composer. A restart rebuilds on first access. If persistence is ever needed, pickling the OrderedDict would suffice, but no user pain has surfaced to justify it.

**Q: What is the relationship between `notebook_uid` and `run_id`?**
A: `notebook_uid` identifies *which notebook* produced the data; `run_id` identifies *which execution* of it. Both keys are needed for the HydraCache because the same notebook can be re-executed against multiple past runs as baseline, and each `(notebook, past_run)` pair is a distinct cache entry. The notebook_uid is generated lazily so notebooks that never use Hydra never acquire one.

**Q: How does noted reconcile DAG-produced runs with Run Manager-produced runs in the Composer's run dropdown?**
A: The Airflow DAG emits an identical `hydra/` artifact tree via its `log_hydra_lineage` task. Any run with a `hydra/` artifact and the expected project tag surfaces in the dropdown regardless of origin. True parity is the goal - a DAG-produced run is resurrectable in the notebook just like a Run Manager run.
