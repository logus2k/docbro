# 9. Configuration Composer + Time Machine
## 9.1 Concept primer

The Configuration Composer is noted's UI-over-Hydra. It lets the user view, modify, and apply a composed configuration without writing YAML, without opening a shell, and without restarting the kernel. Time Machine is the same UI surface with the baseline source flipped from the project's working tree to a past MLflow run's archived bundle - turning the Composer into a config-reproducer for past experiments.

The two features share the same codepath. The only difference between "edit the current config" and "replay a past run" is which `HydraSource` implementation is servicing the composition (Chapter 1.3.1). This is the architectural payoff of the source abstraction - a single composition engine, two user-visible modes, zero duplicated logic.

Four design properties are worth naming:

1. **The Composer is read-write against *selections*, not templates.** Clicking Apply persists group selections and overrides to the notebook's metadata. It never edits a YAML file on disk. Template edits are still made the usual way (open `config.yaml` in the editor, save). This keeps the "templates vs config" distinction (Chapter 1.1) visible in the UI.
2. **The baseline badge is the integrity assertion.** After every Apply, the badge recomputes its state (green/orange/red) by comparing current selections against the active source. The user cannot be wrong about whether their config matches a baseline - the badge always tells the truth.
3. **Time Machine is composition, not checkout.** Loading a past run into the Composer does not modify any file in the working tree. No git checkout happens. The past run's bundle is composed against its archived templates, the result is shown in the Composer, and the user decides whether to Apply it (making it the new baseline for the notebook) or dismiss it.
4. **The HydraCache is the Composer's perf budget.** Each `(notebook_uid, run_id)` pair is one cache entry, backed by MLflow's artifact download on cache miss. Populated entries make Composer interactions instantaneous; cold cache entries take a few seconds for the initial fetch.

## 9.2 Where Time Machine lives

The Composer UI is one file: `frontend/js/panels/explorer/ExplorerHydraViews.js`. The Time Machine is not a separate component - it is the `Experiment Run` mode of the same Composer panel. A mode toggle (Local vs Experiment Run, rendered as two buttons at the top of the panel) swaps which `HydraSource` the backend uses for subsequent compositions.

Backend surface lives at `backend/app/routers/hydra.py` (the four endpoints in Section 1.3.5) plus three manager classes:

- `HydraManager` (`backend/app/managers/hydra_manager.py`) - composition, schema discovery, bundle assembly.
- `HydraSource` + `LocalSource` + `MlflowSource` (`backend/app/managers/hydra_source.py`) - source abstraction.
- `HydraCache` (`backend/app/managers/hydra_cache.py`) - in-memory LRU.

`frontend/js/NotebookEditor.js` owns the badge state and the notebook metadata contract.

## 9.3 The HydraSource abstraction

The filesystem-shaped contract (line 27 of `hydra_source.py`):

```python
class HydraSource:
    def exists(self, path: str) -> bool: ...
    def read_text(self, path: str) -> str: ...
    def walk(self) -> Iterator[tuple[str, list[str], list[str]]]: ...
```

Three methods: "does this file exist", "read this file as text", "enumerate the tree". That is the complete contract. HydraManager's composition code uses only these three methods; any backing store that can satisfy them is a valid source.

### 9.3.1 LocalSource

`LocalSource` (line 56) reads from the project's on-disk `config/` directory. It is a thin wrapper over `pathlib.Path` - `exists()` does `path.exists()`, `read_text()` does `path.read_text()`, `walk()` does `os.walk`.

Used when `baseline_source = "project://config/"` (the default for any notebook that opted into Hydra).

### 9.3.2 MlflowSource

`MlflowSource` (line 113) reads from an MLflow run's `hydra/` artifact tree. It does not hold the tree directly - it delegates to `HydraCache.fetch_from_mlflow(notebook_uid, run_id)` which downloads the tree on first access and flattens it into `dict[str, bytes]`.

`_load_bundle()` (line 129) is the lazy loader. `walk()` (line 175) reconstructs the directory tree structure from the flat bundle keys by splitting on `/` and grouping prefixes.

Used when `baseline_source = "mlflow://<run_id>"`.

### 9.3.3 The `walk()` recursion bug (fixed 2026-04-12)

A load-bearing historical detail: the original `MlflowSource.walk()` only yielded the root directory. Subdirectories were never added to `all_dirs`, so `assemble_bundle_from_source()` copied only top-level files when the source was MlflowSource. This caused runs made against a pinned MLflow baseline to archive *incomplete* config/ trees (only `config/config.yaml`, no group files), which broke downstream composition.

The fix walks each nested dir in the first pass, ensuring subdirectories are enumerated. Bundles produced before the fix remain permanently incomplete - they cannot be replayed. New Run Manager runs produce correct bundles.

## 9.4 The HydraCache

`HydraCache` (`backend/app/managers/hydra_cache.py:27`) is an `OrderedDict` keyed by `(notebook_uid, run_id)` tuples, mapping to bundle dicts. `MAX_ENTRIES=500` (line 24); overflow evicts the oldest entry FIFO.

Methods:

- `get(key)` (line 37) - fast-path read.
- `put(key, bundle)` (line 42) - store; evict oldest if full.
- `fetch_from_mlflow(notebook_uid, run_id)` (line 64) - download the `hydra/` artifact tree from the specified run, flatten into a bundle dict, store and return.

The cache is ephemeral (no disk backing, lost on backend restart). This is intentional - MLflow is the ground truth; the cache exists only to avoid downloading the same bundle on every Composer interaction within a session.

**Cache key design.** Including `notebook_uid` as part of the key is deliberate. The same run can be loaded as a baseline from multiple notebooks, and each notebook may make different override changes on top. Keying by `run_id` alone would conflate these sessions; keying by `(notebook_uid, run_id)` keeps them distinct.

## 9.5 The four Composer endpoints

`backend/app/routers/hydra.py` exposes the Composer's backend surface.

### 9.5.1 `GET /api/hydra/experiments/{project_id}` (line 182)

Returns the list of MLflow experiments that have at least one run tagged with `noted.project_id == project_id`. Used by the Experiment dropdown in Time Machine mode. The filter by project tag is what keeps the dropdown from showing unrelated experiments from other projects.

### 9.5.2 `GET /api/hydra/runs/{project_id}/{experiment_id}` (line 233)

Returns runs within the experiment that have a `hydra/` artifact bundle. Each entry includes `run_id`, `run_name`, `start_time`, `status`, and the `noted.hydra_config_hash` tag. The run dropdown shows these; sort order is newest-first.

Runs without a `hydra/` bundle are filtered out - they cannot serve as a Time Machine baseline, so surfacing them would be misleading.

### 9.5.3 `POST /api/hydra/load-bundle` (line 307)

The critical endpoint. Request body: `{notebook_uid, run_id}`. Response body: the archived selections, the archived overrides, the resolved yaml, and a validation flag.

Server logic:

1. Call `HydraCache.fetch_from_mlflow(notebook_uid, run_id)` to get the bundle.
2. Parse `selections.json` to get archived group selections and overrides.
3. Re-compose from `MlflowSource(notebook_uid, run_id)` using those selections.
4. Compare the new `sha256(resolved_yaml)` with the run's stored `noted.hydra_config_hash` tag.
5. Return `{group_selections, overrides, resolved_yaml, experiment_id, hash_matches: bool}`.

If `hash_matches: false`, the Composer UI surfaces a red X on the badge and refuses to apply. This is the guardrail for replay: if recomposition does not reproduce the archived hash, the bundle is corrupt or the composer code is buggy, and either way applying it would be lying about the run's identity.

### 9.5.4 `POST /api/hydra/compose-mlflow` (line 277)

For live composition while the user is tweaking overrides on top of an MLflow baseline. Request body: `{notebook_uid, run_id, group_selections, overrides}`. Server composes against `MlflowSource` with the user's modifications applied and returns the resulting `resolved_yaml` and `hash`. Used by the Composer to render a preview without calling Apply.

## 9.6 Composer UI state machine

`ExplorerHydraViews.js` manages the Composer's state. The key state fields (stored on `panel.content`):

- `mode` - `local` or `mlflow`.
- `experimentId`, `runId` - valid only when mode is `mlflow`.
- `groupSelections` - `{data: "jena_full_dataset", model: "gru_baseline", scaler: "standard"}`.
- `overrides` - `{seed: 42, "training.epochs": 10, ...}`.
- `schema` - the schema object returned by `get-schema-from-source` (group options + override fields).
- `resolved` - the last-known resolved config (for preview display).

User actions and state transitions:

- **Click Local button** (`_switchToLocal`) - set mode to `local`. Preserve `experimentId` and `runId` for later (preview-only). Reload schema against `project://config/`. Repopulate group dropdowns with schema defaults, preserve user's previous group_selections if they are still valid against the schema (D13 contract: switching modes does not wipe selections).
- **Click Experiment Run button** (`_switchToMlflow`) - set mode to `mlflow`. Load experiments list. Do not Apply yet. Update badge state immediately via `_updateApplyButtonEnabled`.
- **Pick experiment** - load the run list for that experiment.
- **Pick run** - enable Apply button.
- **Click Apply** - call `load-bundle`, then `NotebookEditor.setHydraSelections(...)`, then `_refreshActiveSchema(...)`, then update the badge.
- **Edit any override field** - trigger a debounced `compose-mlflow` (or `compose-local`) call to update the preview pane. Does not modify notebook metadata until Apply.

The Apply button is disabled by default (`rm-btn:disabled` CSS applied) until the state satisfies: mode + experiment + run are all selected (for mlflow) or at least one selection differs from defaults (for local).

## 9.7 Baseline badge state machine

`frontend/js/NotebookEditor.js:2106` is `_updateBaselineBadge()`. Three labels, three dot states.

**Labels:**

- `BASELINE` (neutral gray) - when `hydra_baseline_source === "project://config/"`.
- `RUN xxxxxx` (purple, short hash of the run_id) - when `hydra_baseline_source` is `mlflow://...`.

**Dots:**

- Green check - `no drift`. Current selections match the baseline: for Local mode, match schema defaults; for MLflow mode, match archived selections.
- Orange exclamation - `drift`. Current selections differ from baseline. Tooltip includes a `Drift:` section listing the specific keys that differ.
- Red X - `unreachable`. The baseline source could not be loaded. For MLflow, means the run was deleted or the bundle was corrupt. For Local, means the config tree is missing or malformed (rare).

`_computeBaselineBadgeState()` (line 2166) is the state computer. It walks the current selections against the schema (which was refreshed against the current baseline) and per-key compares to the default. Mismatches accumulate into the drift list.

`_selectionsEqual()` (line 2250) treats undefined / null / empty-string as "not selected" - this is what prevents stale empty metadata from triggering phantom drift.

`_refreshActiveSchema()` (line 2399) is called after every Apply to re-fetch the schema against the (possibly changed) baseline source. Without this call, the badge would compare against a stale schema and could falsely report drift.

## 9.8 Operations

### Add a new Composer override input

For an override to surface in the Composer UI, the corresponding leaf must exist in `config.yaml` (not in a group file - see Chapter 1.4 / the known gap). Once added to `config.yaml`, reload the Composer panel. `_extract_schema` will discover the new leaf and render it as an input.

### Fix a stale-metadata notebook

If the badge is stuck orange and the tooltip lists keys that are not visibly wrong, the notebook likely has legacy flat-format `hydra_selections`. Open the Composer in Local mode, pick defaults for all groups, clear overrides, click Apply. This rewrites the metadata to the current nested format.

### Clear the HydraCache

No UI for this; the cache is process-local. Restarting the backend clears it. On the next Composer interaction, the cache will refill from MLflow on first access.

### Debug a `load-bundle` hash mismatch

1. Check the run's `hydra/` artifact tree in MLflow UI. Verify it contains `config.yaml`, `selections.json`, `resolved.yaml`, plus the group files.
2. If files are missing, the run was produced before the 2026-04-12 `walk()` fix. The bundle is permanently incomplete; the run cannot be replayed.
3. If files are present but the hash still mismatches, diff the local composition's `resolved.yaml` against the archived one. The first divergence is the bug.

## 9.9 Discussion-ready talking points

**Q: Why is Time Machine built on the same UI as the Composer rather than a separate panel?**
A: Because the action of loading a past run as a baseline *is* composition - just from a different source. Spinning up a second panel would duplicate all the dropdowns, override inputs, badge logic, and compose endpoint wiring. The source abstraction (LocalSource vs MlflowSource) is the right axis to split on; swapping the source behind a single UI is cheaper and keeps the user's mental model clean.

**Q: What does the Apply button actually do?**
A: Three things in sequence. (1) Write `hydra_selections` and `hydra_baseline_source` to the notebook's metadata via `PATCH /api/notebooks/...`. (2) Fetch a fresh schema against the new baseline source to seed the drift comparison. (3) Recompute the badge. Nothing about the user's code is modified; no Python is executed; no kernel state changes. Apply is pure metadata mutation plus UI refresh.

**Q: Why validate the hash on `load-bundle` instead of trusting the archived bundle?**
A: Because the archive is only useful if composition against it reproduces the identity the run recorded. A bundle that composes to a different hash means either the bundle is corrupt, the composer code has regressed, or the Hydra library behavior has changed. Any of these is a reason to refuse the load rather than silently serve a subtly-different config. The red X on the badge is the user-visible evidence that the system caught a regression.

**Q: What happens to notebooks that have no `notebook_uid`?**
A: They are treated as Hydra-unaware. The Composer panel still renders, but Apply generates a UUID on the fly and writes it into metadata before proceeding. The first Apply is therefore the moment a notebook commits to being a Hydra-using notebook. Until then, `cfg` is not injected, and the notebook behaves like a regular Jupyter notebook.

**Q: Why does switching Local vs Experiment Run mode not immediately wipe the run selection?**
A: Because the user may flip modes to preview the alternative without committing. The D13 design contract is "switching modes is preview-only; Apply is the commit point". This lets the user click around freely in the Composer without fear of losing state. State is only persisted when Apply is explicitly clicked.

**Q: Why is the badge part of NotebookEditor rather than a separate module?**
A: Because the badge has to know about three pieces of notebook state simultaneously: metadata (baseline source, selections), live Composer state (is the user currently editing?), and the schema (what are the defaults?). All three are owned by NotebookEditor, so colocating the badge logic avoids cross-module coupling. The alternative (a separate Badge module subscribing to notebook events) would be more modular but would require propagating schema changes across a module boundary for no obvious win.

**Q: Does Time Machine work across projects?**
A: Implicitly yes, but the UI does not encourage it. The `experiments/{project_id}` endpoint filters by project tag, so the dropdown only shows experiments tagged for the current project. If an identical notebook existed in two projects, their Time Machine lists would be disjoint by design. Cross-project replay would require opening the bundle directly via a run_id the user enters manually - no UI for that at v0.1.

**Q: What is the relationship between the Composer and the Knowledge Graph?**
A: The Knowledge Graph (Module 12) reads the same MLflow tags and Hydra bundles the Composer does, but it visualizes the graph of runs / datasets / configs instead of showing one run at a time. A future feature would let the user click on a node in the graph to load its config into the Composer - the data is already in place; only the UI wiring is missing.
