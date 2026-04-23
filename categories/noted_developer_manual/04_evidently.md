# 4. Evidently
## 4.1 Concept primer

Evidently is a data and model monitoring framework. Where MLflow tracks run-level metadata (params, metrics, artifacts) and DVC tracks bytes, Evidently tracks *statistical properties of the data* itself: distributions, correlations, null rates, outliers, and how these change between two datasets. The output is always a report - an HTML document with charts, stats, and a verdict - that can be reviewed as a one-off or persisted in a workspace for time-series monitoring.

Four concepts structure every Evidently integration:

1. **Reports and presets.** A `Report` is a container for `Metric` objects (e.g. `ColumnDriftMetric`, `DatasetSummaryMetric`). A *preset* is a curated bundle of metrics wrapped in a single class: `DataSummaryPreset` (distribution summaries per column), `DataDriftPreset` (pairwise distribution comparison between two datasets), `RegressionPreset` (error metrics for a regression model's predictions). Presets are the default entry point; custom metric lists are for advanced use.
2. **Snapshots.** When a `Report` is run and saved into a workspace, the result is a *snapshot* - a serialized JSON payload with metadata (timestamp, tags) and the full metric output. Snapshots are the unit of persistence; they are also what the Evidently UI renders as dashboard rows.
3. **Workspace and project model.** A *workspace* is a storage root (local directory or remote HTTP service). Inside it, *projects* are named folders; inside each project, a time-ordered list of snapshots. The HTTP service (`evidently/evidently-service`) exposes a JSON API over the same data.
4. **The three preset families noted uses.** `DataSummaryPreset` (quality check on a single dataset), `DataDriftPreset` (train-vs-test or train-vs-prod comparison), and `RegressionPreset` (error distribution on held-out predictions). The Tutorial 3 notebook exercises the first two; `RegressionPreset` is reserved for production-monitoring loops not yet wired in.

**What Evidently does not do.** It does not ingest data, schedule jobs, or raise alerts. It computes statistics over what you hand it. Integration with a scheduler (Airflow), a tracker (MLflow), or an alerting system is the caller's responsibility. In noted, that glue is thin by design: Evidently is the stats engine; MLflow holds the run that the stats describe; Airflow runs the job that computes them.

**Tagging is the hinge.** Every snapshot in noted carries tags like `["data-quality", "jena-weather", "pipeline"]` or `["drift", "jena-weather", "run-1"]`. Tags make snapshots filterable in the Evidently UI and scopable in the backend's health endpoints. Without tags, the workspace becomes a flat pile of unlabelled snapshots that nobody can triage.

## 4.2 Where Evidently lives in the notebook

The Tutorial 3 notebook exercises Evidently in two cells: **cell 114** (data quality) and **cell 119** (drift). Both connect to the same `RemoteWorkspace` at `http://noted-evidently:8000`, both write into the same "Jena Weather" project, and both produce one snapshot per execution.

### Cell 114 - data quality

```python
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataSummaryPreset
from evidently.ui.workspace import RemoteWorkspace

ws = RemoteWorkspace("http://noted-evidently:8000")
# get or create project
projects = [p for p in ws.list_projects() if p.name == "Jena Weather"]
project = projects[0] if projects else ws.create_project("Jena Weather")

summary_report = Report(metrics=[DataSummaryPreset()],
                        tags=["data-quality", "jena-weather", "run-1"])
snapshot = summary_report.run(current_data=Dataset.from_pandas(df_features))
ws.add_run(project.id, snapshot, include_data=False)
```

The point of this cell is to write a baseline quality snapshot against the training data *before* any model is trained. The resulting snapshot surfaces distribution shapes, null counts, unique-value counts, and basic correlation. The Evidently UI (embedded in noted via the nginx proxy) renders it as a dashboard row that persists across sessions.

`include_data=False` means the snapshot carries only aggregates, not the raw rows. This keeps the `evidently-data` volume compact and avoids re-storing the dataset the DVC layer already has.

### Cell 119 - drift report

```python
from evidently.presets import DataDriftPreset

ref_dataset = Dataset.from_pandas(df_train[feature_cols])
cur_dataset = Dataset.from_pandas(df_test[feature_cols])
drift_report = Report(metrics=[DataDriftPreset()],
                      tags=["drift", "jena-weather", "run-1"])
snapshot = drift_report.run(current_data=cur_dataset, reference_data=ref_dataset)
ws.add_run(project.id, snapshot, include_data=False)
```

This is a train-vs-test drift check. Reference = train split, current = test split. `DataDriftPreset` computes per-feature distribution divergence (PSI / Wasserstein / chi-squared depending on dtype) and a rolled-up `dataset_drift_share` metric. The UI renders red for features whose drift score exceeds the preset's threshold.

At v0.1 of this manual, **cell 119 does not link the snapshot to the MLflow run_id** - that linkage exists only in the Airflow DAG. The notebook path therefore produces an orphaned drift snapshot from the Compose/Time Machine perspective: visible in Evidently, but not cross-navigable to the training run that produced the model. Closing this gap is a one-line change (`metadata={"run_id": active_run_id}` on the Report) tracked for post-demo work.

**Known runtime caveat.** Both cells assume `df_train`, `df_test`, and `final_feature_cols` are live in the kernel namespace. If the kernel was restarted between training and cell 114/119, those variables are gone and the cell raises `NameError: name 'final_feature_cols' is not defined`. The workaround is "Run All" from the top before reaching cells 114/119. This is a user-observable Run All vs Run Manager foot-gun addressed in Chapter 5.

## 4.3 How noted bridges to Evidently

noted treats Evidently as a thin integration: the charts live in the Evidently UI, the health signal lives in noted. The bridge is three pieces: an embedded service, a health-endpoint shim in the backend, and a tree-node dot in the frontend.

### 4.3.1 The Evidently service

`services/docker-compose.yml:308-318`:

```yaml
evidently:
  image: evidently/evidently-service:latest
  container_name: noted-evidently
  ports:
    - "8009:8000"
  volumes:
    - evidently-data:/app/workspace
```

The service runs the official Evidently HTTP server. Workspace data (projects, snapshots, tags) persists in the `evidently-data` named volume, declared at line 345. The volume was added on 2026-04-13 after a container rebuild wiped an entire notebook-run's snapshots - prior to the fix the workspace was ephemeral with no user warning.

The port mapping `8009:8000` exposes the service on the host for debug inspection. Inside the compose network, other services reach it as `http://noted-evidently:8000`, which is what the notebook cells use.

### 4.3.2 Nginx proxy and the Service tab

`services/nginx/nginx.conf:157-188` defines the `/evidently/` location block:

```nginx
location = /evidently { return 301 /evidently/; }
location ^~ /evidently/ {
    proxy_pass http://evidently:8000/;
    sub_filter '"/api/' '"/evidently/api/';
    sub_filter_once off;
}
```

The `sub_filter` is a subpath-aware rewrite: Evidently's SPA hardcodes `/api/` as the base path for its own AJAX calls, so the proxy rewrites it to `/evidently/api/` on the fly. Without this, the embedded UI would send API calls to noted's own `/api/` namespace and collapse into 404s.

`frontend/js/app-tabs.js:47` lists `evidently` alongside `mlflow`, `airflow`, and `minio` as a `service`-typed tab. When the user clicks the Evidently icon in the side icon bar, `frontend/js/menu-commands.js:93` calls `app._onIconBarClick('evidently')`, which adds a new tab containing an `<iframe>` pointed at `/evidently/`. The iframe loads the Evidently UI through the nginx proxy.

### 4.3.3 Backend proxy endpoints and Data Health

`backend/app/routers/evidently.py` exposes a small health-endpoint shim:

- `GET /api/evidently/projects` (line 16) - lists all projects in the workspace.
- `GET /api/evidently/projects/{project_id}/data-health` (line 36) - returns `{status: "green"|"yellow"|"red", summary: "..."}` derived from the most recent `data-quality`-tagged snapshot.
- `GET /api/evidently/projects/{project_id}/drift-status` (line 41) - returns a drift status computed from the latest `drift`-tagged snapshot's `dataset_drift_share`: green (<= 20%), yellow (> 20%), red (> 50%).

The manager layer (`backend/app/managers/evidently_manager.py`) calls Evidently's HTTP API directly. No Evidently Python library is used on the backend - the manager does `requests.get("http://noted-evidently:8000/api/...")` and parses JSON. This keeps the backend's dependency graph clean and insulates it from Evidently Python version drift.

`frontend/js/panels/explorer/ExplorerDataViews.js:213` (`updateDataHealthBadge`) calls these endpoints, aggregates across projects, and stores the worst-case status in `_dataHealthStatus`. `applyDataHealthDot` (line 236) renders an 8 px colored dot (green `#4caf50`, yellow `#ff9800`, red `#f44336`) on the Data tree's root node with a tooltip showing the summary text. This is the only noted-native surfacing of Evidently state; all other inspection happens in the embedded UI.

### 4.3.4 The Airflow DAG tasks

`dags/jena_training_pipeline.py` mirrors the notebook with two dedicated tasks:

- `evidently_quality` (line 285) - runs `DataSummaryPreset()` on the engineered features with tags `["data-quality", "jena-weather", "pipeline"]`. Runs in parallel with training (line 556).
- `evidently_drift` (line 519) - runs `DataDriftPreset()` on train-vs-test with tags `["drift", "jena-weather", "pipeline"]`. Runs *after* training completes (line 560) because it needs the test split to exist.

Line 536 is the critical linkage the notebook is missing:

```python
drift_report.set_metadata({"run_id": train_result["run_id"]})
```

The drift snapshot produced by the DAG carries the MLflow run_id of the training run it describes. In the Evidently UI, this appears as a metadata field on the snapshot; in a custom drill-down, a user can take that run_id and open the MLflow run to see the trained model's params, metrics, and Hydra config bundle.

### 4.3.5 Not yet implemented: quality gates

Evidently supports `TestSuite` - a report where each metric is wrapped in a pass/fail assertion with user-defined thresholds. At v0.1 of this manual, noted does not use Test Suites. All snapshots are *profiling* reports - they produce statistics, not verdicts. `EvidentlyManager.get_data_health_status()` (line 118) notes in a comment: "this DAG/notebook currently uses profiling reports only".

The quality-gate pattern is the right next step: wrap `DataSummaryPreset` in a `TestSuite` that fails if null rate exceeds X, unique-value count drops below Y, or distribution shape shifts outside a Kolmogorov-Smirnov bound. A failing Test Suite would turn the Data Health dot red automatically instead of requiring a manual `dataset_drift_share` threshold. This is in the backlog, not on the critical path for Tutorial 3.

## 4.4 Operations

### Filter snapshots by tag

1. Open the Evidently tab from the icon bar.
2. Navigate to the "Jena Weather" project.
3. The project page shows all snapshots. Use the tag filter to narrow to `data-quality`, `drift`, or a specific run label.
4. Click into a snapshot to see the full report.

### Configure a custom dashboard panel

Evidently's UI supports user-authored dashboard panels (lines, bar charts, text) that aggregate metrics across snapshots over time. At v0.1 of this manual the project has no custom dashboard - only the default per-snapshot views. Adding one is a UI-only action inside Evidently; noted does not inject dashboards programmatically.

### Link a drift finding back to the model trained on that split

For **DAG-produced** drift snapshots:
1. Open the drift snapshot.
2. Read the `run_id` field from its metadata.
3. In noted, open the Registry or MLflow view, navigate to the run, and inspect params and the Hydra bundle.
4. Open the Composer in Experiment Run mode with that run selected to see the exact config that trained the model.

For **notebook-produced** drift snapshots (v0.1): the `run_id` linkage does not exist. The user has to correlate by timestamp or by tag, which is less reliable. Add the `set_metadata({"run_id": active_run.info.run_id})` call in cell 119 to close this gap.

### Future quality-gate Test Suite pattern

When implemented, the pattern is:

```python
from evidently.tests import TestColumnShareOfMissingValues, TestColumnDrift
from evidently.test_suite import TestSuite

ts = TestSuite(tests=[
    TestColumnShareOfMissingValues("T_degC", lt=0.01),
    TestColumnDrift("T_degC", stattest="wasserstein", stattest_threshold=0.1),
])
ts.run(reference_data=ref_dataset, current_data=cur_dataset)
ws.add_run(project.id, ts.as_dict(), tags=["quality-gate", "jena-weather"])
```

The backend's `data-health` endpoint can then be extended to read the pass/fail status from the Test Suite output instead of inferring from the profiling report's aggregates.

## 4.5 Discussion-ready talking points

**Q: Why does noted treat Evidently as a thin integration (badges in noted, charts in Evidently UI)?**
A: Because Evidently's UI is already comprehensive and maintained upstream. Re-implementing its charts inside noted would duplicate work and force noted to stay in lockstep with Evidently's internal data model. Embedding it via iframe + nginx proxy gives users the full upstream UI with one-click access. noted's contribution is the *health dot* - the summary signal that tells a user "is there something to look at?" without forcing them to open the embedded UI on every glance. One-glance signal + one-click deep dive is the right UX shape.

**Q: Why is the train-vs-test drift framing meaningful for Tutorial 3?**
A: Because Tutorial 3 uses a time-ordered split: train is earlier months, test is later months in the same calendar year. Any distribution drift between these two is a *signal that the training assumption of "past resembles future" is weakening*. It is a weaker version of the production-drift question ("does today's traffic look like last month's training data?"), applied at dataset-preparation time rather than at inference time. Detecting meaningful drift in this framing is a justification for training on a shorter time window, for retraining more frequently, or for adding features that are more time-invariant.

**Q: What does it mean when drift is flagged on a feature that was specifically engineered?**
A: Engineered features can drift for two reasons. (1) The raw inputs drifted and the engineered feature inherited the drift. (2) The engineering logic has a subtle bug whose outputs are sensitive to a distribution the reference split did not cover. Reading the drift report next to the raw-feature drift report disambiguates: if both show drift, the root cause is upstream; if only the engineered feature drifts, the engineering code is the suspect. This is why running `DataSummaryPreset` on both raw and engineered features is valuable even when only one is fed to the model.

**Q: Why does noted poll Evidently for health instead of subscribing to a push event?**
A: Because Evidently's service does not expose event streams - its API is HTTP request/response. Polling on the Data tab open is cheap (a few hundred bytes per project) and does not block the UI. A future improvement would be to cache the last known status and only refresh on explicit user action or on a long debounce, which is a small optimization that is not yet warranted at the current data volume.

**Q: Why are the DAG and notebook Evidently tasks near-duplicates instead of sharing a library function?**
A: Deliberately, to keep the notebook self-documenting. The cells that a reviewer is most likely to read contain the full call to `Report([DataSummaryPreset()]).run(...).add_run(project.id, snapshot)` - no level of indirection to follow. The DAG duplicates the logic because it runs in a different execution context (Airflow worker, not a kernel with user-scope variables). A shared helper would simplify the code at the cost of making the notebook cells harder to read in isolation. For pedagogical notebooks like Tutorial 3, readability wins.

**Q: What is the risk of the `include_data=False` choice on snapshots?**
A: The risk is that, years from now, a reviewer wanting to replay exactly what the distribution looked like cannot - the aggregates are all they have. The alternative (`include_data=True`) would embed the raw rows into the snapshot, bloating the `evidently-data` volume by gigabytes. The right pattern is to store hashes alongside: the snapshot's metadata can carry the DVC md5 of the dataset it was computed on, and replay is possible by fetching that dataset version. This would bring data-lineage to Evidently in the same way it already exists for MLflow runs. Not yet implemented.

**Q: How does the "Jena Weather" project get created on first run?**
A: The notebook's cell 114 does `ws.create_project("Jena Weather")` if no project with that name exists. The DAG's `evidently_quality` task does the same. Either path creates the project; subsequent calls find it and reuse it. The race condition between the two is benign because Evidently's service serializes project creation. A more robust pattern would be to create the project once at noted-startup time (via a backend bootstrap task), but the lazy-creation approach is adequate for a single-user demo stack.

**Q: Can drift snapshots be deleted programmatically?**
A: Yes, via `ws.delete_run(project_id, run_id)`, but noted does not expose this in its UI. The intended workflow is "snapshots accumulate, tags filter" rather than "snapshots get pruned". If the volume grows large enough to matter, a retention policy based on tag + age would be the right shape (e.g. keep all `drift` snapshots, keep last 30 days of `data-quality`). Post-demo work.
