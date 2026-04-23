# 5. End-to-End Scenarios
The first four chapters cover each MLOps tool in isolation: Hydra composes configs, MLflow tracks runs, DVC versions datasets, Evidently monitors data. This chapter combines them into workflows a developer actually performs. Every scenario below is a sequence of user actions that crosses at least three of the four integrations, and each maps to a demonstrable moment in the Tutorial 3 discussion.

## 5.1 Reproduce a past run

The reproducibility story is the one the project was built around. The claim is that any previous run - however old, however many template edits have happened since - can be re-executed from its archived bundle and will produce byte-identical metrics (modulo GPU nondeterminism and seeded variance that is out of noted's scope).

**Preconditions.** A past run exists in MLflow with a `hydra/` artifact tree. For runs produced before the `MlflowSource.walk()` fix on 2026-04-12, the bundle may be incomplete; those cannot be reproduced reliably. Runs produced after the fix are replayable.

**The sequence:**

1. **Pick the target run.** Open the Composer in the target notebook. Click the mode toggle to switch to **Experiment Run**. Pick the experiment from the first dropdown and the run from the second. The dropdowns filter to runs tagged with this notebook's project id.
2. **Apply.** Clicking Apply calls `POST /api/hydra/load-bundle`, which fetches the run's `hydra/` artifact tree into `HydraCache`, validates that composition reproduces the archived hash, and pre-populates the Composer with the archived `group_selections` and `overrides`. The notebook's `hydra_baseline_source` is now `mlflow://<run_id>` and the badge reads `RUN xxxxxx` with a green dot (no drift yet).
3. **Execute.** Click Run in the Run Manager. The prelude injects `cfg` composed from the archived bundle, the `run:execute` handler resolves `cfg.data.file` and injects the matching `dvc.data_hash`, and MLflow begins a fresh run. The prelude's silent execution of `get_run_start_code()` installs the metrics monkey-patch (Chapter 2.3.2).
4. **Verify.** On completion, compare the new run's `test_mae` (and any other metric you care about) against the original. For pure-CPU, single-threaded, seeded training, they should be byte-identical. For GPU training, tiny floating-point differences in the 5th-6th decimal are normal and indicate the config reproduction worked - the variance is in the hardware, not in the config.

**What has been proved?** That the configuration identity survives time. Every template file may have been edited, every group file renamed, every notebook cell restructured - and the archived bundle composed with the archived selections still produces the same resolved config, same hash, same run. This is what the `noted.hydra_config_hash` tag is for and why the badge surfaces drift at the first opportunity.

**Failure modes.**

- *Hash mismatch on `load-bundle`.* Composition against the archived source did not reproduce the stored hash. Most commonly: a pre-2026-04-12 partial bundle. Detected and surfaced as a red X on the badge. The user has to pick a different run or resign themselves to an approximate replay.
- *`dvc.data_hash` is missing from the archived run.* Legacy DAG runs or pre-2026-04-13 Run Manager runs may not carry the dataset hash. The replay will still run, but it uses whatever `cfg.data.file` resolves to in the current working tree - which may be a different version of the same filename than what produced the original. The fix is to re-run `dvc pull` against the git commit pinned at run creation, which `noted.git_commit` records.

## 5.2 Compare two runs

The Compare panel is how a user answers "which run was better, and why?". Two runs in MLflow are selected; noted renders a side-by-side metric diff, a side-by-side params diff, and a Hydra config diff.

**The sequence:**

1. **Select both runs.** Open the Registry or MLflow view and shift-click two runs in the run list, or open two Time Machine snapshots from the Composer and add them to Compare.
2. **Open the Compare panel.** The panel shows:
   - A metrics table with the delta for every overlapping metric.
   - A params table with the delta for every overlapping param.
   - A Hydra config diff: either "HASH MATCH: runs saw identical configs" or "HASH DIFFER: see diff below" with a key-by-key comparison of the two `resolved.yaml` files.
3. **Trace the diff.**
   - *Same hash, different metrics.* The configuration was identical, so the metric delta comes from something outside Hydra's responsibility: GPU variance, different dataset (check `dvc.data_hash`), different code (check `noted.git_commit`), different library versions in the env.
   - *Different hash.* The configuration differed. The key-by-key diff tells you which leaf changed. Each leaf maps back to either a Composer override input or a group selection. "Differed: `training.epochs: 50 -> 10`" means someone overrode epochs in the Composer.
4. **Open the offending run's Composer.** Clicking the hash diff with a run loaded navigates back to the Composer with that run as Experiment Run baseline. The current selections and overrides are displayed, and the diff is now explorable per field.

**Why this works.** Because all metric/param/config comparison is grounded in MLflow's own queryable metadata plus the archived Hydra bundle. noted does not store a separate comparison index - it asks MLflow, parses the bundle, diffs the YAMLs. Two runs with identical hashes are guaranteed to have seen identical configs, so the diff UI never displays "different configs" when they were actually the same.

## 5.3 Promote and serve

The promote-and-serve scenario is the shortest path from "I have a better model" to "the production client is using it". It touches MLflow (registry + aliases), noted-serving (the container), and jena_client (the external caller).

**The sequence:**

1. **Train.** Run the notebook via Run Manager. The prelude opens an MLflow run; cell 94 trains the model; cell 116 calls `mlflow.tensorflow.log_model(...)` to add the model artifact to the run; cell 117 calls `register_and_promote(...)`.
2. **Register and promote.** `register_and_promote` reads the current `@champion`'s `test_mae`, registers the new model as version N+1, and - if `new_mae < champion_mae` - reassigns `@champion` to version N+1 via `client.set_registered_model_alias(...)`. This is a single MLflow API call. No file moves, no redeploy.
3. **Serving health refresh.** noted-serving polls `@champion` on a health-check interval. The next health tick resolves the alias to the new version number. If the serving process is already holding the previous champion in VRAM, it unloads and reloads - or, if "hot-swap" is not yet implemented, the Deploy button in the Registry view surfaces the new champion with a one-click reload prompt.
4. **jena_client picks up the new champion.** The standalone `jena_client` demo app queries noted-serving's `/predict` endpoint. It has no knowledge of version numbers - it sends a payload, receives a prediction, and inverse-transforms using the `target_mean` / `target_std` that noted-serving returns alongside the prediction. When `@champion` hops, the next request hits the new model transparently.

**What has been proved?** That aliases decouple version identity from deployment state. Rolling back is `client.set_registered_model_alias("Jena Weather Forecaster", "champion", 6)` - done, one line, no notebook re-run, no container rebuild. This is the MLOps idiom that makes the rest of the system tolerable to operate.

**Observability during the deploy.** The Deploy button in the Registry view streams NDJSON from `/api/serving/load`: phases like `starting`, `downloading_artifact`, `loading_model`, `ready`, or `failed` appear in the UI as they happen. A failure at any phase surfaces the traceback inline; the button does not silently revert.

## 5.4 Drift investigation

The drift scenario starts from a concerning Evidently snapshot and ends at a retraining decision. It crosses all four integrations in a single narrative.

**The sequence:**

1. **Notice the signal.** The Data Health dot on the Data tree root is yellow or red (Chapter 4.3.3). The user opens the tooltip: "Drift flagged on 4 features."
2. **Open Evidently.** Click the Evidently icon in the side bar. The embedded UI shows the "Jena Weather" project. Filter by tag `drift`. Pick the most recent snapshot.
3. **Identify drifted features.** The snapshot lists per-feature drift scores. Say `T_degC` and `rh` are flagged red, `wv` is yellow, `p` is green. Click a drifted feature to see its distribution comparison chart.
4. **Navigate back to the MLflow run.** If the snapshot is DAG-produced (Chapter 4.3.4), its metadata includes `run_id`. Copy that run_id into the Registry or MLflow view to open the training run. If it is notebook-produced (v0.1 gap), correlate by tag and timestamp.
5. **Inspect the run's Hydra config.** From the run detail, open the `hydra/` artifact tree. Read `selections.json` to see what data group was selected (`jena_full_dataset` or `jena_2012_dataset`). Read `resolved.yaml` to see the features and splits the run actually used.
6. **Decide.** Did the drift occur because the train split included an anomalous period? Because the feature engineering was sensitive to a rare distribution? Because the dataset has actual real-world drift? The combined view (Evidently chart + Hydra config + MLflow params + DVC data hash) is what enables the decision.
7. **Act.** Possible follow-ups:
   - Retrain on a narrower window: switch Composer to `jena_2012_dataset`, click Run Manager, and compare against the full-series champion.
   - Add a feature more robust to the drifted variable (e.g. a rolling mean of `T_degC` instead of the raw value).
   - Re-engineer the feature whose drift was a bug, not a real signal.

**What has been proved?** That every MLOps signal in noted is cross-navigable. Evidently drift -> MLflow run -> Hydra config -> DVC data hash -> git commit. The chain is linear and unambiguous, which is the property that lets an engineer diagnose a drift finding in minutes rather than reconstructing context from scattered logs.

## 5.5 Failure modes

The scenarios above assume the happy path. This section lists the non-happy paths a reviewer is most likely to probe during the discussion.

### Deleted experiment foot-gun

MLflow's soft-delete on an experiment leaves the experiment in a weird zombie state: `get_experiment_by_name(...)` returns the deleted experiment, `start_run` fails cryptically, and the user sees `RESOURCE_DOES_NOT_EXIST` with no actionable guidance. The current `except: pass` in `RUN_START_CODE` masks this failure, so the symptom surfaces much later as "my metrics are not appearing in the panel".

The post-demo fix is to (a) remove the swallowing `except: pass` in `RUN_START_CODE`, and (b) detect the zombie state at prelude time, surface a notification in the frontend, and offer "Restore experiment" or "Purge experiment" actions. Until then, the workaround is manual: open the MLflow UI, find the soft-deleted experiment, purge it from the trash, and let noted recreate it on next run.

### Run All vs Run Manager

The notebook has two execution paths. **Run Manager** goes through `execute_run()`, which installs the full prelude: cfg injection, metrics monkey-patch, run-start hook, dataset hash logging. **Run All** (the Play-all-cells button) goes through `cell:execute` one cell at a time, which injects `cfg` (Chapter 1.3.2) but does *not* install the metrics patch. Consequence: live metrics do not stream during Run All, and the MLflow run opened inside cell 116 is different from the one the prelude would have opened.

Symptom: after Run All, the MLflow run has no epoch-level metrics (only the final `log_metric` calls) and the live panel stays empty. The fix is either to always use Run Manager, or to document that Run All is a "quick sanity check" path and not the tracked execution path. The demo-appropriate workaround is "always use Run Manager for training"; the engineering fix is to have `cell:execute` also install the prelude when a run is about to start.

### Missing scaler stats

`target_mean` and `target_std` are logged as params (Chapter 2.3.5). jena_client reads them on every prediction to invert the scaling. If a run is promoted without having logged these params - e.g. a legacy run, a DAG run before the params were added - jena_client falls back to returning scaled predictions or raises a `KeyError`.

Symptom: jena_client returns "0.27 degrees Celsius" (a scaled value) or an error. Diagnostic: open the champion run in MLflow, check the params table for `target_mean` / `target_std`. If missing, either retrain and re-promote, or manually set the params on the run via `client.log_param(run_id, ...)`.

### DVC remote unreachable

Symptom: `dvc pull` fails or `cfg.data.file` resolves to a path that does not exist. The notebook crashes at cell 16 with `FileNotFoundError`.

Diagnostic: check that `noted-minio` is running (`docker ps | grep noted-minio`), that the `.dvc/config` points at the right endpoint, and that the working tree has the file after `dvc pull`. No silent fallback - the error is loud by design (Chapter 3.5).

### Composer Apply on an empty selection

Before the 2026-04-13 fix, clicking Apply in Experiment Run mode without first picking a run would wipe `group_selections` to `{}`. The badge would go red, subsequent runs would fail to compose, and the user would not know why.

The fix disables the Apply button until a run is selected, and CSS styles disabled buttons distinctively (gray bg, not-allowed cursor) so the state is visible. An older notebook that carries a cleared `group_selections` from before the fix may still need its metadata manually repaired - reopen the Composer in Local mode, pick a set of groups, and re-Apply.

### Stale notebook metadata

A notebook created before the M1 refactor may carry legacy flat-format `hydra_selections` like `{"data": "default", "model": "gru_baseline", "scaler": "standard", "training": "default"}`. The Composer validates each entry against the current schema and falls back to defaults for invalid values, but the badge will go orange until the user re-Applies to refresh the metadata into nested format. Cell 95 markdown that references `cfg.training.epochs` will also still work because the Hydra composition applies the inlined `training` block from `config.yaml` regardless of the legacy selection.

### Bundle archival racing with MLflow artifact upload

The `_log_hydra_bundle_for_run` fire-and-forget thread can in theory race with the notebook's own `tensorflow.log_model` call. In practice MLflow's artifact upload is idempotent and append-only under `hydra/`, so the race is benign. If a future bug surfaces where the bundle is missing from a run while the tag is present, the diagnostic is to check the `_log_hydra_bundle_for_run` logs for a silent exception - historically this has been the `MlflowSource.walk()` recursion bug (Chapter 1.3.5). Failures are logged but not re-raised.

### Container restart wipes the Evidently workspace

Before the 2026-04-13 named-volume fix, a container rebuild wiped `/app/workspace` inside the Evidently container. Symptom: all snapshots disappear from the Evidently UI after a `docker compose up --build`. The fix (the `evidently-data` named volume in `docker-compose.yml`) persists the workspace across rebuilds. No action required except to rebuild after pulling the fix.

---

**What this chapter proves.** Each of the four MLOps integrations is individually useful. Combined, they produce workflows whose coherence comes from three cross-cutting primitives: the Hydra hash (config identity), the DVC md5 (data identity), and the MLflow run id (execution identity). Every scenario above is a walk through the graph these three primitives build. A reviewer asking "how does noted tie everything together?" is being asked to notice that the answer is the graph, not a centralized coordinator.
