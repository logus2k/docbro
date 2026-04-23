# 3. DVC
## 3.1 Concept primer

DVC (Data Version Control) is a thin layer over git that adds one capability git refuses to do well: version large binary files. The mental model is simple: instead of committing a 43 MB CSV into git, you commit a tiny *pointer file* that contains the CSV's content hash, its filename, and its size. The actual bytes live in a remote object store (MinIO, S3, GCS, whatever), addressable by their hash. `dvc pull` reads the pointer, fetches the bytes into the working tree. `dvc push` uploads the bytes to the remote. Git continues to version the pointer.

Four concepts do most of the work:

1. **`.dvc` files.** A YAML manifest with three fields per tracked output: `md5`, `size`, `path`. Committing a `.dvc` file to git pins an exact content hash to a commit. Checkout an old commit, run `dvc pull`, and you get the bytes that matched that commit - guaranteed by the md5, not by the filename.
2. **Remote storage.** Configured in `.dvc/config`. A single remote has a URL (`s3://bucket/prefix`) and credentials. DVC uploads/downloads to/from this remote based on hash addressing. The remote is content-addressed: the same md5 is never uploaded twice.
3. **Stages and pipelines.** `dvc.yaml` defines processing stages with inputs, outputs, and a command. `dvc repro` runs stages whose inputs changed. This is *not* what noted uses - Airflow is the pipeline tool in noted's stack. DVC in noted is strictly the data-versioning primitive, not the orchestrator.
4. **Content addressing.** Two datasets with identical bytes have identical md5s regardless of filename. This means DVC naturally deduplicates: two `.dvc` files with the same md5 point at the same object in the remote. It also means changing a single byte produces a new object.

**Why DVC and not git LFS?** Git LFS stores large files on a git hosting provider (GitHub LFS, GitLab LFS). DVC stores them on *any* object store the team already owns. For a local-first, provider-agnostic stack like noted's, DVC's indifference to the host is the right shape. The other difference: LFS is tightly coupled to git's checkout mechanism; DVC is a separate tool you invoke explicitly. That separation is what lets noted read `.dvc` files without needing a git hook.

**What DVC does not do.** It does not schedule, train, or track experiments. It does not understand semantic data quality (that is Evidently). It produces no alerts. It is a hash-addressed blob store with a pointer-file syntax.

## 3.2 Where DVC lives in the notebook

The Tutorial 3 notebook makes no direct DVC calls. The one load-bearing line is in cell 16:

```python
DATASET_PATH = str(PROJECT_ROOT / cfg.data.file)
df = ingest(DATASET_PATH)
```

`cfg.data.file` resolves to either `data/jena_climate_2009_2016.csv` (full 2009-2016 series) or `data/jena_climate_2012.csv` (one-year subset), depending on the Composer's data selection. Both files are DVC-tracked, both have corresponding `.dvc` manifests next to them, and both must exist in the working tree for the notebook to run.

The project's DVC state at v0.1 of this manual:

```
jena_weather/
  .dvc/config                            <- MinIO remote configuration
  .dvcignore                             <- empty
  data/
    jena_climate_2009_2016.csv          <- 43.2 MB, md5 959915f0...
    jena_climate_2009_2016.csv.dvc      <- pointer file, committed to git
    jena_climate_2012.csv               <-  5.2 MB, md5 d3956bd0...
    jena_climate_2012.csv.dvc           <- pointer file, committed to git
  src/data/
    filter_year.py                       <- derivation script
    ingestion.py                         <- CSV loader used by the notebook
```

`src/data/filter_year.py` (21 lines) is the *derivation script* that produced the 2012 subset. It takes `jena_climate_2009_2016.csv` as input, parses the `Date Time` column (format `DD.MM.YYYY HH:MM:SS`), and writes rows whose year matches the target. The script is committed to git but is not a DVC stage - it was run once manually, the output was `dvc add`-ed, and now the `.dvc` file is the artifact. This is a deliberate choice discussed in Section 3.5: Airflow is the pipeline tool, so DVC stages would duplicate orchestration responsibilities.

`src/data/ingestion.py` (line 7-13) is `load_dataset(file_path)` - a one-line `pd.read_csv(path)` wrapper with datetime parsing. It is filename-agnostic: whatever path you hand it gets read. The caller is responsible for ensuring the file exists (i.e. that `dvc pull` has been run at least once for the repo's current state).

## 3.3 How noted bridges to DVC

noted treats DVC as read-only metadata. The backend never runs `dvc` commands - it parses `.dvc` files directly because they are just YAML, and YAML is cheaper to read than to shell out to a CLI tool. This makes the integration resilient to DVC version drift: a future DVC release that adds new keys to the manifest does not break noted's hash extraction.

### 3.3.1 `DvcManager`: reading `.dvc` files

`backend/app/managers/dvc_manager.py` is the single parser.

- `DvcManager.status()` (line 217-284) walks a project directory, finds every `*.dvc` file, loads it as YAML, and extracts each `outs` block's `{path, md5, size}`. Returns a dict with `tracked_files: [{path, hash, size}, ...]`.
- `DvcManager.data_overview()` (line 390-423) loops over all registered projects and aggregates a single catalog view.

No DVC CLI is invoked. There is no dependency on DVC being installed on the backend server. The only requirement is that `.dvc` files exist in the expected shape, which is a git-committed contract.

### 3.3.2 Dataset hash injection on run start

`backend/app/managers/auto_instrumentation.py:152-162` defines `_get_dataset_logging_code(dataset_hashes: dict)`. It emits Python code that, when executed in the kernel, calls:

```python
mlflow.log_param("dvc_data_hash", hash_value)
mlflow.set_tag("dvc.data_hash", hash_value)
mlflow.set_tag("dvc.data_file", file_path)
```

`get_run_start_code()` (line 134-149) appends this blob to the Run Manager prelude whenever `dataset_hashes` is non-empty. The prelude executes before any notebook cell, so by the time cell 94 starts training, the active MLflow run already carries its data-lineage tags.

### 3.3.3 The `run:execute` handler's hash resolution

The backend decides which hashes to log. `backend/app/main.py:673-749` is the handler:

1. Receive the payload from the frontend. If `hydra_config` is present in the payload (i.e. the notebook has a Hydra baseline set), ignore the frontend's `datasets[]` array entirely.
2. Compose the Hydra config and read `cfg.data.file`.
3. Call `dvc_mgr.status(project_repo)` and build a lookup `{path: hash}`.
4. Resolve `cfg.data.file` to its hash. Pass `{cfg.data.file: hash}` as `dataset_hashes` to `execution_bridge.execute_run()`.
5. If no `hydra_config` is in the payload (legacy non-Hydra notebook), fall back to the frontend's `datasets[]` selections.

This bypass is deliberate. Before the 2026-04-13 fix, the Run Manager had its own dataset checkbox picker *in addition* to the Composer's data group selector. Two UIs could drift. Now, for any Hydra-using notebook, the Composer is the single source of truth and the Run Manager renders a read-only row showing the currently-selected dataset.

### 3.3.4 Data Catalog tree and version history

`frontend/js/panels/explorer/ExplorerDataViews.js` renders the Data tab.

- `loadDataFiles()` (line 41-66) fetches from `/api/dvc/status` and caches per-file metadata under `_dataFileMeta`.
- `showDataFileDetail()` (line 68-176) renders a detail card for a selected file: path, size, md5 in monospace, and a version-history list.
- Version history (line 93-176) fetches from `/api/dvc/file-history`, which resolves `.dvc` file contents across git history. Each historical version renders as a row with short commit, message, date, and size. Non-current versions have a "Checkout" button (line 140-167) which restores that version via a git + `dvc pull` sequence.

`applyDataHealthDot()` (line 236-251) adds a colored indicator to the Data tree's root node based on the most recent Evidently quality snapshot. This is the cross-chapter tie-in to Chapter 4 - the dot reads from Evidently's health endpoint but lives on the Data tree UI.

### 3.3.5 RunManagerPanel: Hydra-aware vs legacy mode

`frontend/js/RunManagerPanel.js` renders the Run tab. Two code paths:

- **Hydra-driven (line 262-292)** - if `getHydraDataFile()` returns a file (notebook has a Hydra baseline set), renders a single read-only row: `[hydra icon] data/jena_climate_2012.csv    from Hydra config`. If the file is not DVC-tracked, an orange warning appears inline.
- **Legacy (line 295-334)** - if the notebook is non-Hydra, renders a multi-select checkbox list of all DVC-tracked files. Each checkbox toggles membership in `run.datasets[]`, which the backend uses as the fallback hash list.

The first path is the demo story. The second exists because noted does not force Hydra adoption on every notebook.

### 3.3.6 MinIO remote and `.dvc/config`

The jena_weather project's `.dvc/config` (lines 1-8) points at:

```ini
[core]
    remote = minio
['remote "minio"']
    url = s3://noted-dvc
    endpointurl = http://noted-minio:9000
    access_key_id = admin
    secret_access_key = password
```

`noted-minio` is the in-compose hostname. From outside the compose network (e.g. a user's host running `dvc pull`), the endpoint has to be `http://localhost:9000` - this is the one configuration that differs between container and host. No `.dvc/config.local` is checked in; container-baked credentials are acceptable here because MinIO is not exposed beyond the compose network in the default deployment.

The bucket is `noted-dvc`, shared by all noted projects. Content addressing makes the shared bucket safe: identical bytes produce identical paths inside the bucket, regardless of which project pushed them.

### 3.3.7 Airflow DAG dataset handling

`dags/jena_training_pipeline.py:202-223` defines `ingest_data()`. It reads `cfg['data']['file']` (default `data/jena_climate_2009_2016.csv` if not overridden) and loads the CSV. MLflow logging (line 371-397) logs the file path as a parameter but, at v0.1 of this manual, does **not** log the `dvc.data_hash` tag that the Run Manager injects via `_get_dataset_logging_code()`. This is an inconsistency flagged for post-demo correction: DAG-produced runs appear in the Composer's Experiment Run dropdown but without the full data-lineage tag surface. The fix is to add an explicit `client.log_param(run_id, "dvc_data_hash", hash)` inside `log_hydra_lineage` or as its own task, resolving the hash the same way `main.py:706-715` does.

## 3.4 Operations

### Add a new tracked dataset

1. Place the file in a location inside the project, e.g. `data/new_dataset.csv`.
2. From the project root: `dvc add data/new_dataset.csv`. This creates `data/new_dataset.csv.dvc` and adds the file to `.gitignore`.
3. `dvc push` to upload to MinIO.
4. `git add data/new_dataset.csv.dvc .gitignore && git commit -m "Add new_dataset"`.
5. In the Composer, either add it as an option to an existing group (`config/data/new_dataset.yaml` with `file: data/new_dataset.csv`) or add an override input in `config.yaml` so it can be selected.
6. Reopen the Data tab - the new file appears automatically. No noted restart.

### Pull a specific version

1. `git log data/jena_climate_2012.csv.dvc` to find the commit that pins the version you want.
2. `git checkout <commit> -- data/jena_climate_2012.csv.dvc`
3. `dvc pull data/jena_climate_2012.csv.dvc` - DVC reads the pinned md5, downloads from MinIO.
4. Alternatively, use the Data tab's version-history Checkout button - it runs the same two commands under the hood.

### Switch dataset via the Composer

1. Open the Composer.
2. In the `data` dropdown, pick `jena_2012_dataset` or `jena_full_dataset`.
3. Click Apply. The notebook's `hydra_selections.group_selections.data` is updated and its `hydra_config_hash` recomputed.
4. On next Run Manager execute, `main.py` resolves `cfg.data.file` and injects its md5 as the `dvc_data_hash` param. The Run's MLflow lineage now carries the dataset identity.
5. No manual `dvc pull` is needed if the file is already in the working tree, which it will be as long as the previous pull produced both variants.

### Flow of the dataset choice into a run's lineage

The hash injection chain is:

`Composer selection` -> `hydra_selections.group_selections.data` -> `run:execute payload` -> `main.py` composes cfg -> `cfg.data.file` -> `dvc_mgr.status()` lookup -> `dataset_hashes = {file: md5}` -> `get_run_start_code(dataset_hashes)` -> prelude `mlflow.log_param("dvc_data_hash", md5)` -> run tags include `dvc.data_hash` and `dvc.data_file` -> Composer Time Machine filters and compare views can use these tags.

## 3.5 Discussion-ready talking points

**Q: Why two separate dataset files instead of a single versioned one?**
A: Because the Composer dropdown presents both as parallel options for user choice, and the user flow demands both be available simultaneously. If it were a single file with two versions, picking "2012" would require a `dvc pull` of that specific version, overwriting the file and forcing all notebooks currently pointing at the "full" version to re-pull. Two files lets every notebook see both datasets side-by-side with no git checkout gymnastics. The content-addressed storage means the duplication is only in pointer files, not in bytes.

**Q: Why DVC + MinIO instead of a full data lake?**
A: Because the project-scoped, local-first deployment model does not justify the operational cost of a data lake. MinIO gives S3-compatible storage in a single compose service. DVC gives git-native pointer files that are easy to review in PRs. A data lake would add catalog, governance, table formats, and ingestion pipelines - none of which are wanted at this scale. The upgrade path exists: a future deployment that needs cataloguing can swap the DVC remote to an S3-backed one and add Iceberg or Delta on top without changing the notebook code.

**Q: What is the role of the derivation script `filter_year.py`?**
A: It is the *recipe* for how `jena_climate_2012.csv` was produced from `jena_climate_2009_2016.csv`. It is committed to git as source code, not registered as a DVC stage. The choice is deliberate: stages would make DVC an orchestrator, and noted already has Airflow in that role. For one-shot derivations, a committed script + `dvc add` of the output is simpler and does not require every contributor to install DVC or run `dvc repro`. The provenance is readable from the repo - anyone can see the script that produced the file.

**Q: What happens when the DVC remote is unreachable?**
A: `dvc pull` fails at the network layer. The notebook will then fail at cell 16 with a `FileNotFoundError`. noted does not proactively pull on notebook open - it assumes the user has pulled as part of project setup. A future improvement is to detect a missing file at `cfg.data.file` resolve time and surface a one-click "Pull from DVC" action in the Data tab, which would shell out to `dvc pull` with the right target. Until then, failure is loud: a Python exception in the cell output. Loud failure is better than silent degradation.

**Q: Why does noted read `.dvc` files directly instead of using the DVC Python API?**
A: Because `.dvc` files are small, stable YAML. Loading them with `yaml.safe_load` is three lines and has no transitive dependencies. The DVC Python API would pull in DVC's full package graph, its global config, and its command-line surface. None of that is needed for noted's read-only hash extraction. The trade-off: noted does not implement arbitrary DVC features (stages, pipelines, experiments) because those require the DVC runtime. noted's contract is "read hashes from `.dvc` files", and everything else stays in the DVC CLI the user runs directly.

**Q: How does noted know which project a dataset belongs to?**
A: By the project's directory structure. `dvc_mgr.status(project_repo)` walks a specific path and finds every `.dvc` file under it. Projects are registered via `NOTED.md` (or the project registry), and each registered project has a root path. The same DVC remote can back many projects, and each project's `.dvc` files are scoped to its tree. Cross-project dataset sharing would require copying `.dvc` files, which is why the content-addressed remote is valuable: same bytes, no re-upload.

**Q: Is `dvc.data_hash` the single source of truth for "which dataset did this run see"?**
A: For Run-Manager runs produced in a noted-backed notebook, yes. Combined with the `hydra_config_hash` (which includes `cfg.data.file` in the resolved YAML) and the run's git commit tag, it triangulates the data identity with high confidence. For DAG runs at v0.1 of this manual, the `dvc.data_hash` tag is not yet logged, which is tracked as a known gap. Closing that gap brings DAG runs to parity with Run Manager runs in the Compose/Time Machine dropdown.
