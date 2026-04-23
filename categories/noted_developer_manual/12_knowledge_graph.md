# 12. Knowledge Graph
## 12.1 Concept primer

The Knowledge Graph is noted's answer to "show me how everything connects". Each MLOps subsystem produces its own entities: MLflow has runs and models, DVC has datasets and versions, Airflow has DAGs and tasks, Hydra has configs and groups, noted has projects and notebooks. Each subsystem also has its own navigation surface. The Knowledge Graph is a *unified view* over all of them - one screen where a user can see a run's training data, its config, its DAG run, its champion promotion, its Evidently snapshot, and its downstream serving state, all as nodes and edges.

Three properties drive the design:

1. **Read-only aggregation.** The graph service owns no ground truth. It scans MLflow, Airflow, DVC, Hydra, and the filesystem on demand, assembles an entity-relationship graph, caches it, and serves it to the frontend. Mutations to the underlying data are done through the original subsystems; the graph reflects them on the next scan.
2. **Perspectives as filters.** A full graph with 100+ runs quickly becomes unreadable. Perspectives (Lineage, Performance, Versioning, Pipeline, Overview, Tags) are named filters that surface only the entity types and relationships relevant to a specific question. The same underlying graph is rendered six different ways.
3. **Three.js force-directed 3D.** The visual is a WebGL scene with force-directed layout. It looks more interactive than a static diagram and scales to hundreds of nodes without becoming illegible. Clicking a node opens a draggable detail panel; clicking "Open in Explorer" teleports back to the relevant noted view.

The graph is a **separate microservice** (`noted-graph` container) rather than part of the backend. This isolates its dependencies (it does not need any ML libraries) and keeps its scan operations from blocking the main backend's event loop.

## 12.2 The `noted-graph` service

`graph/Dockerfile` (14 lines) is Alpine Linux + Python 3.12. It installs a minimal set: `fastapi`, `uvicorn`, `requests`, `pyyaml`. No ML libraries, no socket.io, no heavy dependencies. Listens on port 5523.

`docker-compose.yml` defines the service:

```yaml
noted-graph:
  build: ../graph
  container_name: noted-graph
  environment:
    MLFLOW_TRACKING_URI: http://mlflow:5000
    AIRFLOW_API_URL: http://noted-airflow-apiserver:8080
    AIRFLOW_BASE_PATH: /airflow
    AIRFLOW_USERNAME: airflow
    AIRFLOW_PASSWORD: airflow
    NOTED_API_URL: http://noted:8123
    PROJECTS_DIR: /app/data/projects
    MOUNTS_DIR: /app/mounts
    GRAPH_PORT: "5523"
  volumes:
    - ../data:/app/data:ro
  ports:
    - "5523:5523"
```

The service reads `data/` read-only and calls MLflow / Airflow / noted's REST APIs for the rest of its inputs.

## 12.3 Entity and edge model

`graph/app/models.py` (lines 7-31) defines two pydantic models: `Entity` and `Graph`.

**Entity** is `{id, type, label, properties, tags}`. The `id` is `"{type}:{source_id}"` - e.g. `run:abc123...`, `data_file:jena_climate_2012.csv`, `model_version:Jena Weather Forecaster:v7`. The prefix disambiguates across sources; the source_id is the primary key in the source system.

Entity types (from the design doc):

- **Projects** - `project`, `file`, `notebook`.
- **MLflow** - `experiment`, `run`, `snapshot`, `model`, `model_version`, `tag`.
- **DVC** - `data_file`, `data_version`.
- **Hydra** - `config`, `config_group`, `config_option`.
- **Airflow** - `dag`, `dag_task`, `dag_run`.
- **Environment** - `environment` (virtual env metadata).

**Relationship types** (edges):

- `contains`, `belongs_to`, `version_of`, `snapshot_of`.
- `produces`, `uses_data`, `uses_config`, `executed_by`, `executed_as`.
- `has_task`, `depends_on` (DAG topology).
- `parameterized_by`, `runs_in`, `tagged_with`.
- `promoted_to`, `derived_from`, `code_at`, `scheduled_as`.

Each relationship carries a `properties` dict with the metadata that links the endpoints (e.g. `{hash: "sha256:abc..."}` on a `uses_config` edge so a user can see *which* config hash ties the run to that config group).

## 12.4 Scanners: populating the graph

`graph/app/graph_builder.py:24-88` orchestrates five scanners, run in sequence:

1. **`filesystem_scanner.py`** - walks `PROJECTS_DIR` and `MOUNTS_DIR`, discovers projects, notebooks, files. Produces `project`, `notebook`, `file` entities and `contains` edges.
2. **`mlflow_scanner.py`** (lines 18-80+) - queries `MLFLOW_TRACKING_URI`. Discovers experiments (line 26), runs (line 40), snapshots (runs tagged with `noted.snapshot=true`, line 52), registered models (line 71), and versions. Emits `experiment`, `run`, `model`, `model_version`, `snapshot` entities plus `belongs_to`, `snapshot_of`, `version_of`, `promoted_to` edges.
3. **`dvc_scanner.py`** - parses `.dvc` files plus git log to discover tracked files and historical versions. Emits `data_file` + `data_version` entities.
4. **`hydra_scanner.py`** - scans each project's `config/` directory. Emits `config`, `config_group`, `config_option` entities plus `contains` edges.
5. **`airflow_scanner.py`** (lines 21-80+) - queries Airflow's REST API (`GET /dags`, `/dags/{id}/tasks`, `/dags/{id}/dagRuns`). Emits `dag`, `dag_task`, `dag_run` entities plus `has_task`, `depends_on`, `executed_as` edges.

After scanning, `relationship_resolver.py` (lines 17-258) builds cross-source edges. Its job is "this MLflow run has a `noted.hydra_config_hash` tag, find the config node with that hash, add a `uses_config` edge". Key resolvers:

- `run -> data` via `dvc.data_hash` tag (line 60).
- `run -> config` via `noted.hydra_config_hash` tag (line 90).
- `snapshot -> commit` via git (line 114).
- `dag_run -> mlflow_run` via run_id in task logs (line 131).
- `notebook -> environment` via venv name (line 150).
- `project -> experiment` via naming convention (line 232).

This is the layer that makes the graph *a graph* rather than five disconnected per-source clusters.

## 12.5 Perspectives: filtered views

`graph/app/views.py:18-86` defines `BUILTIN_VIEWS`:

- `overview` - top-level entities (project, experiment, dag, model) with radial layout.
- `lineage` - data_file, run, model_version focus with emphasized `uses_data`, `produces`, `uses_config`. Hierarchical layout.
- `performance` - run + snapshot with color-by-metric and size-by-metric.
- `versioning` - data_version, model_version, snapshot, timeline layout, color-by-recency.
- `pipeline` - dag, dag_task, dag_run, hierarchical layout, color-by-status.
- `tags` - user-selected tags define the view dynamically.

`apply_view(graph, view)` (lines 91-131) takes the full graph and:

1. Filters out entities whose type is not in the view's `primary`/`secondary` sets.
2. Filters out relationships whose endpoints no longer exist after step 1.
3. Annotates each remaining entity with `_view_role` (primary / secondary / tertiary) - used by the frontend for color and size.
4. Annotates relationships with `_emphasized` if they are in the view's emphasized list.

Custom perspectives are persistable. `views.py:135-214` reads and writes `.noted/graph_views/*.json` per project. `save_custom_view` writes; `list_views` merges built-ins and custom.

## 12.6 Three.js rendering

`frontend/js/knowledge-graph/KnowledgeGraph3D.js` is the WebGL scene.

Construction (line 13): Three.js scene + camera + WebGLRenderer (lines 61-72), OrbitControls (line 76), ambient + 2 directional lights (lines 83-90). Mesh bookkeeping via `nodeMeshes` dict, `edgeLines` array, HTML overlay `labels` dict.

Force-directed layout (`_computeForceLayout()`, lines 142-229):

- Initialize positions uniformly random in a sphere.
- For 200 iterations:
  - For each node pair, compute repulsion `F = repulsion / dist^2`.
  - For each edge, compute attraction toward the other endpoint.
  - Apply a weak gravity toward the origin.
  - Damp velocity by 0.85 per step.
- Scale final positions by 0.15.

Node rendering (`_createNode()`, lines 233-276) picks geometry by entity type from `ENTITY_STYLES` (sphere / box / cylinder / octahedron / cone) and material by view role (primary = saturated, secondary = medium, tertiary = dim).

Edge rendering (lines 280-302) uses `THREE.LineBasicMaterial`. Emphasized edges are blue; normal edges are gray; dotted where semantically weaker.

Interactions:

- Hover (lines 341-479) raycasts per mouse move, scales hovered node, shows a floating detail panel.
- Click pins the detail panel and calls `onEntityClick(entity)`.
- Drag a node (lines 407-430) moves it via a plane intersection; on release, runs a 90-frame settling simulation.

The detail panel (`_showDetailPanel`, lines 483-583) is an HTML overlay with the entity's icon, label, property rows, and two buttons: "Open in Explorer" (calls `onEntityNavigate(entity)`) and "Pin". The panel is draggable.

## 12.7 Graph proxy

`backend/app/routers/graph_proxy.py` (42 lines) is a tiny pass-through. A single catch-all endpoint:

```python
@router.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE"])
```

Forwards all `/api/graph/*` requests to `GRAPH_URL` (env var, default `http://noted-graph:5523`). Preserves query params. Returns 503 on `ConnectionError`.

This keeps the frontend from having to know the graph service's hostname and lets the backend transparently add auth or caching later.

## 12.8 Knowledge Graph endpoints

`graph/app/routers/graph.py` (lines 27-128):

- `GET /graph/{project_id}` (line 32) - full scanned graph with entities + relationships.
- `GET /graph/{project_id}/neighborhood/{entity_id}` (lines 54-80) - BFS N hops from a seed entity; used by the "show neighborhood" action.
- `GET /graph/{project_id}/entity/{entity_id}` (line 110) - single entity + direct relationships.
- `POST /graph/{project_id}/invalidate` (line 126) - invalidate cache; forces a rescan next request.

`graph/app/routers/search.py` (lines 28-51):

- `GET /search/{project_id}?q=...` - text match on entity labels and properties. Supports type filtering, metric threshold queries (`val_loss < 0.1`), and tag queries (`#deployed`).

`graph/app/routers/views.py` (lines 14-71):

- `GET /views/{project_id}` - list all views (built-in + custom).
- `GET /views/{project_id}/{view_name}` - graph filtered by the named view.
- `POST /views/{project_id}` - save a custom view.
- `DELETE /views/{project_id}/{view_name}` - remove a custom view.

## 12.9 Perspectives UI

`frontend/js/knowledge-graph/GraphPanel.js` (lines 22-146) is the panel shell. A jsPanel-hosted window with:

- A search input and search button (line 57/64).
- A view selector dropdown with the built-in views (lines 71-81).
- A refresh button that invalidates the graph cache and re-scans.
- A 3D rendering area wired to `KnowledgeGraph3D`.
- An info bar showing entity count, relationship count, and current view name (line 93-96).

On view change (line 113), the panel fetches `GET /api/graph/views/{project_id}/{view_name}` and re-renders.

Search result dropdown (lines 193-229) shows up to 15 matches; clicking a result highlights the entity in the 3D scene.

## 12.10 Dagre and hierarchical layouts

The design doc mentions dagre for hierarchical/left-to-right layouts in the Lineage and Pipeline views. At v0.1 of this manual, the layout algorithm is force-directed only - dagre is a planned enhancement. When implemented, it will apply to specific views where the node type ordering has a clear topological meaning.

## 12.11 Operations

### Rebuild the graph

1. Click the refresh button in the GraphPanel, or
2. `POST /api/graph/{project_id}/invalidate`, then reload.

A rescan takes a few seconds for a small project, ~20s for 100+ runs.

### Add a new entity type

1. Define the type in the scanner responsible for its data source.
2. Give it a unique id prefix (`mytype:`).
3. Emit `{id, type, label, properties, tags}` from the scanner.
4. Add visual style to `ENTITY_STYLES` in the frontend (`GraphNodeRenderer.js`).
5. Decide which built-in views should show it and update their `primary`/`secondary` lists in `views.py`.
6. Rebuild.

### Add a new perspective

1. Define the view in `BUILTIN_VIEWS` in `views.py` with `primary`, `secondary`, `emphasized`, `layout`, optional `color_by`/`size_by`.
2. Add an option to the GraphPanel's view selector.
3. Test with a populated graph.

### Debug a missing edge

1. Pick two entities that should be connected.
2. Check the `relationship_resolver.py` logic for the relevant edge type - it is the most likely source of "should have but didn't".
3. Verify the tags/properties on both endpoints: for a `uses_config` edge, both the run and the config must share the same `noted.hydra_config_hash`.

## 12.12 Discussion-ready talking points

**Q: Why a separate microservice instead of bolting the graph into the main backend?**
A: Because the scan operations are I/O-heavy (multiple HTTP calls to MLflow, Airflow, plus filesystem walks) and do not share much with the rest of noted's backend. Running them in the main process would block the event loop on every rebuild. The isolated service can cache aggressively, scan asynchronously, and fail independently without affecting notebook execution.

**Q: Why Three.js instead of a 2D graph library like D3?**
A: Because 3D gives more visual room before overlapping edges become unreadable. With hundreds of nodes, 2D layouts quickly degrade; 3D with force-directed layout and a rotatable camera lets the user resolve clutter by viewing angle. The cost is that 3D requires a WebGL-capable device; the fallback (an HTML table of nodes and edges) is available for environments that do not support it.

**Q: Why read-only scans instead of real-time subscriptions?**
A: Because none of the source systems (MLflow, Airflow, DVC) emit structured change events that the graph could subscribe to. Polling them individually would multiply traffic; scanning on demand with a cache is simpler and bounded. The invalidate endpoint lets the user force a refresh when they know they changed something.

**Q: Why are perspectives baked into the backend instead of computed on the frontend?**
A: Because the frontend has to render every node and edge it receives. Filtering in the backend reduces payload size significantly - a typical Lineage view is 10-15% of the full graph. This matters both for network transfer and for the Three.js scene construction cost.

**Q: How do you handle the case where MLflow has hundreds of runs?**
A: Two mitigations. (1) Time range filters (a query param filters runs by date). (2) Neighborhood queries (`/graph/.../neighborhood/{entity_id}` returns a bounded subgraph around a seed). The Overview view avoids showing individual runs at all - it aggregates at the experiment level. For a true "show me everything", the force layout settles on clumps that are still navigable by zoom.

**Q: What is the relationship between the Knowledge Graph and the Composer?**
A: Both read from the same MLflow tags and Hydra bundles. The Composer surfaces one run at a time for editing; the Graph surfaces many runs for context. They are complementary views over the same data. A future feature would let the user right-click a run in the graph and "Load into Composer" to immediately edit its config.

**Q: Impact Analysis - what breaks if I change this?**
A: Mentioned in the README roadmap as T-5.1. The idea: right-click a node, get a directed BFS traversal downstream (what depends on this?) and upstream (what does this depend on?). For a `data_file` node, downstream traversal shows every run that used that file, every model promoted from those runs, every deployment that serves those models. Not yet implemented; the data is already in the graph.

**Q: How does the graph interact with the AI assistant?**
A: Via the `query_knowledge_graph` tool (Chapter 11.6). The LLM can ask "what runs use the jena_2012_dataset?" and the tool runs a filtered query against the graph service. This is the canonical way for the assistant to reason about lineage without having to re-implement scanning logic.

**Q: What is the maximum graph size that the 3D scene can handle?**
A: Empirically, ~500 entities render smoothly; ~2000 become sluggish on mid-range hardware. Beyond that, the force-directed iteration cost dominates. The mitigation is to use views to filter before rendering; Overview is explicitly designed to keep node count bounded by aggregating at the experiment level rather than the run level.
