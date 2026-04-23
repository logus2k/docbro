# 14. Infrastructure & Deployment

## 14.1 Concept primer

noted ships as a **Docker Compose stack** of 13 services. The stack is designed to run on a single host - either the user's workstation or a small single-node server - without requiring Kubernetes, a service mesh, or a managed database. A handful of named volumes persist state; the rest is rebuildable from the compose file.

Three architectural choices drive every infrastructure decision:

1. **Local-first.** Every service binds to the same Docker network (`noted-network`) and addresses peers by hostname. There is no external service dependency: MLflow, MinIO, Airflow (with its Postgres + Redis), Evidently, the Knowledge Graph, and the serving container all live inside the compose stack.
2. **File-first state.** The filesystem is the source of truth for projects, notebooks, configs, and data. Databases (Postgres for Airflow, SQLite for MLflow) are internal plumbing; the user-facing state lives in files that git can version. This is the property that makes NOTED.md and `documents.json` work as registries.
3. **Filesystem + process isolation, not network isolation.** Each service's private state (Postgres data, MLflow tracking dir, MinIO buckets) sits in a named volume. Containers do not share volumes unless they must. Network isolation inside the compose network is flat (every service can reach every other service) because everything is local.

The compose file is the canonical deployment artifact. `docker-compose.yml` in `services/` is what a reviewer or a new operator reads to understand what runs where.

## 14.2 The compose graph

`services/docker-compose.yml` declares the services. Grouped by role:

**Noted core**

- `noted` - the main backend + frontend. Image `logus2k/noted`, built from the repo root. Ports `8123` (HTTP) and `3719` (websocket-only in some deployments).

**Model lifecycle**

- `mlflow` (`ghcr.io/mlflow/mlflow:latest`) - tracking server, port 5000, SQLite backend at `/mlflow/mlflow.db`, artifact root `/mlflow/artifacts`. Command line installs `plotly` at runtime for chart rendering in the MLflow UI.
- `minio` (`minio/minio:latest`) - object storage, ports 9000 (S3 API) and 9001 (web UI). Used for DVC remote storage and MLflow artifacts if configured.

**Pipelines**

- `postgres` (Postgres 16) - Airflow metadata DB.
- `redis` (Redis 7.2-bookworm) - Airflow Celery broker.
- `airflow-apiserver` (port 8080), `airflow-scheduler`, `airflow-dag-processor`, `airflow-worker` (Celery worker), `airflow-triggerer` - the full Airflow 3.x worker stack.
- `airflow-init` - one-shot DB migrations + admin user creation; terminates after startup.
- `airflow-cli` (profile `debug`) - ad-hoc Airflow CLI container for debugging.
- `flower` (profile `flower`) - Celery monitoring UI, port 5555.

**Monitoring**

- `evidently` (`evidently/evidently-service:latest`) - data monitoring, port 8009:8000. Persists workspace in the `evidently-data` volume (Chapter 4.3.1).

**Auxiliary services**

- `noted-graph` - the Knowledge Graph service (Module 12), port 5523.
- `noted-serving` - the model serving container (Module 10), port 5522.

Each service attaches to the shared `noted-network`. Hostnames inside the network mirror the service names: `mlflow`, `minio`, `noted-evidently`, `noted-graph`, `noted-serving`, etc.

## 14.3 Named volumes

Four named volumes at the bottom of `docker-compose.yml` persist state across container rebuilds:

- `postgres-data` - Airflow's metadata database.
- `mlflow-data` - MLflow's SQLite DB plus artifact root (runs, models, Logged Models).
- `minio-data` - MinIO buckets (DVC storage, any artifact stores backed by S3).
- `evidently-data` - Evidently's workspace (projects, snapshots, tags). Added on 2026-04-13 after a rebuild wiped every Evidently snapshot for a user.

A compose-level `down -v` removes them; a normal `down` keeps them. User data is therefore safe across routine rebuilds and upgrades as long as the operator does not use `-v`.

Bind mounts handle user-editable state:

- `../data:/app/data` on the `noted` service - the project directory, documents catalog, skills, environments. All user-facing state lives here.
- `../.noted:/app/.noted` - noted's internal config (agents, view customizations).
- `../data:/app/data:ro` on `noted-graph` - read-only access to the same tree.

Airflow mounts its own dag / log / plugin directories from the repo (lines 28-33).

## 14.4 Auto-generated mount file

noted supports bind-mounting external project directories via YAML frontmatter in `data/NOTED.md` (Chapter 7.8). The compose file does not include these mounts directly - they are generated on demand.

The pattern:

1. User edits `data/NOTED.md` to add a mount: `mounts: [{name: jena_weather, host_path: /mnt/data/jena_weather}]`.
2. A helper (scripts/generate_mounts.py or similar) reads the frontmatter and writes `data/docker-compose.mounts.yml` with the corresponding `volumes:` section for the `noted` service.
3. The operator starts the stack with `docker compose -f services/docker-compose.yml -f data/docker-compose.mounts.yml up -d`.

The comment in `docker-compose.yml:59` documents this pattern: `# Include with: -f ../data/docker-compose.mounts.yml`.

This keeps the compose file itself stable - operators who do not use mounts see a clean file; operators who do get a parallel mounts file generated from their NOTED.md edits.

## 14.5 Nginx proxy

An nginx reverse proxy fronts the stack in production-like deployments. Its config (`services/nginx/nginx.conf`, mentioned in Chapter 4.3.2) handles:

- `/` -> `noted` backend (port 8123).
- `/mlflow/*` -> `mlflow` (port 5000) with static prefix rewriting.
- `/airflow/*` -> `airflow-apiserver` (port 8080) with proxy-fix headers.
- `/evidently/*` -> `evidently` (port 8000) with SPA base-path rewriting (the `sub_filter` from Chapter 4.3.2).
- `/minio/*` -> `minio` (port 9001 for the web UI).
- `/llm/*` -> the agent_server (port 7701) for the local LLM.

The proxy is what lets every service present a unified URL space under one origin (e.g. `https://logus2k.com/`) rather than forcing users to remember 7 different ports.

## 14.6 Environment variables and `.env`

The `services/.env` file holds all the service-level config the compose file references. Template is `services/.env.example`; operators copy it and fill in their values.

Critical variables:

- `NOTED_TERMINAL_SECRET` - Gate for terminal and Claude-model LLM access (Chapter 7.9, 11.3).
- `ANTHROPIC_API_KEY` - Optional; enables Claude backends.
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` - MinIO admin credentials, used by DVC's `.dvc/config` as `access_key_id` / `secret_access_key`.
- `_AIRFLOW_WWW_USER_USERNAME`, `_AIRFLOW_WWW_USER_PASSWORD` - Airflow admin credentials.
- `AIRFLOW_UID` - UID for the Airflow container's owner (defaults to 50000).
- `_PIP_ADDITIONAL_REQUIREMENTS` - extra pip installs added to Airflow workers on startup (for project-specific imports).

The compose file references these via `${VAR:-default}` syntax. Missing `.env` falls back to defaults, which works for first-boot but not for anything that needs an API key.

## 14.7 Service dependencies and startup order

Airflow has a known-complicated startup dance because its metadata DB must be migrated before workers can connect:

1. `postgres` starts, healthcheck waits for `pg_isready`.
2. `redis` starts, healthcheck waits for `redis-cli ping`.
3. `airflow-init` runs (DB migrate, create admin user), exits with status 0.
4. `airflow-apiserver`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-worker`, `airflow-triggerer` all start, waiting for `airflow-init` to complete and for postgres/redis to be healthy.

The `depends_on` + `condition: service_healthy` / `service_completed_successfully` blocks in the compose file enforce this order. `restart: always` on the long-running services handles the case where a service starts slightly too eagerly.

Other services have looser dependencies:

- `noted-graph` depends on nothing explicitly - it polls MLflow / Airflow / filesystem and degrades if any are unavailable.
- `noted-serving` depends on nothing - it lazy-loads models on demand.
- `evidently` is standalone.
- `mlflow` and `minio` are standalone.

This keeps the startup graph shallow and avoids tangling non-critical services with critical ones.

## 14.8 Healthchecks

Every long-running service has a healthcheck. A sample:

- Airflow apiserver: `curl --fail http://localhost:8080/api/v2/version`
- Airflow scheduler: `curl --fail http://localhost:8974/health`
- Postgres: `pg_isready -U airflow`
- Redis: `redis-cli ping`
- Flower: `curl --fail http://localhost:5555/`
- Airflow worker: `celery inspect ping`

`noted` itself does not define a healthcheck (the container's uvicorn process is trusted). Adding one would be a simple `curl --fail http://localhost:8123/api/system/info`.

## 14.9 Operational patterns

### Starting the stack

```bash
cd services/
docker compose up -d
```

Wait ~30-60 seconds for Airflow's init to complete. Then open `http://localhost:8123` for noted, `http://localhost:5000` for MLflow, `http://localhost:8080/airflow` for Airflow, etc.

### Starting with mounts

```bash
docker compose -f docker-compose.yml -f ../data/docker-compose.mounts.yml up -d
```

(Assuming `data/docker-compose.mounts.yml` has been generated from `NOTED.md`.)

### Rebuilding a single service

```bash
docker compose build noted          # rebuild image
docker compose up -d noted          # recreate with new image
```

For the `noted` service specifically: because `frontend/` and `backend/` are COPY'd into the image (not bind-mounted), changes require `--build`. Only `data/` is bind-mounted for live updates.

### Attaching to logs

```bash
docker compose logs -f noted        # single service
docker compose logs -f              # all services
```

### Running Airflow CLI

```bash
docker compose --profile debug run --rm airflow-cli <command>
```

The `debug` profile is necessary because the CLI container is not started by default.

### Stopping and cleanup

```bash
docker compose down                 # stop, keep volumes (user data preserved)
docker compose down -v              # stop, remove volumes (DESTRUCTIVE)
```

The `-v` form nukes every named volume. Only run it when you mean "wipe all tracked state".

### Rebuilding after changes to data/

No action required. `data/` is bind-mounted into the `noted` and `noted-graph` containers, so edits appear live. In-memory caches (e.g. the DocumentManager catalog) may need a container restart (`docker compose restart noted`) to pick up changes to `data/documents/documents.json`.

## 14.10 Resource considerations

Approximate footprint at idle:

- `noted` - 500 MB - 2 GB RAM depending on loaded kernels.
- `mlflow` - ~200 MB.
- `minio` - ~100 MB.
- `evidently` - ~200 MB.
- `noted-graph` - ~50 MB (Alpine-based, minimal deps).
- `postgres` - ~150 MB.
- `redis` - ~30 MB.
- Airflow stack (5 services) - ~1.5 GB total.
- `noted-serving` - ~500 MB - 4 GB RAM + VRAM depending on loaded model.

Total idle: ~3-4 GB RAM. Under active training, the serving container plus kernel memory can push this to 8-16 GB depending on model and batch size.

Disk: MinIO + MLflow + Airflow logs + DVC cache can grow to tens of GB over time. The named volumes live under Docker's managed storage area; `docker system df` shows usage.

## 14.11 GPU access

For CUDA-enabled training and serving:

1. Install the NVIDIA Container Toolkit on the host.
2. Add `runtime: nvidia` and `environment: NVIDIA_VISIBLE_DEVICES=all` to the services that need GPU (`noted` and `noted-serving`).
3. Verify inside the container with `nvidia-smi`.

The default `docker-compose.yml` in the repo does not include GPU settings because the compose file should work on CPU-only hosts too. A separate `docker-compose.gpu.yml` overlay can be included with `-f` for GPU deployments.

## 14.12 Discussion-ready talking points

**Q: Why Docker Compose instead of Kubernetes?**
A: Because the target deployment is a single host. Kubernetes would add etcd, an API server, kubelet, CNI, and a learning curve for a deployment that fits on one machine. Compose is the right tool for "I want these N services to run together on this host". Scaling to multi-node would justify Kubernetes; at noted's current scope, it would be a distraction.

**Q: Why so many services (13) instead of consolidating?**
A: Because each service has a well-defined boundary and an upstream image that is maintained independently. Consolidating MLflow + MinIO + Airflow into a single mega-container would mean owning each of their upgrade cycles. Keeping them separate lets the noted project focus on integration code; upstream projects focus on their own code; compose handles the orchestration.

**Q: Why is Airflow's Celery stack kept instead of running LocalExecutor?**
A: Because the DAGs in noted's target use case (Tutorial 3's `jena_training_pipeline`) are non-trivial in compute and isolating workers from the scheduler reduces crash blast radius. LocalExecutor would simplify the stack but would mean one Python process running both DAG parsing and task execution. For the demo, CeleryExecutor is a bit of overkill; for production it is the correct default.

**Q: Why does `noted` serve frontend static files itself instead of having nginx do it?**
A: Because in the default single-host deployment there is no need for a separate static-file server. FastAPI's `StaticFiles` mount is fast enough at noted's traffic volume. In a deployment with nginx, nginx can still proxy `/static/*` directly to the frontend dir if the operator wants; the noted service would continue to work.

**Q: How is horizontal scaling supposed to work?**
A: Not trivially. The noted backend holds kernel sessions, socket.io rooms, and in-process manager state - none of which are shareable across instances. Scaling out requires extracting kernels to a dedicated service, routing socket.io through Redis, and treating project state as a networked resource (Chapter 7.11). This is post-demo infrastructure work, not a compose file change.

**Q: What is the backup story?**
A: Named volumes can be backed up with `docker run --rm -v <volume>:/source -v $(pwd):/dest alpine tar czf /dest/<volume>.tar.gz -C /source .`. For a complete snapshot, stop the stack first. DVC-tracked files are already in the MinIO remote and the git repo, so they replay on next `dvc pull`. The MLflow + Evidently + Airflow + Postgres volumes are what need explicit backup.

**Q: What breaks if the host reboots mid-operation?**
A: `restart: always` policies bring services back. Running kernels are lost (they are in-process state of the noted container). Any in-flight cell execution is lost; the user has to re-execute after reconnecting. MLflow runs that were mid-stream may be left in `RUNNING` state in the DB; Airflow tasks that were running on a worker are marked failed and can be retried.

**Q: How is secret management handled?**
A: Via the `.env` file on disk. This is adequate for single-host local deployments. For production, the operator should replace it with Docker secrets, a secrets manager (Vault, AWS Secrets Manager, etc.), or at minimum an encrypted volume. noted does not force a secret management story; the `.env` is a sensible default that most small deployments can start with.

**Q: Why bind-mount `../data/` instead of copying it into the image?**
A: Because `data/` is user-facing state that changes frequently. Copying it into the image would mean every dataset edit, every new notebook, every document addition requires a rebuild. Bind-mount keeps the state on the host where git can see it and the user can edit it with their normal tools. The cost is that the image is not self-contained - it needs a matching `data/` directory at runtime - but for a developer tool that is the right trade.

**Q: What is the upgrade path?**
A: Pull the latest `docker-compose.yml` + `noted` image, run `docker compose pull && docker compose up -d`. Named volumes survive, so MLflow runs, Evidently snapshots, and Airflow history are preserved. If a migration is required (e.g. a new Postgres version), the operator follows the specific migration step listed in the release notes. This is the pattern Airflow itself uses and is well-documented in the Airflow upgrade guides.

---

**End of Volume 2. The Developer Manual v1.0 now covers both the MLOps integration chapters (Volume 1) and the platform architecture modules (Volume 2). Future revisions will add chapters as new subsystems ship or existing ones undergo significant refactors.**
