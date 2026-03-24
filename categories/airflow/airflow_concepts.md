# Apache Airflow Core Concepts

## DAG

A **DAG** (Directed Acyclic Graph) is the workflow definition. It describes what tasks exist, how they depend on each other, and when the workflow should run. No execution logic lives in the DAG itself - it is purely structural.

```python
from airflow import DAG
from datetime import datetime

with DAG(
    dag_id="my_pipeline",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    pass  # tasks go here
```

Key parameters:
- `schedule` - cron expression, preset (`@daily`), timedelta, or Asset list
- `catchup` - whether to backfill missed runs since `start_date`
- `default_args` - dict of args cascaded to all tasks (e.g. `retries`, `email_on_failure`)

---

## Operator

An **Operator** is a class (template) defining *how* to perform a unit of work. You never run an operator directly - you instantiate it to create a Task.

Airflow ships with many built-in operators:

```python
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# BashOperator - runs a shell command
run_script = BashOperator(
    task_id="run_script",
    bash_command="echo 'hello' && python /opt/scripts/process.py",
)

# PythonOperator - calls a Python function
def my_function(**context):
    print(f"Running for logical date: {context['logical_date']}")

run_fn = PythonOperator(
    task_id="run_fn",
    python_callable=my_function,
)

# EmptyOperator - no-op, useful as a gate or checkpoint
start = EmptyOperator(task_id="start")
```

Provider packages extend this with `SparkSubmitOperator`, `BigQueryOperator`, `S3CopyObjectOperator`, etc.

---

## Task

A **Task** is a specific instance of an Operator bound to a DAG with a `task_id`. It is the node in the graph. Dependencies are declared between tasks using `>>` / `<<` or `.set_downstream()`.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

def extract():
    return {"rows": 42}

def transform(**context):
    data = context["ti"].xcom_pull(task_ids="extract")
    print(f"Transforming {data['rows']} rows")

with DAG("etl", schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False) as dag:
    t_extract = PythonOperator(task_id="extract", python_callable=extract)
    t_transform = PythonOperator(task_id="transform", python_callable=transform)
    t_load = BashOperator(task_id="load", bash_command="echo 'loading...'")

    # Declare dependencies
    t_extract >> t_transform >> t_load
```

The mental model: **Operator = class, Task = instance, Task Instance = one execution at a specific logical date.**

---

## TaskFlow API

The modern way to write tasks in Airflow 2+. The `@task` decorator wraps a Python function as a `PythonOperator` and handles **XCom passing automatically** via return values and function arguments.

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False)
def etl_pipeline():

    @task
    def extract() -> dict:
        return {"rows": [1, 2, 3]}

    @task
    def transform(data: dict) -> dict:
        return {"processed": [x * 2 for x in data["rows"]]}

    @task
    def load(data: dict):
        print(f"Loading: {data['processed']}")

    # Data flow infers dependencies automatically
    raw = extract()
    processed = transform(raw)
    load(processed)

etl_pipeline()
```

No explicit `>>` needed - Airflow infers the dependency graph from the data flow.

---

## XCom

**XCom** (cross-communication) is the mechanism for passing small values between tasks within the same DAG run. Values are stored in Airflow's metadata database.

```python
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from datetime import datetime

# --- TaskFlow style (automatic) ---
@dag(schedule=None, start_date=datetime(2025, 1, 1))
def xcom_example():

    @task
    def produce() -> str:
        return "hello from upstream"

    @task
    def consume(value: str):
        print(f"Got: {value}")  # prints "hello from upstream"

    consume(produce())

# --- Classic style (manual push/pull) ---
def push_fn(**context):
    context["ti"].xcom_push(key="my_key", value={"status": "ok"})

def pull_fn(**context):
    val = context["ti"].xcom_pull(task_ids="push_task", key="my_key")
    print(val)  # {"status": "ok"}
```

> **Important:** XCom is for small metadata (IDs, counts, file paths). Never push large payloads - store data in S3/MinIO and push the reference.

---

## Asset (formerly Dataset)

An **Asset** is a logical URI representing a data entity (file, table, topic). Assets enable **data-driven scheduling**: a DAG can trigger when another DAG marks an asset as updated, decoupling pipelines through data lineage rather than explicit cross-DAG dependencies.

```python
from airflow import DAG, Dataset
from airflow.decorators import task
from datetime import datetime

# Define the asset
processed_data = Dataset("s3://my-bucket/processed/data.parquet")

# --- Producer DAG ---
with DAG("producer", schedule="@hourly", start_date=datetime(2025, 1, 1)) as producer_dag:

    @task(outlets=[processed_data])  # marks the asset as updated on success
    def produce():
        # ... write to s3://my-bucket/processed/data.parquet
        print("Data written")

    produce()

# --- Consumer DAG - runs automatically when processed_data is updated ---
with DAG("consumer", schedule=[processed_data], start_date=datetime(2025, 1, 1)) as consumer_dag:

    @task
    def consume():
        print("Consuming updated data")

    consume()
```

---

## DAG Run and Task Instance

A **DAG Run** is one execution of a DAG, tied to a `logical_date`. A **Task Instance** is one execution of a Task within a DAG Run. These are the runtime objects Airflow tracks in its metadata DB.

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False)
def run_context_demo():

    @task
    def show_context(**context):
        ti = context["ti"]
        print(f"DAG Run ID:    {ti.run_id}")
        print(f"Logical date:  {context['logical_date']}")
        print(f"Task ID:       {ti.task_id}")
        print(f"Try number:    {ti.try_number}")

    show_context()

run_context_demo()
```

Task Instances have states: `queued`, `running`, `success`, `failed`, `skipped`, `up_for_retry`.

---

## Sensor

A **Sensor** is a special operator that polls a condition repeatedly until it is met (or times out). Used to wait for external events before proceeding.

```python
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG("sensor_example", schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False) as dag:

    wait_for_file = FileSensor(
        task_id="wait_for_file",
        filepath="/data/input/ready.flag",
        poke_interval=30,   # check every 30 seconds
        timeout=3600,       # fail after 1 hour
        mode="reschedule",  # release the worker slot between pokes
    )

    def process():
        print("File is ready, processing...")

    process_file = PythonOperator(task_id="process", python_callable=process)

    wait_for_file >> process_file
```

`mode="reschedule"` is preferred over `mode="poke"` in production - it frees the worker slot between checks.

---

## Pool

A **Pool** limits concurrency for a named group of tasks - useful when tasks share a constrained resource (DB connections, API rate limits, GPU slots).

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Pool "db_pool" with 5 slots is created in the Airflow UI or via CLI:
# airflow pools set db_pool 5 "Max 5 concurrent DB tasks"

with DAG("pool_example", schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False) as dag:

    tasks = []
    for i in range(20):
        t = PythonOperator(
            task_id=f"query_{i}",
            python_callable=lambda: print("running query"),
            pool="db_pool",        # only 5 of these run at once
            pool_slots=1,          # this task consumes 1 slot
        )
        tasks.append(t)
```

---

## Connection and Hook

A **Connection** stores credentials and endpoint config (host, login, password, extras) in Airflow's secret store, referenced by a `conn_id`. A **Hook** is a client class that uses a Connection to interact with an external system - operators delegate to hooks internally, but you can use them directly in tasks too.

```python
from airflow.decorators import dag, task
from airflow.hooks.base import BaseHook
from datetime import datetime

@dag(schedule=None, start_date=datetime(2025, 1, 1))
def hook_example():

    @task
    def check_connection():
        # Retrieve connection stored in Airflow (UI or env var AIRFLOW_CONN_MY_DB)
        conn = BaseHook.get_connection("my_db")
        print(f"Host: {conn.host}, Schema: {conn.schema}")

    @task
    def query_db():
        # Using a typed hook directly
        from airflow.providers.postgres.hooks.postgres import PostgresHook
        hook = PostgresHook(postgres_conn_id="my_db")
        records = hook.get_records("SELECT count(*) FROM events")
        print(records)

    check_connection() >> query_db()

hook_example()
```

Connections can be stored in the metadata DB (UI), environment variables (`AIRFLOW_CONN_<CONN_ID>`), or a secrets backend (Vault, AWS SSM, etc.).

---

## Variable

**Variables** are key-value config pairs stored in Airflow's metadata DB (or a secrets backend). Useful for environment-specific config that should not be hardcoded in DAG files.

```python
from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import datetime

@dag(schedule=None, start_date=datetime(2025, 1, 1))
def variable_example():

    @task
    def use_variable():
        # Simple string
        env = Variable.get("environment", default_var="dev")

        # JSON value, deserialized automatically
        config = Variable.get("pipeline_config", deserialize_json=True)
        # e.g. {"batch_size": 1000, "retries": 3}

        print(f"Running in {env} with batch size {config['batch_size']}")

    use_variable()

variable_example()
```

> Avoid calling `Variable.get()` at module level (outside tasks) - DAG files are parsed frequently by the scheduler, and each parse would hit the DB.

---

## Concept Relationships

```
DAG  (workflow definition + schedule)
 |
 +-- Task  (Operator instance, node in the graph)
 |    |
 |    +-- Task Instance  (one execution at a specific logical_date)
 |         |
 |         +-- XCom  (passes values between Task Instances in the same Run)
 |
 +-- DAG Run  (one execution of the full DAG)

Asset  (URI representing a data entity)
 +-- produced by Tasks (via outlets=[...])
 +-- consumed as schedule trigger by other DAGs

Pool       -> limits concurrency across Tasks
Connection -> stores credentials, used by Hooks
Hook       -> typed client for an external system, used by Operators
Variable   -> runtime config key-value store
```
