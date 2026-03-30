# noted - The Integrated MLOps Command Center

In a modern MLOps workflow, practitioners juggle half a dozen tools: notebooks for exploration, MLflow for tracking, Airflow for orchestration, DVC for data versioning, Hydra for configuration, MinIO for storage. Each with its own interface, its own mental model, its own browser tab. **noted** eliminates the context-switching. It is an integrated MLOps platform where every step of the machine learning lifecycle - from data ingestion to model serving - happens in a single, collaborative web interface. The underlying tools remain the engines. noted is the cockpit.

In this video, noted is used to demonstrate the complete MLOps lifecycle using the Jena Climate dataset for weather temperature forecasting:

1. **Infrastructure** - 12+ containers launched with a single Docker Compose command
2. **Hydra Configuration** - hierarchical YAML configs with visual composition and SHA-256 hash tracking
3. **Live Training** - real-time GRU model training with metrics streaming to MLflow
4. **Airflow Pipeline** - 4-stage DAG triggered, monitored, and logged from one interface
5. **Experiment Snapshots** - full reproducibility captured in one click
6. **Model Registry and Serving** - register, promote to champion, and run live predictions

<div class="embedded-video">
    <video controls>
        <source src="https://logus2k.com/docbro/categories/mlops/videos/emi_group3_tutorial2_demo.mp4" type="video/mp4">
    </video>
</div>
