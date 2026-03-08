# Project Overview - Integrated MLOps Platform

**noted** is a collaborative, web-based MLOps platform designed to unify the fragmented landscape of machine learning tools into a **single, integrated interface**. It evolves from an interactive notebook environment into a comprehensive cockpit for managing the full ML lifecycle—from raw data ingestion to production deployment.

<div class="mxgraph"
     data-mxgraph='{
        "url": "https://logus2k.com/docbro/categories/noted/diagrams/noted_architecture.drawio",
        "lightbox": false,
        "nav": true,
        "resize": false,
        "auto-fit": true,
        "auto-crop": true
     }'>
</div>

#### **1. Core Vision and Problem Statement**
Currently, ML practitioners must context-switch between disparate tools: DVC for data, MLflow for tracking, Airflow for orchestration, and various consoles for storage and serving. **noted** eliminates this friction by integrating these canonical engines into a unified experience while ensuring the backend services remain the **authoritative sources of truth**. 

#### **2. Existing Foundation**
The project is built upon a functional Proof of Concept (PoC) that already supports:
*   **Real-time collaboration** with cell-level locking and Socket.io presence.
*   **Multi-runtime kernel execution** supporting Python 3.10–3.14.
*   **Environment management** with isolated virtual environments and PTY-streaming package installation.
*   **GPU acceleration** via CUDA runtime and LD_LIBRARY_PATH injection.

#### **3. Key Capabilities (In Scope)**
The platform is expanding to include five primary MLOps domains:
*   **Data Versioning (DVC + Git):** Backend-managed Git repositories automate dataset tracking without exposing Git complexity to users.
*   **Experiment Tracking (MLflow):** Live metrics streaming, run comparisons, and automatic instrumentation are integrated directly into the notebook sidebar.
*   **Configuration Management (Hydra):** Dynamic UI forms are generated from Hydra structured configs to allow type-validated, swappable experiment parameters.
*   **Pipeline Orchestration (Airflow):** Users can generate Airflow DAGs from project metadata and monitor execution status through real-time node graphs.
*   **Model Registry & Serving:** Supports promoting models to `@champion` status and testing predictions via an integrated FastAPI serving container.

#### **4. Development Roadmap**
The project follows a phased delivery plan to ensure incremental value:
*   **Phase 0: Infrastructure Verification** – Confirming interoperability between the core container and backend services like MinIO, PostgreSQL, and Airflow.
*   **Phase 1: Tracking and Data** – Delivering high-value experiment tracking and dataset versioning.
*   **Phase 2: Configuration and Orchestration** – Transitioning from interactive tools to production-grade pipeline management.
*   **Phase 3: Registry and Serving** – Completing the data-to-deployment lifecycle with model governance.
*   **Phase 4: Integration and Polish** – Adding an activity feed, cross-service event correlation, and end-to-end workflow refinement.

#### **5. Technical Architecture**
The platform utilizes a **single-container proxy pattern**. The **noted** backend acts as the sole intermediary between the vanilla ES6 frontend and the backend services (MLflow, Airflow, MinIO, etc.), ensuring centralized security and atomic cross-service operations.
