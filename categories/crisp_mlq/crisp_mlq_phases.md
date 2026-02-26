# CRISP-ML(Q) Phases

The **CRISP-ML(Q)** process model establishes a comprehensive quality assurance (QA) methodology for machine learning applications across six distinct phases. This methodology focuses on identifying and mitigating technical risks that can affect project success.

The following are the essential quality assurance methods for each phase:

### 1. Business and Data Understanding
This initial phase focuses on defining scope, success criteria, and verifying data quality.
*   **Measurable Success Criteria:** Define success on three levels: **business** (e.g., failure rate < 3%), **ML** (e.g., accuracy > 97%), and **economic** (e.g., KPI for cost savings).
*   **Feasibility Assessment:** Conduct literature searches for similar applications and use a **Proof of Concept (PoC)** to demonstrate feasibility before full development.
*   **Data Quality Verification:** Perform data exploration and visualization to understand the data generation process. Define a **schema** for data requirements (e.g., expected ranges, format) and verify all data against it, discarding non-plausible samples.
*   **Data Version Control:** Use version control for datasets to ensure **reproducibility** and track errors during development.

### 2. Data Preparation
This phase aims to produce a high-quality dataset for modeling.
*   **Feature Selection:** Discard underutilized features to avoid the "curse of dimensionality" and instability. Use **domain expert analysis** to identify and remove features that might cause spurious correlations.
*   **Balanced Sampling:** When classes are skewed, use over-sampling or under-sampling techniques and compare their results to minimize introducing bias.
*   **Noise and Artifact Reduction:** Use signal processing filters for noise reduction and compare different **imputation techniques** to avoid introducing substitution artifacts in missing data.
*   **Normalization:** Apply consistent normalization to both training and test sets using the same parameters.

### 3. Modeling
Modeling involves selecting and training models that satisfy performance and resource constraints.
*   **Comprehensive Quality Measures:** Evaluate models based on six properties: **performance, robustness, scalability, explainability, model complexity, and resource demand**.
*   **Baseline Comparisons:** Start with low-capacity models as baselines and validate if increasing complexity actually improves quality. Validate incorporated domain knowledge in isolation against a baseline.
*   **Reproducibility Assurance:** Document the algorithm, dataset, hyperparameters, and runtime environment. Assess **result reproducibility** by validating mean performance and variance across different random seeds.
*   **Robustness Enhancement:** Use ensemble methods to create more fault-tolerant systems and compute uncertainty estimates.

### 4. Evaluation
This phase validates the model's performance and robustness before deployment.
*   **Disjoint Testing (Blind-Test):** Hold back a dedicated test set that is never used for training or validation to measure final performance.
*   **Sliced Analysis:** Perform performance analysis on specific classes or time slices to find hidden weaknesses.
*   **Robustness Estimation:** Statistically estimate local and global robustness by adding noise or adversarial inputs to the data.
*   **Explainability Check:** Use explanation methods to ensure the model has learned plausible features rather than spurious correlations.

### 5. Deployment
Deployment addresses the practical use of the model in its target environment.
*   **Production Condition Evaluation:** Iteratively evaluate the model under incrementally increasing production conditions to identify model degradation caused by hardware or environment differences.
*   **User Acceptance Testing:** Build prototypes and run field tests with end users to examine usability and acceptance.
*   **Risk Mitigation Strategies:** Implement a **fall-back plan** (e.g., rolling back to a previous version or a rule-based system) and **software safety cages** to control model outputs.
*   **Incremental Rollout:** Use an incremental deployment strategy to minimize the impact of undetected errors.

### 6. Monitoring and Maintenance
This phase manages the lifecycle of the model to prevent performance degradation over time.
*   **Continuous Monitoring:** Compare the statistics of incoming production data and labels against the training data statistics to detect distribution shifts. Use the predefined data schema to validate incoming data and treat non-compliant inputs as anomalies.
*   **Updating Protocol:** Fine-tune models with new data when performance drops below a certain threshold. Every updated model must undergo a full evaluation (as in Phase 4) before being redeployed.
*   **Operational Controls:** Track application usage and use A/B testing for deployment. Implement automated or human-controlled fallback mechanisms to previous stable models.
