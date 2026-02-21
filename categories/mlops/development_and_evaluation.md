# Section 6: Model Development and Offline Evaluation

This section covers the iterative process of selecting, training, and debugging machine learning models, as well as the specialized techniques required for distributed training, AutoML, and thorough offline evaluation.

## I. Model Development and Training

The modeling phase is where the work put into data and feature engineering is transformed into a system that can generate predictions.

> "For me, this has always been the most fun step, as it allows me to play around with different algorithms and techniques, even the latest ones. This is also the first step where I can see all the hard work I’ve put into data and feature engineering transformed into a system whose outputs (predictions) I can use to evaluate the success of my effort."

### **Six Tips for Model Selection**
When choosing an algorithm, practitioners should look beyond raw accuracy and consider factors like compute cost, latency, and interpretability.

1.  **Avoid the State-of-the-Art (SOTA) Trap:** 
    > "Researchers often only evaluate models in academic settings, which means that a model being state of the art often means that it performs better than existing models on some static datasets. It doesn’t mean that this model will be fast enough or cheap enough for you to implement."
2.  **Start with the Simplest Models:** Simple models are easier to deploy, serve as strong baselines, and are easier to debug.
3.  **Avoid Human Biases in Selection:** Different engineers may have preferences for certain architectures, leading them to tune those models more extensively.
4.  **Evaluate Performance Now vs. Later:** Some models (like neural networks) may perform worse than others (like tree-based models) with small data but scale much better as more data is collected.
5.  **Evaluate Trade-offs:** Consider accuracy versus latency and false positives versus false negatives.
6.  **Understand Model Assumptions:** 
    > "The statistician George Box said in 1976 that 'all models are wrong, but some are useful.' The real world is intractably complex, and models can only approximate using assumptions."

---

## II. Ensembles

Ensembling combines multiple "base learners" to produce a stronger final prediction. This is a common strategy in ML competitions but can add significant complexity in production.

> "Ensembling combines 'multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.' While it can give your ML system a small performance improvement, ensembling tends to make a system too complex to be useful in production, e.g., slower to make predictions or harder to interpret the results."

*   **Bagging (Bootstrap Aggregating):** Samples data with replacement to train different models independently. **Random Forest** is a classic example.
*   **Boosting:** An iterative method where subsequent models focus on the errors of previous models. **XGBoost** and **LightGBM** are popular implementations.
*   **Stacking:** A meta-learner is trained to combine the predictions of several base learners.

---

## III. Experiment Tracking and Versioning

Systematically logging every experiment is crucial for reproducibility and debugging.

> "The process of tracking the progress and results of an experiment is called experiment tracking. The process of logging all the details of an experiment for the purpose of possibly recreating it later or comparing it with other experiments is called versioning. These two go hand in hand with each other."

Data versioning remains more challenging than code versioning because of the sheer size of datasets and the complexity of identifying "diffs" in binary files.

---

## IV. Debugging ML Models

ML models fail uniquely because they often fail "silently"—producing incorrect results while appearing to run normally.

> "Debugging ML models is hard because of their cross-functional complexity. There are many components in an ML system: data, labels, features, ML algorithms, code, infrastructure, etc... When an error occurs, it could be because of any of these components or a combination of them."

### **Debugging Techniques**
*   **Start Simple:** Build a tiny version of the model first and add complexity step-by-step.
*   **Overfit a Single Batch:** 
    > "After you have a simple implementation of your model, try to overfit a small amount of training data and run evaluation on the same data to make sure that it gets to the smallest possible loss... If it can’t overfit a small amount of data, there might be something wrong with your implementation."
*   **Set a Random Seed:** This ensures consistency across different runs.

---

## V. Distributed Training

Modern models often require splitting workloads across multiple machines using either **Data Parallelism** or **Model Parallelism**.

*   **Data Parallelism:** The model is replicated across GPUs, but each receives a different slice of data. A major challenge is whether to use **Synchronous SGD** (wait for all workers) or **Asynchronous SGD** (update weights as gradients arrive).
*   **Model Parallelism:** Different parts of the model (e.g., specific layers) are hosted on different hardware.
*   **Pipeline Parallelism:** 
    > "The key idea is to break the computation of each machine into multiple parts. When machine 1 finishes the first part of its computation, it passes the result onto machine 2, then continues to the second part, and so on."

---

## VI. AutoML

AutoML aims to automate the time-consuming parts of model development, like hyperparameter tuning and architecture design.

> "Instead of paying a group of 100 ML researchers/engineers to fiddle with various models and eventually select a suboptimal one, why not use that money on compute to search for the optimal model?"

*   **Soft AutoML:** Focuses on hyperparameter tuning (e.g., learning rate, dropout).
*   **Hard AutoML:** Involves **Neural Architecture Search (NAS)**, where the system discovers the best arrangement of layers, and **Learned Optimizers**, which replace hand-designed rules like Adam with neural networks.

---

## VII. Four Phases of ML Development

Adopting ML should be an incremental process:
1.  **Phase 1: Before ML:** Use non-ML heuristics (chronological feeds, common averages).
2.  **Phase 2: Simple Models:** Use logistic regression or tree-based models as baselines.
3.  **Phase 3: Optimizing Simple Models:** Apply hyperparameter search and feature engineering.
4.  **Phase 4: Complex Models:** Experiment with deep learning and large-scale architectures.

---

## VIII. Model Offline Evaluation

Evaluation metrics only provide value when compared against a baseline.

> "Evaluation metrics, by themselves, mean little. When evaluating your model, it’s essential to know the baseline you’re evaluating it against."

### **Baseline Types**
*   **Random:** The score achieved by guessing.
*   **Simple Heuristic:** The performance of a basic rule-based system.
*   **Zero Rule:** Always predicting the most common class.

### **Specialized Evaluation Methods**
*   **Perturbation Tests:** Adding noise to the test set to see if the model is robust.
*   **Invariance Tests:** Ensuring changes to sensitive attributes (like race or gender) do not change the output.
*   **Model Calibration:** Ensuring that if a model predicts a 70% probability of an event, that event actually happens 70% of the time.
*   **Slice-based Evaluation:** 
    > "Slicing means to separate your data into subsets and look at your model’s performance on each subset separately. A common mistake... is that they are focused too much on coarse-grained metrics like overall F1 or accuracy on the entire data and not enough on sliced-based metrics."
