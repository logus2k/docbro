# Section 11: The Human Side of Machine Learning

This final section explores the non-technical aspects of machine learning systems, focusing on how the probabilistic nature of ML affects user experience, how teams should be structured for success, and the critical framework for developing **Responsible AI**.

## I. User Experience (UX)

ML systems differ from traditional software because they are **probabilistic rather than deterministic**. This creates unique challenges for providing a consistent and helpful user experience.

> "First, ML systems are probabilistic instead of deterministic... Second, due to this probabilistic nature, ML systems’ predictions are mostly correct, and the hard part is we usually don’t know for what inputs the system will be correct! Third, ML systems can also be large and might take an unexpectedly long time to produce a prediction."

### **1. Ensuring Consistency**
Users expect software to behave predictably. Because ML models change as they are retrained on new data, their outputs can shift, leading to user frustration.

> "ML predictions are probabilistic and inconsistent, which means that predictions generated for one user today might be different from what will be generated for the same user the next day... This is known as the consistency–accuracy trade-off, since the recommendations deemed most accurate by the system might not be the recommendations that can provide user consistency."

### **2. Combatting "Mostly Correct" Predictions**
Large models (like GPT-3) can generate impressive results that are often "mostly correct" but still contain errors. This is useful for experts who can edit the output, but dangerous for non-experts.

> "These mostly correct predictions can be useful for users who can easily correct them... However, these mostly correct predictions won’t be very useful if users don’t know how to or can’t correct the responses... To overcome this, an approach is to show users multiple resulting predictions for the same input to increase the chance of at least one of them being correct."

### **3. Smooth Failing**
When a complex model takes too long to generate a prediction, a backup system should be in place to ensure the user isn't left waiting.

> "Some companies... use a backup system that is less optimal than the main system but is guaranteed to generate predictions quickly. These systems can be heuristics or simple models. They can even be cached precomputed predictions."

---

## II. Team Structure

Building ML systems requires a mix of data science, DevOps, and domain expertise. The way an organization structures these roles determines its iteration speed and system reliability.

### **1. Cross-functional Collaboration**
Subject Matter Experts (SMEs) such as doctors or lawyers must be involved throughout the entire project lifecycle, not just for data labeling.

> "An ML system would benefit a lot to have SMEs involved in the rest of the lifecycle, such as problem formulation, feature engineering, error analysis, model evaluation, reranking predictions, and user interface... It’s important to involve SMEs early on in the project planning phase and empower them to make contributions without having to burden engineers."

### **2. The End-to-End Data Scientist**
There is a major debate over whether data scientists should own the process from development to production ("Approach 2") or hand off models to a separate engineering team ("Approach 1").

*   **Approach 1 (Separate Teams):** Leads to communication overhead, debugging challenges, and "finger-pointing."
*   **Approach 2 (End-to-End):** Can lead to "grumpy unicorns" who are overwhelmed by infrastructure boilerplate like Kubernetes and Docker.

> "In Netflix’s model, the specialists—people who originally owned a part of the project—first create tools that automate their parts... Data scientists can leverage these tools to own their projects end-to-end."

---

## III. Responsible AI

Responsible AI is the practice of ensuring AI systems are **fair, private, transparent, and accountable**.

> "Responsible AI is the practice of designing, developing, and deploying AI systems with good intention and sufficient awareness to empower users, to engender trust, and to ensure fair and positive impact to society."

### **Lessons from Irresponsible AI (Case Studies)**
1.  **Ofqual's Grading Algorithm:** Failed by choosing the wrong objective (school-level fairness over student-level fairness) and lacking transparency.
2.  **Strava Heatmap:** Exposed sensitive military base locations despite data being "anonymized," showing that anonymization alone is often insufficient for privacy.

### **A Framework for Responsible AI**
Practitioners should follow these steps to ensure their systems do more good than harm:

*   **Discover Bias Sources:** Evaluate training data, labeling processes, feature engineering (watch for **disparate impact**), and model objectives for hidden biases.
*   **Understand Trade-offs:** Be aware that improving one aspect (like privacy or model size) can harm another (like accuracy for minority groups).
    *   *Privacy vs. Accuracy:* "the accuracy of differential privacy models drops much more for the underrepresented classes and subgroups."
    *   *Compactness vs. Fairness:* "compression techniques [like pruning] amplify algorithmic harm when the protected feature... is in the long tail of the distribution."
*   **Act Early:** "The earlier in the development cycle... that you can start thinking about how this system will affect the life of users... the cheaper it will be to address these biases."
*   **Create Model Cards:** Use standardized documents to report model details, intended use, evaluation data, and ethical considerations.

> "Model cards are short documents accompanying trained ML models that provide information on how these models were trained and evaluated... [and] standardize ethical practice and reporting by allowing stakeholders to compare candidate models."
