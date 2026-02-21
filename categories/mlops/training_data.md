# Section 4: Training Data

This section explores the strategies for selecting, labeling, and augmenting data to create high-quality training sets, which form the foundation of any successful ML model.

## I. Sampling Techniques
Sampling is an integral part of the ML workflow, used to select a subset of data that is feasible to process while minimizing biases.

> "Sampling happens in many steps of an ML project lifecycle, such as sampling from all possible real-world data to create training data; sampling from a given dataset to create splits for training, validation, and testing; or sampling from all possible events that happen within your ML system for monitoring purposes."

### **Probability vs. Nonprobability Sampling**
*   **Nonprobability Sampling:** Selection is not based on probability (e.g., convenience sampling, snowball sampling). While easy, these often result in selection biases.
*   **Simple Random Sampling:** Every sample has an equal chance of selection. It is easy to implement but can miss rare categories.
*   **Stratified Sampling:** The population is divided into groups (strata), and each group is sampled separately to ensure representation.
*   **Weighted Sampling:** Each sample is given a weight determining its probability of selection, allowing practitioners to leverage domain expertise or account for distribution differences.
*   **Reservoir Sampling:** Specifically useful for streaming data. It ensures every item in a stream of unknown size has an equal probability of being selected at any point.
*   **Importance Sampling:** Allows sampling from a distribution (P) when only a different, cheaper-to-sample distribution (Q) is accessible.

---

## II. The Labeling Challenge
Most production ML models are supervised, requiring labeled data. The sources highlight that labeling has evolved into a "core function" of ML teams.

### **Hand Labels vs. Natural Labels**
*   **Hand Labeling:** Often expensive, slow, and raises privacy concerns. It also faces the problem of **label multiplicity**, where different experts provide conflicting labels for the same data.
*   **Natural Labels:** Inferred automatically from user behavior (e.g., a user clicking a recommendation). These are common in recommender systems and time-of-arrival estimations.

> "The canonical example of tasks with natural labels is recommender systems... A recommendation that gets clicked on can be presumed to be good (i.e., the label is POSITIVE) and a recommendation that doesn’t get clicked on after a period of time... can be presumed to be bad (i.e., the label is NEGATIVE)."

### **Feedback Loop Length**
The time between a prediction and its feedback is the feedback loop length. Short loops (minutes) allow for rapid model updates, while long loops (weeks/months, like in fraud detection) make real-time monitoring difficult.

---

## III. Handling Lack of Labels
When labels are scarce, several techniques can bridge the gap:

*   **Weak Supervision:** Uses "labeling functions" (heuristics) to programmatically generate noisy labels. Tools like **Snorkel** denoise and reweight these functions to create training sets.
*   **Semi-supervision:** Leverages a small set of "seed" labels and structural assumptions (like clusters or small perturbations) to generate more labels.
*   **Transfer Learning:** Reuses a model pretrained on a massive, data-rich task (e.g., language modeling) for a new downstream task.
*   **Active Learning:** An iterative process where the model (active learner) chooses which specific unlabeled samples would be most beneficial for a human to label next (e.g., based on uncertainty).

---

## IV. Class Imbalance
Real-world data is rarely balanced. Rare events (frauds, terminal illnesses) are often the most important to detect but have the fewest examples.

### **Challenges**
1.  **Insufficient Signal:** Models have too few examples to learn minority patterns.
2.  **Simple Heuristics:** Models may "cheat" by always predicting the majority class, achieving high accuracy while failing the task.
3.  **Asymmetric Costs:** A false negative in cancer detection is far costlier than a false positive.

### **Techniques for Handling Imbalance**
*   **Metrics:** Move beyond accuracy to **F1, Precision, Recall, and AUC-ROC** to better understand performance on minority classes.
*   **Data-level (Resampling):** **Oversampling** (adding copies of minority samples or using **SMOTE** to synthesize them) or **Undersampling** (removing majority samples, e.g., via **Tomek links**).
*   **Algorithm-level:** Modifying the loss function. Examples include **Cost-sensitive learning** (higher penalties for certain errors) and **Focal loss** (giving higher weights to samples the model finds difficult).

---

## V. Data Augmentation
Techniques used to increase training data volume and improve model robustness against noise or adversarial attacks.

> "Augmented data can make our models more robust to noise and even adversarial attacks."

*   **Simple Transformations:** Randomly cropping, flipping, or rotating images; replacing words with synonyms in text.
*   **Perturbation:** Adding small amounts of noise. In vision, changing a single pixel can sometimes "fool" a model (adversarial attack). Training on these perturbed samples makes the model more robust.
*   **Data Synthesis:** Creating entirely new samples. In NLP, this can involve templates. In vision, **mixup** combines two images and their labels to generate continuous training data.
