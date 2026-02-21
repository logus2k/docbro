# Section 5: Feature Engineering

This section focuses on the transition from raw data to the inputs required by ML models. It covers the techniques for creating features, handling data quality issues, and the critical problem of data leakage.

## I. Importance of Features
Despite the rise of complex architectures, the quality of features remains the most significant factor in model performance.

> "In 2014, the paper 'Practical Lessons from Predicting Clicks on Ads at Facebook' claimed that having the right features is the most important thing in developing their ML models. Since then, many of the companies that I’ve worked with have discovered time and time again that once they have a workable model, having the right features tends to give them the biggest performance boost compared to clever algorithmic techniques such as hyperparameter tuning."

## II. Learned Features Versus Engineered Features
The promise of deep learning is often to eliminate manual feature engineering, but in production, handcrafted features are still prevalent.

> "The promise of deep learning is that we won’t have to handcraft features. For this reason, deep learning is sometimes called feature learning. Many features can be automatically learned and extracted by algorithms. However, we’re still far from the point where all features can be automated."

### **Limitations of Automatic Extraction**
While text and images can often be processed directly by neural networks, most enterprise applications require additional context (metadata about users, threads, or transactions) that must be engineered.

> "An ML system will likely need data beyond just text and images... The process of choosing what information to use and how to extract this information into a format usable by your ML models is feature engineering."

---

## III. Common Feature Engineering Operations

### **1. Handling Missing Values**
Not all missing data is created equal. Understanding the cause of missingness is crucial for selecting the right handling strategy.

*   **Missing Not at Random (MNAR):** 
    > "This is when the reason a value is missing is because of the true value itself."
*   **Missing at Random (MAR):** 
    > "This is when the reason a value is missing is not due to the value itself, but due to another observed variable."
*   **Missing Completely at Random (MCAR):** 
    > "This is when there’s no pattern in when the value is missing... However, this type of missing is very rare."

Strategies include **Deletion** (risky for MAR/MNAR) and **Imputation** (filling with defaults, mean, or median).

> "In general, you want to avoid filling missing values with possible values, such as filling the missing number of children with 0—0 is a possible value for the number of children. It makes it hard to distinguish between people whose information is missing and people who don’t have children."

### **2. Scaling**
Models struggle when features have vastly different ranges. Scaling ensures features contribute proportionally to the learning process.

> "Before inputting features into models, it’s important to scale them to be similar ranges. This process is called feature scaling. This is one of the simplest things you can do that often results in a performance boost for your model."

Common methods include **Rescaling** (min-max), **Standardization** (zero mean, unit variance), and **Log Transformation** (to handle skewed distributions).

### **3. Discretization**
Also known as binning, this process turns continuous features into categories to make patterns easier to learn.

> "Discretization is the process of turning a continuous feature into a discrete feature. This process is also known as quantization or binning... Instead of having to learn an infinite number of possible incomes, our model can focus on learning only three categories, which is a much easier task to learn."

### **4. Encoding Categorical Features**
A major challenge in production is the "unknown" category—new brands, users, or domains that appear after training.

> "In production, categories change... your model crashes because it encounters a brand it hasn’t seen before and therefore can’t encode... One solution to this problem is the hashing trick... you use a hash function to generate a hashed value of each category. The hashed value will become the index of that category."

### **5. Feature Crossing**
This technique combines multiple features to model nonlinear relationships.

> "Feature crossing is the technique to combine two or more features to generate new features. This technique is useful to model the nonlinear relationships between features."

### **6. Positional Embeddings**
Essential for parallel processing architectures like Transformers, where the sequence order is not implicit.

> "If we use a model like a transformer, words are processed in parallel, so words’ positions need to be explicitly inputted so that our model knows the order of these words ('a dog bites a child' is very different from 'a child bites a dog')."

---

## IV. Data Leakage
Data leakage is a disastrous problem that makes models appear highly accurate during development but causes them to fail spectacularly in production.

> "Data leakage refers to the phenomenon when a form of the label 'leaks' into the set of features used for making predictions, and this same information is not available during inference."

### **Common Causes of Leakage**
*   **Time-correlated data:** 
    > "To prevent future information from leaking into the training process and allowing models to cheat during evaluation, split your data by time, instead of splitting randomly, whenever possible."
*   **Scaling/Imputation before splitting:** 
    > "Always split your data first before scaling, then use the statistics from the train split to scale all the splits."
*   **Data Duplication:** 
    > "Data duplication is quite common in the industry... If you have duplicates or near-duplicates in your data, failing to remove them before splitting your data might cause the same samples to appear in both train and validation/test splits."
*   **Group/Process Leakage:** When data samples share correlations (like scans of the same patient) or indicators of the labeling process (like specific machine settings) that won't exist in production.

---

## V. Engineering Good Features
Having too many features creates technical debt, increases latency, and causes overfitting. Practitioners must evaluate features based on **importance** and **generalization**.

### **Feature Importance**
Techniques like **SHAP** help quantify how much each feature contributes to a model's prediction.

> "Intuitively, a feature’s importance to a model is measured by how much that model’s performance deteriorates if that feature or a set of features containing that feature is removed from the model."

### **Feature Generalization**
A feature is only useful if it appears frequently and its distribution is consistent between training and inference.

> "There are two aspects you might want to consider with regards to generalization: feature coverage and distribution of feature values... Coverage is the percentage of the samples that has values for this feature... a rough rule of thumb is that if this feature appears in a very small percentage of your data, it’s not going to be very generalizable."
