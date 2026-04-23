# Foundational Concepts for Understanding the Project Goal

This document explains, one by one, every technical concept referenced in the project's introduction and goal statement. It is meant to be read before studying the notebook itself, so that the terminology feels familiar when you encounter it in context.

---

## Table of Contents

1. [Time Series Forecasting](#1-time-series-forecasting)
2. [The Jena Climate Dataset](#2-the-jena-climate-dataset)
3. [Multivariate-Input](#3-multivariate-input)
4. [Univariate-Output](#4-univariate-output)
5. [Multi-Step Prediction](#5-multi-step-prediction)
6. [Putting It Together: The Forecasting Task](#6-putting-it-together-the-forecasting-task)
7. [Lookback Window and Forecast Horizon](#7-lookback-window-and-forecast-horizon)
8. [Baseline Model](#8-baseline-model)
9. [GRU -- Gated Recurrent Unit](#9-gru----gated-recurrent-unit)
10. [Regularization](#10-regularization)
11. [Gradient Clipping](#11-gradient-clipping)
12. [AdamW Optimizer](#12-adamw-optimizer)
13. [Huber Loss](#13-huber-loss)
14. [Hyperparameters vs. Parameters](#14-hyperparameters-vs-parameters)
15. [Evolutionary Optimization / Genetic Algorithm](#15-evolutionary-optimization--genetic-algorithm)
16. [Search Space and Configuration Space](#16-search-space-and-configuration-space)
17. [Population, Individual, Generation](#17-population-individual-generation)
18. [Synthetic Data Generation](#18-synthetic-data-generation)
19. [TimeGAN](#19-timegan)
20. [Data Augmentation and Information Leakage](#20-data-augmentation-and-information-leakage)
21. [Explainable AI (XAI)](#21-explainable-ai-xai)
22. [Permutation Importance (Global Explainability)](#22-permutation-importance-global-explainability)
23. [Gradient Saliency (Local Explainability)](#23-gradient-saliency-local-explainability)
24. [Feature Pruning](#24-feature-pruning)
25. [Efficiency and Resource Analysis](#25-efficiency-and-resource-analysis)
26. [Accuracy--Efficiency Trade-offs](#26-accuracyefficiency-trade-offs)
27. [End-to-End Pipeline](#27-end-to-end-pipeline)

---

## 1. Time Series Forecasting

A **time series** is a sequence of data points collected at successive, equally-spaced moments in time. Examples: hourly temperature readings, daily stock prices, monthly sales figures.

**Time series forecasting** is the task of using past observations to predict future values. Unlike standard regression where samples are independent, time series data has **temporal dependencies** -- what happened an hour ago strongly influences what happens now.

In this project, the time series is a set of meteorological measurements recorded every hour, and the goal is to forecast future air temperature.

---

## 2. The Jena Climate Dataset

The dataset used in this project. It was collected by the Max Planck Institute for Biogeochemistry in Jena, Germany, and contains meteorological measurements (temperature, pressure, humidity, wind, etc.) recorded approximately every 10 minutes from 2009 to 2016.

Key facts:
- **~420,000 rows** at 10-minute resolution (resampled to ~70,000 hourly rows in the notebook)
- **15 original variables** (reduced to 6 for this project)
- Publicly available and widely used as a benchmark for time series deep learning

---

## 3. Multivariate-Input

"Multivariate" means **multiple variables**. A multivariate input means the model receives several different measurements at each time step, not just one.

In this project, at every hour the model sees 6 raw meteorological measurements (plus derived features, totalling 16 input dimensions):

| Variable | What it measures |
|---|---|
| T (degC) | Air temperature |
| p (mbar) | Atmospheric pressure |
| rh (%) | Relative humidity |
| wv (m/s) | Wind speed |
| max. wv (m/s) | Maximum wind speed |
| wd (deg) | Wind direction |

**Why multivariate?** Temperature does not evolve in isolation. A drop in pressure or a shift in wind direction can signal an incoming weather change. By feeding all these variables, the model can learn cross-variable patterns that help it make better temperature predictions.

**Contrast -- univariate input:** If the model only received past temperature values and nothing else, that would be a univariate input.

---

## 4. Univariate-Output

Even though the model receives multiple variables as input, it only needs to predict **one** variable as output: air temperature (`T (degC)`).

This is called **univariate output** -- the prediction target is a single quantity.

**Contrast -- multivariate output:** If the model had to simultaneously predict temperature, pressure, and humidity, that would be a multivariate output problem.

---

## 5. Multi-Step Prediction

Instead of predicting just the next single hour, the model predicts the **next 24 hours** all at once. Each of those 24 future values is a "step", so this is a **multi-step** prediction.

```
Input: hours [t-119, t-118, ..., t]    -->    Output: hours [t+1, t+2, ..., t+24]
       (120 past time steps)                          (24 future time steps)
```

**Contrast -- single-step prediction:** Predicting only `t+1` (the next hour) would be a single-step prediction. Multi-step is harder because errors can compound over the horizon.

---

## 6. Putting It Together: The Forecasting Task

Combining the three terms:

> **Multivariate-input, univariate-output, multi-step prediction**

Means: the model takes a window of historical data with **many sensor channels** (multivariate input), and produces a sequence of future predictions for **one target variable** (univariate output) across **multiple future time steps** (multi-step).

Concretely in this project:
- **Input:** 120 hours x 16 features = a matrix of shape (120, 16)
- **Output:** 24 future temperature values = a vector of shape (24,)

---

## 7. Lookback Window and Forecast Horizon

These two terms define the "shape" of the forecasting problem:

**Lookback window (LOOKBACK):** How many past time steps the model can see. In the baseline, LOOKBACK = 120 hours (5 days of history). The evolutionary search also tested 96 and 144 hours.

**Forecast horizon (HORIZON):** How many future time steps the model must predict. Fixed at HORIZON = 24 hours (one day ahead).

Think of it as a sliding window moving along the time series:

```
                    LOOKBACK = 120h              HORIZON = 24h
              |<------------------------->|<---------------->|
Past data:    [..... input features .......]  [temperature predictions]
              the model reads this             the model outputs this
```

---

## 8. Baseline Model

A **baseline** is a reference model that sets a minimum performance bar. It answers the question: "How well can we do with a simple approach?"

This project uses two baselines:

1. **Persistence baseline:** The simplest possible forecast -- repeat the last observed temperature for all 24 future hours. If it was 15 degC at the last observed hour, predict 15 degC for all of the next 24 hours. Any useful model must beat this.

2. **GRU baseline:** A well-tuned deep learning model carried over from a previous mini-project. This is a stronger reference point. The evolutionary optimization aims to beat this.

Baselines matter because without them you cannot tell whether a model is genuinely good or just looks impressive in isolation.

---

## 9. GRU -- Gated Recurrent Unit

A **GRU** is a type of **Recurrent Neural Network (RNN)** designed to process sequential data (like time series). Standard RNNs struggle with long sequences because gradients either vanish (become too small) or explode (become too large) during training. GRUs solve this with **gating mechanisms**.

### How a GRU works (simplified)

At each time step, the GRU receives:
- The current input (e.g., one hour of 16 meteorological features)
- A hidden state from the previous time step (the "memory")

It uses two gates:

| Gate | Purpose |
|---|---|
| **Reset gate (r)** | Decides how much of the previous memory to forget. If r is near 0, the GRU ignores the past and reacts mainly to the current input. |
| **Update gate (z)** | Decides how much of the new candidate state to mix in vs. keeping the old state. If z is near 1, the GRU keeps the old memory unchanged. |

```
r = sigmoid(W_r * [h_prev, x])          <-- how much past to forget
z = sigmoid(W_z * [h_prev, x])          <-- how much new info to let in
h_candidate = tanh(W * [r * h_prev, x]) <-- new candidate memory
h_new = (1 - z) * h_prev + z * h_candidate  <-- final updated memory
```

### Why GRU for time series?

- It can learn **long-range dependencies** (e.g., seasonal patterns spanning days)
- It is **lighter than LSTM** (a related architecture) because it has 2 gates instead of 3, so it trains faster
- It is well-suited for moderate-length sequences like the 120-hour windows in this project

### Multi-layer GRU

The baseline uses **2 stacked GRU layers** (96 units, then 64 units). The first layer processes the raw input sequence and passes its output sequence to the second layer, which captures higher-level temporal patterns. The final hidden state is then passed through a dense layer to produce the 24-hour forecast.

---

## 10. Regularization

**Regularization** is any technique that prevents a model from **overfitting** -- performing well on training data but poorly on unseen data. An overfitting model memorizes noise rather than learning general patterns.

Regularization techniques used in this project:

| Technique | How it works |
|---|---|
| **Dropout** | During training, randomly sets a fraction of neuron outputs to zero at each step. Forces the network to not rely on any single neuron. Value range in this project: 0.0--0.3 (the EA found 0.0 was best). |
| **L2 regularization** (weight decay) | Adds a penalty proportional to the squared magnitude of weights to the loss function. Discourages large weights, keeping the model simpler. Formula: `Loss_total = Loss_original + lambda * sum(w^2)` |
| **Gaussian noise** | Adds small random noise to layer inputs during training. Acts as a form of data augmentation at the feature level. The EA found 0.0 was best (no noise needed). |
| **Early stopping** | Monitors validation loss during training and stops when it stops improving (after a "patience" number of epochs). Prevents the model from training too long and overfitting. |

---

## 11. Gradient Clipping

During training, the model adjusts its weights using **gradients** (the direction and magnitude of the loss function's slope with respect to each weight). Sometimes gradients become extremely large -- this is called **gradient explosion** and causes unstable training (weights swing wildly).

**Gradient clipping** caps the gradient norm at a maximum value. If the gradient vector's length exceeds the threshold, it is scaled down proportionally.

```
If ||gradient|| > clipnorm:
    gradient = gradient * (clipnorm / ||gradient||)
```

In the baseline, `clipnorm = 2.0`. The EA searched values from 0.5 to 5.0 and found `clipnorm = 5.0` worked best for its configuration.

---

## 12. AdamW Optimizer

An **optimizer** is the algorithm that updates the model's weights during training based on computed gradients.

**Adam** (Adaptive Moment Estimation) is the most popular optimizer in deep learning. It maintains two running averages for each weight:
- The **mean** of recent gradients (momentum -- which direction to go)
- The **variance** of recent gradients (scaling -- how big a step to take per weight)

This gives each weight its own adaptive learning rate, so parameters that need larger updates get them and vice versa.

**AdamW** is a variant that fixes how weight decay (L2 regularization) is applied. In standard Adam, weight decay gets entangled with the adaptive learning rate in unintended ways. AdamW **decouples** them, applying weight decay directly to the weights rather than through the gradient. This leads to better regularization behavior.

**Learning rate:** Controls the overall step size. Too large = unstable; too small = slow convergence. The baseline uses `2e-4` (0.0002); the EA found `3e-4` (0.0003) worked better.

---

## 13. Huber Loss

A **loss function** measures how far the model's predictions are from the true values. The optimizer minimizes this function.

Common choices:
- **MSE (Mean Squared Error):** Squares each error. Heavily penalizes large errors.
- **MAE (Mean Absolute Error):** Takes the absolute value. Treats all errors linearly.

**Huber loss** is a hybrid:

```
For each prediction error e:
    If |e| <= delta:  loss = 0.5 * e^2       (behaves like MSE -- smooth near zero)
    If |e| > delta:   loss = delta * (|e| - 0.5 * delta)  (behaves like MAE -- robust to outliers)
```

The parameter **delta** controls the transition point. The baseline uses `delta = 1` ("huber1").

**Why Huber?** It gets the best of both worlds: smooth gradients near zero (like MSE) for precise optimization, and robustness to outliers (like MAE) for stability. However, the EA discovered that plain **MAE loss** worked better for this task, likely because MAE directly aligns with the evaluation metric.

---

## 14. Hyperparameters vs. Parameters

This distinction is fundamental:

| | Parameters | Hyperparameters |
|---|---|---|
| **What** | The weights and biases inside the neural network | Settings that control the architecture and training process |
| **How they're set** | Learned automatically during training via backpropagation | Set manually by the engineer, or searched automatically (by the EA) |
| **Examples** | GRU gate weights, dense layer weights | Number of layers, units per layer, learning rate, dropout rate, batch size |
| **Count** | Baseline: 86,744 | This project: 17 hyperparameters searched by the EA |

The evolutionary algorithm in this project is searching over **hyperparameters** (and also preprocessing choices like scaler type) to find the best configuration.

---

## 15. Evolutionary Optimization / Genetic Algorithm

A **Genetic Algorithm (GA)** is an optimization method inspired by biological evolution. Instead of trying every possible combination (impossible with 17 dimensions) or relying on calculus-based optimization (which requires a differentiable objective), it uses:

### Core loop

```
1. INITIALIZE a population of random candidate solutions
2. EVALUATE each candidate's "fitness" (how good it is)
3. SELECT the fittest candidates as parents
4. CROSSOVER: combine pairs of parents to produce children
5. MUTATE: randomly alter some genes in the children
6. REPEAT from step 2 for N generations
```

### Operators used in this project

| Operator | Description |
|---|---|
| **Tournament selection** (k=3) | Pick 3 random individuals, keep the best one. Repeat to fill the parent pool. Balances exploration and exploitation. |
| **Uniform crossover** | For each gene (hyperparameter), randomly pick from parent A or parent B with 50/50 chance. Produces children that mix traits from both parents. |
| **Per-gene mutation** (rate=0.2) | Each gene has a 20% chance of being replaced with a new random value from the search space. Introduces diversity to avoid getting stuck. |
| **Elitism** | The best individual(s) survive unchanged into the next generation. Ensures the best solution found so far is never lost. |

### Why "evolutionary"?

The analogy: each candidate configuration is an "organism". Good configurations (low MAE) are "fit" and survive to reproduce. Over generations, the population converges toward better solutions. Unlike grid search or random search, the GA intelligently recombines promising parts of different solutions.

---

## 16. Search Space and Configuration Space

The **search space** is the set of all possible hyperparameter combinations that the GA can explore. In this project, it has **17 dimensions** -- each dimension is a hyperparameter with a defined range of possible values.

**Configuration space** is used interchangeably -- it is the full space of pipeline configurations the search can produce. What makes this project's approach distinctive is that the search space includes not just model architecture, but also:

- **Preprocessing** (which scaler to use)
- **Windowing** (how many hours of history to feed the model)
- **Training** (optimizer, loss function, learning rate, batch size)
- **Regularization** (dropout, L2, noise, gradient clipping)

This makes it an **end-to-end** optimization rather than just tuning a few model knobs.

---

## 17. Population, Individual, Generation

These are the GA's terminology:

| Term | Meaning | In this project |
|---|---|---|
| **Individual** | One candidate solution (a specific combination of hyperparameters) | A dictionary mapping 17 gene names to values, e.g., `{"n_layers": 2, "units1": 64, "loss_name": "mae", ...}` |
| **Population** | The set of all individuals being evaluated at one time | 20 individuals per generation |
| **Generation** | One cycle of evaluation + selection + crossover + mutation | 15 generations total |
| **Genotype** | The encoding of an individual's traits (the dictionary) | The hyperparameter dictionary |
| **Phenotype** | The expressed result of the genotype (the actual model and its performance) | The trained GRU model and its validation MAE |
| **Fitness** | How good an individual is (lower MAE = higher fitness) | Validation MAE in degC |

Total evaluations: 20 individuals x 15 generations = **300 pipeline evaluations**.

---

## 18. Synthetic Data Generation

**Synthetic data** is artificially generated data that mimics the statistical properties of real data. In machine learning, generating synthetic training samples can:

- **Augment** a small training set, giving the model more examples to learn from
- **Balance** underrepresented scenarios
- **Improve generalization** if the synthetic data is realistic enough

The key challenge: synthetic data must be realistic enough to help, not hurt. If it introduces unrealistic patterns, the model may learn wrong relationships.

---

## 19. TimeGAN

**TimeGAN** (Time-series Generative Adversarial Network, Yoon et al., 2019) is a generative model specifically designed for time series data. It combines:

### GAN basics

A standard **GAN** (Generative Adversarial Network) has two networks:
- **Generator:** Creates fake data from random noise
- **Discriminator:** Tries to distinguish real data from fake

They are trained adversarially: the generator tries to fool the discriminator, and the discriminator tries not to be fooled. Over time, the generator learns to produce increasingly realistic data.

### What TimeGAN adds

Standard GANs do not capture temporal dynamics well. TimeGAN adds three innovations:

| Component | Purpose |
|---|---|
| **Embedding network** | Compresses real data into a latent (hidden) representation space |
| **Recovery network** | Reconstructs data from latent space back to original space |
| **Supervisor network** | Learns temporal transitions within the latent space (given the current latent state, predict the next one) |

Together with the standard generator and discriminator, TimeGAN has **5 sub-networks**, all GRU-based in this project.

### Training phases

1. **Autoencoder pretraining:** Embedder + Recovery learn to compress and reconstruct real sequences
2. **Supervisor pretraining:** Supervisor learns latent temporal dynamics
3. **Adversarial training:** Full GAN training with generator, discriminator, and all supporting losses

In this project, TimeGAN produced sequences with good reconstruction quality but insufficient realism for safe use in the final pipeline.

---

## 20. Data Augmentation and Information Leakage

### Data Augmentation

Adding more training examples by transforming or generating data. In computer vision, this might mean rotating/flipping images. In time series, it means generating synthetic sequences.

### Information Leakage

**Information leakage** (or data leakage) occurs when information from outside the training set "leaks" into the model during training, giving it an unfair advantage that won't exist during real deployment.

Common forms:
- **Training on test data:** If the model sees any test data during training, its test performance is artificially inflated
- **Using future data to predict the past:** In time series, shuffling data across time can let the model "peek" at the future
- **Fitting the scaler on the full dataset:** If you compute mean/variance from all data including test, the test set statistics leak into training

This project prevents leakage by:
- Chronological splits (no shuffling)
- Fitting scalers only on training data
- Training TimeGAN only on the training split
- Evaluating EA fitness only on validation data, never on test

---

## 21. Explainable AI (XAI)

Deep learning models are often called "black boxes" -- they make predictions, but it is hard to understand **why**. **Explainable AI** is a set of techniques that make model decisions interpretable.

Two levels:

| Level | Question answered | Technique in this project |
|---|---|---|
| **Global** | "Which features are generally most important across all predictions?" | Permutation importance |
| **Local** | "For this specific prediction, which inputs mattered most?" | Gradient saliency |

XAI is valuable for:
- **Trust:** Understanding why a model makes certain predictions
- **Debugging:** Detecting if the model relies on spurious correlations
- **Refinement:** Identifying unimportant features that can be removed (feature pruning)

---

## 22. Permutation Importance (Global Explainability)

A model-agnostic technique to measure feature importance:

### Algorithm

```
1. Evaluate the model on the test set --> get baseline MAE
2. For each feature:
   a. Shuffle (randomly permute) that feature's values across all samples
   b. Evaluate the model again --> get new MAE
   c. Importance = new MAE - baseline MAE
3. Rank features by importance (higher = more important)
```

### Intuition

If a feature is truly important, shuffling it destroys the information the model relies on, causing MAE to increase sharply. If shuffling a feature barely changes MAE, the model does not depend on it.

In this project, past temperature (`T (degC)`) showed the largest importance by far, confirming it is the dominant predictor.

---

## 23. Gradient Saliency (Local Explainability)

A technique that explains **one specific prediction** by looking at gradients:

### Algorithm

```
1. Take one input sample (120 hours x 16 features)
2. Pass it through the model
3. Compute the gradient of the output with respect to the input (using TensorFlow's GradientTape)
4. Take the absolute value of each gradient --> saliency map
```

### Intuition

The gradient tells you: "If I slightly changed this input value, how much would the prediction change?" Large absolute gradients mean the model is very sensitive to that input -- it is paying close attention to it.

The saliency map has the same shape as the input (120 x 16), so it can be visualized as:
- **Aggregated over time** --> which features matter most (bar chart)
- **Aggregated over features** --> which time steps matter most (line plot)
- **Full 2D heatmap** --> joint view of feature and time importance

Key finding: the model focuses on the **most recent** time steps, especially temperature and pressure.

---

## 24. Feature Pruning

**Feature pruning** means removing input features (variables) that contribute little or nothing to the model's predictions.

### Why prune?

- **Simplicity:** Fewer inputs = simpler model, easier to understand
- **Speed:** Fewer features = faster training and inference
- **Performance:** Removing noisy or redundant features can actually *improve* accuracy by reducing distractions

### How it was done in this project

1. Permutation importance ranked all 16 features
2. The 5 least important were identified: `doy_sin`, `gust_ratio`, `wd (deg)`, `wd_sin`, `wy`
3. The model was retrained on the remaining 11 features
4. Result: the pruned model was **more accurate** (1.575 vs 1.598 degC MAE), confirming those features were slightly harmful

This is an example of using XAI not just for interpretation but for **model refinement**.

---

## 25. Efficiency and Resource Analysis

Beyond prediction accuracy, practical deployment cares about computational cost:

| Metric | What it measures | Why it matters |
|---|---|---|
| **Training time** | How long it takes to train the model | Affects development iteration speed and energy cost |
| **Inference latency** | How long one prediction takes | Critical for real-time applications |
| **Trainable parameters** | Number of weights in the model | Proxy for model complexity and memory needs |
| **Memory usage (RAM/GPU)** | Memory consumed during training/inference | Determines hardware requirements |
| **Throughput** | Predictions per second | Determines if the model can keep up with incoming data |

---

## 26. Accuracy--Efficiency Trade-offs

In machine learning, there is often a tension between accuracy and efficiency:
- Larger, more complex models tend to be more accurate but slower and more expensive
- Smaller models are faster but may sacrifice accuracy

A key finding of this project is that this trade-off is **not inevitable**: the evolutionary search found a model that is simultaneously more accurate, smaller (fewer parameters), and faster to train than the hand-tuned baseline. This happened because the baseline was over-parameterized -- its extra complexity was not helping.

---

## 27. End-to-End Pipeline

An **end-to-end pipeline** means the entire process from raw data to final prediction is treated as a connected system, not as isolated steps:

```
Raw Data --> Preprocessing --> Feature Engineering --> Scaling --> Windowing --> Model --> Prediction
```

In this project, the evolutionary algorithm optimizes across **the entire pipeline**, not just the model. It jointly selects:
- Which scaler to use (preprocessing)
- How many hours of history to look at (windowing)
- The model architecture (how many layers, how many units)
- Training settings (optimizer, loss, learning rate, batch size)
- Regularization strategy (dropout, L2, noise, gradient clipping)

This end-to-end approach is important because choices interact: a different scaler may work better with a different architecture, and a different lookback window may require a different learning rate. Optimizing everything together captures these interactions.
