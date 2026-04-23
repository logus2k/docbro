# Section 11 -- Explainable AI (XAI)

This document explains the goals, concepts, and implementation details of Section 11 of the notebook, which applies global and local explainability techniques to the optimized forecasting model, and then uses the insights for feature pruning.

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [Global vs. Local Explainability](#2-global-vs-local-explainability)
3. [Permutation Importance -- Concept](#3-permutation-importance----concept)
4. [Permutation Importance -- Implementation](#4-permutation-importance----implementation)
5. [Permutation Importance -- Results](#5-permutation-importance----results)
6. [Gradient Saliency -- Concept](#6-gradient-saliency----concept)
7. [Gradient Saliency -- Implementation](#7-gradient-saliency----implementation)
8. [Three Views of Local Saliency](#8-three-views-of-local-saliency)
9. [XAI Summary -- Consistency Between Global and Local](#9-xai-summary----consistency-between-global-and-local)
10. [XAI-Guided Feature Pruning](#10-xai-guided-feature-pruning)
11. [Pruning Results](#11-pruning-results)
12. [The Closed-Loop Methodology](#12-the-closed-loop-methodology)
13. [Implementation Summary](#13-implementation-summary)

---

## 1. Section Goal

Three objectives:

1. **Understand** what the model has learned (which features and time steps matter)
2. **Validate** that the model's behavior aligns with physical intuition (temperature and recent history should dominate)
3. **Improve** the model by removing features that XAI identifies as unimportant

This section transforms XAI from a passive interpretation tool into an active **model refinement** mechanism.

---

## 2. Global vs. Local Explainability

| Level | Question | Technique | Scope |
|---|---|---|---|
| **Global** | "Which features matter most across all predictions?" | Permutation importance | All test samples |
| **Local** | "For this specific prediction, which inputs mattered?" | Gradient saliency | One sample |

Both perspectives are needed:
- Global tells you **what the model generally relies on** -- useful for feature selection
- Local tells you **why a specific prediction was made** -- useful for debugging and trust

---

## 3. Permutation Importance -- Concept

### The idea

If a feature is important, **destroying its information** should hurt the model's predictions. If it's unimportant, destroying it should make little difference.

### Algorithm

```
1. Compute baseline MAE on the test set (predictions with intact features)
2. For each feature:
   a. Shuffle that feature's values across all samples
      (breaks the real relationship between this feature and the target)
   b. Compute MAE on the shuffled data
   c. Importance = shuffled MAE - baseline MAE
3. Repeat n_repeats times and average (reduces noise)
4. Rank features by importance
```

### What "shuffling" means in 3D data

The input tensor has shape `(N_samples, 120_timesteps, 16_features)`. Shuffling feature `k` means: for each sample, replace all 120 time steps of feature `k` with those from a **randomly selected different sample**.

```python
perm = rng.permutation(X.shape[0])              # random sample order
X_permuted[:, :, feat_idx] = X_permuted[perm, :, feat_idx]  # swap entire feature sequences
```

This preserves the feature's **marginal distribution** (same values exist, just mismatched to the wrong samples) while breaking its **conditional relationship** with the target.

### Why not just set to zero?

Zeroing a feature changes the input distribution in a way the model never saw during training, potentially causing unpredictable behavior. Shuffling keeps the values realistic (they come from actual data) while only breaking the sample-specific relationship.

---

## 4. Permutation Importance -- Implementation

```python
def permutation_feature_importance(model, X, y, feature_names,
                                    metric_fn=None, n_repeats=3, random_state=42):
    rng = np.random.default_rng(random_state)

    # Step 1: baseline score
    baseline_pred = model.predict(X, verbose=0)
    baseline_score = metric_fn(y, baseline_pred)

    importances = []

    for feat_idx, feat_name in enumerate(feature_names):
        scores = []

        for _ in range(n_repeats):
            # Step 2a: shuffle this feature
            X_permuted = X.copy()
            perm = rng.permutation(X.shape[0])
            X_permuted[:, :, feat_idx] = X_permuted[perm, :, feat_idx]

            # Step 2b: compute degraded MAE
            perm_pred = model.predict(X_permuted, verbose=0)
            perm_score = metric_fn(y, perm_pred)
            scores.append(perm_score - baseline_score)

        importances.append({
            "feature": feat_name,
            "importance_mean": np.mean(scores),
            "importance_std": np.std(scores),
        })

    return baseline_score, importances
```

### Key details

- **`n_repeats=3`:** Each feature is shuffled 3 times to get a mean and standard deviation. More repeats give more stable estimates but cost more (each requires a full model prediction on the test set).
- **`random_state=42`:** Ensures reproducibility across runs.
- **`importance_std`:** Measures how variable the importance estimate is. Low std means the importance is reliable; high std means it varies across shuffles.
- **Model-agnostic:** This technique works with any model (GRU, LSTM, transformer, random forest) because it only needs predictions, not internal access to the model.

---

## 5. Permutation Importance -- Results

### Ranking (from most to least important)

| Rank | Feature | Importance (MAE increase) | Category |
|---:|---|---|---|
| 1 | **T (degC)** | Very high | Past temperature -- dominant predictor |
| 2 | **doy_cos** | High | Seasonal cycle (cosine) |
| 3 | **hour_cos** | High | Diurnal cycle (cosine) |
| 4 | **hour_sin** | Moderate-high | Diurnal cycle (sine) |
| 5 | **max. wv (m/s)** | Moderate | Wind gust intensity |
| 6 | **rh (%)** | Moderate | Humidity |
| 7-11 | Other features | Low-moderate | Various wind/pressure features |
| 12 | **wd_sin** | Very low | Wind direction (sine) |
| 13 | **wd (deg)** | Very low | Raw wind direction |
| 14 | **gust_ratio** | Very low | Gust ratio |
| 15 | **doy_sin** | Very low | Seasonal cycle (sine) |
| 16 | **wy** | Very low | Wind y-component |

### Interpretation

- **Temperature dominates**: Past temperature is by far the strongest predictor. This makes physical sense -- temperature has strong autocorrelation (today's temperature is the best predictor of tomorrow's).
- **Cyclical features matter**: `doy_cos`, `hour_cos`, `hour_sin` capture the seasonal and diurnal patterns the model relies on.
- **Redundancy visible**: `doy_sin` is unimportant while `doy_cos` is important. This is because the (sin, cos) pair encodes the same cycle -- the model can capture the seasonal signal from just one of them. Similarly, `wd (deg)` is redundant with `wd_sin/wd_cos`.
- **Figure 9**: Horizontal bar chart showing this ranking with error bars.

---

## 6. Gradient Saliency -- Concept

### The idea

For a specific input sample, compute **how sensitive the model's output is to each input value**. High sensitivity = the model is "paying attention" to that value.

### Mathematically

```
saliency(i, j) = |d(output) / d(input[i, j])|
```

The absolute value of the partial derivative of the model's output with respect to each element of the input tensor. This gives a saliency map with the same shape as the input: `(120 time steps, 16 features)`.

### Intuition

The gradient tells you: "If I nudged this input value by a tiny amount, how much would the prediction change?" Large absolute gradient = the model is highly sensitive to that input = it's important for this prediction.

---

## 7. Gradient Saliency -- Implementation

```python
def compute_saliency_map(model, input_sample, forecast_step=None):
    x = tf.convert_to_tensor(input_sample, dtype=tf.float32)  # (1, 120, 16)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x, training=False)  # (1, 24)

        if forecast_step is None:
            target = tf.reduce_mean(y_pred)  # average over all 24 hours
        else:
            target = y_pred[:, forecast_step]  # specific hour

    grads = tape.gradient(target, x)
    saliency = tf.abs(grads).numpy()[0]  # (120, 16)

    return saliency
```

### Key details

- **`GradientTape`:** TensorFlow's mechanism for computing gradients. It records all operations on watched tensors, then computes derivatives via backpropagation.
- **`tape.watch(x)`:** Tells TensorFlow to track gradients with respect to the input (normally only model weights are tracked).
- **`training=False`:** Ensures dropout and batch normalization behave as in inference mode.
- **`forecast_step=None`:** When None, the saliency explains the **mean prediction** across all 24 forecast hours. When an integer, it explains a specific hour.
- **`tf.abs(grads)`:** Takes absolute value because we care about **magnitude** of influence, not direction (positive or negative sensitivity both indicate importance).

### Aggregation functions

```python
def aggregate_saliency_over_time(saliency):
    return np.mean(saliency, axis=0)    # (16,) -- one value per feature

def aggregate_saliency_over_features(saliency):
    return np.mean(saliency, axis=1)    # (120,) -- one value per time step
```

These reduce the 2D saliency map into 1D views for visualization.

---

## 8. Three Views of Local Saliency

For a single test sample (sample 0), the saliency map is visualized three ways:

### View 1: Feature-level importance (Figure 10)

Aggregate saliency across all 120 time steps for each feature. Result: a bar chart showing which features most influenced this specific prediction.

**Finding:** `T (degC)` dominates, followed by `hour_cos`, `doy_cos`, `p (mbar)`. Consistent with the global ranking.

### View 2: Time-step importance (Figure 11)

Aggregate saliency across all 16 features for each time step. Result: a line plot showing which parts of the 120-hour history the model focused on.

**Finding:** Sharp concentration near the **most recent observations** (the last ~10-20 hours). The earlier part of the window has very low saliency. This means the model relies primarily on recent history, with distant past contributing little.

**Physical interpretation:** For short-term temperature forecasting, the most recent conditions are indeed the strongest predictors. Yesterday's temperature tells you more about tomorrow than last week's.

### View 3: Full heatmap (Figure 12)

The complete 2D saliency map: features on the y-axis, time steps on the x-axis, intensity as color.

**Finding:** A bright spot in the bottom-right corner -- recent temperature values. A weaker bright area for recent pressure. Everything else is dim.

This gives the most detailed view: the model's "attention" for this sample is focused on **recent T (degC) and p (mbar)**.

---

## 9. XAI Summary -- Consistency Between Global and Local

| Finding | Global (permutation) | Local (saliency) | Consistent? |
|---|---|---|---|
| Temperature is dominant | Yes (highest importance) | Yes (highest saliency) | Yes |
| Cyclical time features matter | Yes (doy_cos, hour_cos rank high) | Yes (high saliency) | Yes |
| Pressure contributes | Moderate importance | Visible in heatmap | Yes |
| Recent time steps matter most | N/A (global is time-agnostic) | Yes (sharp recency spike) | -- |
| gust_ratio, wd_sin, doy_sin are weak | Yes (bottom of ranking) | Yes (low saliency) | Yes |

The consistency between global and local methods strengthens confidence that both techniques are revealing genuine model behavior, not artifacts.

---

## 10. XAI-Guided Feature Pruning

### The idea

If XAI shows that certain features contribute negligibly, removing them should not hurt (and might help by reducing noise).

### Features removed (5 least important)

| Removed feature | Why it was unimportant |
|---|---|
| `doy_sin` | Redundant with `doy_cos` -- the model only needs one component of the seasonal cycle |
| `gust_ratio` | Redundant with `wind_gap` -- both measure gust intensity, one was enough |
| `wd (deg)` | Redundant with `wd_sin`/`wd_cos` -- the cyclic encoding already captured direction |
| `wd_sin` | Less informative than `wd_cos` for this dataset and location |
| `wy` | Less informative than `wx` -- the east-west wind component matters more in Jena |

### Retained features (11)

`T (degC)`, `p (mbar)`, `rh (%)`, `wv (m/s)`, `max. wv (m/s)`, `hour_sin`, `hour_cos`, `doy_cos`, `wd_cos`, `wx`, `wind_gap`

### Pruning procedure

1. Remove the 5 features from the feature list
2. Re-prepare data (scale, window) using the reduced 11-feature set
3. Rebuild the model with `n_features=11` (same architecture otherwise)
4. Retrain with same protocol (30 epochs, same callbacks)
5. Evaluate on test set

The architecture hyperparameters remain identical -- only the input dimensionality changes. This isolates the effect of pruning.

---

## 11. Pruning Results

### Single-run comparison (Table 15)

| Model | Features | MAE (degC) | RMSE (degC) |
|---|---:|---:|---:|
| Best Evolutionary GRU | 16 | 1.598 | 2.157 |
| **Pruned Evolutionary GRU** | **11** | **1.575** | **2.134** |

Pruning improved MAE by **0.023 degC** and RMSE by **0.023 degC**. The pruned model is both **simpler and more accurate**.

### Multi-seed robustness (Table 16)

| Model | MAE mean +/- std | RMSE mean +/- std |
|---|---:|---:|
| Full EA (16 features) | 1.593 +/- 0.016 | 2.151 +/- 0.023 |
| **Pruned EA (11 features)** | **1.579 +/- 0.009** | **2.134 +/- 0.010** |

The pruned model is:
- **More accurate** on average (1.579 vs 1.593)
- **More stable** across seeds (std 0.009 vs 0.016)
- **Best single seed**: 1.569 degC -- the best result in the entire project

### Why pruning helped

The removed features were not just uninformative -- they were slightly **detrimental**:
- They added 5 extra dimensions of noise the model had to process
- The GRU's first layer had to learn weights for 16 inputs instead of 11, wasting capacity
- Redundant features (like both `doy_sin` and `doy_cos`) could create interference during training

Removing them reduced noise and allowed the model to focus its capacity on the features that actually matter.

---

## 12. The Closed-Loop Methodology

This section demonstrates a complete **optimize -> explain -> prune -> validate** loop:

```
Section 9:  Evolutionary optimization --> Best EA GRU (1.598 degC)
    |
Section 11: Permutation importance --> identify 5 weak features
    |
Section 11: Remove weak features --> Pruned EA GRU (1.575 degC)
    |
Section 11: Multi-seed validation --> 1.579 +/- 0.009 degC (confirmed stable)
```

This closed loop is one of the project's key contributions. XAI is used not just to explain, but to **actively improve** the pipeline. The pruned model becomes the final selected model of the project.

---

## 13. Implementation Summary

### Source modules

| Module | Contents |
|---|---|
| `src/xai/permutation.py` | `permutation_feature_importance()` -- model-agnostic global importance |
| `src/xai/saliency.py` | `compute_saliency_map()`, `aggregate_saliency_over_time()`, `aggregate_saliency_over_features()` -- gradient-based local attribution |

### Figures and tables produced

| Output | Description |
|---|---|
| Table 12 | Permutation importance ranking (all 16 features) |
| Figure 9 | Permutation importance bar chart |
| Table 13 | Local feature saliency for sample 0 |
| Figure 10 | Local feature saliency bar chart |
| Figure 11 | Temporal saliency profile (time steps) |
| Figure 12 | Full 2D saliency heatmap |
| Table 15 | Pruning comparison (16 vs 11 features) |
| Table 16 | Pruned model robustness across seeds |
