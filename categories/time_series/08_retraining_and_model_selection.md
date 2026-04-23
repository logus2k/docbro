# Section 10 -- Retraining and Final Model Selection

This document explains the goals, concepts, and implementation details of Section 10 of the notebook, which retrains the two final candidate models under a fair protocol, evaluates them on the held-out test set, and validates robustness across random seeds.

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [Why Retrain?](#2-why-retrain)
3. [Final Candidates](#3-final-candidates)
4. [Data Preparation for Each Candidate](#4-data-preparation-for-each-candidate)
5. [Retraining Procedure](#5-retraining-procedure)
6. [Training Curves and Convergence](#6-training-curves-and-convergence)
7. [Test Set Evaluation](#7-test-set-evaluation)
8. [Scaled vs. Original Metrics](#8-scaled-vs-original-metrics)
9. [Final Model Comparison](#9-final-model-comparison)
10. [Robustness Across Random Seeds](#10-robustness-across-random-seeds)
11. [Model Selection Decision](#11-model-selection-decision)

---

## 1. Section Goal

The EA found its best configuration under a limited budget (20 epochs per individual). Before declaring a winner, both the baseline and the EA-optimized model must be retrained under **identical, expanded conditions** and evaluated on the **test set for the first time**.

This section answers three questions:
1. Does the EA advantage hold when both models get a full training budget?
2. How do the models perform on completely unseen test data?
3. Is the EA result robust across different random seeds, or was it a lucky initialization?

---

## 2. Why Retrain?

During the EA search, each candidate was trained for only **20 epochs** to keep the search computationally feasible (300 evaluations x 20 epochs). This limited budget might not let a model reach its full potential.

For a fair final comparison, both candidates are retrained from scratch with:
- **50 epochs** (2.5x the EA budget)
- Same callbacks (early stopping patience=6, LR reduction patience=3)
- Fresh weight initialization (no warm-starting from EA weights)

This ensures neither model is disadvantaged by training budget.

---

## 3. Final Candidates

| Candidate | Source | Scaler | Lookback | Parameters |
|---|---|---|---:|---:|
| **GRU Baseline Official** | Mini-project best config | StandardScaler | 120h | 86,744 |
| **Best Evolutionary GRU** | EA search (Section 9) | RobustScaler | 144h | 63,512 |

Their full configurations (Table 9 in the notebook) differ across nearly every hyperparameter -- architecture, optimizer settings, loss function, preprocessing, and windowing. This is a comparison of **complete pipelines**, not just architectures.

---

## 4. Data Preparation for Each Candidate

Because the two models use **different scalers and lookback windows**, each needs its own data preparation:

```python
def prepare_data_for_cfg(cfg, df_train, df_val, df_test, final_feature_cols,
                          target_idx, lookback, horizon):
    scaler = get_scaler(cfg["scaler_name"])

    X_train_scaled = scaler.fit_transform(df_train[final_feature_cols])
    X_val_scaled   = scaler.transform(df_val[final_feature_cols])
    X_test_scaled  = scaler.transform(df_test[final_feature_cols])

    X_train, y_train = make_windows(X_train_scaled, target_idx, lookback, horizon)
    X_val, y_val     = make_windows(X_val_scaled, target_idx, lookback, horizon)
    X_test, y_test   = make_windows(X_test_scaled, target_idx, lookback, horizon)

    return scaler, X_train, y_train, X_val, y_val, X_test, y_test
```

### Why separate preparation matters

| Aspect | Baseline | EA model |
|---|---|---|
| Scaler | StandardScaler (mean/std) | RobustScaler (median/IQR) |
| Lookback | 120 hours | 144 hours |
| X shape | (N, 120, 16) | (N, 144, 16) |
| Number of samples | Slightly more (shorter window = more valid positions) | Slightly fewer |

The scaler is **fit on train only** in both cases, and the same scaler object is kept for later inverse-scaling of predictions.

---

## 5. Retraining Procedure

Both models are built from scratch and trained with `train_model()`:

```python
baseline_final_history = train_model(
    model=baseline_final_model,
    X_train=X_train_b, y_train=y_train_b,
    X_val=X_val_b, y_val=y_val_b,
    batch_size=baseline_cfg_final["batch_size"],  # 128
    epochs=50,
)

evolutionary_final_history = train_model(
    model=evolutionary_final_model,
    X_train=X_train_e, y_train=y_train_e,
    X_val=X_val_e, y_val=y_val_e,
    batch_size=evolutionary_cfg_final["batch_size"],  # 256
    epochs=50,
)
```

Both use the same callback protocol:
- **Early stopping** (patience=6, restore best weights)
- **LR reduction on plateau** (patience=3, factor=0.5, min_lr=1e-5)

This is the same `train_model()` function used throughout the project, ensuring consistency.

---

## 6. Training Curves and Convergence

### Baseline (Figure 7)

- Best validation loss at **epoch 14** (0.0346)
- Best validation MAE at **epoch 20** (0.194)
- Training stopped early -- the baseline converges quickly and plateaus

### EA model (Figure 8)

- Best validation loss and MAE both at **epoch 43** (0.1376 and 0.1354)
- Continued improving for much longer than the baseline
- Benefited from the full 50-epoch budget

### What this tells us

The EA-optimized model has a **longer learning curve** -- it keeps extracting useful patterns from the data even after 40 epochs. The baseline converges faster but at a worse level. This is consistent with the EA model using:
- MAE loss (smoother gradients, slower but steadier convergence)
- Larger batch size (256 vs 128, more stable gradient estimates, slower per-epoch progress)
- Longer lookback (144h, more temporal context to learn from)

---

## 7. Test Set Evaluation

This is the **first and only time** the test set is used for evaluation. Both models predict on their respective test tensors:

```python
y_pred_baseline_test = baseline_final_model.predict(X_test_b)
y_pred_evo_test = evolutionary_final_model.predict(X_test_e)
```

Metrics are computed in both scaled space and original temperature scale.

### Inverse scaling

Since the two models use different scalers, each prediction is inverse-scaled using its **own scaler**:

```python
# Baseline: uses StandardScaler statistics
y_test_b_inv = inverse_target_with_scaler(y_test_b, baseline_scaler, target_idx, n_features)
y_pred_baseline_inv = inverse_target_with_scaler(y_pred_baseline_test, baseline_scaler, ...)

# EA: uses RobustScaler statistics
y_test_e_inv = inverse_target_with_scaler(y_test_e, evo_scaler, target_idx, n_features)
y_pred_evo_inv = inverse_target_with_scaler(y_pred_evo_test, evo_scaler, ...)
```

After inverse scaling, both are in degrees Celsius and directly comparable.

---

## 8. Scaled vs. Original Metrics

### Scaled space

| Model | MAE (scaled) | RMSE (scaled) |
|---|---:|---:|
| GRU Baseline | 0.1909 | 0.2537 |
| Best EA GRU | 0.1295 | 0.1748 |

The scaled metrics show a large gap. However, **scaled values are not directly comparable** across models when different scalers are used. StandardScaler and RobustScaler produce different numeric ranges, so 0.19 under StandardScaler and 0.13 under RobustScaler are measured in different units.

### Original scale (degrees Celsius) -- the primary metric

| Model | MAE (degC) | RMSE (degC) |
|---|---:|---:|
| GRU Baseline | 1.650 | 2.193 |
| Best EA GRU | 1.598 | 2.157 |

After inverse scaling, both are on the same physical scale. The EA model wins by **0.052 degC MAE** (-3.2%) and **0.036 degC RMSE** (-1.6%).

### Why the percentage gap looks smaller in degC

In scaled space, the EA model appears ~32% better. In degC, it's ~3.2% better. This is because:
- RobustScaler produces a different normalization range than StandardScaler
- The absolute temperature range (~50 degC from coldest to warmest) makes small improvements look smaller in percentage terms
- But 0.05 degC improvement across 10,000+ test samples is statistically meaningful

---

## 9. Final Model Comparison

| Model | MAE (degC) | RMSE (degC) | Parameters | vs. Baseline |
|---|---:|---:|---:|---|
| GRU Baseline Official | 1.650 | 2.193 | 86,744 | reference |
| **Best Evolutionary GRU** | **1.598** | **2.157** | **63,512** | **-3.2% MAE, 27% fewer params** |

The EA model is **simultaneously more accurate and more compact**. This is significant because it means the baseline was over-parameterized -- the extra 23,000 parameters were not contributing to better predictions.

---

## 10. Robustness Across Random Seeds

### The problem

The EA optimized on the validation set using seed 42. What if the result is specific to that seed? A different random initialization might produce different weights, and the "best" configuration might only work well with one particular initialization.

### The solution

Retrain the EA configuration from scratch with **3 different seeds** (7, 21, 42) and evaluate each on the test set:

```python
def evaluate_model_with_seed(seed, cfg, ...):
    set_global_seed(seed)          # Reset all random states
    # Full pipeline: scale -> window -> build -> train -> predict -> evaluate
    ...
    return {"seed": seed, "mae_degC": ..., "rmse_degC": ...}
```

### Results (Table 11)

| Seed | MAE (degC) | RMSE (degC) |
|---:|---:|---:|
| 7 | ~1.579 | ~2.129 |
| 21 | ~1.610 | ~2.175 |
| 42 | ~1.590 | ~2.148 |
| **Mean +/- std** | **1.593 +/- 0.016** | **2.151 +/- 0.023** |

### Interpretation

- All three seeds produce MAE in the range **1.58-1.61 degC** -- a tight band
- Standard deviation of 0.016 degC is very small relative to the mean
- The worst seed (1.61) still beats the baseline (1.65)
- This confirms the EA result is **robust**, not a lucky accident

### Why 3 seeds?

Three seeds provide a basic estimate of mean and variability. More seeds (e.g., 10) would give a more precise estimate, but each requires a full training run (~minutes), and three is sufficient to check that the result is not dependent on a single initialization. The notebook acknowledges that more seeds would strengthen the analysis (listed as future work).

---

## 11. Model Selection Decision

Based on all evidence:

| Criterion | Winner |
|---|---|
| Test MAE | EA model (1.598 vs 1.650) |
| Test RMSE | EA model (2.157 vs 2.193) |
| Parameter count | EA model (63,512 vs 86,744) |
| Multi-seed stability | EA model (1.593 +/- 0.016) |

The **Best Evolutionary GRU** is selected as the final model for subsequent XAI analysis (Section 11) and efficiency profiling (Section 12). It is later further improved through XAI-guided feature pruning.

### What happens next

```
Best EA GRU (selected here)
    |
    v  Section 11: XAI analysis reveals 5 unimportant features
    |
    v  Pruning: retrain with 11 features instead of 16
    |
Pruned EA GRU (1.575 degC MAE -- even better)
    |
    v  Section 12: Efficiency comparison of all 3 models
```
