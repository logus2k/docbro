# Section 6 -- Split, Scaling and Windowing

This document explains the goals, concepts, and implementation details of Section 6 of the notebook. This section completes the data preparation pipeline by splitting the data chronologically, normalizing features, and converting the flat time series into supervised learning tensors. After this section, the data is ready for model training.

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [Temporal Train/Validation/Test Split](#2-temporal-trainvalidationtest-split)
3. [Why No Shuffling?](#3-why-no-shuffling)
4. [Feature Scaling](#4-feature-scaling)
5. [The Three Scalers](#5-the-three-scalers)
6. [Fit on Train Only -- Preventing Leakage](#6-fit-on-train-only----preventing-leakage)
7. [Target Index](#7-target-index)
8. [Supervised Windowing](#8-supervised-windowing)
9. [The Sliding Window Procedure Step by Step](#9-the-sliding-window-procedure-step-by-step)
10. [Implementation: `make_windows()`](#10-implementation-make_windows)
11. [Tensor Shapes and What They Mean](#11-tensor-shapes-and-what-they-mean)
12. [Sanity Check](#12-sanity-check)
13. [Complete Data Pipeline Summary](#13-complete-data-pipeline-summary)

---

## 1. Section Goal

Transform the engineered dataframe (`df_feat`, 70,041 rows x 16 features) into the final **NumPy tensors** that the GRU model will consume:

| Output | Shape | Content |
|---|---|---|
| `X_train` | (N_train, 120, 16) | Training input windows |
| `y_train` | (N_train, 24) | Training target sequences |
| `X_val` | (N_val, 120, 16) | Validation input windows |
| `y_val` | (N_val, 24) | Validation target sequences |
| `X_test` | (N_test, 120, 16) | Test input windows |
| `y_test` | (N_test, 24) | Test target sequences |

Three operations are applied in order: **split**, then **scale**, then **window**.

---

## 2. Temporal Train/Validation/Test Split

### What happens

```python
n = len(df_feat)              # ~70,041

train_end = int(n * 0.70)    # first 70%
val_end   = int(n * 0.85)    # next 15%

df_train = df_feat.iloc[:train_end].copy()        # 2009 -- mid 2014
df_val   = df_feat.iloc[train_end:val_end].copy()  # mid 2014 -- early 2016
df_test  = df_feat.iloc[val_end:].copy()            # early 2016 -- end 2016
```

### The three sets and their roles

| Set | Fraction | Approx. period | Purpose | Who sees it |
|---|---|---|---|---|
| **Train** | 70% (~49,028 rows) | 2009 -- mid 2014 | Model learns weights from this data | The model during training |
| **Validation** | 15% (~10,506 rows) | mid 2014 -- early 2016 | Tune hyperparameters, EA fitness, early stopping decisions | The EA and callbacks, but NOT the model's weight updates |
| **Test** | 15% (~10,507 rows) | early 2016 -- end 2016 | Final evaluation only -- completely untouched during development | Nobody until the very end |

### Why 70/15/15?

This is a common split for time series. The training set needs to be large enough to learn seasonal patterns (at least 4-5 years). The validation and test sets each need to span enough time to include different weather conditions (each covers roughly a year).

### Concept: validation vs. test

This distinction is critical and often confused:

- **Validation set:** Used repeatedly during development to compare models, tune hyperparameters, and make decisions (e.g., the EA evaluates fitness on validation). Because decisions are made based on it, there is a risk of **indirectly overfitting** to it.
- **Test set:** Used **exactly once** at the very end. No decisions are made based on it. This gives an unbiased estimate of how the model would perform on truly unseen future data.

If you use the test set to make decisions and then report test performance, you are cheating -- the number no longer reflects real-world performance.

---

## 3. Why No Shuffling?

In standard machine learning (e.g., image classification), you typically shuffle data before splitting so that each set contains a random mix of samples. **In time series, this is forbidden.**

### What would go wrong with shuffling

Imagine shuffling and then splitting. Your training set might contain observations from December 2016, and your test set might contain observations from March 2016. The model would be trained on data from **after** the test period -- it would have seen the future.

```
Shuffled (WRONG):
Train: [Jan 2009, Jul 2016, Mar 2012, Dec 2016, ...]  <-- future data in train
Test:  [Mar 2016, Aug 2010, ...]                        <-- past data in test

Chronological (CORRECT):
Train: [Jan 2009 ... Jun 2014]    <-- all past
Val:   [Jul 2014 ... Jan 2016]    <-- all between
Test:  [Feb 2016 ... Dec 2016]    <-- all future
```

### The deployment argument

In real deployment, you train on historical data and predict the future. The chronological split **simulates this exact scenario**. If your model performs well on the test set (the most recent data), you have evidence that it would work in production.

---

## 4. Feature Scaling

### Why scale at all?

The 16 input features have very different numeric ranges:

| Feature | Typical range | Scale |
|---|---|---|
| T (degC) | -20 to 35 | Tens |
| p (mbar) | 850 to 1050 | Hundreds to thousands |
| rh (%) | 0 to 100 | Tens to hundreds |
| hour_sin | -1 to 1 | Unit |
| gust_ratio | 1 to 10+ | Single digits |

Without scaling, pressure (~1000) would dominate the gradient updates simply because its values are numerically larger, not because it is more important. Scaling ensures all features contribute proportionally to their actual informational content.

### What happens

```python
SCALER_NAME = "standard"
scaler = get_scaler(SCALER_NAME)

X_train_scaled = scaler.fit_transform(X_train_df)   # fit AND transform
X_val_scaled   = scaler.transform(X_val_df)          # transform only
X_test_scaled  = scaler.transform(X_test_df)          # transform only
```

---

## 5. The Three Scalers

The scaling module (`src/features/scaling.py`) provides three options:

```python
def get_scaler(name="standard"):
    if name == "standard":  return StandardScaler()
    elif name == "robust":  return RobustScaler()
    elif name == "minmax":  return MinMaxScaler()
```

### StandardScaler

```
x_scaled = (x - mean) / std
```

- Centers data at 0 with standard deviation 1
- Assumes data is roughly **normally distributed**
- Sensitive to outliers: a single extreme value shifts the mean and inflates std, compressing all normal values

### RobustScaler

```
x_scaled = (x - median) / IQR
```

Where IQR = 75th percentile - 25th percentile (the interquartile range).

- Centers data at the median (not the mean)
- Scales by the middle 50% of the data (not the full spread)
- **Resistant to outliers:** extreme values don't affect median or IQR
- The EA chose this one for the best model

### MinMaxScaler

```
x_scaled = (x - min) / (max - min)
```

- Squeezes all values into [0, 1]
- **Very sensitive to outliers:** a single extreme value stretches the range, pushing all normal values into a narrow band
- Preserves zero entries (useful for sparse data, not relevant here)

### Visual comparison

Imagine temperature data where 99% of values are between -10 and 35 degC, but one reading is 45 degC (a sensor error or heat wave):

| Scaler | Effect of the 45 degC outlier |
|---|---|
| **Standard** | Mean shifts up, std inflates. Normal values get slightly compressed. |
| **Robust** | Median and IQR barely change. Normal values keep their spread. |
| **MinMax** | The range becomes [-10, 45] instead of [-10, 35]. All normal values are squeezed into [0, 0.82] instead of the full [0, 1]. |

This is why the EA preferred RobustScaler -- weather data inevitably contains extreme events.

---

## 6. Fit on Train Only -- Preventing Leakage

### The rule

```python
scaler.fit_transform(train)    # learn statistics from train
scaler.transform(val)          # apply train's statistics to val
scaler.transform(test)         # apply train's statistics to test
```

### Why not fit on all data?

`fit` computes statistics (mean/std, median/IQR, or min/max) from the data you give it. If you fit on the entire dataset:

```python
# WRONG -- leaks future information into scaling
scaler.fit_transform(all_data)
```

The scaler's mean would include test-period temperatures. When you then scale the training data, it is centered relative to a mean that incorporates future values the model should not know about. This is **data leakage** -- subtle, invisible, and it inflates your test metrics.

### What `fit_transform` vs. `transform` means

| Method | What it does |
|---|---|
| `fit(data)` | Computes and stores statistics (mean, std, etc.) from data |
| `transform(data)` | Applies the stored statistics to scale data |
| `fit_transform(data)` | Does both in one call (convenience shorthand) |

After `fit_transform(train)`, the scaler "remembers" the training set's statistics. Every subsequent `transform()` call uses those same statistics, regardless of what data is passed in.

### Consequence for validation and test

Validation and test values might fall slightly outside the range the scaler was fitted on (e.g., the test set might have a temperature colder than anything in training). This is fine -- the scaler still applies the same formula, producing a scaled value that is simply outside the typical [-2, 2] range. This reflects reality: the model should handle unseen conditions, not be protected from them.

---

## 7. Target Index

### What happens

```python
target_idx = final_feature_cols.index(TARGET_COL)  # = 0
```

`T (degC)` is the first column in the 16-feature list, so `target_idx = 0`.

### Why this matters

During windowing, the input tensor X keeps **all 16 features** (the model needs multivariate input), but the output vector y keeps **only column 0** (temperature). The `target_idx` tells the windowing function which column to extract for y.

```
Scaled data: (N_timesteps, 16)
                                    column 0 = T (degC)  <-- target_idx
                                    column 1 = p (mbar)
                                    ...
                                    column 15 = gust_ratio

X window: all 16 columns  --> shape (120, 16)
y window: only column 0   --> shape (24,)
```

---

## 8. Supervised Windowing

### The concept

A time series is a continuous stream of hourly observations. But a neural network needs discrete (input, output) pairs to train on. **Windowing** (also called "sliding window" or "rolling window") converts the stream into individual samples.

### Analogy

Imagine reading a book to predict the next sentence. The window slides along:

```
Sample 1: read pages 1-5,   predict page 6
Sample 2: read pages 2-6,   predict page 7
Sample 3: read pages 3-7,   predict page 8
...
```

Each sample overlaps with its neighbors. The same data point appears in multiple input windows (and sometimes in an output window of a different sample). This is normal and expected -- it maximizes the number of training examples from a finite dataset.

---

## 9. The Sliding Window Procedure Step by Step

With `LOOKBACK = 120` and `HORIZON = 24`, here is how windows are created from a time series of length T:

```
Time index:  0   1   2  ...  119  120  121 ... 143  144  145 ...
             |<-- LOOKBACK=120 -->|<-- HORIZON=24 -->|
Sample 0:    [  X: hours 0-119   ][ y: hours 120-143 ]

             |<-- LOOKBACK=120 -->|<-- HORIZON=24 -->|
Sample 1:       [  X: hours 1-120   ][ y: hours 121-144 ]

             |<-- LOOKBACK=120 -->|<-- HORIZON=24 -->|
Sample 2:          [  X: hours 2-121   ][ y: hours 122-145 ]

...and so on until the end of the time series.
```

Each sample is created by:
1. Taking a slice of 120 consecutive hours (all 16 features) as **X**
2. Taking the next 24 hours of **temperature only** as **y**
3. Sliding forward by 1 hour and repeating

### How many samples are produced?

```
N_samples = T - LOOKBACK - HORIZON + 1
```

For the training set with ~49,028 hours:
```
N_train = 49,028 - 120 - 24 + 1 = 48,885 samples
```

For the test set with ~10,507 hours:
```
N_test = 10,507 - 120 - 24 + 1 = 10,364 samples
```

---

## 10. Implementation: `make_windows()`

The source code in `src/features/windowing.py`:

```python
def make_windows(data, target_idx, lookback=120, horizon=24):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, target_idx])
    return np.array(X), np.array(y)
```

### Line by line

| Line | What it does |
|---|---|
| `X, y = [], []` | Initialize empty lists to collect samples |
| `for i in range(...)` | Loop over every valid starting position. The range ensures the last sample still has enough data for both lookback and horizon. |
| `data[i:i + lookback]` | Slice rows i through i+119 (120 rows), **all columns** (all 16 features). This becomes one X sample. |
| `data[i + lookback:i + lookback + horizon, target_idx]` | Slice rows i+120 through i+143 (24 rows), **only the target column** (temperature). This becomes one y sample. |
| `np.array(X), np.array(y)` | Convert lists of arrays into 3D and 2D NumPy arrays |

### Why target_idx matters here

The slicing `data[..., target_idx]` is what makes this **multivariate-input, univariate-output**:
- X gets all columns: `data[i:i+lookback]` has shape `(120, 16)`
- y gets one column: `data[..., target_idx]` has shape `(24,)`

If you wanted multivariate output (predict all 16 features), you would simply drop the `target_idx` indexing from the y line.

---

## 11. Tensor Shapes and What They Mean

### Calling the function

```python
LOOKBACK = 120
HORIZON = 24

X_train, y_train = make_windows(X_train_scaled, target_idx, LOOKBACK, HORIZON)
X_val, y_val     = make_windows(X_val_scaled, target_idx, LOOKBACK, HORIZON)
X_test, y_test   = make_windows(X_test_scaled, target_idx, LOOKBACK, HORIZON)
```

### Resulting shapes

| Tensor | Shape | Dimensions meaning |
|---|---|---|
| `X_train` | (48885, 120, 16) | 48,885 samples, each 120 time steps, each 16 features |
| `y_train` | (48885, 24) | 48,885 samples, each 24 future temperature values |
| `X_val` | (~10363, 120, 16) | Same structure, fewer samples |
| `y_val` | (~10363, 24) | |
| `X_test` | (10364, 120, 16) | |
| `y_test` | (10364, 24) | |

### How the GRU reads these

The GRU processes each X sample as a **sequence**:

```
Time step 0:   [T, p, rh, wv, max_wv, wd, hour_sin, ..., gust_ratio]  (16 values)
Time step 1:   [T, p, rh, wv, max_wv, wd, hour_sin, ..., gust_ratio]  (16 values)
...
Time step 119: [T, p, rh, wv, max_wv, wd, hour_sin, ..., gust_ratio]  (16 values)
```

At each time step, the GRU reads the 16-dimensional feature vector, updates its hidden state, and after processing all 120 steps, produces a prediction vector of 24 future temperatures.

### Batch processing

During training, the model does not process one sample at a time. It processes a **batch** (e.g., 128 samples simultaneously). So the actual tensor fed to the model has shape `(128, 120, 16)`, but conceptually each sample in the batch is independent.

---

## 12. Sanity Check

### What happens

```python
print("Input shape:", X_train.shape[1:])           # (120, 16)
print("Forecast horizon:", y_train.shape[1])        # 24
print("Number of input features:", X_train.shape[2]) # 16

print("Example input window shape:", X_train[0].shape)  # (120, 16)
print("Example target shape:", y_train[0].shape)         # (24,)
print("First 5 target values (scaled):", y_train[0][:5])
```

### What to verify

| Check | Expected | Why |
|---|---|---|
| Input shape `(120, 16)` | Matches LOOKBACK x N_features | Confirms the window length and feature count are correct |
| Horizon `24` | Matches HORIZON | Confirms the output length is correct |
| Target values are scaled | Small numbers around 0 | Confirms scaling was applied before windowing |
| No NaN in tensors | All finite values | NaN would cause training to fail silently (loss becomes NaN) |

### Off-by-one errors

The sanity check exists specifically to catch **off-by-one errors** in the windowing loop. A common bug: using `range(len(data) - lookback - horizon)` instead of `range(len(data) - lookback - horizon + 1)`, which would lose one valid sample. Or slicing `data[i:i+lookback+1]` instead of `data[i:i+lookback]`, which would add an extra time step to every window.

---

## 13. Complete Data Pipeline Summary

After Section 6, the entire data pipeline from raw CSV to model-ready tensors is complete:

```
jena_climate_2009_2016.csv
    |
    v  Section 3: Load, clean, resample to hourly, select 6 variables
    |
df_model (70,041 rows x 7 cols)
    |
    v  Section 5: Feature engineering (+4 time + 6 wind = 16 features)
    |
df_feat (70,041 rows x 16 features + datetime)
    |
    v  Section 6.1: Chronological split (70/15/15)
    |
df_train (49,028) | df_val (10,506) | df_test (10,507)
    |
    v  Section 6.2: Scale (fit on train, transform all)
    |
X_train_scaled (49028, 16) | X_val_scaled (10506, 16) | X_test_scaled (10507, 16)
    |
    v  Section 6.4: Sliding window (lookback=120, horizon=24)
    |
X_train (48885, 120, 16)  y_train (48885, 24)
X_val   (10363, 120, 16)  y_val   (10363, 24)
X_test  (10364, 120, 16)  y_test  (10364, 24)
    |
    v  Ready for GRU training (Section 7+)
```

Every step is irreversible in the pipeline but reversible mathematically -- predictions can be **inverse-scaled** back to degrees Celsius using `scaler.mean_` and `scaler.scale_` (for StandardScaler) or equivalent attributes. This inverse transformation is used in Sections 7 and 10 to report metrics in physically interpretable units.
