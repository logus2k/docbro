# Section 7 -- Baseline Models

This document explains the goals, concepts, and implementation details of Section 7 of the notebook, which establishes the two reference models against which all subsequent improvements are measured: the persistence baseline and the official GRU baseline.

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [Why Baselines Matter](#2-why-baselines-matter)
3. [Persistence Baseline](#3-persistence-baseline)
4. [Quick Sanity GRU](#4-quick-sanity-gru)
5. [Official GRU Baseline -- Architecture](#5-official-gru-baseline----architecture)
6. [The GRU Model Builder -- Layer by Layer](#6-the-gru-model-builder----layer-by-layer)
7. [Optimizer and Loss Configuration](#7-optimizer-and-loss-configuration)
8. [Training Procedure and Callbacks](#8-training-procedure-and-callbacks)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Inverse Scaling -- Back to Degrees Celsius](#10-inverse-scaling----back-to-degrees-celsius)
11. [Baseline Results](#11-baseline-results)
12. [Forecast Visualization](#12-forecast-visualization)
13. [Parameter Count Breakdown](#13-parameter-count-breakdown)
14. [What the Baseline Establishes for Later Sections](#14-what-the-baseline-establishes-for-later-sections)

---

## 1. Section Goal

Establish two reference points for performance:

| Baseline | Type | Purpose |
|---|---|---|
| **Persistence** | No-skill benchmark | Proves the model is doing something useful (any learned model must beat this) |
| **GRU Official** | Strong deep learning benchmark | The target the EA must improve upon |

Everything from Section 9 onward (evolutionary optimization, XAI, pruning) is evaluated by asking: "Is this better than the GRU baseline?"

---

## 2. Why Baselines Matter

Without baselines, a model's MAE of 1.65 degC is meaningless. Is that good or bad? Compared to what?

| Question | Baseline that answers it |
|---|---|
| "Is the model learning anything at all?" | Persistence (3.14 degC) -- if the model can't beat repeating the last value, it's useless |
| "Is the EA optimization actually helping?" | GRU baseline (1.65 degC) -- if the EA can't beat a well-tuned hand-crafted model, the search was wasted |

Baselines transform raw numbers into **relative improvements** that are meaningful.

---

## 3. Persistence Baseline

### The strategy

The simplest possible forecast: whatever the temperature was at the last observed hour, assume it stays the same for the next 24 hours.

### Implementation

```python
def persistence_forecast(X):
    last_temp = X[:, -1, target_idx]                     # last time step, temperature column
    return np.repeat(last_temp[:, None], HORIZON, axis=1) # repeat 24 times
```

Step by step:
- `X[:, -1, target_idx]` -- From each sample's input window, take the **last time step** (`-1`) and the **temperature column** (`target_idx = 0`). Result shape: `(10364,)` -- one temperature per sample.
- `last_temp[:, None]` -- Add a dimension to make it `(10364, 1)` -- needed for the repeat operation.
- `np.repeat(..., HORIZON, axis=1)` -- Copy that single value 24 times along the time axis. Result shape: `(10364, 24)`.

### What the prediction looks like

```
True temperature:         [12.3, 12.8, 13.5, 14.1, 14.6, ...]   (changes over 24h)
Persistence prediction:   [12.3, 12.3, 12.3, 12.3, 12.3, ...]   (flat line)
```

### Why it's still useful

Persistence is a surprisingly strong baseline for short horizons. For the next 1-2 hours, temperature rarely changes dramatically, so persistence can be hard to beat. Over 24 hours, however, diurnal cycles cause significant changes (cold at night, warm in afternoon), which persistence completely misses.

---

## 4. Quick Sanity GRU

Before the official baseline, a simple GRU is built as a development checkpoint:

```python
def build_gru_baseline(input_shape, horizon):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(horizon)
    ])
    model.compile(optimizer=Adam(lr=1e-3), loss="mse", metrics=["mae"])
    return model
```

This is a **throwaway model** -- its results are not used for comparison. Its purpose is to verify that the entire pipeline (data loading, windowing, training, prediction, evaluation) works end-to-end before committing to the full official baseline.

| Property | Sanity GRU | Official GRU |
|---|---|---|
| Layers | 1 GRU (64 units) | 2 GRU (96, 64 units) |
| Optimizer | Adam | AdamW |
| Loss | MSE | Huber (delta=1) |
| Regularization | Dropout only | L2 + gradient clipping |
| Purpose | Pipeline check | Official benchmark |

---

## 5. Official GRU Baseline -- Architecture

### Configuration

```python
BASELINE_CFG = {
    "n_layers": 2,              # 2 stacked GRU layers
    "units1": 96,               # first GRU: 96 hidden units
    "units2": 64,               # second GRU: 64 hidden units
    "units3": 96,               # (unused -- only needed if n_layers=3)
    "dropout": 0.0,             # no dropout
    "l2": 1e-6,                 # very light L2 regularization
    "dense_units": 256,         # intermediate dense layer
    "dense_activation": "relu", # ReLU activation for dense layer
    "learning_rate": 2e-4,      # learning rate (0.0002)
    "clipnorm": 2.0,            # gradient clipping threshold
    "optimizer_name": "adamw",  # AdamW optimizer
    "weight_decay": 1e-6,       # AdamW weight decay
    "loss_name": "huber1",      # Huber loss with delta=1.0
    "gaussian_noise_std": 0.0,  # no Gaussian noise
    "batch_size": 128,          # training batch size
}
```

This configuration comes from the **previous mini-project** where it was identified as the best-performing setup. Reusing it ensures methodological continuity.

### Architecture diagram

```
Input (120, 16)
    |
    v
GRU Layer 1 (96 units, return_sequences=True)  --> outputs (120, 96)
    |                                                for each of 120 time steps,
    v                                                produces 96-dim hidden state
GRU Layer 2 (64 units, return_sequences=False)  --> outputs (64,)
    |                                                only the LAST hidden state
    v
Dense (256 units, ReLU activation)              --> outputs (256,)
    |
    v
Dropout (0.0 -- effectively disabled)
    |
    v
Dense (24 units, no activation)                 --> outputs (24,)
    |                                                one value per forecast hour
    v
Output: 24-hour temperature prediction
```

---

## 6. The GRU Model Builder -- Layer by Layer

The model is built in `src/models/gru.py` using the Keras Functional API:

### Input layer

```python
inputs = keras.Input(shape=(L, n_features))  # (120, 16)
```

Defines the expected input shape. `L` is the lookback window, `n_features` is the number of input features.

### Optional Gaussian noise

```python
if gaussian_noise_std and gaussian_noise_std > 0:
    x = layers.GaussianNoise(gaussian_noise_std)(x)
```

Adds random noise to inputs during training as a regularization technique. Disabled in the baseline (`std = 0.0`).

### GRU layers

```python
# Layer 1: returns full sequence (needed as input for layer 2)
x = layers.GRU(units1, return_sequences=(n_layers >= 2), dropout=dropout,
               recurrent_dropout=0.0, kernel_regularizer=reg)(x)

# Layer 2: returns only the last hidden state
if n_layers >= 2:
    x = layers.GRU(units2, return_sequences=(n_layers == 3), dropout=dropout,
                   recurrent_dropout=0.0, kernel_regularizer=reg)(x)
```

**`return_sequences`:** This parameter controls what the GRU layer outputs:
- `True` -- outputs a hidden state for **every** time step. Shape: `(batch, 120, units)`. Needed when another GRU layer follows (it needs a sequence as input).
- `False` -- outputs only the **last** hidden state. Shape: `(batch, units)`. Used for the final GRU layer, since we only need one summary vector to produce the forecast.

**`kernel_regularizer=reg`:** Applies L2 penalty to the GRU's input weights, discouraging large values.

**`recurrent_dropout=0.0`:** Dropout applied to recurrent connections. Set to zero because recurrent dropout can destabilize GRU training and is generally less effective than standard dropout.

### Dense projection layer

```python
if dense_units and dense_units > 0:
    x = layers.Dense(dense_units, activation=dense_activation, kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
```

An intermediate dense layer (256 units with ReLU) that transforms the 64-dimensional GRU output into a higher-dimensional space before the final prediction. This gives the model more capacity to learn complex mappings from the temporal summary to the 24-step forecast.

**ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`. The most common activation function. It introduces non-linearity (essential for learning complex patterns) while being computationally simple.

### Output layer

```python
outputs = layers.Dense(H)(x)  # H = 24, no activation
```

A simple linear layer that maps to 24 output values -- one per forecast hour. **No activation function** because temperature predictions can be any real number (positive or negative), and an activation like ReLU would clip negative values.

---

## 7. Optimizer and Loss Configuration

### Optimizer selection

```python
if optimizer_name.lower() == "adamw":
    opt = keras.optimizers.AdamW(
        learning_rate=learning_rate,   # 2e-4
        weight_decay=weight_decay,     # 1e-6
        clipnorm=clipnorm              # 2.0
    )
else:
    opt = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm
    )
```

The baseline uses **AdamW** with:
- **Learning rate 2e-4:** Relatively conservative -- ensures stable convergence
- **Weight decay 1e-6:** Very light regularization through the optimizer
- **Clipnorm 2.0:** If the gradient vector's norm exceeds 2.0, it is scaled down. Prevents exploding gradients.

### Loss function selection

```python
if loss_name == "mae":
    loss = "mae"
elif loss_name == "huber1":
    loss = keras.losses.Huber(delta=1.0)
elif loss_name == "huber2":
    loss = keras.losses.Huber(delta=2.0)
else:
    loss = "mse"
```

The baseline uses **Huber loss with delta=1.0**. Recap from the concepts document:

```
If |error| <= 1.0:  loss = 0.5 * error^2     (quadratic, like MSE)
If |error| > 1.0:   loss = 1.0 * (|error| - 0.5)  (linear, like MAE)
```

This means:
- Small errors (< 1 degC in scaled space) are penalized quadratically -- the model works hard to minimize them
- Large errors (> 1 degC) are penalized linearly -- outliers don't dominate training

The EA later discovered that plain **MAE** loss worked better, likely because it directly aligns with the evaluation metric.

### Final compilation

```python
model.compile(optimizer=opt, loss=loss, metrics=["mae"])
```

`metrics=["mae"]` means MAE is tracked during training for monitoring purposes, but the optimizer minimizes the **loss** (Huber), not MAE directly.

---

## 8. Training Procedure and Callbacks

### Training function

```python
def train_model(model, X_train, y_train, X_val, y_val, batch_size=128, epochs=60, verbose=1):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_default_callbacks(),
        verbose=verbose
    )
    return history
```

### Callbacks

Two callbacks control training dynamics:

#### Early Stopping

```python
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `monitor` | `val_loss` | Watch the validation loss |
| `patience` | 6 | If val_loss hasn't improved for 6 consecutive epochs, stop training |
| `restore_best_weights` | `True` | After stopping, roll back the model weights to the epoch with the lowest val_loss |

**Why this matters:** Without early stopping, the model might train for 60 epochs, overfitting to training data in the later epochs. With it, training stops at the sweet spot where validation performance peaks.

#### Learning Rate Reduction on Plateau

```python
keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `monitor` | `val_loss` | Watch the validation loss |
| `factor` | 0.5 | Multiply learning rate by 0.5 (halve it) |
| `patience` | 3 | If no improvement for 3 epochs, reduce LR |
| `min_lr` | 1e-5 | Never reduce below 0.00001 |

**Why this matters:** Early in training, a larger learning rate helps the model move quickly toward a good solution. Later, the model is near a minimum and the same step size causes it to oscillate instead of converging. Reducing the learning rate allows finer adjustments.

**Interaction with early stopping:** LR reduction triggers first (patience=3) and gives the model a chance to improve with a smaller step. If it still doesn't improve after 3 more epochs (patience=6 total), early stopping activates.

### What `history` contains

The returned `history` object stores loss and metric values for every epoch:
- `history.history["loss"]` -- training loss per epoch
- `history.history["val_loss"]` -- validation loss per epoch
- `history.history["mae"]` -- training MAE per epoch
- `history.history["val_mae"]` -- validation MAE per epoch

These are used later to plot training curves (Figures 7 and 8 in Section 10).

---

## 9. Evaluation Metrics

Two metrics are computed on the test set:

### MAE (Mean Absolute Error)

```python
MAE = mean(|y_true - y_pred|)
```

The average absolute difference between predicted and actual temperature. Intuitive: "On average, the forecast is off by X degrees."

- Treats all errors equally (a 2-degree error is twice as bad as a 1-degree error)
- Robust to outliers
- Used as the EA's fitness function and the primary evaluation metric

### RMSE (Root Mean Squared Error)

```python
RMSE = sqrt(mean((y_true - y_pred)^2))
```

Like MAE but penalizes large errors more heavily due to squaring.

- A few large errors increase RMSE disproportionately
- Always >= MAE (they are equal only if all errors are the same size)
- Useful as a secondary metric to detect if the model has occasional large misses

### Scaled vs. original metrics

```python
# Scaled -- computed on normalized data
evaluate_scaled_forecasts(y_test, y_pred)

# Original -- computed on inverse-scaled data (degrees Celsius)
evaluate_original_scale_forecasts(y_test_inv, y_pred_inv)
```

| Metric type | Units | Use |
|---|---|---|
| **Scaled** | Dimensionless (standard deviations) | Comparing models trained with the same scaler; monitoring training |
| **Original** | Degrees Celsius | Physical interpretation; final reporting; EA fitness |

Both are reported throughout the notebook, but **original-scale MAE in degC** is the primary metric for all decisions.

---

## 10. Inverse Scaling -- Back to Degrees Celsius

### Why inverse-scale?

The model was trained on scaled (normalized) data. Its predictions are also in scaled space. To interpret them as temperatures, we need to reverse the scaling.

### How it works

For StandardScaler, scaling was:
```
x_scaled = (x - mean) / std
```

So inverse-scaling is:
```
x_original = x_scaled * std + mean
```

### Implementation

```python
target_mean = scaler.mean_[target_idx]     # mean of T (degC) from training data
target_std  = scaler.scale_[target_idx]    # std of T (degC) from training data

def inverse_scale_target(y_scaled, mean, std):
    return y_scaled * std + mean
```

Note: only the temperature column's statistics are needed (not all 16 features), because only temperature is being inverse-scaled (it's the only output).

### Applied to all predictions

```python
y_test_inv              = inverse_scale_target(y_test, target_mean, target_std)
y_pred_persistence_inv  = inverse_scale_target(y_pred_persistence, target_mean, target_std)
y_pred_gru_inv          = inverse_scale_target(y_pred_gru, target_mean, target_std)
```

Now all values are in degrees Celsius and can be compared directly.

---

## 11. Baseline Results

### Scaled metrics

| Model | MAE (scaled) | RMSE (scaled) |
|---|---:|---:|
| Persistence | ~0.36 | ~0.49 |
| GRU Baseline Official | ~0.19 | ~0.25 |

### Original-scale metrics (primary)

| Model | MAE (degC) | RMSE (degC) | Interpretation |
|---|---:|---:|---|
| Persistence | 3.144 | 4.254 | Forecasts are off by ~3.1 degC on average |
| GRU Baseline Official | 1.650 | 2.193 | Forecasts are off by ~1.65 degC on average |

### What this tells us

- The GRU reduces MAE by **47%** compared to persistence (3.14 -> 1.65)
- The GRU captures meaningful temporal patterns that persistence completely misses
- An MAE of 1.65 degC is a strong result -- the 24-hour forecast is typically less than 2 degrees off
- RMSE (2.19) is higher than MAE (1.65), meaning there are some forecasts with larger errors, but the gap is moderate

---

## 12. Forecast Visualization

### What is plotted (Figure 1)

A single 24-hour window from the test set, showing three lines:
- **True** (ground truth) -- actual temperature trajectory with markers
- **Persistence** -- flat line at the last observed temperature
- **GRU Baseline** -- the model's predicted trajectory

```python
sample_idx = 0

plt.plot(y_test_inv[sample_idx], label="True", marker="o")
plt.plot(y_pred_persistence_inv[sample_idx], label="Persistence", linestyle="--")
plt.plot(y_pred_gru_inv[sample_idx], label="GRU Baseline", linestyle="--")
```

### What to observe

| Aspect | Persistence | GRU |
|---|---|---|
| Shape | Flat horizontal line | Follows the general trend |
| Diurnal cycle | Completely missed | Partially captured (warming during day, cooling at night) |
| Amplitude | Zero variation | Under-estimates peaks and troughs |
| Phase | N/A | Roughly aligned with true timing |

The GRU forecast is smoother than reality -- it captures the trend but not the sharp oscillations. This is typical for multi-step forecasts: uncertainty compounds over the horizon, so models tend to "hedge" toward the mean.

---

## 13. Parameter Count Breakdown

The official GRU baseline has **86,744 trainable parameters**:

### GRU parameter formula

Each GRU layer has 3 gates (reset, update, candidate). Per gate:
```
gate_params = input_dim * units + units * units + 2 * units
```
(input weights + recurrent weights + 2 biases per gate in Keras)

Total per GRU layer:
```
layer_params = 3 * (input_dim * units + units * units + 2 * units)
```

### Calculation

| Layer | input_dim | units | Calculation | Parameters |
|---|---:|---:|---|---:|
| GRU 1 | 16 | 96 | 3*(16*96 + 96*96 + 2*96) | 32,832 |
| GRU 2 | 96 | 64 | 3*(96*64 + 64*64 + 2*64) | 31,104 |
| Dense | 64 | 256 | 64*256 + 256 | 16,640 |
| Output | 256 | 24 | 256*24 + 24 | 6,168 |
| **Total** | | | | **86,744** |

For comparison, the EA-optimized model (64->64 GRU) has 63,512 parameters -- **27% fewer**. Despite being smaller, it performs better, showing that the baseline was over-parameterized.

---

## 14. What the Baseline Establishes for Later Sections

The baseline section creates the reference framework used throughout the rest of the notebook:

| What is established | Where it is used later |
|---|---|
| Persistence MAE (3.14 degC) | Minimum bar -- all models are expected to beat this |
| GRU baseline MAE (1.65 degC) | The EA's goal is to find a configuration that beats this |
| `BASELINE_CFG` dictionary | Used in Section 10 for fair retraining comparison |
| `evaluate_scaled_forecasts()` / `evaluate_original_scale_forecasts()` | Same functions used to evaluate EA and pruned models |
| `inverse_scale_target()` | Same function used for all inverse-scaling throughout |
| `train_model()` with standard callbacks | Same training protocol for all final models |
| The 86,744 parameter count | Compared against EA (63,512) and pruned (62,552) in efficiency analysis |

The key principle: **everything uses the same evaluation protocol**. Models are compared on the same test set, using the same metrics, with the same inverse-scaling procedure. This ensures fair comparison.
