# Notebook Study Guide: Advanced Time Series Forecasting, Optimization and Explainability

**Course:** Advanced Topics in Deep Learning (2025/2026) -- Final Project  
**Authors:** Antonio Cruz (140129), Catia Bras (120093), Ricardo Kyaseller (95813)  
**Notebook:** `final_project_v8A.ipynb` (311 cells)

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Environment and Reproducibility (Section 2)](#2-environment-and-reproducibility)
3. [Dataset and Problem Definition (Section 3)](#3-dataset-and-problem-definition)
4. [Data Preparation (Section 4)](#4-data-preparation)
5. [Feature Engineering (Section 5)](#5-feature-engineering)
6. [Split, Scaling and Windowing (Section 6)](#6-split-scaling-and-windowing)
7. [Baseline Models (Section 7)](#7-baseline-models)
8. [Synthetic Data Generation with TimeGAN (Section 8)](#8-synthetic-data-generation-with-timegan)
9. [Evolutionary Optimization (Section 9)](#9-evolutionary-optimization)
10. [Retraining and Final Model Selection (Section 10)](#10-retraining-and-final-model-selection)
11. [Explainable AI (Section 11)](#11-explainable-ai)
12. [Efficiency, Latency and Resource Analysis (Section 12)](#12-efficiency-latency-and-resource-analysis)
13. [Comparative Discussion (Section 13)](#13-comparative-discussion)
14. [Future Work (Section 14)](#14-future-work)
15. [Conclusion (Section 15)](#15-conclusion)
16. [Summary of Tables and Figures](#16-summary-of-tables-and-figures)

---

## 1. Project Goal

The notebook implements an **end-to-end forecasting pipeline** for **24-hour air temperature prediction** using the **Jena Climate dataset**. The core task is a **multivariate-input, univariate-output, multi-step prediction** problem: given a window of historical meteorological measurements (temperature, pressure, humidity, wind speed, max wind speed, wind direction), predict air temperature for the next 24 hours.

The project integrates five advanced techniques:

| Technique | Purpose |
|---|---|
| **GRU Baseline** | Reference deep learning model with regularization, gradient clipping, AdamW, Huber loss |
| **Evolutionary Optimization (GA)** | Automatic search over the entire pipeline configuration (17 dimensions) |
| **TimeGAN** | Synthetic weather sequence generation for data augmentation feasibility |
| **Explainable AI (XAI)** | Global (permutation importance) and local (gradient saliency) explanations + feature pruning |
| **Efficiency Analysis** | Training time, inference latency, parameter counts, memory profiling |

---

## 2. Environment and Reproducibility

**Cells 4--13** | Methodology section

### What is done

- Import all required libraries: NumPy, Pandas, TensorFlow/Keras, Matplotlib, scikit-learn, psutil
- Set a **global random seed** (`SEED = 42`) across Python, NumPy, and TensorFlow for reproducibility
- Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory upfront
- Set up the project root path and add it to `sys.path` for importing custom modules from `src/`
- Import and verify custom utility functions from `src.utils.env`

### Key implementation details

```python
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

GPU memory growth is enabled to prevent out-of-memory errors when running multiple models sequentially. The project uses a modular code structure with reusable modules under `src/`.

---

## 3. Dataset and Problem Definition

**Cells 14--42** | Sections 3.1--3.7

### 3.1 Load Data

The **Jena Climate dataset** is loaded from `src/data/jena_climate_2009_2016.csv`. It contains meteorological measurements collected at approximately 10-minute intervals from 2009 to 2016, with 15 variables.

### 3.2 Initial Inspection

- Check dataset shape, column names, data types
- Identify and remove duplicated rows
- Verify no missing values exist

### 3.3 Datetime Parsing and Temporal Ordering

- Parse `Date Time` column to proper datetime objects (`dayfirst=True`)
- Sort by datetime and reset index
- Inspect time delta distribution to detect sampling irregularities

### 3.4 Hourly Resampling

The raw 10-minute data (~420,224 rows) is resampled to **1-hour intervals** using `resample("1h").mean()`:

- Reduces dataset from **420,224 to ~70,129 rows** (sixfold reduction)
- Preserves dominant meteorological patterns relevant for day-ahead forecasting
- Reduces computational cost

### 3.5--3.6 Post-Resampling Quality Check

- Re-validate for NaN values introduced by resampling
- Drop any rows with NaN and reset index

### 3.7 Variable Selection

Six meteorological variables are retained (plus datetime):

| Variable | Description |
|---|---|
| `T (degC)` | Air temperature -- **the target** |
| `p (mbar)` | Atmospheric pressure |
| `rh (%)` | Relative humidity |
| `wv (m/s)` | Wind speed |
| `max. wv (m/s)` | Maximum wind speed |
| `wd (deg)` | Wind direction |

All other variables from the original 15-column dataset are excluded.

---

## 4. Data Preparation

**Cells 43--47** | Section 4

### Target and Base DataFrame

- **Target variable:** `T (degC)` (air temperature in degrees Celsius)
- `Date Time` is used only for feature engineering (extracting hour/day-of-year), not as a direct model input
- The remaining 5 meteorological variables serve as exogenous inputs
- A `feature_cols` list is defined excluding the `TIME_COL`

---

## 5. Feature Engineering

**Cells 48--57** | Section 5

Two families of derived features are created, all implemented in `src.features.engineering`.

### 5.1 Cyclical Time Features (4 new features)

Hour-of-day and day-of-year are encoded as sine-cosine pairs to avoid discontinuities (e.g., hour 23 being far from hour 0):

| Feature | Encoding | Period |
|---|---|---|
| `hour_sin` | sin(2*pi * hour/24) | 24h (diurnal) |
| `hour_cos` | cos(2*pi * hour/24) | 24h (diurnal) |
| `doy_sin` | sin(2*pi * dayofyear/365.25) | 365.25 days (seasonal) |
| `doy_cos` | cos(2*pi * dayofyear/365.25) | 365.25 days (seasonal) |

### 5.2 Wind-Derived Features (6 new features)

| Feature | Description |
|---|---|
| `wd_sin`, `wd_cos` | Cyclic encoding of wind direction |
| `wx`, `wy` | Cartesian decomposition of wind vector (speed x cos/sin of direction) |
| `wind_gap` | Difference between max and sustained wind speed (gust intensity) |
| `gust_ratio` | Ratio of max to sustained wind speed (epsilon = 1e-6 for stability) |

### 5.3 Final Feature Set

The complete input vector has **16 dimensions**:
- 6 original meteorological variables
- 4 cyclical temporal encodings
- 6 wind-derived features

This feature set is used consistently across all experiments (baseline, EA, TimeGAN). The XAI analysis later evaluates which features matter most.

---

## 6. Split, Scaling and Windowing

**Cells 58--74** | Section 6

### 6.1 Temporal Train/Validation/Test Split

Chronological split with **no shuffling** (prevents data leakage):

| Split | Fraction | Purpose |
|---|---|---|
| Train | 70% | Model training (earliest years) |
| Validation | 15% | Hyperparameter tuning, EA fitness |
| Test | 15% | Final evaluation (most recent data, untouched during optimization) |

### 6.2 Feature Scaling

- **Baseline:** `StandardScaler` (zero mean, unit variance)
- Scaler is **fit only on training data**, then applied to validation and test sets
- The EA also searches over scaler type: `standard`, `robust`, `minmax`
- Implementation in `src.features.scaling.get_scaler()`

### 6.3 Target Index

The position of `T (degC)` within the 16-feature vector is identified. This index is needed to extract the target column from windowed data.

### 6.4 Supervised Windowing

The scaled multivariate time series is converted into supervised learning samples via sliding windows:

| Parameter | Value | Description |
|---|---|---|
| `LOOKBACK` | 120 hours (baseline) | Length of input history window |
| `HORIZON` | 24 hours | Length of forecast output |

- **Input X shape:** `(LOOKBACK, 16)` -- multivariate input
- **Output y shape:** `(HORIZON,)` -- univariate output (temperature only)
- Implementation in `src.features.windowing.make_windows()`

### 6.5 Sanity Check

Final verification of windowed tensor shapes and consistency.

---

## 7. Baseline Models

**Cells 75--113** | Section 7

### 7.1 Persistence Baseline

The simplest possible forecast: repeat the last observed temperature across the entire 24-hour horizon.

- Prediction tensor shape: `(10364, 24)`
- Serves as a "no-skill" reference; any learned model should beat this

### 7.2 Quick Sanity GRU

Before the official baseline, a simple 1-layer GRU with 64 units is trained as a sanity check to confirm the pipeline works and beats persistence.

### 7.3 Official GRU Baseline

The official baseline uses the best configuration from the previous mini-project:

```python
BASELINE_CFG = {
    "n_layers": 2,
    "units1": 96,        # First GRU layer
    "units2": 64,        # Second GRU layer
    "units3": 96,        # (not used with 2 layers)
    "dropout": 0.0,
    "l2": 1e-6,
    "dense_units": 256,  # Intermediate dense layer
    "dense_activation": "relu",
    "learning_rate": 2e-4,
    "clipnorm": 2.0,
    "optimizer_name": "adamw",
    "weight_decay": 1e-6,
    "loss_name": "huber1",  # Huber loss with delta=1
    "gaussian_noise_std": 0.0,
    "batch_size": 128,
}
```

- Built with `build_gru_model()` from `src.models.gru`
- Trained with `train_model()` from `src.models.train_eval` for 60 epochs with early stopping (patience=6)
- **86,744 trainable parameters**

### 7.4--7.6 Evaluation and Comparison

Metrics are computed in both **scaled space** (normalized) and **original temperature scale** (degrees Celsius) via inverse scaling.

| Model | MAE (degC) | RMSE (degC) |
|---|---:|---:|
| Persistence | 3.144 | 4.254 |
| GRU Baseline Official | 1.650 | 2.193 |

### 7.7 Forecast Visualization

**Figure 1** plots a single 24-hour forecast window comparing ground truth, persistence, and GRU baseline predictions.

---

## 8. Synthetic Data Generation with TimeGAN

**Cells 114--150** | Section 8

### 8.1 Motivation

Generate realistic synthetic multivariate weather sequences using **only the training split** (no data leakage). The goal is to demonstrate data augmentation feasibility.

### 8.2 Data Preparation

Training data is converted to fixed-length sequences of `LOOKBACK + HORIZON = 144` time steps. Each generated sequence can be split into input-output pairs matching the forecasting setup.

### 8.3 TimeGAN Architecture

TimeGAN (Yoon et al., 2019) consists of **five GRU-based sub-networks**:

| Network | Function |
|---|---|
| **Embedder** | Data space --> Latent space |
| **Recovery** | Latent space --> Data space |
| **Generator** | Noise --> Latent space |
| **Supervisor** | Temporal dynamics in latent space |
| **Discriminator** | Real vs. synthetic classification |

Each uses 3 GRU layers with hidden dimension 24.

### Training Procedure (3 phases)

**Phase 1 -- Autoencoder Pretraining (Section 8.3.1):**
- Embedder + Recovery jointly trained to minimize reconstruction error (MSE)
- Loss: 0.280 --> 0.0023 over 20 epochs
- **Figure 2:** Autoencoder loss curve

**Phase 2 -- Supervisor Pretraining (Section 8.3.2):**
- Supervisor trained to predict next latent time step
- Loss: 0.0308 --> 0.0076 over 20 epochs
- **Figure 3:** Combined pretraining losses

**Phase 3 -- Adversarial Training (Section 8.3.4):**
- Generator and discriminator optimized adversarially
- Generator objective = adversarial loss + supervised loss (x100 weight) + reconstruction loss (gamma=1.0)
- Discriminator update threshold prevents early dominance
- **Figure 4:** Adversarial training loss curves (discriminator, generator, supervised, reconstruction)

### 8.4 Quality Assessment

- **Reconstruction check:** Real sequences passed through embedder --> recovery; MSE measured
- **Figure 5:** Real vs. reconstructed temperature sequence
- **Figure 6:** Real vs. synthetic (fully generated) sequence comparison
- **Conclusion:** Good reconstruction quality but insufficient synthetic realism for safe integration into the final pipeline. TimeGAN was **not used** in the final forecasting pipeline.

---

## 9. Evolutionary Optimization

**Cells 151--170** | Section 9

### 9.1 Motivation

Improve the forecasting pipeline beyond the fixed baseline by systematically exploring alternative configurations using a **genetic algorithm (GA)**, replacing the Bayesian optimization from previous work.

### 9.2 Search Space (17 dimensions)

| Category | Parameters | Range |
|---|---|---|
| **Architecture** | `n_layers` | 1--2 |
| | `units1`, `units2` | 64--128 |
| | `units3` | 32--96 |
| | `dense_units` | 0--256 |
| | `dense_activation` | ReLU, GELU, ELU, LeakyReLU |
| **Regularization** | `dropout` | 0.0--0.3 |
| | `l2` | 0 to 1e-4 |
| | `gaussian_noise_std` | 0.0 or 0.01 |
| | `clipnorm` | 0.5--5.0 |
| **Training** | `optimizer_name` | Adam, AdamW |
| | `weight_decay` | 0 to 1e-4 |
| | `learning_rate` | 1e-4 to 1e-3 |
| | `loss_name` | MSE, MAE, Huber(d=1), Huber(d=2) |
| | `batch_size` | 128, 256 |
| **Preprocessing** | `scaler_name` | standard, robust, minmax |
| **Windowing** | `lookback` | 96, 120, 144 hours |

### 9.3 Representation and Constraints

Each individual is a **genotype** (dictionary mapping gene names to values). Constraints enforce valid configurations (e.g., GRU layer widths are constrained). Implementation in `src.evolution.genotype`.

### 9.4 Fitness Function

The fitness function encapsulates the **full pipeline** for each candidate:
1. Feature scaling with candidate's scaler
2. Supervised window generation with candidate's lookback
3. GRU model construction with candidate's architecture
4. Training on training split
5. Evaluation on **validation set only** (MAE in degC)

Implementation in `src.evolution.fitness.evaluate_individual()`.

### 9.5 Search Execution

| GA Parameter | Value |
|---|---|
| Population size | 20 |
| Generations | 15 |
| Total evaluations | 300 |
| Selection | Tournament (k=3) |
| Crossover | Uniform |
| Mutation rate | 0.2 per gene |
| Elitism | Yes |
| Training budget per individual | 20 epochs (with early stopping) |
| Fitness metric | Validation MAE (degC) |

Implementation in `src.evolution.ga_evolutionary_search.run_evolutionary_search()`.

### 9.6 Best Configuration Found

```
n_layers = 2          units1 = 64           units2 = 64
units3 = 32           dropout = 0.0         l2 = 1e-5
dense_units = 256     dense_activation = relu
learning_rate = 3e-4  batch_size = 256      clipnorm = 5.0
optimizer_name = adamw weight_decay = 0.0   loss_name = mae
gaussian_noise_std = 0.0
scaler_name = robust  lookback = 144
```

**Key discoveries vs. baseline:**
- **RobustScaler** instead of StandardScaler
- **MAE loss** instead of Huber
- **Compact architecture** (64-->64 vs 96-->64) with fewer parameters
- **144-hour lookback** instead of 120 (windowing was effectively optimized)
- No dropout or Gaussian noise needed

Best validation MAE: ~1.636 degC.

---

## 10. Retraining and Final Model Selection

**Cells 171--214** | Section 10

### 10.1 Final Candidates

Two models are compared under identical retraining conditions:
1. **GRU Baseline Official** (from mini-project)
2. **Best Evolutionary GRU** (from GA search)

### 10.2 Retraining Procedure

Both models are retrained from scratch with expanded budget:
- **50 epochs** (vs. 20 during EA search)
- Same callbacks: early stopping, learning rate reduction
- Baseline uses StandardScaler + 120h lookback
- EA model uses RobustScaler + 144h lookback

Data preparation handled by `prepare_data_for_cfg()` helper (dynamic lookback and scaler support).

**Figure 7:** Baseline training/validation loss curves  
**Figure 8:** EA-optimized training/validation loss curves

### 10.3 Test Set Evaluation

| Model | MAE scaled | RMSE scaled | MAE (degC) | RMSE (degC) |
|---|---:|---:|---:|---:|
| GRU Baseline Official | 0.1909 | 0.2537 | 1.650 | 2.193 |
| Best Evolutionary GRU | 0.1295 | 0.1748 | 1.598 | 2.157 |

The evolutionary model reduced MAE by **3.2%** relative to baseline.

### 10.6 Robustness Across Random Seeds

The best EA configuration is re-evaluated across seeds **7, 21, 42** to confirm results aren't dependent on a single initialization.

| Seed | MAE (degC) | RMSE (degC) |
|---|---:|---:|
| 7 | ~1.59x | ~2.15x |
| 21 | ~1.59x | ~2.15x |
| 42 | ~1.59x | ~2.15x |
| **Mean +/- std** | **1.593 +/- 0.016** | **2.151 +/- 0.023** |

---

## 11. Explainable AI

**Cells 215--258** | Section 11

### 11.1 Global Explainability -- Permutation Importance

**Method:** Shuffle each feature's values across samples and measure the increase in MAE. Larger increase = more important feature.

Applied to the Best Evolutionary GRU on the test set.

**Key findings:**
- **`T (degC)`** (past temperature) is by far the most influential
- Cyclical temporal features (`doy_cos`, `hour_cos`, `hour_sin`) are highly important
- Wind-derived and meteorological variables contribute moderately
- Some features have minimal importance (candidates for pruning)

**Figure 9:** Permutation importance bar chart  
Implementation in `src.xai.permutation.permutation_feature_importance()`.

### 11.2 Local Explainability -- Gradient Saliency

**Method:** Compute gradient of model output w.r.t. input using `GradientTape`. Absolute gradient magnitude indicates influence.

Implementation in `src.xai.saliency`.

Three views are presented for sample 0:

**11.2.1 Feature-Level (Figure 10):**
- Aggregated saliency across time steps for each feature
- `T (degC)` dominant, followed by `hour_cos`, `doy_cos`, `p (mbar)`

**11.2.2 Time-Step Level (Figure 11):**
- Aggregated saliency across features at each time step
- Clear concentration of importance near **most recent observations**
- Sharp increase in the final segment of the input window

**11.2.3 Saliency Heatmap (Figure 12):**
- Full 2D heatmap (features x time steps)
- Attribution overwhelmingly in the **final input segment**, focused on `T (degC)` and `p (mbar)`

### 11.4 XAI-Guided Feature Pruning

Based on permutation importance, the **5 least important features** are removed:

| Removed Features |
|---|
| `doy_sin` |
| `gust_ratio` |
| `wd (deg)` |
| `wd_sin` |
| `wy` |

This reduces input dimensionality from **16 to 11 features**.

**Retained features:** `T (degC)`, `p (mbar)`, `rh (%)`, `wv (m/s)`, `max. wv (m/s)`, `hour_sin`, `hour_cos`, `doy_cos`, `wd_cos`, `wx`, `wind_gap`

The pruned model is retrained with the same architecture and hyperparameters.

### 11.5 Pruning Results

| Model | Features | MAE (degC) | RMSE (degC) |
|---|---:|---:|---:|
| Best Evolutionary GRU | 16 | 1.598 | 2.157 |
| **Pruned Evolutionary GRU** | **11** | **1.575** | **2.134** |

Multi-seed robustness (seeds 7, 21, 42):
- Pruned: **1.579 +/- 0.009 degC** MAE
- Full: **1.593 +/- 0.016 degC** MAE
- Pruning improved both accuracy and stability

**Interpretation:** The removed features were not just uninformative but slightly detrimental (redundancy/noise). XAI served as a principled mechanism for feature selection.

---

## 12. Efficiency, Latency and Resource Analysis

**Cells 259--304** | Section 12

Three models are profiled: Baseline, Best EA GRU, Pruned EA GRU.

### 12.3 Training Time

| Model | Params | Total Time | Per Epoch |
|---|---:|---:|---:|
| GRU Baseline Official | 86,744 | ~39s | ~7.8s |
| Best Evolutionary GRU | 63,512 | ~21s | ~4.2s |
| Pruned Evolutionary GRU | 62,552 | ~19s | fastest |

The EA model trains **~46% faster** than the baseline.

### 12.4 Memory Usage

RAM consumption measured before/after training. GPU memory tracked where available.

### 12.5 Trainable Parameters

| Model | Parameters |
|---|---:|
| GRU Baseline | 86,744 |
| Best Evolutionary GRU | 63,512 (27% fewer) |
| Pruned Evolutionary GRU | 62,552 (28% fewer) |

### 12.6 Inference Latency

Measured with warmup runs + multiple repetitions (batch of 32 samples):
- All models: ~12--13ms per batch
- Evolutionary models slightly faster
- Throughput reported in samples/second

### Visualizations

- **Figure 12:** Inference latency comparison (mean +/- std)
- **Figure 13:** Throughput comparison (samples/sec)
- **Figure 14:** Trainable parameter count comparison
- **Figure 15:** Accuracy vs. efficiency scatter (MAE vs. parameters)

### 12.8 Key Takeaway

The evolutionary optimization did **not** trade accuracy for complexity. It found a configuration that is simultaneously **more accurate, more compact, and more efficient** than the hand-tuned baseline.

---

## 13. Comparative Discussion

**Cell 308** | Section 13

### Final Results Summary

| Model | MAE (degC) | RMSE (degC) | Params | vs. Baseline |
|---|---:|---:|---:|---:|
| Persistence | 3.144 | 4.254 | -- | -- |
| GRU Baseline Official | 1.650 | 2.193 | 86,744 | reference |
| Best Evolutionary GRU | 1.598 | 2.157 | 63,512 | -3.2% MAE |
| **Pruned Evolutionary GRU** | **1.575** | **2.134** | **62,552** | **-4.5% MAE** |

Best observed single-seed result: **~1.569 degC MAE** (Pruned model).

### Key Insights from the EA

- **RobustScaler** > StandardScaler for this dataset (more outlier-resistant)
- **MAE loss** > Huber loss (direct alignment with evaluation objective)
- **Compact 64-->64 GRU** > wider 96-->64 (better generalization)
- **144h lookback** > 120h (temporal context matters and should be optimized)
- Little benefit from dropout or Gaussian noise in the best configurations

### Robustness Validation

| Model | MAE mean +/- std (degC) | RMSE mean +/- std (degC) |
|---|---:|---:|
| Best Evolutionary GRU | 1.593 +/- 0.016 | 2.151 +/- 0.023 |
| Pruned Evolutionary GRU | 1.579 +/- 0.009 | 2.134 +/- 0.010 |
| Best single seed (Pruned) | 1.569 | 2.123 |

### Strengths

- End-to-end pipeline with temporal split, leakage prevention, untouched test set
- EA applied to preprocessing, architecture, training, and windowing jointly
- Explicit lookback optimization (windowing not fixed)
- Multi-seed robustness analysis
- XAI analysis followed by principled feature pruning
- Final model is more accurate, simpler, and more efficient than baseline

### Limitations

- EA budget explores only a small portion of the search space
- Forecast horizon (H=24) was fixed; only lookback was optimized
- Only GRU family explored (no transformers/attention models)
- TimeGAN achieved good reconstruction but insufficient synthetic realism

---

## 14. Future Work

- **Larger EA budget:** broader search space, multi-objective formulations
- **Alternative architectures:** Transformers, temporal CNNs, hybrid recurrent-attention
- **Stronger robustness:** more seeds, rolling-origin backtesting
- **Improved TimeGAN:** longer training, diffusion-based generators, better conditioning
- **Iterative XAI loop:** repeated explain --> prune --> validate cycles
- **Deployment focus:** GPU profiling, energy consumption, real-time latency constraints

---

## 15. Conclusion

The project shows that combining a strong baseline with evolutionary optimization, explainability, and XAI-guided pruning produces a pipeline that is **more accurate, more compact, more interpretable, and more efficient** than manual tuning.

The **Pruned Evolutionary GRU** is the final selected model:
- **4.5% MAE improvement** over the GRU baseline
- **28% fewer parameters**
- Best single-seed result: **1.569 degC MAE**
- Closed-loop methodology: **optimize --> explain --> prune --> validate**

---

## 16. Summary of Tables and Figures

### Tables

| Table | Description | Section |
|---|---|---|
| Table 1 | First rows of raw Jena Climate dataset | 3.1 |
| Table 2 | Date range after datetime parsing | 3.3 |
| Table 3 | Selected 6 meteorological variables | 3.7 |
| Table 4 | Cyclical temporal features | 5.1 |
| Table 5 | Wind-derived features | 5.2 |
| Table 6 | Persistence vs. GRU baseline (scaled metrics) | 7.3 |
| Table 7 | Final baseline comparison (scaled + degC) | 7.6 |
| Table 8 | Top EA candidate configurations | 9.7 |
| Table 9 | Final candidate configurations (baseline vs. EA) | 10.1 |
| Table 10 | Final model comparison (test set) | 10.4 |
| Table 11 | EA model across 3 random seeds | 10.6 |
| Table 12 | Permutation feature importance ranking | 11.1 |
| Table 13 | Local feature-level gradient saliency | 11.2.1 |
| Table 15 | Feature pruning: 16 vs. 11 features | 11.4 |
| Table 16 | Pruned model across 3 seeds | 11.4 |
| Tables 17--22 | Efficiency metrics (training time, memory, params, latency) | 12 |
| Table 23 | Final comprehensive comparison | 12.7 |
| Table 24 | Robustness across seeds (all models) | 12.7 |

### Figures

| Figure | Description | Section |
|---|---|---|
| Figure 1 | 24h forecast: truth vs. persistence vs. GRU baseline | 7.7 |
| Figure 2 | TimeGAN autoencoder pretraining loss | 8.3.1 |
| Figure 3 | Combined pretraining losses (autoencoder + supervisor) | 8.3.2 |
| Figure 4 | Adversarial training loss curves | 8.3.4 |
| Figure 5 | Real vs. reconstructed temperature sequence | 8.4.2 |
| Figure 6 | Real vs. synthetic sequence | 8.4.4 |
| Figure 7 | Baseline GRU training/validation loss | 10.2 |
| Figure 8 | EA GRU training/validation loss | 10.2 |
| Figure 9 | Permutation importance bar chart | 11.1 |
| Figure 10 | Local feature saliency bar chart | 11.2.1 |
| Figure 11 | Temporal saliency profile | 11.2.2 |
| Figure 12 | Full saliency heatmap + Inference latency comparison | 11.2.3 / 12.6 |
| Figure 13 | Throughput comparison | 12.6 |
| Figure 14 | Parameter count comparison | 12.7 |
| Figure 15 | Accuracy vs. efficiency scatter | 12.7 |

---

## Key Modules Referenced

| Module | Purpose |
|---|---|
| `src.utils.env` | Seed setting, GPU config, device info |
| `src.features.engineering` | `add_time_features()`, `add_wind_features()`, `get_final_feature_columns()` |
| `src.features.scaling` | `get_scaler()` |
| `src.features.windowing` | `make_windows()` |
| `src.models.gru` | `build_gru_model()` |
| `src.models.train_eval` | `train_model()`, `evaluate_scaled_forecasts()`, `inverse_scale_target()`, `evaluate_original_scale_forecasts()` |
| `src.gan.config` | `TIMEGAN_CONFIG` |
| `src.gan.timegan` | `TimeGAN` class |
| `src.gan.data_prep` | `make_timegan_sequences()`, `split_synthetic_sequences()` |
| `src.evolution.search_space` | `SEARCH_SPACE` |
| `src.evolution.genotype` | `sample_genotype()` |
| `src.evolution.fitness` | `evaluate_individual()` |
| `src.evolution.ga_evolutionary_search` | `run_evolutionary_search()` |
| `src.evolution.phenotype` | `inverse_target_with_scaler()` |
| `src.xai.permutation` | `permutation_feature_importance()` |
| `src.xai.saliency` | `compute_saliency_map()`, `aggregate_saliency_over_time()`, `aggregate_saliency_over_features()` |
| `src.utils.profiling` | `count_trainable_params()`, `measure_inference_latency()`, `get_ram_usage_mb()` |
