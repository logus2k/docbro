# Executive Summary

## Advanced Time Series Forecasting, Optimization and Explainability

**Course:** Advanced Topics in Deep Learning — 2nd Semester 2025/2026
**Team:** António Cruz (140129), Cátia Brás (120093), Ricardo Kyaseller (95813)
**Dataset:** Jena Climate (2009–2016) — hourly temperature forecasting

---

## 1. Problem Statement

The objective of this project was to build a systematic, optimized forecasting system for predicting air temperature 24 hours ahead, using 5 days of historical meteorological data. The task is formulated as a **multivariate-input, univariate-output, multi-step forecasting** problem: 6 meteorological variables (temperature, pressure, humidity, wind speed, max wind speed, wind direction) serve as input, and only future temperature is predicted.

Unlike the earlier mini-project where interpretation took priority, this Final Project treats **performance as a primary goal** — achieved through evolutionary optimization and supported by explainability and efficiency analysis.

## 2. Pipeline Overview

The project follows a structured end-to-end pipeline:

1. **Data preparation:** Hourly resampling (10min → 1h), quality assurance, temporal train/val/test split (70/15/15)
2. **Feature engineering:** 16 input features derived from 6 raw variables — cyclical temporal encodings (hour, day-of-year as sin/cos pairs) and wind decompositions (Cartesian components, gust metrics)
3. **Baseline establishment:** Persistence forecast + hand-tuned 2-layer GRU
4. **Synthetic data generation:** TimeGAN for weather sequence synthesis (bonus)
5. **Evolutionary optimization:** Genetic algorithm over 16 hyperparameters
6. **Explainability:** Permutation importance (global) + gradient saliency (local) + XAI-guided feature pruning
7. **Efficiency profiling:** Training time, inference latency, memory, parameter counts

## 3. Key Design Decisions

### Feature Engineering
We applied **sine–cosine encoding** for all cyclic quantities (hour, day-of-year, wind direction) to avoid artificial discontinuities, and decomposed wind into **Cartesian components** (wx, wy) rather than using polar coordinates directly. This expanded the input from 6 to 16 features while respecting the physical structure of the data.

### GRU Baseline Configuration
The baseline was manually designed with modern best practices: 2-layer GRU (96→64 units), AdamW optimizer with weight decay, Huber loss (robust to outliers), gradient clipping, and an intermediate dense layer (256 units). This configuration, tuned from the mini-project, provided a strong reference at **1.665°C MAE** on the initial test evaluation, later refined to **1.650°C** after full retraining.

### Evolutionary Algorithm Design
Rather than tuning isolated hyperparameters, we designed the EA to optimize the **full pipeline jointly**. Each individual encodes 16 genes spanning:
- Architecture (layers, units, dense layer, activation)
- Regularization (dropout, L2, Gaussian noise, gradient clipping)
- Training (optimizer, learning rate, loss function, batch size, weight decay)
- Preprocessing (scaler type)
- Windowing (lookback length)

The GA used **tournament selection** (k=3), **uniform crossover**, **per-gene mutation** (20% rate), and **elitism** (top individuals preserved). The budget was set to **population 20 × 15 generations = 300 evaluations**, with each individual trained for up to 20 epochs with early stopping. Fitness was defined as **MAE in °C on the validation set**, ensuring invariance to the scaler choice.

### TimeGAN: Built but Not Applied
A TimeGAN was successfully trained to generate synthetic weather sequences (reconstruction MSE = 0.0012). However, the quality assessment revealed that generated sequences did not sufficiently replicate real temporal dynamics. Rather than forcing augmentation that could degrade performance, we made the principled decision to **demonstrate the pipeline without applying synthetic data to the final models**. This reflects scientific honesty over checkbox-checking.

## 4. Main Findings

### 4.1 Evolutionary Search Results

The EA converged within the first 4 generations, finding a best validation fitness of **1.636°C MAE** (Generation 4). The discovered configuration challenges several assumptions from the hand-tuned baseline:

| Aspect | Baseline | EA Discovery |
|---|---|---|
| Architecture | 2-layer, 96→64 units | 2-layer, 64→64 units |
| Parameters | 86,744 | 63,512 (–27%) |
| Scaler | StandardScaler | **RobustScaler** |
| Loss function | Huber (δ=1.0) | **MAE** |
| Lookback | 120 hours | **144 hours** |
| Batch size | 128 | 256 |
| Gradient clipping | 2.0 | 5.0 |
| Regularization | L2=1e-6, no noise | L2=1e-5, no noise |

Key insight: the EA found that a **smaller, simpler model** with the right preprocessing (RobustScaler handles meteorological outliers better) and loss function (MAE, simpler than Huber) **outperforms** a larger model with suboptimal choices.

### 4.2 Final Performance Comparison

| Model | MAE (°C) | RMSE (°C) | Parameters | vs. Baseline |
|---|---|---|---|---|
| Persistence | 3.144 | 4.254 | — | — |
| GRU Baseline | 1.650 | 2.193 | 86,744 | reference |
| EA-Optimized GRU | 1.598 | 2.157 | 63,512 | **–3.2%** |
| **Pruned EA GRU** | **1.575** | **2.134** | **62,552** | **–4.5%** |

The EA-optimized model achieves a 3.2% MAE improvement over the baseline with 27% fewer parameters. After XAI-guided feature pruning, the best result is **1.575°C MAE** — a 4.5% improvement with 28% fewer parameters.

### 4.3 Robustness Validation

Multi-seed evaluation (seeds 7, 21, 42) confirmed stability:

| Model | MAE mean ± std (°C) |
|---|---|
| EA-Optimized GRU | 1.593 ± 0.016 |
| Pruned EA GRU | 1.579 ± 0.009 |

The pruned model has both **lower mean error and lower variance**, confirming that removing noisy features improves consistency across initializations.

### 4.4 Explainability Insights

**Permutation importance** revealed a clear feature hierarchy: temperature history dominates overwhelmingly, followed by diurnal cycle (hour_sin, hour_cos) and humidity. Five features (doy_sin, gust_ratio, wd_deg, wd_sin, wy) had negligible contribution and were removed in the pruning step.

**Gradient saliency** maps confirmed the model concentrates attention on recent time steps and the temperature channel, consistent with the strong autoregressive nature of the task. For certain forecast windows, seasonal encoding (doy_cos) and atmospheric pressure rank higher locally, suggesting context-dependent feature utilization.

The key contribution here was using XAI not just for interpretation but as an **active model improvement tool**: the closed loop of **optimize → explain → prune → validate** produced the project's best model.

### 4.5 Efficiency Analysis

| Metric | Baseline | EA-Optimized | Pruned EA |
|---|---|---|---|
| Parameters | 86,744 | 63,512 | 62,552 |
| Training time | ~39s | ~21s | ~19s |
| Inference latency | ~13ms | ~12ms | ~12ms |

The EA-optimized model is **faster, smaller, and more accurate** than the baseline — contradicting the common assumption that better accuracy requires more complexity.

## 5. Conclusions

This project demonstrates that **systematic evolutionary optimization** over an end-to-end forecasting pipeline can discover configurations that are simultaneously more accurate and more efficient than careful manual tuning. The final pruned model (1.575°C MAE, 62,552 parameters) achieves a 4.5% improvement over the baseline (1.650°C, 86,744 parameters) while being simpler, faster, and more robust.

The main contributions are:
1. **Joint pipeline optimization** — searching over architecture, training, preprocessing, and windowing simultaneously
2. **XAI-driven refinement** — using explainability as a tool for feature selection, not just interpretation
3. **Honest synthetic data assessment** — building the TimeGAN pipeline while acknowledging its limitations
4. **Comprehensive evaluation** — combining accuracy, robustness (multi-seed), explainability, and efficiency into a holistic analysis

Future directions include expanding the EA budget, incorporating Transformer architectures into the search space, integrating improved synthetic data generation, and exploring multi-objective optimization (accuracy vs. efficiency on a Pareto front).
