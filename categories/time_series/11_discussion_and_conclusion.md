# Sections 13-15 -- Discussion, Future Work and Conclusion

This document covers the final three sections of the notebook, which synthesize all findings, acknowledge limitations, propose future improvements, and state the final conclusions.

---

## Table of Contents

1. [Section 13: Comparative Discussion](#1-section-13-comparative-discussion)
2. [Summary of Results](#2-summary-of-results)
3. [Impact of Evolutionary Optimization](#3-impact-of-evolutionary-optimization)
4. [Robustness Validation](#4-robustness-validation)
5. [Explainability Insights](#5-explainability-insights)
6. [Accuracy-Efficiency Trade-offs](#6-accuracy-efficiency-trade-offs)
7. [Strengths and Limitations](#7-strengths-and-limitations)
8. [Section 14: Future Work](#8-section-14-future-work)
9. [Section 15: Conclusion](#9-section-15-conclusion)

---

## 1. Section 13: Comparative Discussion

This section synthesizes findings across all experiments. It does not introduce new results -- it connects the threads from Sections 7-12 into a coherent narrative.

---

## 2. Summary of Results

### Final results table

| Model | MAE (degC) | RMSE (degC) | Parameters | vs. Baseline |
|---|---:|---:|---:|---|
| Persistence | 3.144 | 4.254 | -- | -- |
| GRU Baseline Official | 1.650 | 2.193 | 86,744 | reference |
| Best Evolutionary GRU | 1.598 | 2.157 | 63,512 | -3.2% MAE |
| **Pruned Evolutionary GRU** | **1.575** | **2.134** | **62,552** | **-4.5% MAE** |

Best single-seed result (Pruned model): **~1.569 degC MAE** -- the strongest forecast in the entire project.

### How to read this progression

1. **Persistence -> Baseline:** The GRU learns temporal dependencies, cutting MAE by 47%. This proves the deep learning approach is working.
2. **Baseline -> EA Best:** Evolutionary search finds a leaner, better pipeline. -3.2% MAE with 27% fewer parameters.
3. **EA Best -> Pruned:** XAI-guided pruning removes noise features. -1.4% more MAE improvement with even fewer parameters.
4. **Total:** The pruned model is 4.5% better than the baseline while being 28% smaller.

---

## 3. Impact of Evolutionary Optimization

The EA's main discoveries, and why they matter:

| Discovery | Significance |
|---|---|
| **RobustScaler** over StandardScaler | Weather data contains outliers (heat waves, storms). RobustScaler's median/IQR approach handles these better than mean/std. The EA found this without any meteorological domain knowledge. |
| **MAE loss** over Huber | Direct alignment between training objective and evaluation metric. Huber's compromise (quadratic for small errors, linear for large) was unnecessary -- pure MAE worked better. |
| **Compact 64->64 architecture** | The baseline's 96->64 was over-parameterized. The extra capacity didn't improve predictions, just added noise and computation. |
| **144-hour lookback** | 6 days of history is better than 5. This confirms that temporal context length is a meaningful hyperparameter that should not be fixed arbitrarily. The EA was able to discover this because lookback was included in the search space. |
| **No dropout, no Gaussian noise** | The compact architecture with L2 regularization and early stopping was sufficient to prevent overfitting. Additional regularization added nothing. |

The key insight: the EA optimized the **entire pipeline**, not just model architecture. Preprocessing (scaler) and data setup (lookback) contributed as much to the improvement as architectural choices.

---

## 4. Robustness Validation

### Multi-seed results

| Model | MAE mean +/- std (degC) | RMSE mean +/- std (degC) |
|---|---:|---:|
| Best Evolutionary GRU | 1.593 +/- 0.016 | 2.151 +/- 0.023 |
| Pruned Evolutionary GRU | 1.579 +/- 0.009 | 2.134 +/- 0.010 |
| **Pruned (best single seed)** | **1.569** | **2.123** |

### What this tells us

- The EA result is **not a fluke** -- it holds across seeds 7, 21, and 42
- The pruned model is **more stable** (lower std) than the full model -- removing noisy features reduced initialization-dependent variance
- Even the worst seed for the pruned model (1.585) beats the baseline (1.650)
- The standard deviation is small relative to the improvement, confirming statistical significance

---

## 5. Explainability Insights

### Physically plausible behavior

The XAI analysis confirmed that the model behaves in ways consistent with meteorological knowledge:

| Finding | Physical explanation |
|---|---|
| Past temperature is the dominant predictor | Temperature has strong autocorrelation -- today's temperature is the best predictor of tomorrow's |
| Seasonal and diurnal cycles matter | Temperature follows predictable daily and yearly patterns |
| Recent history matters most (saliency) | Short-term weather depends primarily on current conditions |
| Pressure contributes meaningfully | Pressure changes signal weather system movements |
| 5 features were prunable | Some wind-derived features were redundant (multiple encodings of the same information) |

### XAI as a refinement tool

The project demonstrates that XAI can go beyond interpretation:
- **Interpret:** Understand what the model learned
- **Validate:** Confirm it aligns with domain knowledge
- **Refine:** Use importance rankings to simplify the model

This **optimize -> explain -> prune -> validate** loop is one of the project's key methodological contributions.

---

## 6. Accuracy-Efficiency Trade-offs

The optimized models broke the typical accuracy-efficiency trade-off:

| Comparison | Accuracy | Efficiency |
|---|---|---|
| Pruned vs. Baseline | Better (-4.5% MAE) | Better (28% fewer params, 51% faster training) |

This happened because the baseline was over-parameterized. When you start with a model that's too complex, optimization can improve both accuracy (by finding a better configuration) and efficiency (by finding a leaner one) simultaneously.

### When the trade-off would reappear

If you pushed further -- trying to reduce parameters below 62,000 or reduce latency below 12ms -- you would eventually hit a point where simplification hurts accuracy. The current result sits at a sweet spot where further simplification would likely degrade performance.

---

## 7. Strengths and Limitations

### Strengths

| Strength | Evidence |
|---|---|
| End-to-end pipeline with leakage prevention | Chronological split, scaler fit on train only, test untouched during EA |
| EA applied to the full pipeline (not just architecture) | Scaler, lookback, training settings all optimized jointly |
| Explicit lookback optimization | Best config uses 144h, confirming windowing should not be fixed |
| Multi-seed robustness | Mean +/- std reported, not just single-run results |
| XAI analysis followed by principled pruning | 5 features removed based on data, improving accuracy |
| Final model is more accurate, simpler, and more efficient | No trade-offs between these objectives |

### Limitations

| Limitation | Impact | Possible remedy |
|---|---|---|
| EA budget (300 evaluations out of ~10^9) | Only a tiny fraction of the search space was explored | Larger budget, parallel evaluation, smarter search |
| Fixed forecast horizon (H=24) | Only lookback was optimized, not the output horizon | Include H in the search space |
| Only GRU architectures | Transformers or temporal CNNs might perform better | Expand model family in the search space |
| TimeGAN insufficient realism | Synthetic data could not be used for augmentation | Longer training, diffusion models, better architectures |
| 3 random seeds | Limited statistical power | 10+ seeds, or rolling-origin evaluation |
| Single dataset | Results may not generalize to other weather stations | Test on multiple datasets |

---

## 8. Section 14: Future Work

Six directions for future improvement:

### 1. Larger evolutionary budget

More generations, larger population, wider search space. Include more lookback values, alternative horizons, and multi-objective optimization (accuracy vs. complexity as simultaneous objectives).

### 2. Alternative architectures

Expand beyond GRU to include:
- **Transformers** (self-attention for long-range dependencies)
- **Temporal Convolutional Networks** (dilated convolutions for efficient sequence processing)
- **Hybrid models** (recurrent + attention mechanisms)
- **Lightweight sequence models** (for deployment on constrained hardware)

### 3. Stronger robustness analysis

- More random seeds (10+)
- **Rolling-origin evaluation:** Instead of a single test set, evaluate on multiple sliding test windows across different time periods
- Cross-validation for time series (blocked or nested)

### 4. Improved TimeGAN

- Longer adversarial training (15 epochs was short)
- Larger latent dimension
- **Diffusion-based generators** (newer approach, often more stable than GANs)
- Conditional generation (generate sequences conditioned on specific weather regimes)

### 5. Iterative XAI pruning

Instead of a single round of "remove 5 features," iterate:
1. Explain -> identify least important features
2. Remove them -> retrain -> re-evaluate
3. Explain again on the pruned model
4. Repeat until performance degrades

This could find an even smaller optimal feature set.

### 6. Deployment-oriented evaluation

- Precise GPU profiling (not just RAM)
- Energy consumption (kWh per training run)
- Memory peaks during training (not just steady-state)
- Real-time latency under production conditions

---

## 9. Section 15: Conclusion

### What the project accomplished

A comprehensive end-to-end forecasting pipeline for **24-hour air temperature prediction** integrating:
- Evolutionary optimization (EA)
- Explainable AI (XAI)
- Efficiency analysis
- TimeGAN (proof-of-concept)

### Key findings

1. **The GRU baseline** (1.650 degC MAE) was a strong starting point, cutting persistence error by 47%

2. **The EA** discovered a better configuration (1.598 degC MAE) that was also more compact:
   - RobustScaler > StandardScaler
   - MAE loss > Huber loss
   - 64->64 GRU > 96->64 GRU
   - 144h lookback > 120h lookback

3. **XAI-guided pruning** produced the best model (1.575 degC MAE, best seed 1.569):
   - 5 features removed based on permutation importance
   - Accuracy improved (less noise)
   - Stability improved (lower seed variance)

4. **TimeGAN** showed feasibility of synthetic generation but insufficient realism for integration

5. **Efficiency analysis** confirmed the best model is also the most efficient -- no accuracy-efficiency trade-off

### The closed-loop methodology

The project's core contribution is the **optimize -> explain -> prune -> validate** loop:

```
Strong baseline
    |
    v  EA optimization (Section 9) --> better + smaller model
    |
    v  XAI analysis (Section 11) --> identify weak features
    |
    v  Pruning (Section 11) --> even better + even smaller model
    |
    v  Multi-seed validation (Sections 10, 11) --> confirmed robust
    |
Final model: Pruned Evolutionary GRU
    - MAE: 1.575 degC (main run), 1.569 degC (best seed)
    - 4.5% better than baseline
    - 28% fewer parameters
    - More stable across seeds
```

### Final model selection

The **Pruned Evolutionary GRU** is selected as the project's final model because it achieved:
- The strongest main evaluation result (1.575 degC MAE)
- The best observed single-seed result (1.569 degC MAE)
- The highest multi-seed stability (1.579 +/- 0.009 degC)
- The lowest parameter count (62,552)
- The fastest training time

It is simultaneously **more accurate, more compact, more interpretable, and more efficient** than the manually defined baseline.
