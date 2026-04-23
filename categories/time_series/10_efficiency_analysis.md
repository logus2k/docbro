# Section 12 -- Efficiency, Latency and Resource Analysis

This document explains the goals, concepts, and implementation details of Section 12 of the notebook, which profiles the three final models across training time, memory usage, parameter counts, and inference latency.

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [Why Efficiency Matters](#2-why-efficiency-matters)
3. [Models Compared](#3-models-compared)
4. [Training Time](#4-training-time)
5. [Memory Usage (RAM / GPU)](#5-memory-usage-ram--gpu)
6. [Trainable Parameters](#6-trainable-parameters)
7. [Inference Latency](#7-inference-latency)
8. [Consolidated Comparison](#8-consolidated-comparison)
9. [Visualizations](#9-visualizations)
10. [Key Finding: No Accuracy-Efficiency Trade-off](#10-key-finding-no-accuracy-efficiency-trade-off)
11. [Implementation Summary](#11-implementation-summary)

---

## 1. Section Goal

Answer the question: **beyond accuracy, what does each model cost?**

A model that is 0.05 degC more accurate but takes 10x longer to train or 100x more memory might not be worth it in practice. This section provides the data to make that judgment.

---

## 2. Why Efficiency Matters

| Dimension | Why it matters |
|---|---|
| **Training time** | Affects development iteration speed. Faster training = faster experimentation. Also affects energy cost and CO2 footprint. |
| **Memory usage** | Determines hardware requirements. A model that exceeds GPU memory cannot be deployed on that hardware. |
| **Parameter count** | Proxy for model complexity. Fewer parameters = less storage, faster loading, lower overfitting risk. |
| **Inference latency** | Critical for real-time applications. If a weather forecast takes 10 seconds to compute but conditions change every minute, the system can't keep up. |
| **Throughput** | How many predictions per second. Relevant when processing large batches (e.g., forecasting for many locations simultaneously). |

---

## 3. Models Compared

Three models are profiled side by side:

| Model | Features | Parameters | Origin |
|---|---:|---:|---|
| GRU Baseline Official | 16 | 86,744 | Mini-project best config |
| Best Evolutionary GRU | 16 | 63,512 | EA search (Section 9) |
| Pruned Evolutionary GRU | 11 | 62,552 | XAI pruning (Section 11) |

This allows three comparisons:
- Baseline vs. EA: effect of evolutionary optimization
- EA vs. Pruned: effect of feature pruning
- Baseline vs. Pruned: total improvement from both techniques combined

---

## 4. Training Time

### How it's measured

```python
def measure_training_time(model_builder_fn, X_train, y_train,
                           X_val, y_val, batch_size, epochs=5):
    model = model_builder_fn()
    start = time.perf_counter()
    history = train_model(model, X_train, y_train, X_val, y_val,
                          batch_size=batch_size, epochs=epochs, verbose=0)
    elapsed = time.perf_counter() - start
    return {"total_time_s": elapsed, "per_epoch_s": elapsed / epochs}
```

`time.perf_counter()` provides the highest-resolution clock available, measuring wall-clock time (not CPU time). Both total and per-epoch times are reported.

### Results

| Model | Total time | Per epoch | vs. Baseline |
|---|---:|---:|---|
| GRU Baseline | ~39s | ~7.8s | reference |
| Best EA GRU | ~21s | ~4.2s | 46% faster |
| Pruned EA GRU | ~19s | fastest | 51% faster |

### Why the EA model trains faster

- **Fewer parameters** (63,512 vs 86,744): smaller weight matrices = fewer multiplications per step
- **Larger batch size** (256 vs 128): fewer gradient updates per epoch (though each update processes more data)
- **Simpler architecture** (64->64 vs 96->64): smaller GRU layers

---

## 5. Memory Usage (RAM / GPU)

### How it's measured

```python
def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)
```

RAM is measured before and after model inference using `psutil`, which reads the process's **Resident Set Size** (RSS) -- the amount of physical memory currently allocated.

GPU memory is measured via `nvidia-smi` where available.

### Concepts

| Metric | What it measures |
|---|---|
| **RAM before** | Memory used by the Python process before loading/running the model |
| **RAM after** | Memory after running inference |
| **RAM delta** | Difference -- the incremental memory cost of the model |
| **GPU memory** | Video memory allocated on the graphics card (if GPU is used) |

### Why measure memory?

- Models that exceed available GPU memory cannot run on that hardware
- In shared environments (cloud, multi-user servers), memory is a constrained resource
- Smaller models can run on cheaper hardware (e.g., edge devices, smaller cloud instances)

---

## 6. Trainable Parameters

### How it's counted

```python
def count_trainable_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
```

Iterates over all trainable weight tensors in the model, computes the size of each (product of dimensions), and sums them.

### Results

| Model | Parameters | vs. Baseline |
|---|---:|---|
| GRU Baseline | 86,744 | reference |
| Best EA GRU | 63,512 | -27% |
| Pruned EA GRU | 62,552 | -28% |

### Why the pruned model has fewer parameters than the full EA model

Both have the same GRU architecture (64->64, dense 256). The difference comes from the **first GRU layer's input weights**:

```
Full EA:   first GRU input weights = 16 features * 64 units * 3 gates = 3,072
Pruned EA: first GRU input weights = 11 features * 64 units * 3 gates = 2,112
Difference: 960 parameters
```

63,512 - 62,552 = 960. Exactly the reduction from having 5 fewer input features.

---

## 7. Inference Latency

### How it's measured

```python
def measure_inference_latency(model, sample_input, n_warmup=10, n_runs=30):
    # Warmup: run the model several times to fill caches and trigger JIT compilation
    for _ in range(n_warmup):
        y = model(sample_input, training=False)
        y.numpy()
        _sync_if_needed()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        y = model(sample_input, training=False)
        y.numpy()
        _sync_if_needed()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "latency_mean_ms": mean(times) * 1000,
        "latency_std_ms": std(times) * 1000,
        "latency_p50_ms": percentile(times, 50) * 1000,
        "latency_p95_ms": percentile(times, 95) * 1000,
        "throughput_samples_s": batch_size / mean(times),
    }
```

### Key implementation details

| Detail | Why |
|---|---|
| **Warmup runs** (10) | First few runs are slow due to TensorFlow graph compilation, GPU kernel loading, and cache filling. Warmup ensures timed runs reflect steady-state performance. |
| **GPU synchronization** (`_sync_if_needed()`) | GPU operations are asynchronous -- `model(x)` returns immediately while the GPU is still computing. Without sync, you'd measure the time to *launch* the computation, not *complete* it. `tf.constant(0.).numpy()` forces TensorFlow to wait for all pending GPU work. |
| **Multiple runs** (30) | Averaging over 30 runs reduces measurement noise from OS scheduling, memory allocation, etc. |
| **Percentiles** (p50, p95) | p50 (median) is the typical latency. p95 is the "worst reasonable case" -- 95% of predictions complete within this time. Important for SLA guarantees. |

### Results

| Model | Mean latency | p95 latency | Throughput |
|---|---:|---:|---:|
| GRU Baseline | ~13ms | ~14ms | ~2,500 samples/s |
| Best EA GRU | ~12ms | ~13ms | ~2,700 samples/s |
| Pruned EA GRU | ~12ms | ~13ms | ~2,700 samples/s |

Latency is similar across all three models (~12-13ms for a batch of 32). This is expected -- at these model sizes, inference is fast regardless. The differences would become more significant with larger models or on constrained hardware.

---

## 8. Consolidated Comparison

### Table 22 -- Full comparison

| Model | MAE (degC) | RMSE (degC) | Parameters | Training time | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| GRU Baseline | 1.650 | 2.193 | 86,744 | ~39s | ~13 |
| Best EA GRU | 1.598 | 2.157 | 63,512 | ~21s | ~12 |
| **Pruned EA GRU** | **1.575** | **2.134** | **62,552** | **~19s** | **~12** |

### Table 24 -- Robustness across seeds (all models)

Combined robustness results showing that both EA-derived models are stable across seeds 7, 21, and 42.

---

## 9. Visualizations

Four figures are produced:

| Figure | Type | What it shows |
|---|---|---|
| **Figure 12** | Bar chart with error bars | Inference latency comparison (mean +/- std in ms) |
| **Figure 13** | Bar chart | Throughput comparison (samples per second) |
| **Figure 14** | Bar chart | Trainable parameter count for each model |
| **Figure 15** | Scatter plot | MAE (degC) vs. trainable parameters -- shows the accuracy-efficiency frontier |

Figure 15 is the most informative: it plots accuracy on one axis and complexity on the other. The ideal model is in the **bottom-left** (low MAE, few parameters). The Pruned EA GRU is closest to this ideal.

---

## 10. Key Finding: No Accuracy-Efficiency Trade-off

The central finding of this section is that the evolutionary optimization and XAI pruning **did not trade accuracy for efficiency**. The relationship was:

| Model | More accurate? | More efficient? |
|---|---|---|
| EA vs. Baseline | Yes (-3.2% MAE) | Yes (27% fewer params, 46% faster) |
| Pruned vs. EA | Yes (-1.4% MAE) | Yes (1.5% fewer params, slightly faster) |
| Pruned vs. Baseline | Yes (-4.5% MAE) | Yes (28% fewer params, 51% faster) |

This happened because the baseline was **over-parameterized**. Its 86,744 parameters and wide GRU layers (96->64) were more complex than the task required. The EA discovered that a leaner architecture generalizes better, and XAI pruning further confirmed that 5 of the 16 input features were adding noise rather than signal.

The conclusion: **the best-performing model is also the simplest and most efficient**. This is the ideal outcome and it supports selecting the Pruned Evolutionary GRU as the final model.

---

## 11. Implementation Summary

### Source module

| Module | Contents |
|---|---|
| `src/utils/profiling.py` | `count_trainable_params()`, `measure_inference_latency()`, `get_ram_usage_mb()`, `timed_call()` |

### Profiling utilities

| Function | What it measures | How |
|---|---|---|
| `count_trainable_params(model)` | Total trainable weights | Sum of all weight tensor sizes |
| `measure_inference_latency(model, input, n_warmup, n_runs)` | Prediction speed | Timed model calls with GPU sync |
| `get_ram_usage_mb()` | Process memory | `psutil.Process().memory_info().rss` |
| `timed_call(fn, *args)` | Arbitrary function timing | `time.perf_counter()` wrapper |
