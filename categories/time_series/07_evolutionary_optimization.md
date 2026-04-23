# Section 9 -- Evolutionary Optimization of the Forecasting Pipeline

This document explains the goals, concepts, and implementation details of Section 9 of the notebook, which uses a Genetic Algorithm (GA) to automatically discover better forecasting pipeline configurations than the hand-tuned baseline.

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [Motivation: Why Evolutionary Search?](#2-motivation-why-evolutionary-search)
3. [Search Space -- The 17 Dimensions](#3-search-space----the-17-dimensions)
4. [Genotype: How an Individual Is Represented](#4-genotype-how-an-individual-is-represented)
5. [Genotype Constraints](#5-genotype-constraints)
6. [Fitness Function: Evaluating an Individual](#6-fitness-function-evaluating-an-individual)
7. [Inverse Scaling Across Different Scalers](#7-inverse-scaling-across-different-scalers)
8. [The GA Loop Step by Step](#8-the-ga-loop-step-by-step)
9. [Tournament Selection](#9-tournament-selection)
10. [Uniform Crossover](#10-uniform-crossover)
11. [Per-Gene Mutation](#11-per-gene-mutation)
12. [Elitism](#12-elitism)
13. [GA Parameters and Budget](#13-ga-parameters-and-budget)
14. [The Best Configuration Found](#14-the-best-configuration-found)
15. [Key Discoveries vs. the Baseline](#15-key-discoveries-vs-the-baseline)
16. [Top Candidate Analysis](#16-top-candidate-analysis)
17. [Overfitting Mitigation](#17-overfitting-mitigation)
18. [Implementation Summary](#18-implementation-summary)

---

## 1. Section Goal

Automatically search over the **entire forecasting pipeline configuration** -- not just model architecture, but also preprocessing, training settings, and windowing -- to find a configuration that outperforms the hand-tuned GRU baseline.

The key constraint: the search must use only the validation set for fitness evaluation. The test set remains completely untouched.

---

## 2. Motivation: Why Evolutionary Search?

### The problem with manual tuning

The baseline was manually configured based on the previous mini-project. While effective, manual tuning has limitations:

- A human can only test a handful of configurations
- Interactions between hyperparameters are hard to reason about (e.g., does a larger learning rate work better with a smaller batch size?)
- Fixed choices (like StandardScaler) may not be optimal

### Why not grid search or random search?

| Method | Limitation for this problem |
|---|---|
| **Grid search** | With 17 dimensions, the grid would have billions of points -- computationally impossible |
| **Random search** | Explores blindly -- doesn't learn from good configurations |
| **Bayesian (Optuna)** | More sample-efficient, but doesn't satisfy the course requirement for evolutionary methods |

### Why GA?

The GA explores intelligently: it recombines parts of good solutions (crossover) while introducing novelty (mutation). Over generations, the population converges toward good regions of the search space. It also aligns with the course's focus on evolutionary/nature-inspired computation.

---

## 3. Search Space -- The 17 Dimensions

The search space defines all possible values each hyperparameter can take:

### Architecture (6 genes)

| Gene | Possible values | What it controls |
|---|---|---|
| `n_layers` | [1, 2] | Number of stacked GRU layers |
| `units1` | [64, 96, 128] | Hidden units in first GRU layer |
| `units2` | [64, 96, 128] | Hidden units in second GRU layer |
| `units3` | [32, 64, 96] | Hidden units in third GRU layer (if used) |
| `dense_units` | [0, 64, 128, 256] | Intermediate dense layer size (0 = no dense layer) |
| `dense_activation` | [relu, gelu, elu, leaky_relu] | Activation function for the dense layer |

### Regularization (4 genes)

| Gene | Possible values | What it controls |
|---|---|---|
| `dropout` | [0.0, 0.1, 0.2, 0.3] | Dropout rate |
| `l2` | [0.0, 1e-6, 1e-5, 1e-4] | L2 regularization strength |
| `gaussian_noise_std` | [0.0, 0.01] | Gaussian noise added to inputs |
| `clipnorm` | [0.5, 1.0, 2.0, 5.0] | Gradient clipping threshold |

### Training (5 genes)

| Gene | Possible values | What it controls |
|---|---|---|
| `optimizer_name` | [adam, adamw] | Optimizer type |
| `weight_decay` | [0.0, 1e-6, 1e-5, 1e-4] | AdamW weight decay |
| `learning_rate` | [1e-3, 5e-4, 3e-4, 2e-4, 1e-4] | Learning rate |
| `loss_name` | [mse, mae, huber1, huber2] | Loss function |
| `batch_size` | [128, 256] | Training batch size |

### Preprocessing (1 gene)

| Gene | Possible values | What it controls |
|---|---|---|
| `scaler_name` | [standard, robust, minmax] | Feature normalization strategy |

### Windowing (1 gene)

| Gene | Possible values | What it controls |
|---|---|---|
| `lookback` | [96, 120, 144] | Hours of history used as model input |

### Total search space size

The combinatorial size is approximately:
```
2 * 3 * 3 * 3 * 4 * 4 * 4 * 4 * 2 * 5 * 2 * 4 * 4 * 2 * 3 * 3 = ~10^9
```

The GA evaluates only 300 configurations out of roughly a billion possibilities. This is why intelligent search (not brute force) is necessary.

---

## 4. Genotype: How an Individual Is Represented

Each individual in the population is a **genotype** -- a Python dictionary mapping gene names to values:

```python
def sample_genotype():
    genotype = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    return apply_genotype_constraints(genotype)
```

Example genotype:
```python
{
    "n_layers": 2,
    "units1": 96,
    "units2": 64,
    "units3": 32,
    "dropout": 0.1,
    "l2": 1e-5,
    "dense_units": 128,
    "dense_activation": "relu",
    "learning_rate": 3e-4,
    "batch_size": 256,
    "clipnorm": 2.0,
    "optimizer_name": "adamw",
    "weight_decay": 1e-5,
    "loss_name": "mae",
    "gaussian_noise_std": 0.0,
    "scaler_name": "robust",
    "lookback": 144,
}
```

This single dictionary fully specifies a complete forecasting pipeline -- from how data is scaled, through how the model is built and trained, to how many hours of history are used.

---

## 5. Genotype Constraints

Not all gene combinations are valid. Constraints are enforced after every initialization, crossover, and mutation:

```python
def apply_genotype_constraints(genotype):
    g = deepcopy(genotype)

    # GRU layers must be monotonically non-increasing in width
    if g["units2"] > g["units1"]:
        g["units2"] = g["units1"]
    if g["units3"] > g["units2"]:
        g["units3"] = g["units2"]

    # Weight decay only meaningful for AdamW
    if g["optimizer_name"] == "adam":
        g["weight_decay"] = 0.0

    return g
```

### Why these constraints?

| Constraint | Reasoning |
|---|---|
| `units2 <= units1` | A GRU that widens after the first layer is architecturally unusual and often unproductive. Funneling (wide -> narrow) is the standard pattern. |
| `units3 <= units2` | Same logic for the third layer |
| `weight_decay = 0` for Adam | Standard Adam does not implement decoupled weight decay. Setting it would have no effect, wasting a gene's value. Only AdamW uses it meaningfully. |

### When constraints are applied

After **every** genetic operation:
- `sample_genotype()` -- initial population
- `crossover_genotypes()` -- after combining parents
- `mutate_genotype()` -- after random gene changes

This ensures that every individual ever evaluated is a valid pipeline configuration.

---

## 6. Fitness Function: Evaluating an Individual

The fitness function (`evaluate_individual()`) runs the **complete forecasting pipeline** for a single genotype:

```
Step 1: Get scaler based on cfg["scaler_name"]
    |
Step 2: Fit scaler on training data, transform train and val
    |
Step 3: Create sliding windows using cfg["lookback"]
    |
Step 4: Build GRU model using all architecture/training genes
    |
Step 5: Train model (up to 20 epochs with early stopping)
    |
Step 6: Predict on validation set
    |
Step 7: Inverse-scale predictions to degrees Celsius
    |
Step 8: Compute validation MAE in degC --> this is the fitness
```

### Implementation

```python
def evaluate_individual(cfg, df_train, df_val, final_feature_cols,
                        target_idx, lookback, horizon, epochs, verbose):
    # Step 1-2: Scale
    scaler = get_scaler(cfg["scaler_name"])
    X_train_scaled = scaler.fit_transform(df_train[final_feature_cols])
    X_val_scaled = scaler.transform(df_val[final_feature_cols])

    # Step 3: Window
    effective_lookback = cfg.get("lookback", lookback)
    X_train, y_train = make_windows(X_train_scaled, target_idx, effective_lookback, horizon)
    X_val, y_val = make_windows(X_val_scaled, target_idx, effective_lookback, horizon)

    # Step 4: Build model
    model = build_gru_model(L=effective_lookback, n_features=..., H=horizon, **cfg)

    # Step 5: Train
    train_model(model, X_train, y_train, X_val, y_val, ...)

    # Step 6: Predict
    y_pred_val = model.predict(X_val)

    # Step 7: Inverse-scale
    y_val_inv = inverse_target_with_scaler(y_val, scaler, target_idx, n_features)
    y_pred_val_inv = inverse_target_with_scaler(y_pred_val, scaler, target_idx, n_features)

    # Step 8: Fitness
    original_metrics = evaluate_original_scale_forecasts(y_val_inv, y_pred_val_inv)

    return {"cfg": cfg, "fitness": original_metrics["mae"], "metrics": {...}}
```

### Why fitness in degC (not scaled)?

Different individuals may use different scalers (Standard, Robust, MinMax). Scaled MAE values are not comparable across scalers because the scale is different. By inverse-transforming to degrees Celsius first, all individuals are compared on the **same physical scale**, regardless of which scaler they use.

---

## 7. Inverse Scaling Across Different Scalers

A subtle implementation challenge: the baseline uses `y_scaled * std + mean` for inverse scaling, which only works for StandardScaler. Since the EA can choose any scaler, a general-purpose inverse function is needed:

```python
def inverse_target_with_scaler(y_scaled, scaler, target_idx, n_features):
    y_flat = y_scaled.reshape(-1)

    # Build a dummy array with zeros everywhere except the target column
    dummy = np.zeros((len(y_flat), n_features))
    dummy[:, target_idx] = y_flat

    # Use sklearn's inverse_transform (works for any scaler)
    inv = scaler.inverse_transform(dummy)[:, target_idx]

    return inv.reshape(y_scaled.shape)
```

### How it works

Scikit-learn's `inverse_transform()` expects a 2D array with all features. Since we only have the target column, we create a **dummy array** filled with zeros and place our values in the target column. After inverse transforming the full array, we extract only the target column. The zeros in other columns don't affect the target's inverse transformation because each column is scaled independently.

This trick works for StandardScaler, RobustScaler, and MinMaxScaler -- all of them apply column-wise transformations.

---

## 8. The GA Loop Step by Step

The main evolutionary loop in `run_evolutionary_search()`:

```
INITIALIZE: Create 20 random genotypes (population)

FOR each generation (1 to 15):
    |
    EVALUATE: Train and score all 20 individuals
    |         (each runs the full pipeline on validation set)
    |
    SORT: Rank by fitness (lowest MAE = best)
    |
    ELITISM: Copy the top 2 directly to next generation
    |
    FILL: Generate 18 new individuals:
    |     1. Select parent A via tournament (k=3)
    |     2. Select parent B via tournament (k=3)
    |     3. Crossover: combine A and B into child
    |     4. Mutation: randomly alter some genes (rate=0.2)
    |     5. Add child to next generation
    |
    TRACK: Record best fitness and all results
    |
    REPEAT

RETURN: Best individual across all generations
```

### Implementation

```python
def run_evolutionary_search(..., population_size=20, generations=15,
                             mutation_rate=0.2, elitism=2):
    population = [sample_genotype() for _ in range(population_size)]
    best_result = None

    for gen in range(generations):
        # Evaluate all individuals
        population_results = []
        for cfg in population:
            result = evaluate_individual(cfg, ...)
            population_results.append(result)

        # Sort by fitness (lower MAE = better)
        population_results = sorted(population_results, key=lambda x: x["fitness"])

        # Track global best
        if best_result is None or population_results[0]["fitness"] < best_result["fitness"]:
            best_result = population_results[0]

        # Elitism: top 2 survive unchanged
        next_population = [res["cfg"] for res in population_results[:elitism]]

        # Fill rest via selection + crossover + mutation
        while len(next_population) < population_size:
            parent1 = tournament_selection(population_results)
            parent2 = tournament_selection(population_results)
            child = crossover_genotypes(parent1, parent2)
            child = mutate_genotype(child, mutation_rate=mutation_rate)
            next_population.append(child)

        population = next_population

    return best_result, history
```

---

## 9. Tournament Selection

### How it works

```python
def tournament_selection(population_results, k=3):
    candidates = random.sample(population_results, k=min(k, len(population_results)))
    candidates = sorted(candidates, key=lambda x: x["fitness"])
    return candidates[0]["cfg"]
```

1. Randomly pick 3 individuals from the population
2. Return the one with the lowest fitness (best MAE)

### Why tournament selection?

| Property | Benefit |
|---|---|
| **Selection pressure** | Good individuals are more likely to be selected, but not guaranteed -- even mediocre individuals have a chance if they happen to be the best in their tournament |
| **Diversity** | The randomness preserves variety in the population, preventing premature convergence to a single solution |
| **Simplicity** | Easy to implement and tune (just change k) |
| **k controls pressure** | Higher k = more selective (the best of more candidates). k=3 is a moderate value -- selective enough to converge, diverse enough to explore. |

### Example

Population of 20, sorted by fitness (MAE in degC):
```
Individual  1: 1.63  (best)
Individual  2: 1.64
...
Individual 10: 1.72
...
Individual 20: 1.85  (worst)
```

Tournament picks individuals 5, 12, 17. Best is individual 5. That becomes a parent.
Another tournament picks individuals 1, 8, 15. Best is individual 1. That becomes the other parent.

---

## 10. Uniform Crossover

### How it works

```python
def crossover_genotypes(parent1, parent2):
    child = {}
    for key in SEARCH_SPACE.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return apply_genotype_constraints(child)
```

For **each gene independently**, flip a coin:
- Heads: take parent 1's value
- Tails: take parent 2's value

### Example

```
Gene            Parent 1     Parent 2     Child (random)
n_layers        2            1            2       (from P1)
units1          96           64           64      (from P2)
loss_name       huber1       mae          mae     (from P2)
scaler_name     standard     robust       standard (from P1)
lookback        120          144          144     (from P2)
```

The child inherits a mix of traits. If Parent 1 had a good architecture and Parent 2 had a good scaler, the child might get both. Or it might get the worst of both -- selection in the next generation will weed out bad combinations.

### Why constraints are re-applied

Crossover can create invalid combinations. If Parent 1 has `units1=64` and Parent 2 has `units2=128`, the child could inherit `units1=64, units2=128`, violating the `units2 <= units1` constraint. `apply_genotype_constraints()` fixes this to `units2=64`.

---

## 11. Per-Gene Mutation

### How it works

```python
def mutate_genotype(genotype, mutation_rate=0.2):
    child = deepcopy(genotype)
    for key, values in SEARCH_SPACE.items():
        if random.random() < mutation_rate:
            child[key] = random.choice(values)
    return apply_genotype_constraints(child)
```

For each of the 17 genes:
- With 20% probability: replace with a **random value** from the search space
- With 80% probability: keep the inherited value

### Expected mutations per individual

With 17 genes and rate 0.2: `17 * 0.2 = 3.4` genes mutated on average. So each child differs from its crossover result by about 3-4 genes.

### Why mutation matters

Without mutation, the population can only recombine existing gene values. If no individual in the initial population has `lookback=144`, no future individual can either (crossover can only mix what already exists). Mutation introduces **new values** that may not have been present, preventing the population from getting trapped.

### Why constraints are re-applied

A mutation might change `units1` from 128 to 64 while `units2` is still 128 from the parent, creating an invalid genotype. Constraints fix this.

---

## 12. Elitism

```python
next_population = [res["cfg"] for res in population_results[:elitism]]  # elitism=2
```

The **top 2 individuals** survive unchanged into the next generation. No crossover, no mutation -- their exact genotypes are preserved.

### Why elitism is essential

Without elitism, the best solution found so far could be lost. Crossover and mutation are random -- they might accidentally destroy a good configuration. Elitism guarantees that the best-ever fitness can only improve or stay the same, never get worse across generations.

### Trade-off

Elitism = 2 out of 20 (10% of the population) is a moderate setting. Too much elitism reduces diversity (the population becomes copies of the best). Too little risks losing good solutions.

---

## 13. GA Parameters and Budget

| Parameter | Value | Meaning |
|---|---|---|
| `population_size` | 20 | Individuals per generation |
| `generations` | 15 | Number of evolutionary cycles |
| `mutation_rate` | 0.2 | 20% chance per gene |
| `elitism` | 2 | Top 2 survive unchanged |
| `tournament k` | 3 | Select best of 3 random candidates |
| `epochs per individual` | 20 | Training budget per evaluation (with early stopping) |
| **Total evaluations** | **300** | 20 * 15 = 300 complete pipeline runs |

### Computational cost

Each evaluation involves:
1. Scaling the dataset
2. Creating sliding windows
3. Building a GRU model
4. Training for up to 20 epochs
5. Predicting on validation
6. Computing metrics

This is why the EA search takes many hours -- it is effectively training 300 separate models.

---

## 14. The Best Configuration Found

After 15 generations, the best individual (lowest validation MAE) was:

```
n_layers = 2              units1 = 64           units2 = 64
units3 = 32               dropout = 0.0         l2 = 1e-5
dense_units = 256         dense_activation = relu
learning_rate = 3e-4      batch_size = 256      clipnorm = 5.0
optimizer_name = adamw    weight_decay = 0.0    loss_name = mae
gaussian_noise_std = 0.0  scaler_name = robust  lookback = 144
```

Validation MAE: **~1.636 degC**

---

## 15. Key Discoveries vs. the Baseline

| Gene | Baseline | EA Best | Insight |
|---|---|---|---|
| `scaler_name` | standard | **robust** | Weather data has outliers; RobustScaler handles them better |
| `loss_name` | huber1 | **mae** | Direct alignment with the evaluation metric works better than Huber's compromise |
| `units1` | 96 | **64** | Smaller layers generalize better -- the baseline was over-parameterized |
| `units2` | 64 | **64** | Same (narrower funnel not needed) |
| `lookback` | 120 | **144** | 6 days of history is better than 5 -- temporal context matters |
| `learning_rate` | 2e-4 | **3e-4** | Slightly faster learning works with the compact architecture |
| `batch_size` | 128 | **256** | Larger batches provide more stable gradient estimates |
| `clipnorm` | 2.0 | **5.0** | The compact model has smaller gradients; a looser clip is fine |
| `dropout` | 0.0 | **0.0** | Confirmed -- no dropout needed |
| `gaussian_noise_std` | 0.0 | **0.0** | Confirmed -- no input noise needed |
| `weight_decay` | 1e-6 | **0.0** | Light L2 via `l2=1e-5` is sufficient; no extra weight decay needed |

### Parameter count comparison

| Model | Parameters |
|---|---:|
| Baseline (96->64, dense 256) | 86,744 |
| EA best (64->64, dense 256) | 63,512 |
| Reduction | 27% fewer |

The EA found a model that is **smaller AND better** -- a clear sign the baseline was over-parameterized.

---

## 16. Top Candidate Analysis

The notebook displays the top 5 unique configurations across all generations (Table 8). This reveals:

- Whether the population **converged** to a narrow region (all top 5 similar) or found **diverse** good solutions
- Which genes are consistent across top candidates (likely important) vs. which vary (less critical)
- Whether the best individual is an outlier or representative of a competitive cluster

---

## 17. Overfitting Mitigation

The EA optimizes on the validation set, which creates a risk: after 300 evaluations, the best configuration might be "overfit" to validation data -- performing well on validation but not on test.

Three safeguards address this:

| Safeguard | How it helps |
|---|---|
| **Test set untouched** | The test set is never used during the EA. Test metrics are computed only once, at the very end (Section 10). |
| **Limited training budget** | Each individual trains for only 20 epochs (with early stopping), reducing the chance of memorizing validation patterns |
| **Multi-seed robustness** (Section 10.6) | The best configuration is re-evaluated across 3 different random seeds (7, 21, 42). If it works well across seeds, it's genuinely good -- not lucky. |

---

## 18. Implementation Summary

### Source modules

| Module | Contents |
|---|---|
| `src/evolution/search_space.py` | `SEARCH_SPACE` dictionary -- defines all gene ranges |
| `src/evolution/genotype.py` | `sample_genotype()`, `apply_genotype_constraints()` |
| `src/evolution/operators.py` | `mutate_genotype()`, `crossover_genotypes()` |
| `src/evolution/fitness.py` | `evaluate_individual()` -- full pipeline evaluation, `inverse_target_with_scaler()` |
| `src/evolution/ga_evolutionary_search.py` | `run_evolutionary_search()` -- main GA loop, `tournament_selection()` |

### Data flow per generation

```
Population (20 genotypes)
    |
    v  evaluate_individual() x 20
    |  [scale -> window -> build -> train -> predict -> inverse -> MAE]
    |
Ranked results (sorted by fitness)
    |
    v  Elitism: top 2 survive
    v  Tournament selection + crossover + mutation: 18 new children
    |
Next population (20 genotypes)
    |
    v  Repeat for 15 generations
    |
Best individual across all generations --> used in Section 10
```
