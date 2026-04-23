# Section 8 -- Synthetic Data Generation with TimeGAN

This document explains the goals, concepts, and implementation details of Section 8 of the notebook, which explores generating synthetic weather sequences using a Time-series Generative Adversarial Network (TimeGAN).

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [Motivation: Why Generate Synthetic Data?](#2-motivation-why-generate-synthetic-data)
3. [GAN Fundamentals](#3-gan-fundamentals)
4. [What TimeGAN Adds Over a Standard GAN](#4-what-timegan-adds-over-a-standard-gan)
5. [The Five Sub-Networks](#5-the-five-sub-networks)
6. [TimeGAN Configuration](#6-timegan-configuration)
7. [Training Data Preparation](#7-training-data-preparation)
8. [Phase 1: Autoencoder Pretraining](#8-phase-1-autoencoder-pretraining)
9. [Phase 2: Supervisor Pretraining](#9-phase-2-supervisor-pretraining)
10. [Phase 3: Adversarial Training](#10-phase-3-adversarial-training)
11. [The Generator Loss Function -- Detailed Breakdown](#11-the-generator-loss-function----detailed-breakdown)
12. [Discriminator Threshold](#12-discriminator-threshold)
13. [Generating Synthetic Sequences](#13-generating-synthetic-sequences)
14. [Quality Assessment](#14-quality-assessment)
15. [Why TimeGAN Was Not Used in the Final Pipeline](#15-why-timegan-was-not-used-in-the-final-pipeline)
16. [Implementation Summary](#16-implementation-summary)

---

## 1. Section Goal

Demonstrate the feasibility of **synthetic time series generation** for training data augmentation using TimeGAN. The section trains the generative model, assesses synthetic quality, and determines whether the generated sequences are realistic enough to improve forecasting.

**Outcome:** TimeGAN achieved strong reconstruction quality but insufficient generative realism. It was retained as a proof-of-concept but **not integrated** into the final forecasting pipeline.

---

## 2. Motivation: Why Generate Synthetic Data?

### The augmentation idea

If a model is trained on 49,000 real hourly sequences, would adding 10,000 synthetic sequences help? Potential benefits:

- **More training data** -- more examples to learn from, possibly reducing overfitting
- **Rare event coverage** -- the generator might produce plausible extreme weather scenarios underrepresented in training
- **Robustness** -- exposure to a wider variety of patterns during training

### The risk

If synthetic data is not realistic enough, it introduces **noise** that the model learns as if it were real weather. This can degrade performance rather than improve it.

### The leakage constraint

The generative model is trained **only on the training split**. Validation and test data are never exposed to TimeGAN. This ensures that even if synthetic data were used, no future information would leak into training.

---

## 3. GAN Fundamentals

A **Generative Adversarial Network** trains two networks against each other:

### Generator (G)

- **Input:** Random noise (a vector of random numbers)
- **Output:** A fake data sample
- **Goal:** Produce samples realistic enough to fool the discriminator

### Discriminator (D)

- **Input:** A data sample (either real or fake)
- **Output:** A probability that the sample is real
- **Goal:** Correctly classify samples as real or fake

### The adversarial game

```
Generator: "Here's a fake sequence"
Discriminator: "That's fake, I can tell because..."
Generator: "OK, let me adjust..." (learns from the feedback)
...many rounds later...
Generator: "Here's another fake sequence"
Discriminator: "I honestly can't tell if that's real or fake"
```

The training is a **minimax game**: the generator minimizes what the discriminator maximizes. When the discriminator can no longer distinguish real from fake (accuracy ~50%), the generator has learned to produce realistic data.

### Why standard GANs struggle with time series

A standard GAN treats each generated sample as an independent data point. For images, this is fine -- each pixel's relationship to its neighbors is spatial, and convolutional layers handle that well.

For time series, each time step depends on previous steps in complex temporal patterns. A standard GAN has no mechanism to enforce that step 5 follows logically from step 4. The result: individual time steps might look plausible, but the sequence as a whole lacks coherent temporal dynamics.

---

## 4. What TimeGAN Adds Over a Standard GAN

TimeGAN (Yoon et al., 2019) addresses the temporal coherence problem by adding three components:

| Standard GAN | TimeGAN addition | Purpose |
|---|---|---|
| Generator | **Embedder** | Learn a compressed latent representation where temporal patterns are easier to model |
| Discriminator | **Recovery** | Reconstruct data from latent space (inverse of embedder) |
| -- | **Supervisor** | Learn temporal transition rules in latent space: given current state, predict next state |

The key insight: it is easier to learn temporal dynamics in a **learned latent space** than directly in data space. The embedder compresses the data, the supervisor learns how the compressed representation evolves over time, and the generator produces sequences in this latent space rather than in the raw data space.

---

## 5. The Five Sub-Networks

All five are built from the same template: stacked GRU layers followed by a TimeDistributed Dense output.

```python
def _build_rnn_network(self, output_dim, input_dim, name):
    inputs = keras.Input(shape=(self.seq_len, input_dim))
    x = inputs
    for i in range(self.num_layers):       # 3 GRU layers
        x = layers.GRU(self.hidden_dim, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(output_dim))(x)
    return keras.Model(inputs, outputs)
```

### Network details

| Network | Input dim | Output dim | Function |
|---|---:|---:|---|
| **Embedder** | 16 (n_features) | 24 (hidden_dim) | Compresses real data into latent space |
| **Recovery** | 24 (hidden_dim) | 16 (n_features) | Reconstructs data from latent space |
| **Generator** | 24 (hidden_dim) | 24 (hidden_dim) | Transforms random noise into latent sequences |
| **Supervisor** | 24 (hidden_dim) | 24 (hidden_dim) | Predicts next latent state from current |
| **Discriminator** | 24 (hidden_dim) | 1 | Classifies latent sequences as real/fake |

### `return_sequences=True` everywhere

Unlike the forecasting GRU (which outputs only the last state), all TimeGAN GRUs output a hidden state at **every time step**. This is because TimeGAN operates on full sequences -- every time step of the input produces a corresponding time step in the output.

### `TimeDistributed(Dense(...))`

Applies the same Dense layer independently to each time step. This means:
- Time step 0 goes through the Dense layer
- Time step 1 goes through the same Dense layer (same weights)
- ...
- Time step 143 goes through the same Dense layer

This produces an output with shape `(144, output_dim)` -- one output vector per time step.

---

## 6. TimeGAN Configuration

```python
TIMEGAN_CONFIG = {
    "seq_len": 144,        # LOOKBACK (120) + HORIZON (24)
    "hidden_dim": 24,      # latent space dimensionality
    "num_layers": 3,       # GRU layers per sub-network
    "batch_size": 128,     # training batch size
    "ae_epochs": 20,       # autoencoder pretraining epochs
    "sup_epochs": 20,      # supervisor pretraining epochs
    "adv_epochs": 15,      # adversarial training epochs
    "learning_rate": 1e-3, # Adam learning rate
    "gamma": 1.0,          # reconstruction loss weight in adversarial phase
}
```

### Why `hidden_dim = 24`?

The latent space (24 dimensions) is larger than the input space (16 features). This might seem counterintuitive -- compression usually reduces dimensionality. But the latent space must capture not just the features but their **temporal dynamics**. The extra dimensions give the embedder room to encode temporal patterns that are implicit in the raw data.

### Why `seq_len = 144`?

Each generated sequence has 144 time steps, matching `LOOKBACK + HORIZON = 120 + 24`. This way, a generated sequence can be split directly into:
- An input window (first 120 steps, all 16 features)
- A target sequence (last 24 steps, temperature only)

This makes synthetic sequences directly compatible with the forecasting pipeline.

---

## 7. Training Data Preparation

```python
timegan_train_sequences = make_timegan_sequences(X_train_scaled, TIMEGAN_SEQ_LEN)
```

### Implementation

```python
def make_timegan_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)
```

This is the same sliding window approach used for forecasting, but:
- The window is 144 steps (not split into input/output yet)
- All 16 features are kept (no target extraction)
- Only training data is used

Result shape: `(N, 144, 16)` -- N overlapping 144-step sequences from the training set.

---

## 8. Phase 1: Autoencoder Pretraining

### Goal

Train the Embedder and Recovery networks to compress and reconstruct real sequences with minimal information loss.

### How it works

```python
# Autoencoder: real data -> embedder -> recovery -> reconstructed data
X = keras.Input(shape=(144, 16))
H = self.embedder(X)       # (144, 16) -> (144, 24)  compress
X_tilde = self.recovery(H) # (144, 24) -> (144, 16)  reconstruct
```

The autoencoder is trained to minimize **MSE** between the input and its reconstruction:

```
Loss = mean((X - X_tilde)^2)
```

The training target is the input itself -- this is **self-supervised learning**. The autoencoder learns by trying to reproduce what it was given.

### Data flow

```
Real sequence (144, 16)
    |
    v  Embedder (3 GRU layers)
    |
Latent representation (144, 24)
    |
    v  Recovery (3 GRU layers)
    |
Reconstructed sequence (144, 16)
    |
    v  Compare with original -> MSE loss
```

### Results

| Epoch | MSE Loss |
|---:|---:|
| 1 | 0.280 |
| 5 | ~0.010 |
| 20 | 0.0023 |

The rapid convergence indicates the autoencoder can compress 16 features into 24 latent dimensions and reconstruct them with very high fidelity. This is a good sign -- the latent space captures meaningful structure.

### Figure 2

Loss curve showing steady decrease from 0.28 to 0.0023 over 20 epochs.

---

## 9. Phase 2: Supervisor Pretraining

### Goal

Train the Supervisor to predict the **next** latent time step from the **current** one, learning temporal dynamics in latent space.

### How it works

First, all training sequences are embedded into latent space using the (now-trained) Embedder:

```python
H = self.embedder.predict(sequences)  # (N, 144, 24)
```

Then the Supervisor is trained on a shifted prediction task:

```python
Input:  H[:, :-1, :]   # latent steps 0 to 142  (given this...)
Target: H[:, 1:, :]    # latent steps 1 to 143  (...predict this)
```

This is a **next-step prediction** task in latent space. The Supervisor learns: "Given the latent state at hour t, what should the latent state at hour t+1 look like?"

### Loss

```
Loss = mean((H_true_next - H_predicted_next)^2)
```

### Results

| Epoch | MSE Loss |
|---:|---:|
| 1 | 0.0308 |
| 20 | 0.0076 |

Convergence is less dramatic than Phase 1, which is expected -- predicting the future (even one step) is inherently harder than reconstructing the present.

### Figure 3

Combined loss curves for both pretraining phases.

---

## 10. Phase 3: Adversarial Training

### Goal

Train the Generator to produce realistic latent sequences from random noise, while the Discriminator tries to distinguish real from fake.

### The generation path

```
Random noise Z ~ Uniform(0, 1)    shape: (batch, 144, 24)
    |
    v  Generator
    |
Raw latent sequence E_hat          shape: (batch, 144, 24)
    |
    v  Supervisor (refines temporal dynamics)
    |
Refined latent sequence H_hat     shape: (batch, 144, 24)
    |
    v  Recovery (maps back to data space)
    |
Synthetic data X_hat              shape: (batch, 144, 16)
```

The Generator does not produce data directly. It produces a **latent sequence**, which the Supervisor refines for temporal coherence, and then Recovery maps it back to the original feature space.

### The adversarial step

Each training batch performs two updates:

**1. Discriminator update:**
- Embed real data: `H_real = Embedder(X_real)`
- Generate fake latent: `H_fake = Supervisor(Generator(Z))`
- Discriminator scores both: `Y_real = D(H_real)`, `Y_fake = D(H_fake)`
- Loss: binary cross-entropy on correct classification

**2. Generator + Supervisor update:**
- Generate fake latent and reconstruct to data space
- Compute three loss components (see next section)
- Update Generator and Supervisor weights jointly

---

## 11. The Generator Loss Function -- Detailed Breakdown

The generator's total loss combines three objectives:

```python
g_loss = g_loss_u + (100.0 * g_loss_s) + (gamma * g_loss_v)
```

### Component 1: Adversarial loss (`g_loss_u`)

```python
g_loss_u = BinaryCrossEntropy(ones, Y_fake)
```

"Fool the discriminator." The generator wants the discriminator to classify its fake sequences as real (label = 1). This is the core GAN objective.

### Component 2: Supervised loss (`g_loss_s`) -- weighted x100

```python
H_hat = Supervisor(H_real)
g_loss_s = mean((H_real[:, 1:, :] - H_hat[:, :-1, :])^2)
```

"Preserve temporal dynamics." Even during adversarial training, the Supervisor should still correctly predict the next latent state from the current one. This prevents the adversarial game from destroying the temporal structure learned in Phase 2.

**Why x100 weight?** Temporal coherence is critical for time series. Without a strong weight, the adversarial loss could push the Generator toward producing samples that fool the Discriminator but have incoherent temporal dynamics. The 100x weight ensures temporal quality is prioritized.

### Component 3: Reconstruction loss (`g_loss_v`) -- weighted by gamma (1.0)

```python
X_fake = Recovery(H_fake)
g_loss_v = mean((X_real - X_fake)^2)
```

"Stay close to real data in data space." This encourages the Generator to produce latent sequences that, when decoded by Recovery, resemble real data. It acts as an anchor preventing the Generator from drifting too far into unrealistic territory.

### Balance of the three losses

| Component | Weight | What it prevents |
|---|---|---|
| Adversarial (g_loss_u) | 1.0 | Generator ignoring the discriminator (not learning to be realistic) |
| Supervised (g_loss_s) | 100.0 | Loss of temporal coherence during adversarial training |
| Reconstruction (g_loss_v) | 1.0 (gamma) | Generator producing latent sequences that decode into unrealistic data |

---

## 12. Discriminator Threshold

```python
if d_loss > 0.15:
    self.d_optimizer.apply_gradients(zip(d_grads, d_vars))
```

The Discriminator is **only updated when its loss exceeds 0.15**. This is a stabilization technique:

### The problem it solves

In GAN training, if the Discriminator becomes too strong too early, the Generator receives uninformative gradients ("everything you produce is obviously fake") and cannot learn. This leads to **mode collapse** -- the Generator produces the same output regardless of input.

### How the threshold works

| Condition | Action | Reasoning |
|---|---|---|
| `d_loss > 0.15` | Update Discriminator | It's making enough mistakes; it can afford to improve |
| `d_loss <= 0.15` | Skip Discriminator update | It's already very accurate; let the Generator catch up |

This creates a feedback loop: when the Discriminator is too good, it stops improving, giving the Generator time to learn. When the Generator improves enough that the Discriminator starts making errors again, the Discriminator resumes training.

---

## 13. Generating Synthetic Sequences

After training, generating new sequences follows the full path:

```python
def generate(self, n_samples):
    # Step 1: Sample random noise
    Z = np.random.uniform(0, 1, size=(n_samples, seq_len, hidden_dim))

    # Step 2: Generator transforms noise into raw latent sequences
    E_hat = self.generator.predict(Z)

    # Step 3: Supervisor refines temporal dynamics
    H_hat = self.supervisor.predict(E_hat)

    # Step 4: Recovery decodes latent sequences back to data space
    X_hat = self.recovery.predict(H_hat)

    return X_hat  # shape: (n_samples, 144, 16)
```

### The path visualized

```
Random noise          Generator         Supervisor         Recovery
(n, 144, 24) ------> (n, 144, 24) ---> (n, 144, 24) ---> (n, 144, 16)
Uniform[0,1]         raw latent         refined latent     synthetic data
```

Each generated sequence has shape `(144, 16)` -- 144 time steps across 16 features -- and could be split into a forecasting input (120 steps) and target (24 steps) using `split_synthetic_sequences()`.

---

## 14. Quality Assessment

Three checks are performed to evaluate the synthetic sequences:

### Check 1: Reconstruction quality (autoencoder path)

```python
reconstructed = timegan.autoencoder.predict(real_sequences[:256])
reconstruction_mse = mean((real - reconstructed)^2)
```

**Real -> Embedder -> Recovery -> Reconstructed.** This tests the autoencoder in isolation.

Result: MSE ~ 0.0012. **Very good** -- the autoencoder faithfully compresses and reconstructs.

### Check 2: Visual comparison -- real vs. reconstructed (Figure 5)

A single temperature sequence is plotted alongside its reconstruction. The two lines overlap closely, confirming the autoencoder's high fidelity.

### Check 3: Visual comparison -- real vs. fully synthetic (Figure 6)

A real training sequence is plotted alongside a fully generated synthetic sequence. This tests the **complete generation path** (noise -> Generator -> Supervisor -> Recovery).

**Result: Significant quality gap.** The synthetic sequence shows:
- Much lower variability than the real signal
- Reduced amplitude (flatter)
- Missing the temporal dynamics (warming/cooling cycles) visible in real data

This gap between reconstruction quality and generation quality is the critical finding. The autoencoder works well, but the Generator has not learned to produce diverse, realistic latent sequences that capture the full range of real weather dynamics.

### Why the gap exists

| Path | Quality | Why |
|---|---|---|
| **Reconstruction** (real -> embed -> recover) | High | The embedder receives structured real data; the recovery just inverts it |
| **Generation** (noise -> generate -> supervise -> recover) | Low | The generator must transform unstructured random noise into realistic latent dynamics -- a much harder task |

The Generator must learn the **entire distribution** of weather patterns from the training data. With only 15 adversarial epochs and a 24-dimensional latent space, it does not have enough capacity or training time to capture the full complexity of 8 years of weather.

---

## 15. Why TimeGAN Was Not Used in the Final Pipeline

The decision is clearly stated in the notebook and follows from the quality assessment:

| Factor | Assessment |
|---|---|
| Autoencoder reconstruction | Excellent (MSE ~ 0.0012) |
| Synthetic generation | Insufficient realism |
| Temporal dynamics | Flatter and less variable than real data |
| Risk of augmentation | Would introduce noise/artifacts into training |
| Conclusion | **Not used** -- proof-of-concept only |

### What would be needed for it to work

- **Longer adversarial training** (15 epochs is short for GAN training)
- **Larger latent dimension** (24 may be too small for 16 features with complex dynamics)
- **Alternative architectures** (diffusion-based generators, conditional generation)
- **Quantitative evaluation** (not just visual -- e.g., discriminative score, predictive score from the TimeGAN paper)

These are listed as future work in Section 14 of the notebook.

---

## 16. Implementation Summary

### Source modules

| Module | Contents |
|---|---|
| `src/gan/config.py` | `TIMEGAN_CONFIG` dictionary with all hyperparameters |
| `src/gan/data_prep.py` | `make_timegan_sequences()` for creating overlapping sequences; `split_synthetic_sequences()` for splitting generated sequences into (X, y) pairs |
| `src/gan/timegan.py` | `TimeGAN` class with all five sub-networks, three training phases, and generation method |

### Training flow

```
Phase 1: pretrain_autoencoder()     -- 20 epochs, MSE loss
    Embedder + Recovery learn to compress/reconstruct
    |
Phase 2: pretrain_supervisor()      -- 20 epochs, MSE loss
    Supervisor learns latent temporal dynamics
    |
Phase 3: fit()                      -- 15 epochs, adversarial
    Generator + Discriminator compete
    Supervised and reconstruction losses maintain quality
    |
Generation: generate(n_samples)
    Noise -> Generator -> Supervisor -> Recovery -> Synthetic data
    |
Assessment: visual comparison + reconstruction MSE
    Result: reconstruction good, generation insufficient
```

### Figures produced

| Figure | Content |
|---|---|
| Figure 2 | Autoencoder pretraining loss curve |
| Figure 3 | Combined pretraining losses (autoencoder + supervisor) |
| Figure 4 | Adversarial training losses (5 components) |
| Figure 5 | Real vs. reconstructed sequence (temperature) |
| Figure 6 | Real vs. fully synthetic sequence (temperature) |
