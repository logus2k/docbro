# Matrix Shapes & Transposes

## Terminology

- **batch size**: number of samples in the current mini-batch.
- **feature dimension**: number of features per sample.

---

## 1) The default layout

**Rows = Samples and Columns = Features**
Remember: **B × F**: “Batch by Features”

* Input (batch): $X \in \mathbb{R}^{B \times d_{\text{in}}}$
  where $B$ = batch size (samples), $d_{\text{in}}$ = feature dimension.

---

## 2) Weights (always Out × In)

* Layer weights: $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
* Bias: $b \in \mathbb{R}^{d_{\text{out}} \times 1}$ (one bias per output neuron)

> **Mnemonic:** Weights are **Out × In**. Never include the **batch size** in $W$.

---

## 3) Forward pass (rows = samples)

Pre-activation:
$$
Z = X,W^\top + \mathbf{1},b^\top
\quad\Rightarrow\quad
(B \times d_{\text{in}})\cdot(d_{\text{in}} \times d_{\text{out}})
= B \times d_{\text{out}}.
$$

Activation (element-wise):
$$
H = \phi(Z) \in \mathbb{R}^{B \times d_{\text{out}}}.
$$

> **Mnemonic:** *Data is $B\times F$, Weights are $\text{Out}\times\text{In}$, Output is $B\times\text{Out}$.*

---

## 4) The “bias trick” (one matmul instead of matmul + add)

Fold the bias into the multiply by augmenting **features**, not samples.

* Augment input with a **column of 1s**:
  $$
  X_{\text{aug}} = [,X\ \mathbf{1},] \in \mathbb{R}^{B \times (d_{\text{in}}+1)}.
  $$
* Augment weights by **appending $b$ as a column**:
  $$
  W_{\text{aug}} = [,W\ b,] \in \mathbb{R}^{d_{\text{out}} \times (d_{\text{in}}+1)}.
  $$
* Compute:
  $$
  Z = X_{\text{aug}}, W_{\text{aug}}^\top \in \mathbb{R}^{B \times d_{\text{out}}}.
  $$

> When rows = samples, **add a column of 1s** to $X$ (not a row).

---

## 5) Layer-to-layer shape flow

Let layer widths be $d_0 \to d_1 \to d_2 \to \cdots$ with $d_0=d_{\text{in}}$.

* $W^{(1)} \in \mathbb{R}^{d_1 \times d_0}$ and $Z^{(1)}, H^{(1)} \in \mathbb{R}^{B \times d_1}$
* $W^{(2)} \in \mathbb{R}^{d_2 \times d_1}$ and $Z^{(2)}, H^{(2)} \in \mathbb{R}^{B \times d_2}$
* …rows stay $B$; columns change to the current layer width.

---

## 6) Alternative layout (less common): Columns = Samples

If you store data as **features × samples**:

* $X \in \mathbb{R}^{d_{\text{in}} \times B}$, $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
* Forward:
  $$
  Z = W X + b,\mathbf{1}^\top \in \mathbb{R}^{d_{\text{out}} \times B}.
  $$
* Bias trick here: **add a row of 1s** to $X$ (not a column).

> Same rule still holds: **Weights = Out × In**. Batch size never appears in $W$.

---

## 7) Sanity checks (fast)

* **Inner dims must match:** if $(A_{m\times n})(B_{n\times p})$ → OK; else transpose the right thing.
* **Bias aligns with outputs:** $\text{len}(b) = d_{\text{out}}$.
* **Activation keeps shape:** $H$ has the **same shape** as $Z$.

Unit-check trick:
$$
X,(B \times d_{\text{in}})\quad W^\top,(d_{\text{in}} \times d_{\text{out}})
;\Rightarrow; Z,(B \times d_{\text{out}}).
$$

---

## 8) Common mistakes (and fixes)

* ❌ Using **batch size** in weight shapes (e.g., $3\times4$ because there are 4 samples).
  ✅ Weights depend only on **features in** and **neurons out**.

* ❌ Adding bias along the **sample axis** (a row of 1s) when rows = samples.
  ✅ Add a **column** of 1s to $X$ (augment **features**).

* ❌ Applying softmax **across samples**.
  ✅ Apply softmax **across classes (columns)** *per row* (per sample).

---

## 9) Tiny worked example

Given $B=4$, $d_{\text{in}}=2$, hidden widths $3 \to 4 \to 3$, output $=1$:

* $X: 4\times2$
* $W_1: 3\times2$ ⇒ $Z_1,H_1: 4\times3$
* $W_2: 4\times3$ ⇒ $Z_2,H_2: 4\times4$
* $W_3: 3\times4$ ⇒ $Z_3,H_3: 4\times3$
* $W_4: 1\times3$ ⇒ $Z_4: 4\times1$ → sigmoid → $Y: 4\times1$

(With the bias trick at each step, temporarily augment $X/H$ by one **extra column** and $W$ by one **extra column**.)

---

## 10) Quick code/Excel patterns

**NumPy (rows = samples):**

```python
Z = X @ W.T + b.ravel()     # (B, d_out)
H = np.maximum(Z, 0)        # ReLU
```

**NumPy (bias trick):**

```python
X_aug = np.hstack([X, np.ones((B,1))])   # (B, d_in+1)
W_aug = np.hstack([W, b])                # (d_out, d_in+1) with b as (d_out,1)
Z = X_aug @ W_aug.T
```

**Excel (rows = samples, bias trick):**

```
=MMULT( X_aug , TRANSPOSE(W_aug) )
```

---

## 11) Two-line memory aid

* **“Data is $B\times F$; Weights are $\text{Out}\times\text{In}$; Output is $B\times\text{Out}$.”**
* **“If inner dims don’t match, transpose the weights (or the data) — not the batch.”**

---
