# Parameter-Efficient Fine-Tuning (PEFT) with LoRA & QLoRA

## 1) Why PEFT?

Fine-tuning all weights of large pretrained models is effective but **expensive** (memory, compute, storage). **Parameter-Efficient Fine-Tuning (PEFT)** adapts a model by training **only a small subset of additional parameters** while keeping the original weights **frozen**. Benefits:

* **Compute/Memory:** Train and store **millions** of parameters instead of **billions**.
* **Stability:** Lower risk of catastrophic forgetting; base model remains intact.
* **Modularity:** Maintain many task-specific adapters and hot-swap them at inference.
* **Latency/Deploy:** Optionally **merge** adapters into base weights for single-file deployment.

---

## 2) LoRA: The Core Idea

### 2.1 Low-Rank Updates

Instead of optimizing a full weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$, **LoRA** learns a low-rank update $\Delta W$:

$$
\Delta W = B A, \quad
B \in \mathbb{R}^{d_{out} \times r}, ;
A \in \mathbb{R}^{r \times d_{in}}, ;
r \ll \min(d_{in}, d_{out})
$$

The effective weight during training/inference is:
$$
W' = W + \frac{\alpha}{r} , B A
$$

* $W$ is **frozen** (pretrained).
* $A, B$ are **trainable**.
* $\alpha$ is a scaling hyperparameter (often $\alpha \approx r$).
* Initialization: typically $B$ initialized near zero (so $W' \approx W$ at start).

**Parameter count:** full update has $d_{out} \cdot d_{in}$ parameters; LoRA trains only $r(d_{in}+d_{out})$.
If $d_{in}=d_{out}=4096$ and $r=16$: full = 16.8M vs. LoRA ≈ 131k (**~128× fewer**).

### 2.2 Where to Apply LoRA

Apply to the most sensitive/profitable matrices in transformer blocks:

* **Attention projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
* **MLP projections:** `up_proj`, `gate_proj`, `down_proj` (optional; add if underfitting)

Starting point: only `q_proj` and `v_proj`. If accuracy stalls, include `o_proj` and selective MLP layers.

### 2.3 Training Objective

Same as standard fine-tuning (e.g., cross-entropy for language modeling). Only LoRA parameters (and any new head) receive gradients. Use the **same optimizer** families (AdamW, etc.) with **smaller learning rates** than scratch.

---

## 3) QLoRA: LoRA on Quantized Bases

**QLoRA** combines LoRA with **low-bit quantization** of the **frozen base weights** to reduce memory further:

* **Base model weights:** 4-bit (commonly **NF4** quantization) or 8-bit; kept frozen.
* **LoRA adapters:** kept in higher precision (fp16/bf16) and trained as usual.
* **Compute:** weights are dequantized on-the-fly for matmuls; adapters add low-rank deltas.

**Why it works:** Most adaptation capacity lives in the **learned low-rank deltas**, while quantization compresses the large, frozen backbone. This enables fine-tuning **very large models** on a single modern GPU.

---

## 4) Practical Recipes

### 4.1 Choosing Hyperparameters

* **Rank ($r$):** 8–64.

  * Start with **$r=8$ or $16$**. Increase if the model underfits or the task is complex.
* **Alpha ($\alpha$):** 16–128, often set to $\alpha=r$ or $2r$.
* **Dropout (LoRA only):** 0.05–0.1 when data is small/noisy.
* **Target modules:** Start with `q_proj`, `v_proj`. Add `o_proj`, then MLP if needed.
* **Precision:**

  * **LoRA:** fp16/bf16 training is common.
  * **QLoRA:** 4-bit base (nf4) + fp16/bf16 adapters.

### 4.2 Optimization & Scheduling

* **LR (adapters):** $1\mathrm{e}{-4}$ to $5\mathrm{e}{-4}$ (language models); smaller for vision backbones.
* **Warmup:** ~5–10% of total steps.
* **Weight decay:** small or zero on adapter matrices (task-dependent).
* **Gradient clipping:** norm 1.0 is a safe default.
* **Batching:** Use gradient accumulation to reach effective batch sizes of ≥ 128 tokens per device (LLMs) if memory is tight.

### 4.3 Normalization & Freezing

* Keep **base model frozen** (weights, and typically LayerNorm/Bias unless you intentionally use bias-only tuning like BitFit).
* If you unfreeze anything else, use **discriminative LRs** (lower for earlier layers).

### 4.4 Data & Regularization

* **Small datasets:** favor **lower $r$**, **dropout**, **early stopping**, and **strong task-specific augmentation** (vision/audio).
* **Text tasks:** consider packing or fused sequences to improve GPU utilization.

---

## 5) Evaluation & Diagnostics

* **Start with a quick baseline:** a **linear head** or very low-rank LoRA to sanity-check labels and preprocessing.
* Track **learning curves** (loss vs. steps) and **validation metrics**; LoRA should converge quickly.
* If **underfitting**: increase $r$, add more target modules, modestly raise LR, or train longer.
* If **overfitting**: add dropout, reduce $r$, use early stopping, or collect/augment more data.
* Report **calibration** (e.g., ECE/Brier) when decisions depend on probability estimates.

---

## 6) Inference & Deployment

### 6.1 Adapter Composition

* You can **stack or compose** multiple adapters (e.g., base skills + domain adapter). In practice, keep one active per request unless you know they commute well.

### 6.2 Merging Adapters

* For deployment simplicity, **merge** LoRA into the base:
  $W \leftarrow W + \frac{\alpha}{r} BA$ (one-time).
* Pros: single set of weights, lower runtime complexity.
* Cons: loses modularity; you must re-merge to switch tasks.

### 6.3 Quantized Serving (QLoRA)

* Keep base **quantized** and load the **adapter** weights at runtime.
* Ensure your inference stack supports **low-bit matmuls** efficiently, or pre-merge into a higher-precision artifact if latency is critical.

---

## 7) When LoRA/QLoRA vs. Full Fine-Tuning?

**Prefer LoRA/QLoRA when:**

* Compute/memory is constrained or you manage **many tasks**.
* Target data is **small/medium** or close to the pretraining domain.
* You need **fast iteration** and **safety** (frozen base, lower forgetting risk).

**Prefer Full Fine-Tuning when:**

* You have **large, high-quality** labeled data and **significant domain shift**.
* You require deep architectural or representational changes throughout the network.
* You seek the **last few %** on a hard benchmark and can pay the cost.

---

## 8) Worked Example (LLM, PyTorch + PEFT-style)

```python
import torch
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType

# 1) Load a pretrained model (causal LM as an example)
base = load_pretrained_llm(...)  # e.g., AutoModelForCausalLM.from_pretrained(...)

# 2) Configure LoRA
lora_cfg = LoraConfig(
	task_type=TaskType.CAUSAL_LM,
	r=16,                 # rank
	lora_alpha=32,        # scaling (≈ r or 2r is common)
	lora_dropout=0.05,
	target_modules=["q_proj", "v_proj"]  # start minimal; expand if needed
)

# 3) Wrap the model
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()  # sanity check: tiny % of total

# 4) Optimizer/schedule
opt = AdamW(model.parameters(), lr=1e-4)
sched = get_cosine_with_warmup(opt, warmup_steps=0.1 * total_steps)

# 5) Train loop (standard)
model.train()
for batch in train_loader:
	loss = model(**batch).loss
	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
	opt.step(); sched.step(); opt.zero_grad()

# 6) Save only adapters (small) or merge into base for deployment
# model = model.merge_and_unload()  # optional: bake LoRA into base weights
# model.save_pretrained(output_dir)
```

**QLoRA tweak:** load the **base** in 4-bit (e.g., nf4) and keep **adapters** in fp16/bf16; everything else is identical from an API perspective.

---

## 9) Common Pitfalls & Fixes

| Pitfall                        | Symptom                  | Fix                                                                       |
| ------------------------------ | ------------------------ | ------------------------------------------------------------------------- |
| Too low capacity               | Underfitting, flat loss  | Increase $r$; add `o_proj`/MLP adapters; train longer                     |
| Too high capacity on tiny data | Overfitting              | Add LoRA dropout; reduce $r$; stronger regularization/early stop          |
| Wrong target modules           | No gains vs. linear head | Start with `q,v`; then add `o` and selective MLP                          |
| LR too high                    | Unstable loss/val        | Reduce LR, add warmup, enable grad clipping                               |
| BN/LN quirks (vision)          | Train/val mismatch       | Keep base frozen; avoid changing normalization stats unless intentional   |
| Quantization artifacts (QLoRA) | Small accuracy drop      | Use nf4 4-bit or try 8-bit; ensure stable dequant kernels and batch sizes |

---

## 10) Quick-Start Checklists

**Minimal LoRA (LLM):**

* [ ] `target_modules = ["q_proj","v_proj"]`
* [ ] `r = 8–16`, `alpha = r or 2r`, `dropout = 0.05`
* [ ] `lr = 1e-4`, warmup 5–10%, clip grad 1.0
* [ ] Train, validate, then consider expanding targets

**Minimal QLoRA:**

* [ ] Load base in **4-bit (nf4)**, adapters in **fp16/bf16**
* [ ] Same LoRA hyperparams as above
* [ ] Watch memory; use gradient accumulation if needed

---

## 11) Conceptual Summary

* **LoRA** approximates the needed weight change with a **low-rank matrix**. This captures most task-specific variation at a tiny parameter cost.
* **QLoRA** pushes memory even lower by **quantizing** the large, frozen base while still learning expressive low-rank deltas.
* Together, they deliver **near-full-tuning quality** for many tasks with a fraction of the resources, and they scale elegantly to multi-task and on-device settings.

---

## 12) References (for deeper study)

* **LoRA:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).
* **QLoRA:** Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023).
* Related PEFT methods: Adapters, Prefix/Prompt-Tuning, P-Tuning, BitFit, IA³.

---
