# Transfer Learning Techniques: A Practical Guide

## Overview

Transfer learning reuses knowledge from a **source task** to speed up and stabilize learning on a **target task**. In deep networks, **low-level features** (edges, textures, n-grams) tend to be broadly useful, while **high-level features** are task/domain specific.

**Why it works in practice**

* **Faster convergence** and fewer epochs to reach strong performance.
* **Data efficiency** when labeled target data is scarce.
* **Lower compute** vs. training from scratch.
* **Better generalization** via rich priors from pretraining.

> Rule of thumb: the **closer** your target data distribution to the source, the **more** you can reuse; the **farther**, the **more** you must adapt (or re-train).

---

## Quick-Start Decision Guide

1. **How much labeled target data do you have?**

   * Tiny (≤1k): start with **feature extraction** or **PEFT** (LoRA/adapters).
   * Small (1k–50k): **partial fine-tune** or **PEFT**.
   * Medium/Large (>50k): **full fine-tune** is viable.

2. **How similar is the domain to the source?**

   * High similarity: **unfreeze more**.
   * Low similarity: **freeze more** or consider **domain adaptation**/**self-supervised pretraining** on in-domain data.

3. **Deployment constraints?**

   * Tight memory/latency: prefer **distillation**, **quantization**, or **PEFT** (parameter deltas only).

---

## Choosing a Strategy

| Scenario                           | Target Data      | Domain Similarity | Recommended Strategy                                             | Notes                                      |
| :--------------------------------- | :--------------- | :---------------- | :--------------------------------------------------------------- | :----------------------------------------- |
| High resource / Low similarity     | Large            | Low               | **Train from scratch** or **in-domain SSL pretrain → fine-tune** | Avoid negative transfer.                   |
| Feature extraction                 | Small            | High              | **Freeze backbone, train head**                                  | Strong baseline; cheap and stable.         |
| Partial fine-tuning                | Small–Medium     | Medium–High       | **Freeze early layers; unfreeze late blocks**                    | Good balance of stability/adaptation.      |
| Full fine-tuning                   | Medium–Large     | Medium–High       | **Unfreeze all, small LR**                                       | Highest ceiling; monitor forgetting.       |
| PEFT (Adapters/LoRA/Prompt-tuning) | Any (esp. small) | Any               | **Inject small trainable modules**                               | Large gains with minimal trainable params. |

---

## Core Techniques

### 1) Feature Extraction (Linear/Head Probing)

* **Mechanism:** Freeze the pretrained encoder; learn a new task head.
* **When:** Very small datasets; fast baselines; sanity checks.
* **Tips:** Try **linear-probe first**; if promising, then unfreeze progressively.

### 2) Full Fine-Tuning

* **Mechanism:** Unfreeze all layers; optimize with a **much smaller LR** than scratch.
* **Hyperparams:** Base LR often **10×–100× lower**; strong **weight decay**; **grad clipping**.
* **Progressive unfreezing:** Unfreeze from top → bottom to reduce instability.

### 3) Partial (Layer-wise) Fine-Tuning

* **Mechanism:** Freeze early/generic layers; train deeper/task-specific layers.
* **Patterns:** Unfreeze last *N* blocks; use **discriminative LRs** (lower in shallow layers).

### 4) Parameter-Efficient Fine-Tuning (PEFT)

* **Why:** Keep most weights frozen; adapt via small parameter sets → huge memory/latency wins.
* **Variants:**

  * **Adapters:** Small MLPs inserted between blocks; train only adapters + head.
  * **LoRA:** Low-rank matrices on attention/linear weights; train low-rank factors.
  * **Prefix/Prompt-Tuning / P-Tuning:** Learn soft prompts; no backbone changes.
  * **IA³ / BitFit:** Train only per-channel scales (IA³) or only bias terms (BitFit).
* **When:** Limited compute, frequent task-switching, on-device deployment.

---

## Advanced Paradigms

### Domain Adaptation (DA)

* **Goal:** Bridge **covariate shift** between source and target domains.
* **Techniques:**

  * **Adversarial DA (DANN/CDAN):** Align latent distributions with a domain classifier.
  * **Moment matching (MMD/CORAL):** Align statistics across domains.
  * **Source-free DA:** Adapt using only target data and a frozen source model.
  * **Test-Time Adaptation (TTA):** Update batch-norm or a small subset of params using test data entropy/minimization.
* **Caveat:** Prevent label leakage; constrain updates to avoid drift.

### Multi-Task Learning (MTL)

* **Goal:** Share an encoder across related tasks for regularization and data efficiency.
* **Design:** Shared encoder + per-task heads; consider task balancing (uncertainty weighting).

### Knowledge Distillation (KD)

* **Goal:** Compress a large teacher into a smaller student with minimal accuracy loss.
* **Mechanism:** Match **soft logits**, intermediate **features**, or **attention maps**.
* **Pair with:** Quantization/pruning for deployment.

### Self-Supervised Pretraining (SSL/SSP)

* **Vision:** Contrastive (SimCLR, MoCo), Siamese (BYOL, SimSiam), masked image modeling (MAE).
* **NLP:** Masked language modeling (BERT), seq-to-seq denoising (T5).
* **Audio/Speech:** CPC, wav2vec 2.0, HuBERT.
* **Recipe:** Pretrain on **unlabeled in-domain** data → fine-tune supervised.

### Prompting & Inference-Time Techniques (for Foundation Models)

* **Zero/Few-shot prompting, CoT, Self-Consistency, Retrieval-Augmented Generation (RAG)**.
* **Note:** No gradient updates; fast to iterate; pair with **guardrails** and **evaluation**.

---

## Best Practices

### Optimization & LRs

* **Pretrained layers:** Start around **1e-5–5e-5** (transformers) or **1e-4–3e-4** (CNNs).
* **New heads:** Often **10×** higher than backbone LR.
* **Discriminative LRs:** Decrease LR toward early layers.
* **Schedulers:** Cosine decay with warmup; One-Cycle works well for vision.

### Regularization & Stability

* **Data augmentation:** Crucial with limited labels (AutoAugment/RandAugment, Mixup/CutMix for vision; SpecAugment for audio; back-translation/noise for NLP).
* **Dropout / Stochastic Depth:** Moderate values; avoid over-regularizing tiny heads.
* **Grad clipping:** Prevents exploding grads during early unfreeze.
* **EMA (model averaging):** Stabilizes validation metrics.

### Normalization & Freezing Details

* **BatchNorm (BN):** If freezing convs, either **freeze BN affine + running stats** or **update stats carefully** with sufficient batch size; mismatch causes drift.
* **LayerNorm (LN):** Safer to leave trainable in transformers; consider **BitFit** (bias-only) for tiny updates.

### Data & Splits

* **Stratify** splits; **group by entity** to avoid leakage (e.g., same patient/user in train/test).
* Track **data contamination** (overlap with pretraining corpora).
* Maintain a **clean validation** set; avoid repeated peeking/tuning.

### Evaluation

* Use **learning curves** (samples vs. accuracy) to choose strategy.
* Report **calibration** (ECE/Brier), not just accuracy/F1.
* **Ablate**: linear-probe → +partial → +full/PEFT.
* **Robustness**: test on shifted domains; OOD detection if relevant.

### Reproducibility

* Fix seeds; log **versions, hyperparams, checkpoints**; save **exact preprocessing**.
* Use **mixed precision** (fp16/bf16) and **deterministic ops** where possible.

### Deployment

* Combine **distillation + quantization (int8/int4)**; consider **sparsity** or **pruning**.
* Prefer **PEFT** for rapid multi-task adaptation; store only deltas.

---

## Practical Recipes by Modality

### Vision (Image Classification)

1. Start with **ImageNet-pretrained** CNN/ViT.
2. **Linear probe** (frozen backbone) → assess head accuracy.
3. If underfitting: **unfreeze last 1–3 blocks** with LR_backbone ≈ 1e-4, head ≈ 1e-3; strong aug (RandAugment + Mixup/CutMix), label smoothing 0.1.
4. If still constrained: **LoRA on attention/MLP** (for ViT) or **adapters** in later blocks.

### NLP (Sequence Classification)

1. Start with **BERT/RoBERTa/T5** base.
2. **Linear head** over [CLS] or pooled token; LR_head ≈ 1e-3, LR_backbone ≈ 2e-5 (if unfrozen).
3. Low data: **LoRA/Prefix-tuning/BitFit**; batch size ≥ 16 with grad-accum; warmup 10%.
4. Regularize with **weight decay** (0.01), **dropout** (0.1).

### Speech/Audio (ASR or Emotion)

1. **wav2vec 2.0/HuBERT** pretrained encoder.
2. Freeze encoder; train CTC/seq2seq head.
3. Unfreeze top encoder blocks with small LR; use **SpecAugment**; clip grads.

### Tabular + Text/Images (Multimodal)

* Freeze vision/text backbones; learn a **fusion head** (attention/projection + MLP).
* Consider **adapters/LoRA** inside each backbone before fusion.

---

## Reference Implementation (PyTorch-style Pseudocode)

```python
model = load_pretrained_backbone(name)          # CNN/Transformer
head  = init_task_head(model.out_dim, num_classes)

freeze(model)                                   # feature extraction baseline
opt = AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(E):
    for x, y in loader:
        with torch.no_grad():
            feats = model(x)
        logits = head(feats)
        loss = criterion(logits, y)
        update(opt, loss)

# If promising: partial unfreeze with discriminative LRs
unfreeze_last_n_blocks(model, n=2)
params = [
    {"params": head.parameters(), "lr": 1e-3},
    {"params": last_n_blocks(model, 2).parameters(), "lr": 1e-4},
]
opt = AdamW(params, weight_decay=0.05)
sched = cosine_with_warmup(opt, warmup_steps=500)

train_with_mixup_cutmix_dropout_gradclip(...)
early_stop_on_val(...)
save_best_checkpoint(...)
```

---

## Common Failure Modes & Fixes

### Negative Transfer

* **Symptom:** Worse than training from scratch.
* **Fix:** Increase freezing; try **linear-probe** baseline; switch to **in-domain SSL pretraining** or **DA**.

### Catastrophic Forgetting

* **Symptom:** Rapid loss of general features; val drops early.
* **Fix:** Lower LR, partial unfreeze, **EWC/LwF** (continual-learning regularizers), stronger weight decay.

### Overfitting

* **Symptom:** Train↑, Val↓.
* **Fix:** More augmentation; smaller head; **PEFT** instead of full FT; early stopping; label smoothing.

### BN/LN Mismatch

* **Symptom:** Train ok, test unstable.
* **Fix:** Freeze BN stats and affine when backbone is frozen; ensure adequate batch size if updating stats.

---

## Operational Checklist

* [ ] Establish **linear-probe** baseline first.
* [ ] Decide **partial vs. full vs. PEFT** based on data, similarity, and constraints.
* [ ] Use **discriminative LRs** + **warmup**; clip grads.
* [ ] Lock down **normalization behavior** (BN/LN).
* [ ] Apply **strong augmentations** appropriate to modality.
* [ ] Track **calibration** and **robustness**, not just accuracy.
* [ ] Log **seeds, versions, preprocessing**; archive configs and checkpoints.
* [ ] Validate on **OOD/shifted** splits if relevant.
* [ ] Plan **deployment** (distill/quantize/PEFT) early.

---

## References & Further Reading

* Pan & Yang (2010). *A Survey on Transfer Learning*.
* Yosinski et al. (2014). *How transferable are features in deep neural networks?*
* Howard & Ruder (2018). *ULMFiT: Universal Language Model Fine-tuning for Text Classification*.
* Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
* He et al. (2016). *Deep Residual Learning for Image Recognition (ResNet)*.
* Chen et al. (2020). *SimCLR: A Simple Framework for Contrastive Learning of Visual Representations*.
* Grill et al. (2020). *BYOL: Bootstrap Your Own Latent*.
* He et al. (2022). *MAE: Masked Autoencoders Are Scalable Vision Learners*.
* Radford et al. (2021). *CLIP: Learning Transferable Visual Models From Natural Language Supervision*.
* Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
* Tzeng et al. (2017); Ganin et al. (2016). *Domain-Adversarial Approaches (DANN)*.
* Sun et al. (2016). *CORAL: Deep CORrelation ALignment*.
* Kingma & Ba (2015). *Adam*. Loshchilov & Hutter (2019). *AdamW*.

---

## Appendix: Typical Hyperparameter Ranges (Starting Points)

| Setting                         | Vision (CNN/ViT) | NLP (BERT-base classif.) | Audio (wav2vec 2.0) |
| :------------------------------ | :--------------- | :----------------------- | :------------------ |
| LR (frozen backbone, head only) | 1e-3 – 3e-3      | 5e-4 – 1e-3              | 1e-3                |
| LR (partial FT, backbone)       | 3e-5 – 3e-4      | 1e-5 – 3e-5              | 5e-6 – 3e-5         |
| Weight decay                    | 0.02 – 0.1       | 0.01                     | 0.01                |
| Warmup                          | 5–10% steps      | 10% steps                | 10% steps           |
| Label smoothing                 | 0.05–0.2         | 0.0–0.1                  | —                   |
| Grad clip (norm)                | 1.0              | 1.0                      | 1.0                 |

> Tune empirically per dataset; validate with learning curves and ablations.

---
