# Sequence Models Evolution

One page tracing the evolution of RNNs → LSTMs → GRUs → Attention → Transformers.

---

## 1980s–early 1990s — Foundations of Recurrent Nets

* **Simple (“vanilla”) RNNs** (Elman 1990; Jordan 1986).
  Idea: a hidden state $h_t$ updated by $h_t = \phi(W_x x_t + W_h h_{t-1} + b)$.
* **Backpropagation Through Time (BPTT)** (Williams & Zipser, 1989) makes training feasible.
* **Problem discovered:** **vanishing/exploding gradients** (Hochreiter, 1991). Long-term dependencies are hard.

**Why it mattered:** established recurrence + training, exposed core optimization obstacles.

---

## 1997 — LSTM: Making Long Dependencies Learnable

* **Long Short-Term Memory (LSTM)** (Hochreiter & Schmidhuber, 1997).
  Introduces cell state $c_t$ and gates (input/forget/output) to control information flow.
* **Refinements:** **forget gate** (Gers et al., 2000), **peepholes** (2002), **bidirectional RNNs** (1997).

**Why it mattered:** dramatically improves learning of long-range dependencies; becomes default for many sequence tasks.

---

## Early 2000s — Practical Training & Speech Breakthroughs

* **CTC loss** (Graves et al., 2006) for unsegmented labeling (ASR/handwriting).
* **Reservoir computing / Echo State Networks** (Jaeger, 2001) as lightweight alternatives.

**Why it mattered:** broadened RNN applicability (speech, handwriting) with better training objectives.

---

## 2012–2015 — RNNs Go Mainstream for Seq2Seq

* **RNN Transducer (RNN-T)** (Graves, 2012) for **streaming** sequence transduction (ASR).
* **Seq2Seq encoder–decoder** (Sutskever et al., 2014) for translation; deep LSTMs with input reversal.
* **Neural attention** on RNNs:
  **Additive attention** (Bahdanau et al., 2014), **multiplicative/Luong** (2015).
  Aligns source and target tokens during decoding.
* **GRU (Gated Recurrent Unit)** (Cho et al., 2014):
  $z_t$ (update), $r_t$ (reset), and $\tilde{h}_t$ candidate; **fewer params than LSTM**, similar accuracy.

**Why it mattered:** attention + gated recurrence made seq2seq accurate and trainable at scale.

---

## 2016–2017 — From Recurrence to Attention-First

* **Dilated CNN/TCN alternatives** (e.g., WaveNet 2016) show strong non-RNN temporal modeling.
* **Transformer** (“Attention Is All You Need”, Vaswani et al., 2017):
  Removes recurrence; uses **self-attention** with positional encodings; parallelizable training.

**Why it mattered:** unlocks efficient parallel training, better long-range modeling, and scaling laws.

---

## 2018→today — Transformers Dominate; RNNs Specialize

* **Pretrained Transformers:** BERT (2018), GPT series, T5/BART → transfer learning for NLP.
  Variants for other modalities: **ViT** (vision), **Conformer** (speech: conv + self-attention).
* **Efficient/long-context attention:** Longformer/Performer, etc.
* **RNNs persist** where they shine: **streaming, low-latency, on-device** (e.g., GRU/LSTM/RNN-T).

**Why it mattered:** attention models become general-purpose backbones; RNNs remain valuable for real-time constraints.

---

## Minimal “When to Use What”

* **Offline NLP, long context, transfer:** Transformers (encoder for understanding; decoder for generation; enc-dec for seq2seq).
* **Streaming ASR / low-latency edge:** LSTM/GRU, **RNN-T**, Conformer-RNN-T.
* **Time-series with fixed horizons:** TCN/dilated CNN; Transformers or state-space models when horizons are long.
* **Small models / embedded:** GRU/LSTM preferred for parameter efficiency.

---

## Quick Shape Reminders (carry across models)

* **Batch is not in parameters.** Weights depend on **in** and **out** features/channels, not batch size.
* **Inner dims must match** for matmul/attention/conv.
  Example (MLP layer): $Z = X W^\top + b$ with $X \in \mathbb{R}^{B \times d_{\text{in}}}$, $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ → $Z \in \mathbb{R}^{B \times d_{\text{out}}}$.
* **Bias aligns with outputs** (length $d_{\text{out}}$).
* **Activation is element-wise** → same shape as input.

---

## Citable Anchors (years)

* Elman/Jordan RNNs (1986–1990)
* BPTT (1989)
* Vanishing/exploding (1991)
* **LSTM (1997)**; BiRNNs (1997); Forget gate (2000)
* ESN (2001); CTC (2006)
* RNN-T (2012)
* Seq2Seq + attention (2014–2015)
* **GRU (2014)**
* WaveNet/TCN (2016)
* **Transformer (2017)**
* BERT/GPT/T5 (2018+)

---

### One-sentence takeaway

RNNs established sequence modeling; **LSTM/GRU** solved long-dependencies; **attention on RNNs** improved alignment; **Transformers** removed recurrence, enabling scalable, transfer-friendly models—while **RNNs** still excel for streaming and tight-compute settings.

---
