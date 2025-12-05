# NumPy vs PyTorch

---

## Overview

| Aspect | NumPy (`ndarray`) | PyTorch (`Tensor`) | Notes |
|---|---|---|---|
| **Primary scope** | General array math on **CPU** | Array math + deep learning on **CPU & accelerators** (CUDA, MPS, XLA) | `t = t.to("cuda")` |
| **Autograd** | ❌ | ✅ `requires_grad=True`, `.backward()` | Optimizers use grads |
| **Default dtype** | `float64` | `float32` | Watch cross-lib casts |
| **Device model** | CPU only | `.to(device)`, `.cpu()`, `.cuda()` | Ops must share device |
| **Indexing/broadcast** | Yes | Yes (NumPy-compatible) | Right-aligned shapes |
| **Strides/contiguity** | Strided; contiguity less visible | Strided; some ops need contiguous | `t = t.contiguous()` |
| **In-place ops** | Via slice/`+=` | Ops end with `_` (e.g., `add_`) | Can break autograd if unsafe |
| **Linalg/FFT** | `numpy.linalg`, `numpy.fft` | `torch.linalg`, `torch.fft` | Similar coverage |
| **Randomness** | `np.random`, `Generator` | `torch.rand*`, `torch.Generator` | Per-device RNG |
| **Interop zero-copy** | — | `torch.from_numpy(nd)` / `t.numpy()` (CPU only) | Shared buffer |
| **Serialization** | `np.save/load` | `torch.save/load`, `state_dict` | Pickle + storages |
| **Sparse/special** | Via SciPy | Native sparse (COO/CSR/CSC/BSR), quantized, AMP | Also pinned memory |
| **JIT/compile** | Numba, etc. | `torch.compile` (2.x) | Graph capture/fusion |
| **Ecosystem** | SciPy stack | `nn`, `optim`, `DataLoader`, `torchvision/torchaudio` | End-to-end DL |

---

## Interop

### Zero-copy (CPU only; shared memory)
```python
import numpy as np, torch

# NumPy → Torch
nd = np.arange(6).reshape(2, 3)
t  = torch.from_numpy(nd)        # no copy; shared buffer

# Torch → NumPy
t2 = torch.arange(6).reshape(2, 3)
nd2 = t2.numpy()                 # no copy; shared buffer
```
- **Hazard:** Mutating one mutates the other. Use `.clone()` or copies when independence is needed.

### Force copies (safe round-trip)
```python
t_copy  = torch.tensor(nd)       # copy NumPy → Torch
nd_copy = np.array(t2)           # copy Torch → NumPy
safe    = t2.detach().cpu().numpy().copy()  # fully independent
```

### Dtype & device alignment
```python
# NumPy default float64; cast for PyTorch GPU work
nd = np.random.randn(4, 3).astype(np.float32)
t  = torch.from_numpy(nd).to("cuda")         # zero-copy on CPU → device move
```

### Checklist
- If you see dtype mismatch errors: cast with `.float()/.double()`.
- If you see device mismatch errors: ensure all tensors are on the same device (`cpu`/`cuda`).
- For dataloading from NumPy: consider `torch.from_numpy` + `pin_memory=True` in DataLoader (CPU→GPU faster).

---

## Operation Name Mapping

| Concept | NumPy | PyTorch |
|---|---|---|
| Create range | `np.arange(n)` | `torch.arange(n)` |
| Linspace | `np.linspace(a,b,N)` | `torch.linspace(a,b,steps=N)` |
| Zeros / Ones | `np.zeros(s)`, `np.ones(s)` | `torch.zeros(s)`, `torch.ones(s)` |
| Random uniform | `np.random.rand(*s)` | `torch.rand(s)` |
| Random normal | `np.random.randn(*s)` | `torch.randn(s)` |
| Identity | `np.eye(n)` | `torch.eye(n)` |
| Reshape | `a.reshape(s)` | `t.reshape(s)` |
| Flatten | `a.ravel()` / `a.reshape(-1)` | `t.view(-1)` / `t.flatten()` |
| Transpose | `a.transpose(axes)` | `t.permute(axes)` |
| Swap axes | `np.swapaxes(a,i,j)` | `t.movedim(i,j)` |
| Expand dims | `np.expand_dims(a,axis)` | `t.unsqueeze(dim)` |
| Squeeze | `np.squeeze(a,axis=None)` | `t.squeeze(dim=None)` |
| Concatenate | `np.concatenate([a,b], axis=k)` | `torch.cat([t,u], dim=k)` |
| Stack | `np.stack([a,b], axis=k)` | `torch.stack([t,u], dim=k)` |
| Repeat/Tile | `np.repeat(a,reps,axis)` / `np.tile(a,reps)` | `t.repeat(reps)` / `t.repeat_interleave(reps,dim)` |
| Where | `np.where(c,x,y)` | `torch.where(c,x,y)` |
| Clip | `np.clip(a, lo, hi)` | `torch.clamp(t, min=lo, max=hi)` |
| Argmax | `a.argmax(axis=k)` | `t.argmax(dim=k)` |
| Sort | `np.sort(a,axis=k)` | `t.sort(dim=k)` |
| Top-k | `np.argpartition` combo | `torch.topk(k, dim=k)` |
| Mean/Sum | `a.mean(axis=k)` / `a.sum(axis=k)` | `t.mean(dim=k)` / `t.sum(dim=k)` |
| Std/Var | `a.std(axis=k)` / `a.var(axis=k)` | `t.std(dim=k)` / `t.var(dim=k)` |
| Matmul | `a @ b`, `np.matmul(a,b)` | `t @ u`, `torch.matmul(t,u)` |
| Norm | `np.linalg.norm(a,ord,axis=k)` | `torch.linalg.norm(t, ord=..., dim=k)` |
| In-place add | `a += b` | `t.add_(b)` (or `t += b` with care) |

**Tip:** In PyTorch, many functions have both functional (`torch.*`) and method (`tensor.*`) forms. For autograd safety, avoid in-place ops on tensors needed for gradient computation.

---

## Autograd Essentials

### Minimal pattern
```python
import torch

# 1) Create parameters with grads
w = torch.randn(3, requires_grad=True)

# 2) Build computation (any tensor ops)
x = torch.randn(5, 3)
y = x @ w
loss = (y**2).mean()

# 3) Backprop
loss.backward()       # accumulate grads into w.grad

# 4) Optimizer-like step (manual)
with torch.no_grad():
	w -= 0.1 * w.grad
	w.grad.zero_()
```

### Common rules & gotchas
- Set `requires_grad=True` on **leaf** tensors you want gradients for (e.g., model parameters).
- Backprop with `loss.backward()`; grads accumulate in `.grad`. Zero them between steps.
- **In-place ops** (ending with `_`) can invalidate saved tensors needed for backward and cause errors.
- For inference or when editing parameters: wrap in `with torch.no_grad(): ...`.
- Detach from graph when exporting to NumPy: `t.detach().cpu().numpy()`.

### Autocast (mixed precision) on GPU
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in loader:
	data = data.to("cuda", non_blocking=True)
	target = target.to("cuda", non_blocking=True)

	with autocast():
		out = model(data)
		loss = loss_fn(out, target)

	scaler.scale(loss).backward()
	scaler.step(optimizer)
	scaler.update()
	optimizer.zero_grad(set_to_none=True)
```

---

## Device Guide (CPU, CUDA, MPS, XLA)

### Moving tensors
```python
t_cpu  = torch.randn(2, 3)             # CPU
t_gpu  = t_cpu.to("cuda")              # NVIDIA CUDA
t_mps  = t_cpu.to("mps")               # Apple Silicon (Metal)
t_back = t_gpu.cpu()                   # back to CPU
```
- Check availability: `torch.cuda.is_available()`, `torch.backends.mps.is_available()`.
- Keep operands on the **same device**; mixing devices raises errors.
- Some kernels require contiguous layout; after `permute`, call `.contiguous()` if an op complains.

### Performance tips
- Use `pin_memory=True` in DataLoader for faster CPU→GPU copies.
- Prefer `float32` (default) for training; use AMP (`autocast`) for speed on GPU.
- For reproducibility: set seeds in both libs if mixing (`np.random.seed(...)`, `torch.manual_seed(...)`) and control CUDA determinism if needed.

---

## Quick Practice

**A. NumPy batch → GPU tensor (normalized [0,1])**
```python
imgs_np = np.random.randint(0, 256, (32, 3, 64, 64), dtype=np.uint8)
imgs = torch.from_numpy(imgs_np).float().div_(255).to("cuda")  # zero-copy CPU → cast → normalize → move
```

**B. Safe round-trip without sharing**
```python
a = np.linspace(0, 1, 8, dtype=np.float32).reshape(2, 4)
t = torch.tensor(a)                 # copy
out_np = (t**2).detach().cpu().numpy().copy()
```

**C. Non-contiguous to contiguous**
```python
t = torch.randn(4, 3, 2).permute(2, 0, 1)  # non-contiguous
u = t.contiguous().view(2, -1)             # now safe to view/reshape
```

---
