# CuPy for GPU-Accelerated Python: Study Guide

## 1. Introduction to CuPy

### What is CuPy?

CuPy is an **open-source Python array library** that uses GPU power to accelerate numerical computations. It is designed as a **drop-in replacement** for NumPy and SciPy.

**Key Features:**
- **GPU Acceleration**: Runs computations on NVIDIA CUDA or AMD ROCm platforms
- **NumPy/SciPy Compatibility**: Provides `cupy.ndarray` and functions that mirror NumPy/SciPy APIs
- **High Performance**: Can achieve 100x+ speedups over NumPy for large-scale calculations
- **Interoperability**: Integrates with PyTorch, TensorFlow via DLPack protocol
- **Custom Kernels**: Supports creation of custom CUDA kernels using C++ snippets

**Basic Concept:**
- NumPy arrays live in **host (CPU) memory**
- CuPy arrays live in **device (GPU) memory**
- Operations execute on the GPU automatically

---

## 2. Basic CuPy Usage

### 2.1 Installation and Setup

```python
import numpy as np
import cupy as cp

print("NumPy version:", np.__version__)
print("CuPy version:", cp.__version__)
```

### 2.2 Creating Arrays

**Simple porting pattern:**
```python
# NumPy (CPU)
x_np = np.array([1, 2, 3], dtype=np.float32)

# CuPy (GPU) - identical syntax
x_cp = cp.array([1, 2, 3], dtype=cp.float32)
```

**Common array creation functions:**
- `cp.array()` - Create array from list/NumPy array
- `cp.ones()` - Array of ones
- `cp.zeros()` - Array of zeros
- `cp.linspace()` - Evenly spaced values
- `cp.random.rand()` - Random values

---

## 3. GPU Device Information

### 3.1 Checking GPU Availability

```python
# Get current device
device = cp.cuda.Device()
print("Using GPU device:", device)

# Device properties
props = cp.cuda.runtime.getDeviceProperties(device.id)
print("Name:", props["name"].decode())
print("MultiProcessor count:", props["multiProcessorCount"])
print("Total global memory (GB):", props["totalGlobalMem"] / (1024**3))
```

**What to check:**
- GPU name and compute capability
- Number of streaming multiprocessors (SMs)
- Total GPU memory available

---

## 4. Host-Device Memory Transfer

### 4.1 Understanding Host vs Device

**Memory Locations:**
- **Host (CPU)** memory → NumPy arrays (`np.ndarray`)
- **Device (GPU)** memory → CuPy arrays (`cp.ndarray`)

### 4.2 Transfer Operations

```python
# Start with NumPy array on CPU
a_np = np.random.rand(5, 5).astype(np.float32)

# CPU → GPU transfer
a_cp = cp.asarray(a_np)  # or cp.array(a_np)
print("Type on GPU:", type(a_cp))  # <class 'cupy.ndarray'>

# GPU → CPU transfer
a_back_np = cp.asnumpy(a_cp)
print("Type back on CPU:", type(a_back_np))  # <class 'numpy.ndarray'>
```

**Typical Workflow:**
1. Load data on CPU (from disk, databases, etc.)
2. Transfer to GPU with `cp.asarray()`
3. Process on GPU with CuPy operations
4. Transfer results back to CPU with `cp.asnumpy()` (if needed for visualization/saving)

---

## 5. GPU Operations

### 5.1 Elementwise Operations

```python
# Create large arrays directly on GPU
N = 10_000_000
x = cp.linspace(0, 1, N, dtype=cp.float32)
y = cp.linspace(1, 2, N, dtype=cp.float32)

# Elementwise operations (executed on GPU)
z = x**2 + 2 * y
```

### 5.2 Reduction Operations

```python
# Reductions on GPU
mean_z = cp.mean(z)
sum_z = cp.sum(z)

# Convert GPU scalars to Python for printing
print("Mean(z) on GPU:", float(mean_z))
print("Sum(z) on GPU:", float(sum_z))
```

**Important:** CuPy scalars live on GPU; use `float()` or `int()` to bring values to CPU.

---

## 6. Performance Measurement and Benchmarking

### 6.1 Why CPU Timing is Incorrect for GPU

**Critical concept:** GPU operations are **asynchronous** with respect to CPU execution.

**Common pitfall:**
```python
# WRONG - Only measures kernel launch time, not execution
t0 = time.perf_counter()
result = cp.sum(x_cp)
t1 = time.perf_counter()
# This timing is meaningless!
```

### 6.2 Correct GPU Timing with CUDA Events

```python
import cupy as cp
import time

# Create CUDA events for timing
gpu_t0 = cp.cuda.Event()
gpu_t1 = cp.cuda.Event()

# Record start event
gpu_t0.record()

# GPU operation
result = x_cp.sum()

# Record end event
gpu_t1.record()

# Wait for GPU to finish
gpu_t1.synchronize()

# Get elapsed time (in milliseconds)
gpu_time_ms = cp.cuda.get_elapsed_time(gpu_t0, gpu_t1)
gpu_time_s = gpu_time_ms / 1000.0

print(f"GPU time: {gpu_time_s:.3f} s")
```

**Key functions:**
- **`cp.cuda.Event()`**: Creates synchronization marker for GPU timing
- **`event.record()`**: Enqueues event at current point in stream
- **`event.synchronize()`**: CPU waits until GPU reaches this event
- **`cp.cuda.get_elapsed_time(start, end)`**: Returns time in milliseconds

### 6.3 Warm-up Pattern

**First GPU call includes overhead:**
- CUDA context creation
- Memory setup
- Kernel compilation/caching

**Best practice:**
```python
# Warm-up call
_ = x_cp.sum()
cp.cuda.Stream.null.synchronize()

# Now perform timed execution
gpu_t0.record()
result = x_cp.sum()
gpu_t1.record()
gpu_t1.synchronize()
```

### 6.4 Complete Benchmarking Example

```python
N = 100_000_000

# CPU: NumPy
x_np = np.random.rand(N).astype(np.float32)

cpu_t0 = time.perf_counter()
s_np = x_np.sum()
cpu_t1 = time.perf_counter()
cpu_time = cpu_t1 - cpu_t0

print(f"NumPy sum: {s_np:.4f}, time: {cpu_time:.3f} s")

# GPU: CuPy
x_cp = cp.asarray(x_np)  # Transfer to GPU

# Warm-up
_ = x_cp.sum()
cp.cuda.Stream.null.synchronize()

# Timed execution
gpu_t0 = cp.cuda.Event()
gpu_t1 = cp.cuda.Event()

gpu_t0.record()
s_cp = x_cp.sum()
gpu_t1.record()
gpu_t1.synchronize()

gpu_time = cp.cuda.get_elapsed_time(gpu_t0, gpu_t1) / 1000.0

print(f"CuPy sum: {float(s_cp):.4f}, time: {gpu_time:.3f} s")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

---

## 7. Practical Application: 2D Image Convolution

### 7.1 Image Loading and Preparation

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load and prepare image (CPU side)
img = Image.open("image.jpg").convert("L")  # Grayscale
img = img.resize((1024, 1024), Image.BILINEAR)
img_np = np.asarray(img, dtype=np.float32)

print("Image shape:", img_np.shape)

# Visualize
plt.imshow(img_np, cmap="gray")
plt.title("Original Image (CPU)")
plt.axis("off")
plt.show()
```

### 7.2 Defining Convolution Kernels

```python
# 3x3 box blur kernel
box_kernel_np = np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]], dtype=np.float32
)
box_kernel_np /= box_kernel_np.sum()  # Normalize

# 3x3 edge detection kernel (Sobel-like)
edge_kernel_np = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float32
)
```

### 7.3 CPU Reference Implementation

```python
from scipy.signal import convolve2d

# CPU convolution
t0 = time.perf_counter()
img_blur_cpu = convolve2d(img_np, box_kernel_np, mode="same", boundary="symm")
t1 = time.perf_counter()

print(f"CPU (SciPy) blur time: {t1 - t0:.3f} s")

# Visualize
plt.imshow(img_blur_cpu, cmap="gray")
plt.title("CPU Blur")
plt.axis("off")
plt.show()
```

### 7.4 GPU Implementation with CuPy

```python
import cupyx.scipy.signal as cpsignal

# Transfer to GPU
img_cp = cp.asarray(img_np)
box_kernel_cp = cp.asarray(box_kernel_np)

# Warm-up
_ = cpsignal.convolve2d(img_cp, box_kernel_cp, mode="same", boundary="symm")
cp.cuda.Stream.null.synchronize()

# Timed execution
gpu_t0 = cp.cuda.Event()
gpu_t1 = cp.cuda.Event()

gpu_t0.record()
img_blur_gpu = cpsignal.convolve2d(img_cp, box_kernel_cp, mode="same", boundary="symm")
gpu_t1.record()
gpu_t1.synchronize()

gpu_time = cp.cuda.get_elapsed_time(gpu_t0, gpu_t1) / 1000.0

print(f"GPU (CuPy) blur time: {gpu_time:.3f} s")

# Transfer back and visualize
img_blur_gpu_np = cp.asnumpy(img_blur_gpu)

plt.imshow(img_blur_gpu_np, cmap="gray")
plt.title("GPU Blur")
plt.axis("off")
plt.show()
```

**CuPy SciPy Functions:**
- Available in `cupyx.scipy` module
- API-compatible with SciPy
- Common functions: `convolve2d`, `fft`, `linalg` operations

---

## 8. Performance Considerations

### 8.1 When GPU Acceleration Shines

**GPU is beneficial for:**
- Very large arrays (millions+ elements)
- Computationally intensive operations
- Highly parallelizable tasks (element-wise operations)
- Repeated operations on same data

### 8.2 When CPU May Be Competitive

**GPU overhead matters for:**
- Small arrays (thousands of elements)
- Simple operations with low computational intensity
- Single operations (transfer cost dominates)
- Complex control flow

### 8.3 Transfer Costs

**Key insight:** Data transfer between CPU and GPU is expensive.

**Best practices:**
- Minimize transfers between host and device
- Keep data on GPU for multiple operations
- Batch transfers when possible
- Use `float32` instead of `float64` when precision allows (faster transfer and computation)

---

## 9. Common Patterns and Best Practices

### 9.1 Typical CuPy Workflow

```python
# 1. Load/prepare data on CPU
data_np = load_data()  # NumPy array

# 2. Transfer to GPU once
data_cp = cp.asarray(data_np)

# 3. Perform multiple operations on GPU
result1_cp = cp.mean(data_cp, axis=0)
result2_cp = cp.std(data_cp, axis=0)
normalized_cp = (data_cp - result1_cp) / result2_cp

# 4. Transfer final result back (if needed)
final_result = cp.asnumpy(normalized_cp)

# 5. Use result on CPU (save, visualize, etc.)
save_results(final_result)
```

### 9.2 Synchronization Best Practices

```python
# Always synchronize before timing
cp.cuda.Stream.null.synchronize()

# Use CUDA events for GPU timing
gpu_t0 = cp.cuda.Event()
gpu_t1 = cp.cuda.Event()

gpu_t0.record()
# GPU operation here
gpu_t1.record()
gpu_t1.synchronize()

elapsed_ms = cp.cuda.get_elapsed_time(gpu_t0, gpu_t1)
```

### 9.3 Data Type Optimization

```python
# Prefer float32 over float64 when possible
data_np = np.random.rand(N).astype(np.float32)  # NOT float64

# Reasons:
# - Faster GPU computation (more single-precision units)
# - Faster memory transfer (half the bandwidth)
# - Less GPU memory usage
```

---

## 10. CuPy vs NumPy API Comparison

### 10.1 Direct Replacements

| NumPy | CuPy | Notes |
|-------|------|-------|
| `np.array()` | `cp.array()` | Create array |
| `np.zeros()` | `cp.zeros()` | Array of zeros |
| `np.ones()` | `cp.ones()` | Array of ones |
| `np.linspace()` | `cp.linspace()` | Evenly spaced values |
| `np.sum()` | `cp.sum()` | Sum reduction |
| `np.mean()` | `cp.mean()` | Mean reduction |
| `np.dot()` | `cp.dot()` | Dot product |
| `np.random.rand()` | `cp.random.rand()` | Random values |

### 10.2 SciPy Compatibility

| SciPy | CuPy | Module |
|-------|------|--------|
| `scipy.signal.convolve2d` | `cupyx.scipy.signal.convolve2d` | Signal processing |
| `scipy.fft.fft` | `cupyx.scipy.fft.fft` | FFT operations |
| `scipy.linalg` | `cupyx.scipy.linalg` | Linear algebra |

---

## 11. Advanced Example: Mandelbrot Set

### 11.1 Mathematical Background

The Mandelbrot set computation is **perfectly data-parallel**:

$$z_0 = 0, \quad z_{n+1} = z_n^2 + c$$

For each complex number $c$:
- Iterate until $|z_n| > 2$ (diverges) or reach max iterations
- Record iteration count for coloring
- Each point is completely independent

### 11.2 CuPy Implementation Pattern

```python
def mandelbrot_cupy(x_min, x_max, y_min, y_max, width, height, max_iter):
    # 1. Create coordinate grid on GPU
    x = cp.linspace(x_min, x_max, width, dtype=cp.float64)
    y = cp.linspace(y_min, y_max, height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y, indexing="ij")
    
    # 2. Complex plane on GPU
    C = X + 1j * Y
    
    # 3. Initialize arrays on GPU
    Z = cp.zeros_like(C, dtype=cp.complex128)
    mat = cp.zeros(C.shape, dtype=cp.float64)
    mask = cp.ones(C.shape, dtype=bool)
    
    # 4. Vectorized iteration loop
    for n in range(max_iter):
        # Update active points
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        
        # Check divergence
        escaped = cp.abs(Z) > 2.0
        newly_escaped = escaped & mask
        
        if newly_escaped.any():
            # Smooth coloring
            z_new = Z[newly_escaped]
            mat[newly_escaped] = (
                n + 1 - cp.log(cp.log(cp.abs(z_new))) / cp.log(2.0)
            )
            mask[newly_escaped] = False
        
        if not mask.any():
            break
    
    return mat
```

**Key techniques demonstrated:**
- GPU array creation with `cp.linspace`, `cp.meshgrid`
- Complex number operations on GPU
- Boolean masking for selective updates
- Early termination optimization
- All computation stays on GPU until final result

---

## 12. Key Takeaways

### 12.1 CuPy Strengths

- **Easy to use**: Minimal code changes from NumPy
- **High performance**: Leverages massive GPU parallelism
- **Familiar API**: NumPy/SciPy compatible
- **Comprehensive**: Covers most common numerical operations

### 12.2 Best Practices Summary

1. **Always use CUDA events** for GPU timing (not `time.perf_counter()`)
2. **Warm up GPU** before benchmarking
3. **Minimize CPU-GPU transfers** (keep data on GPU)
4. **Use float32** when precision allows
5. **Synchronize explicitly** when timing matters
6. **Batch operations** on GPU rather than single operations
7. **Profile first** - not all operations benefit equally from GPU

### 12.3 When to Use CuPy

**Ideal use cases:**
- Large-scale array operations (millions+ elements)
- Image/signal processing on large datasets
- Scientific computing with NumPy-style operations
- Data preprocessing for machine learning
- Iterative algorithms on large arrays

**Not ideal for:**
- Small arrays (< 10,000 elements)
- Single operations with high transfer overhead
- Complex control flow and conditionals
- Operations not supported by CuPy API

### 12.4 Relationship with Other Tools

**CuPy complements:**
- **NumPy**: Drop-in GPU replacement
- **Numba**: CuPy for high-level ops, Numba for custom kernels
- **PyTorch/TensorFlow**: CuPy for scientific computing, DL frameworks for neural networks
- **SciPy**: GPU acceleration for scientific algorithms

---

## 13. Common Pitfalls and Solutions

### 13.1 Incorrect Timing

**Problem:**
```python
# WRONG
t0 = time.perf_counter()
result = cp.sum(x_cp)
t1 = time.perf_counter()
```

**Solution:**
```python
# CORRECT
gpu_t0 = cp.cuda.Event()
gpu_t1 = cp.cuda.Event()
gpu_t0.record()
result = cp.sum(x_cp)
gpu_t1.record()
gpu_t1.synchronize()
time_ms = cp.cuda.get_elapsed_time(gpu_t0, gpu_t1)
```

### 13.2 Excessive CPU-GPU Transfers

**Problem:**
```python
for i in range(1000):
    data_cp = cp.asarray(data_np[i])  # Transfer each iteration
    result_cp = process(data_cp)
    result_np = cp.asnumpy(result_cp)  # Transfer back each iteration
```

**Solution:**
```python
# Transfer all data once
all_data_cp = cp.asarray(data_np)
for i in range(1000):
    result_cp = process(all_data_cp[i])  # Work on GPU
# Transfer final results once
final_results = cp.asnumpy(result_cp)
```

### 13.3 Not Converting GPU Scalars

**Problem:**
```python
mean_value = cp.mean(x_cp)
print(f"Mean: {mean_value}")  # Slow, triggers implicit transfer
```

**Solution:**
```python
mean_value = float(cp.mean(x_cp))  # Explicit conversion
print(f"Mean: {mean_value}")  # Fast
```

---
