# Numba for GPU-Accelerated Python: Study Guide

## 1. Introduction to Numba

### What is Numba?

Numba is a **just-in-time (JIT) compiler** for Python that focuses on numerical code. It translates a subset of Python and NumPy into **fast machine code** using LLVM (via `llvmlite`).

**Key Capabilities:**
- Targets **CPU** (single-threaded or multi-threaded)
- Targets **GPU** via CUDA backend (`numba.cuda`)
- Works best on **tight loops** and **numerical operations**

**Basic Workflow:**
1. Write normal Python functions
2. Decorate with `@jit` or related decorators
3. First call compiles to optimized machine code
4. Subsequent calls execute much faster

### JIT Compilation vs Interpretation

- **Interpreted languages** (standard Python): Execute code line-by-line at runtime
- **Compiled languages** (C, C++): Translate entire program to machine code before execution
- **JIT compilation** (Numba): Compiles functions to machine code at runtime, only when called

---

## 2. CPU Acceleration with `@njit`

### 2.1 Pure Python Baseline

Consider a sum of squares function:

$$f(x) = \sum_{i=0}^{N-1} x_i^2$$

**Pure Python implementation:**
```python
def sum_of_squares(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * x[i]
    return total
```

**Performance characteristics:**
- Slow due to Python interpreter overhead
- Dynamic type checking on every iteration
- No optimization of loop structure

### 2.2 Using `@njit` Decorator

**`@njit`** = "no Python mode" - compiles to machine code and disallows Python object mode

```python
from numba import njit

@njit
def sum_of_squares_numba(x):
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i] * x[i]
    return total
```

**Performance characteristics:**
- **First call**: Slower (includes compilation time)
- **Subsequent calls**: Much faster than pure Python
- Numba **specializes** the function for the input types (`float64`, etc.)

**Key insight:** Compilation overhead is paid once; subsequent calls reuse compiled code.

---

## 3. Multi-Core CPU Parallelization

### 3.1 Using `prange` for Parallel Loops

Numba can generate **multithreaded code** for CPUs using a data-parallel programming model.

**Key decorators and functions:**
- **`prange`**: Parallel range - indicates loop can execute in parallel
- **`parallel=True`**: Enables parallel execution in decorated function
- **`fastmath=True`**: Enables aggressive floating-point optimizations

```python
from numba import prange

@njit(parallel=True, fastmath=True)
def sum_of_squares_numba_parallel(x):
    total = 0.0
    for i in prange(x.shape[0]):
        total += x[i] * x[i]
    return total
```

**Benefits:**
- Exploits multiple CPU cores automatically
- Significant speedup for large arrays
- No manual thread management required

---

## 4. Creating Universal Functions with `@vectorize`

### 4.1 What are Universal Functions (ufuncs)?

NumPy's **ufuncs** operate element-wise on arrays and support broadcasting. Numba's `@vectorize` creates similar functions that:
- Support scalar or array inputs
- Work efficiently on CPU
- Can optionally target GPU

### 4.2 CPU Vectorization Example

```python
import numpy as np
from numba import vectorize

@vectorize(['float64(float64)'], target='cpu')
def gauss_cpu(x):
    return np.exp(-x * x)
```

**Signature components:**
- `['float64(float64)']`: Type signature (output(input))
- `target='cpu'`: Compile for CPU execution

**Available targets:**
- `target='cpu'`: Default CPU execution
- `target='parallel'`: Multi-threaded CPU ufunc
- `target='cuda'`: GPU execution (requires CUDA-capable GPU)

---

## 5. GPU Programming with Numba

### 5.1 GPU Architecture Overview

**Key GPU information:**
```python
from numba import cuda

my_gpu = cuda.get_current_device()
print("GPU name:", my_gpu.name)
print("Compute capability:", my_gpu.compute_capability)
print("Streaming Multiprocessors:", my_gpu.MULTIPROCESSOR_COUNT)
```

**CUDA Cores per SM by compute capability:**
- Compute capability 7.x (Volta/Turing): 64 cores/SM
- Compute capability 8.x (Ampere): 128 cores/SM
- Compute capability 9.x (Ada Lovelace): 128 cores/SM

### 5.2 GPU Vectorization with `@vectorize`

**Critical limitation:** Cannot use NumPy functions inside GPU ufuncs. Must use Python `math` module instead.

```python
import math
from numba import vectorize

@vectorize(['float32(float32)'], target='cuda')
def gauss_gpu(x):
    return math.exp(-x * x)  # Use math.exp, NOT np.exp
```

**Automatic GPU features:**
- **Automatic memory transfer**: NumPy arrays transferred CPU ↔ GPU automatically
- **Automatic work scheduling**: Work distributed across all GPU threads
- **Automatic memory management**: GPU memory freed automatically when objects destroyed

**Memory cleanup (optional):**
```python
cuda.current_context().memory_manager.deallocations.clear()
```

### 5.3 Multi-Input Vectorization Example

```python
@vectorize(['float32(float32, float32)', 'float64(float64, float64)'], target='cuda')
def gpu_sincos(x, y):
    return math.sin(x) * math.cos(y)
```

**Performance characteristics:**
- CPU vectorization similar to NumPy (math operations dominate)
- GPU vectorization significantly faster for large arrays
- GPU overhead matters for small arrays

---

## 6. Numba and CuPy Interoperability

### 6.1 Complementary Roles

**CuPy provides:**
- GPU arrays (`cp.ndarray`)
- High-level operations (FFT, linear algebra, convolutions)

**Numba provides:**
- Custom compiled functions (CPU/GPU)
- Fine-grained control over execution

### 6.2 Common Pattern: CPU Pre-processing + GPU Computation

```python
from numba import njit, prange
import cupy as cp

@njit(parallel=True, fastmath=True)
def normalize(x):
    """CPU-side pre-processing with Numba"""
    m = 0.0
    for i in prange(x.shape[0]):
        m += x[i]
    m /= x.shape[0]
    
    s2 = 0.0
    for i in prange(x.shape[0]):
        diff = x[i] - m
        s2 += diff * diff
    s2 /= x.shape[0]
    s = np.sqrt(s2)
    
    for i in prange(x.shape[0]):
        x[i] = (x[i] - m) / s
    
    return x

# Workflow
x_cpu = np.random.rand(N).astype(np.float64)
x_cpu_normalized = normalize(x_cpu)  # Fast CPU pre-processing
x_gpu = cp.asarray(x_cpu_normalized)  # Move to GPU
result = cp.dot(x_gpu, x_gpu)  # GPU computation
```

---

## 7. Performance Benchmarking Best Practices

### 7.1 Proper Benchmarking Pattern

**NumPy (CPU):**
```python
t0 = time.perf_counter()
result_numpy = np.sum(x_np * x_np)
t1 = time.perf_counter()
numpy_time = t1 - t0
```

**Numba (CPU):**
```python
# Warm-up compilation
_ = sumsq_numba(x_np)

# Timed execution
t0 = time.perf_counter()
result_numba = sumsq_numba(x_np)
t1 = time.perf_counter()
numba_time = t1 - t0
```

**CuPy (GPU):**
```python
x_cp = cp.array(x_np)  # Copy to GPU

# Warm-up
_ = cp.sum(x_cp * x_cp)
cp.cuda.Stream.null.synchronize()

# Timed execution with CUDA events
t0 = cp.cuda.Event()
t1 = cp.cuda.Event()
t0.record()
result_cupy = cp.sum(x_cp * x_cp)
t1.record()
t1.synchronize()
cupy_time = cp.cuda.get_elapsed_time(t0, t1) / 1000.0  # Convert to seconds
```

### 7.2 Performance Considerations

**When GPU shines:**
- Very large arrays
- Computationally intensive operations
- Operations that can be heavily parallelized

**When CPU + Numba is competitive:**
- Smaller arrays (GPU overhead not amortized)
- Operations with complex control flow
- Limited parallelism potential

---

## 8. Key Takeaways

### 8.1 Numba Strengths
- **Easy to use**: Minimal code changes (add decorators)
- **Flexible**: Targets CPU or GPU
- **Composable**: Works well with NumPy, CuPy
- **No new syntax**: Pure Python code

### 8.2 Best Practices
1. **Always warm up** compiled functions before benchmarking
2. **Use appropriate targets**: CPU for small data, GPU for large
3. **Profile before optimizing**: Measure actual bottlenecks
4. **Combine tools**: Numba + CuPy for complete workflows

### 8.3 Limitations
- GPU vectorize cannot use NumPy functions (use `math` module)
- Best for numerical operations (not general Python)
- Compilation overhead on first call
- Not all Python/NumPy features supported

### 8.4 What's Next
This guide covers high-level Numba decorators (`@njit`, `@vectorize`). The next level involves:
- **`@cuda.jit`** decorator for custom CUDA kernels
- Explicit control over **threadIdx**, **blockIdx**
- Direct GPU memory management
- Manual grid/block configuration

These advanced features provide fine-grained control similar to CUDA C but from Python.

---
