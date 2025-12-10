# CuPy Glossary

---

## **A**

### **Array API Compatibility**

CuPy provides a GPU-accelerated array object (`cupy.ndarray`) with an API closely matching NumPy.
Most NumPy code can be converted to CuPy by simply replacing `numpy` with `cupy`, enabling drop-in GPU acceleration.

* Covers most common NumPy operations: element-wise operations, broadcasting, slicing, reductions, linear algebra.
* Not all NumPy features supported (e.g., some advanced indexing, object arrays).
* Includes SciPy-compatible functions in `cupyx.scipy` module.

### **Asynchronous Execution**

GPU operations in CuPy are **asynchronous** with respect to CPU execution.
Kernel launches return immediately to the CPU, allowing the CPU to continue execution while the GPU works.

* Critical for understanding timing: CPU timing utilities measure launch time, not execution time.
* Requires explicit synchronization for accurate performance measurement.
* See **Device Synchronization** and **CUDA Events**.

---

## **B**

### **Broadcasting**

NumPy's mechanism for performing operations on arrays of different shapes, fully supported in CuPy.
Dimensions are aligned from right to left, and dimensions of size 1 are stretched to match.

* Same rules as NumPy broadcasting.
* Executed on GPU with minimal memory overhead.
* Essential for vectorized computations without explicit loops.

---

## **C**

### **Coalesced Memory Access**

GPU optimization where threads in a warp access consecutive memory addresses.
Results in combined memory transactions, drastically improving throughput.

* Critical for GPU kernel performance.
* CuPy's built-in operations generally ensure coalesced access.
* Custom kernels require careful attention to access patterns.

### **CUDA**

A parallel computing platform and API model from NVIDIA used by CuPy to execute array operations directly on the GPU.
Operations run as compiled CUDA kernels.

* Developed by NVIDIA for general-purpose GPU computing.
* CuPy abstracts CUDA complexity while exposing key features.
* Also supports AMD ROCm platform for non-NVIDIA GPUs.

### **CUDA Events**

Synchronization markers used for accurate GPU timing and stream coordination.

```python
gpu_t0 = cp.cuda.Event()
gpu_t1 = cp.cuda.Event()
gpu_t0.record()
# GPU operation
gpu_t1.record()
gpu_t1.synchronize()
elapsed_ms = cp.cuda.get_elapsed_time(gpu_t0, gpu_t1)
```

* Essential for benchmarking GPU operations.
* Measure actual GPU execution time, not just kernel launch time.
* More accurate than CPU-based timing for GPU workloads.

### **CUDA Kernel**

A function executed in parallel on the GPU.
CuPy uses both:

* **Pre-built kernels** for standard operations (e.g., addition, multiplication, reductions)
* **JIT-compiled kernels** (RawKernel, ElementwiseKernel, ReductionKernel) for custom GPU functions

All CuPy operations ultimately execute as CUDA kernels on the GPU.

### **CUDA Stream**

See **GPU Stream**.

### **CuPy-CuPy Interoperability**

CuPy integrates with other GPU libraries via standard protocols:

* **DLPack**: Zero-copy tensor exchange with PyTorch, TensorFlow, JAX.
* **CUDA Array Interface**: Standard protocol for GPU array sharing.
* Enables building pipelines across different GPU frameworks.

### **cupyx.scipy**

CuPy's SciPy-compatible submodule providing GPU-accelerated scientific computing functions.

```python
import cupyx.scipy.signal as cpsignal
import cupyx.scipy.ndimage as cpndimage
import cupyx.scipy.fft as cpfft
```

* Available modules: `signal`, `ndimage`, `fft`, `linalg`, `sparse`, `special`
* API-compatible with SciPy for easy porting.
* Particularly useful for signal processing, image filtering, FFTs.

### **cupy.ndarray**

CuPy's GPU array type, analogous to `numpy.ndarray` but stored in device memory.
Supports:

* Vectorized operations
* Broadcasting
* Slicing and indexing
* Reductions (sum, mean, max, min, etc.)
* Advanced indexing (subset)
* Matrix operations

Most operations dispatch to GPU kernels under the hood.
All data resides in GPU memory unless explicitly transferred.

---

## **D**

### **Data Transfer Overhead**

The time cost of moving data between CPU (host) and GPU (device) memory.

* Can dominate performance for small arrays or single operations.
* Minimize by:
  * Keeping data on GPU for multiple operations
  * Batching transfers
  * Using `float32` instead of `float64` when appropriate
* Critical consideration when deciding whether to use GPU acceleration.

### **Device**

The GPU hardware and its associated memory (VRAM).
Contrasts with **host** (CPU and system RAM).

* CuPy arrays live in device memory.
* Operations execute on device processors.
* Multiple devices can be managed with `cupy.cuda.Device()`.

### **Device Memory**

Memory allocated on the GPU (VRAM), distinct from host RAM.
CuPy arrays reside in device memory; moving data between host and device requires explicit transfer.

* Typical size: 4-80+ GB depending on GPU model.
* Much faster access from GPU than accessing host memory.
* Managed automatically by CuPy's memory pool.
* Transfer operations: `cupy.asarray()` (host→device), `cupy.asnumpy()` (device→host).

### **Device Synchronization**

Forcing the CPU to wait until all pending GPU operations complete.
Most CuPy operations are asynchronous; synchronization occurs when:

* Transferring data back to host with `cupy.asnumpy()`
* Calling `cupy.cuda.Stream.null.synchronize()`
* Invoking certain blocking operations
* Accessing scalar values (implicit synchronization)

Essential for accurate timing and ensuring results are ready before use.

### **DLPack**

A standard protocol for zero-copy tensor exchange between GPU frameworks.
CuPy supports conversion to/from PyTorch, TensorFlow, JAX without memory copying.

```python
# CuPy → PyTorch
torch_tensor = torch.from_dlpack(cupy_array)
# PyTorch → CuPy
cupy_array = cupy.from_dlpack(torch_tensor)
```

---

## **E**

### **Element-wise Operations**

Operations applied independently to each element of an array.

* Examples: addition, multiplication, trigonometric functions, exponentials.
* Highly parallelizable and ideal for GPU execution.
* CuPy executes these as optimized CUDA kernels.
* Support broadcasting for arrays of different shapes.

### **ElementwiseKernel**

A CuPy utility to JIT-compile custom element-wise GPU kernels using a concise DSL (Domain-Specific Language).

```python
square = cupy.ElementwiseKernel(
    'float32 x',        # input
    'float32 y',        # output
    'y = x * x;',       # operation
    'square_kernel'     # kernel name
)
```

* Generates optimized CUDA kernel at runtime.
* Supports multiple inputs/outputs and complex expressions.
* Automatically handles threading and indexing.

---

## **F**

### **Fusion (Kernel Fusion)**

Automatic optimization that combines multiple element-wise operations into a single GPU kernel.
Reduces memory reads/writes and kernel launch overhead.

* Used implicitly in some CuPy operations.
* Explicitly enabled via `@cupy.fuse()` decorator.
* Particularly effective for chains of element-wise operations.

```python
@cupy.fuse()
def fused_operation(x):
    return (x ** 2 + 1) / (x + 2)
```

---

## **G**

### **GPU (Graphics Processing Unit)**

Specialized hardware designed for massively parallel computation.
Originally for graphics, now widely used for general-purpose computing.

* Contains hundreds to thousands of cores.
* Optimized for throughput over latency.
* CuPy leverages GPU parallelism for array operations.

### **GPU Stream**

A CUDA stream represents an execution queue on the GPU.
Operations in the same stream execute sequentially; operations in different streams can execute concurrently.

```python
stream = cupy.cuda.Stream()
with stream:
    # operations execute in this stream
    result = cupy.sum(array)
```

* Default stream: `cupy.cuda.Stream.null`
* Used for overlapping compute and memory transfers.
* Advanced feature for performance optimization.

### **Global Memory (GPU)**

The main GPU memory (VRAM), accessible by all threads.

* Largest GPU memory space (gigabytes).
* Highest latency (hundreds of cycles).
* All CuPy arrays stored in global memory.
* Efficient access requires coalesced memory patterns.

---

## **H**

### **Host**

The CPU and its main system memory (RAM).
Contrasts with **device** (GPU and VRAM).

* Where NumPy arrays reside.
* Where Python code executes.
* Data must be transferred from host to device for GPU computation.

### **Host-Device Transfer**

Moving data between CPU (host) memory and GPU (device) memory.

```python
# Host → Device (NumPy → CuPy)
gpu_array = cp.asarray(numpy_array)

# Device → Host (CuPy → NumPy)
numpy_array = cp.asnumpy(gpu_array)
```

* Relatively expensive operation (limited by PCIe bandwidth).
* Minimize transfers for best performance.
* See **Data Transfer Overhead**.

---

## **J**

### **JIT Compilation (CuPy Kernels)**

Just-In-Time compilation: CuPy uses NVRTC (NVIDIA Runtime Compilation) to compile CUDA C/C++ source code into kernels at runtime.

Used by:

* `RawKernel` - custom CUDA kernels
* `RawModule` - multiple kernels from source
* `ElementwiseKernel` - element-wise operations
* `ReductionKernel` - reduction operations

* JIT kernels are cached, so subsequent runs don't recompile.
* Enables high-performance custom operations without pre-compilation.

---

## **M**

### **Memory Coalescing**

See **Coalesced Memory Access**.

### **Memory Pool**

CuPy implements a caching memory allocator that reuses GPU memory blocks.
Reduces overhead of repeated `cudaMalloc`/`cudaFree` calls.

Benefits:

* Faster execution (avoids frequent allocation/deallocation)
* Lower GPU memory fragmentation
* Automatically enabled by default

Controllable via:

```python
cupy.get_default_memory_pool().free_all_blocks()  # Clear cache
cupy.cuda.set_allocator()  # Custom allocator
```

### **Multi-GPU Support**

CuPy supports multiple CUDA devices in a single system.

```python
# Switch to device 1
cupy.cuda.Device(1).use()

# Context manager
with cupy.cuda.Device(1):
    array = cupy.zeros(100)
```

* Arrays remain bound to the device where created.
* Requires explicit peer-to-peer transfers between GPUs.
* Useful for workload distribution across multiple GPUs.

---

## **N**

### **NumPy Compatibility**

CuPy provides a NumPy-compatible API for GPU acceleration.

* Most NumPy functions have CuPy equivalents.
* Same function names and signatures.
* Enables porting code by replacing `np` with `cp`.

Supported:

* Array creation, manipulation, indexing
* Mathematical operations, broadcasting
* Linear algebra, FFTs, random number generation
* Reductions, sorting, searching

Not fully supported:

* Object arrays
* Some advanced indexing patterns
* String operations
* Certain specialized NumPy functions

### **NumPy Interoperability**

CuPy supports easy conversion between CPU and GPU:

* **Host → GPU**: `cupy.asarray(numpy_array)` or `cupy.array(numpy_array)`
* **GPU → Host**: `cupy.asnumpy(cupy_array)`

Many NumPy functions have equivalent CuPy implementations with identical APIs.

### **NVRTC (NVIDIA Runtime Compiler)**

NVIDIA's runtime CUDA compiler.
Used by CuPy to compile raw CUDA kernels from Python at runtime.

* Compiles CUDA C/C++ source code to GPU binaries.
* No offline compilation required.
* Kernels cached to avoid recompilation.
* Enables dynamic kernel generation and optimization.

---

## **P**

### **Parallel Execution**

The simultaneous execution of operations across many GPU cores.

* Fundamental to GPU computing and CuPy's performance.
* Element-wise operations naturally parallel.
* Thousands of threads execute concurrently.
* CuPy operations automatically leverage GPU parallelism.

### **Pinned Memory (Page-Locked Memory)**

Host memory that is locked in physical RAM and cannot be paged to disk.
Allows faster GPU transfers and asynchronous memory copies.

```python
pinned = cupy.cuda.alloc_pinned_memory(size)
```

* Faster host-device transfer speeds.
* Enables asynchronous transfers.
* Limited resource (consumes physical RAM).
* Use for frequently transferred data.

---

## **R**

### **RawKernel**

A JIT-compiled kernel defined from CUDA C source code, giving full control over GPU programming.

```python
code = '''
extern "C" __global__
void add(const float* a, const float* b, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
'''
kernel = cupy.RawKernel(code, 'add')
kernel((grid_size,), (block_size,), (a, b, out, n))
```

* Used for high-performance or specialized GPU kernels.
* Requires knowledge of CUDA programming.
* Maximum flexibility and control.

### **Reduction Operations**

Operations that combine many values into one (e.g., sum, mean, max, min, std).

* Highly optimized in CuPy using specialized CUDA kernels.
* Return scalar values (triggers synchronization).
* Examples: `cupy.sum()`, `cupy.mean()`, `cupy.max()`, `cupy.min()`

### **ReductionKernel**

Specialized utility for custom reduction operations.

```python
sum_kernel = cupy.ReductionKernel(
    'T x',              # input
    'T y',              # output
    'x',                # map expression
    'a + b',            # reduce expression
    'y = a',            # post map
    '0',                # identity
    'my_sum'            # name
)
```

* Automatically handles parallel reduction tree.
* More efficient than naive approaches.
* For advanced custom reductions.

---

## **S**

### **Scalar Conversion**

Converting GPU scalar values to Python native types.

```python
gpu_value = cupy.array([42.0])
python_value = float(gpu_value)  # Explicit conversion
```

* CuPy scalars live on GPU.
* Implicit conversion triggers synchronization and transfer.
* Explicit conversion with `float()`, `int()` recommended.

### **SciPy Compatibility**

See **cupyx.scipy**.

### **Streams (CUDA Streams)**

Used for scheduling asynchronous GPU operations.
CuPy uses the default stream if none is specified, but custom streams allow overlapping transfers and computation.

* Enable concurrent kernel execution.
* Allow overlapping compute and data transfer.
* Advanced optimization technique.

See **GPU Stream** for details.

### **Synchronization**

See **Device Synchronization**.

---

## **T**

### **Throughput-Oriented Architecture**

GPU design philosophy prioritizing maximum total work completed over minimum latency per operation.

* Achieves high throughput via massive parallelism.
* Contrasts with CPU's latency-oriented design.
* Ideal for data-parallel workloads.
* Explains why GPUs excel at array operations.

### **Thrust Backend**

Some CuPy operations (sorting, scanning, reduction) rely on NVIDIA Thrust.
Thrust is a high-level C++ template library built on CUDA.

* Provides optimized parallel algorithms.
* Used internally by CuPy.
* Users interact through CuPy's Python API.

### **Transfer Cost**

See **Data Transfer Overhead**.

---

## **U**

### **Universal Functions (UFuncs)**

Element-wise functions that operate on arrays, supporting broadcasting.
CuPy implements GPU-accelerated versions of most NumPy ufuncs.

Examples:

* Mathematical: `cupy.sin`, `cupy.exp`, `cupy.sqrt`
* Comparison: `cupy.greater`, `cupy.equal`
* Logical: `cupy.logical_and`, `cupy.logical_or`

* Many are kernel-fused for performance.
* Support all NumPy ufunc features (broadcasting, output arrays, etc.).

---

## **V**

### **Vectorization**

Expressing operations on entire arrays rather than explicit loops.

* Natural programming style in NumPy/CuPy.
* Enables GPU parallelism automatically.
* Example: `c = a + b` instead of `for i: c[i] = a[i] + b[i]`
* Essential for GPU performance.

### **VRAM (Video RAM)**

The GPU's dedicated memory, also called device memory or global memory.

* Separate from system RAM.
* High bandwidth for GPU access.
* Size varies by GPU model (4GB to 80GB+).
* Where all CuPy arrays reside.

---

## **W**

### **Warm-up Call**

A preliminary GPU operation to initialize CUDA context and compile kernels before performance measurement.

```python
# Warm-up
_ = cupy.sum(array)
cupy.cuda.Stream.null.synchronize()

# Now measure performance
```

* Essential for accurate benchmarking.
* First GPU call includes initialization overhead.
* Subsequent calls reuse compiled kernels.

---

## **Z**

### **Zero-Copy**

Techniques to access data without explicit memory copying.

* **DLPack**: Share arrays between frameworks without copying.
* **Pinned memory**: Can enable zero-copy host-device access in some scenarios.
* Minimizes memory overhead and transfer time.
* Important for multi-framework pipelines.

---

## **Cross-Reference**

### **CuPy vs NumPy**

* **NumPy**: CPU-based array library, standard for numerical Python.
* **CuPy**: GPU-accelerated, NumPy-compatible API.
* **Conversion**: `cp.asarray()` (CPU→GPU), `cp.asnumpy()` (GPU→CPU).
* **Performance**: CuPy excels on large arrays; NumPy better for small arrays.

### **CuPy vs Numba**

* **CuPy**: High-level GPU arrays with NumPy-compatible operations.
* **Numba**: JIT compiler for custom kernels (CPU and GPU).
* **Use together**: CuPy for GPU array management; Numba for custom compiled functions.
* **Interoperability**: Both work with NumPy; can be combined in workflows.

### **Recommended Workflow**

1. Start with NumPy on CPU for prototyping and correctness.
2. Identify computational bottlenecks through profiling.
3. For large arrays and standard operations, port to CuPy.
4. Minimize host-device transfers by batching operations on GPU.
5. Use CUDA events for accurate GPU timing.
6. For custom GPU algorithms, consider Numba CUDA or CuPy RawKernel.
7. Use `cupyx.scipy` for scientific computing operations.

---

## **General Summary**

CuPy provides a highly efficient, NumPy-compatible API that offloads array operations to the GPU using CUDA.

Core features:

* **GPU arrays** (`cupy.ndarray`) with NumPy-compatible API
* **Automatic parallelization** of array operations
* **JIT-compiled CUDA kernels** for custom operations
* **Memory pools** and efficient memory management
* **Streams** for asynchronous execution
* **Multi-GPU support** for distributed workloads
* **NumPy and SciPy compatibility** for easy code porting
* **Interoperability** with PyTorch, TensorFlow, JAX via DLPack

Best for: Large-scale array operations, image processing, signal processing, scientific computing, data preprocessing for machine learning.

---
