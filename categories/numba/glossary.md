# Numba Glossary

## **A**

### **Ahead-of-Time (AOT) Compilation**

* Compiles Numba-decorated functions into a **shared library (.so/.pyd)** *before* runtime.
* Implemented via `numba.pycc.CC`.
* Eliminates JIT warm-up cost and is useful for deployment.
* Less flexible than JIT: **signatures must be explicitly provided** and runtime specialization is not possible.
* Does **not** support all Python features; same restrictions as JIT in nopython mode.

### **Auto-Parallelization**

* Numba's ability to automatically parallelize supported operations (e.g., reductions, array expressions) when `parallel=True` is set.
* Controlled by the underlying LLVM/OpenMP toolchain.
* Works best on **embarrassingly parallel** workloads where iterations are independent.

---

## **B**

### **Blocking Operations**

* Operations that **pause execution** (e.g., I/O, disk access, network calls).
* Numba cannot JIT-compile I/O code; such sections fall back to **object mode** or fail to compile.
* Numba is designed for CPU-bound numeric operations, not for asynchronous or I/O workloads.

### **Broadcasting**

* NumPy's mechanism for performing operations on arrays of different shapes.
* Numba supports broadcasting in nopython mode for most standard array operations.
* Follows the same rules as NumPy: dimensions are aligned from right to left, and dimensions of size 1 are stretched.

---

## **C**

### **Cache / Cached Compilation**

* Optional caching of compiled machine code to disk via `@njit(cache=True)`.
* Subsequent runs load the cached binary, avoiding recompilation unless source or environment changes.
* Particularly useful in production environments to eliminate first-call overhead.

### **Compilation Overhead**

* The time taken to analyze, optimize, and generate machine code on the first call to a JIT-decorated function.
* Can be significant for complex functions but is amortized across subsequent calls.
* Mitigated by **warm-up calls** in benchmarking or **caching** in production.

### **CPU-Bound Tasks**

* Workloads limited by CPU computation (e.g., numerical loops, vector math).
* Numba is highly optimized for these via LLVM and optional parallelization.
* Contrast with memory-bound tasks (limited by data transfer) and I/O-bound tasks.

### **CUDA**

* NVIDIA's parallel computing platform and programming model for GPU computing.
* Numba provides a CUDA backend via `numba.cuda` for writing GPU kernels in Python.
* See **CUDA Target / GPU Compilation** for details.

### **Cython**

* A separate technology for compiling Python-like syntax to C extensions.
* Requires manual type annotations and explicit C-level control.
* Numba is simpler for numeric kernels but Cython is more flexible for full-module extension development.

---

## **D**

### **Data-Parallel Computation**

* Computational pattern where the same operation is applied independently to many data elements.
* Ideal for both CPU parallelization (`prange`) and GPU execution (`cuda.jit`).
* Examples: element-wise array operations, image filtering, Monte Carlo simulations.

### **Device**

* In GPU computing, refers to the **GPU** as opposed to the **host (CPU)**.
* Device memory is separate from host memory and requires explicit data transfer.
* See **Host vs Device** for details.

### **Dynamic Typing**

* Python's runtime typing model where variable types can change during execution.
* Numba replaces it with **static typing** inferred at compile time.
* Highly dynamic constructs (changing types, polymorphic containers) often **cannot** be compiled in nopython mode.

---

## **E**

### **Embarrassingly Parallel**

* Workloads where each iteration is completely independent with no communication or synchronization required.
* Ideal for Numba's `prange`, which distributes loop iterations across threads.
* Examples: element-wise transformations, Monte Carlo simulations, independent parameter sweeps.

---

## **F**

### **`fastmath`**

* A Numba option (`@njit(fastmath=True)`) that enables aggressive floating-point optimizations.
* May violate strict IEEE 754 compliance (e.g., reordering operations, ignoring NaN/Inf edge cases).
* Can significantly improve performance for numerical kernels where strict precision is not critical.
* Trade-off: performance vs. numerical reproducibility and standards compliance.

### **Function Signature**

* A static type specification for a Numba-compiled function (e.g., `"float64(float64, float64)"`).
* Supplying explicit signatures forces immediate compilation and avoids type inference overhead.
* Used in `@vectorize` and `@jit` decorators to pre-specify input/output types.

---

## **G**

### **Global Interpreter Lock (GIL)**

* CPython's lock preventing concurrent execution of Python bytecode across multiple threads.
* Numba **releases the GIL in nopython mode**, enabling true multithreaded execution for numeric kernels.
* This is a key advantage over standard Python multithreading for CPU-bound tasks.

### **GPU (Graphics Processing Unit)**

* Specialized hardware designed for massively parallel computation.
* Originally for graphics rendering, now widely used for general-purpose computing (GPGPU).
* Numba supports GPU programming via `numba.cuda` for NVIDIA GPUs.
* See **CUDA Target / GPU Compilation** for implementation details.

---

## **H**

### **Host vs Device**

* **Host**: The CPU and its main memory (RAM).
* **Device**: The GPU and its dedicated memory (VRAM).
* Data must be explicitly transferred between host and device for GPU computations.
* Numba CUDA provides utilities for managing these transfers (`cuda.to_device`, `cuda.copy_to_host`).

---

## **I**

### **Intermediate Representation (IR)**

* Numba lowers Python bytecode to LLVM IR before generating machine code.
* Enables optimizations such as dead-code elimination, loop unrolling, vectorization, constant folding, etc.
* LLVM IR is a platform-independent representation that can be optimized and then compiled to native code.

---

## **J**

### **Just-In-Time (JIT) Compilation**

* Compiles Python functions at runtime on first call or upon explicit request.
* Default mechanism used via `@jit` or `@njit`.
* Creates specialized machine code **per input type signature**.
* First call incurs compilation overhead; subsequent calls run at native speed.
* Contrast with **Ahead-of-Time (AOT)** compilation.

---

## **K**

### **Kernel**

* In GPU computing, a function executed in parallel by many threads on the GPU.
* Written with `@cuda.jit` decorator in Numba.
* Each thread executes the same kernel code but operates on different data.
* See **Numeric Kernel** for CPU context.

---

## **L**

### **LLVM**

* Low Level Virtual Machine - a compiler infrastructure used by Numba.
* Provides the backend for generating optimized machine code from IR.
* Handles platform-specific optimizations, vectorization, and code generation.
* Via `llvmlite`, Numba interfaces with LLVM without requiring full LLVM installation.

### **`llvmlite`**

* A lightweight Python wrapper around LLVM's core functionality.
* Used by Numba to generate and compile LLVM IR without the complexity of the full LLVM API.
* Provides just enough functionality for JIT compilation scenarios.

### **Loop Unrolling**

* Optimization where short loops are expanded into repeated statements.
* Applied by LLVM when iteration count is known and small, improving instruction-level efficiency.
* Reduces loop overhead (branching, counter increments) at the cost of code size.

---

## **M**

### **Machine Code**

* CPU-native binary instructions executed directly by the processor without interpretation.
* Numba ultimately emits LLVM-generated machine code for supported functions.
* Provides performance comparable to C/C++ for numerical operations.

### **Memory Coalescing**

* GPU optimization where threads in a warp access consecutive memory addresses.
* Results in combined (coalesced) memory transactions, drastically improving throughput.
* Non-coalesced access patterns cause multiple separate transactions and increase latency.
* Critical for efficient GPU kernel performance.

---

## **N**

### **nopython Mode**

* The mode where Numba generates **pure machine code with no Python object operations**.
* Enabled explicitly by `@njit` or `@jit(nopython=True)`.
* Fastest execution mode; Python operations not supported result in compilation failure.
* Preferred mode for maximum performance.

### **Numeric Kernel**

A numeric kernel is the core computation-heavy part of a program, typically expressed as tight loops or vector operations over numerical data. It is the portion of code that benefits most from JIT compilation and parallelization.

Numeric kernels have these characteristics:

* **Arithmetic-focused:** Perform large amounts of numeric work (e.g., additions, multiplications, dot products, iterative updates).
* **Statically typable:** Use numeric scalars and arrays without relying on Python objects or dynamic typing.
* **No I/O or Python runtime features:** Avoids file I/O, printing, exceptions, dynamic attributes, and other interpreter-dependent behaviors.
* **Parallelizable or vectorizable:** Iterations are usually independent, making them ideal for thread parallelism (`prange`) or SIMD vectorization.
* **Performance-critical:** Represent the "hot loops" of the algorithm where most execution time is spent.

Numba is designed to optimize numeric kernels by compiling them into efficient machine code, removing Python overhead, enabling SIMD execution, and (when configured) distributing work across multiple CPU cores.

### **NumPy Integration**

* Numba supports a large subset of NumPy, translating array operations directly to optimized native loops.
* Includes: basic indexing, slicing, broadcasting, reductions, element-wise operations.
* Not all NumPy features are supported (e.g., object arrays, some advanced indexing semantics, certain functions).
* Performance often matches or exceeds NumPy for operations compiled in nopython mode.

---

## **O**

### **Object Mode**

* Fallback mode where Numba executes unsupported operations using Python's interpreter.
* Greatly slower than nopython mode due to interpreter overhead.
* Usually avoided by using `@njit` (which forbids object mode and raises compilation errors instead).
* Can be useful for debugging but should not be used in performance-critical code.

### **Occupancy**

* In GPU programming, the ratio of active warps to the maximum possible warps on a Streaming Multiprocessor (SM).
* High occupancy allows the GPU to hide memory latency by switching between warps.
* Reduced by excessive register usage or shared memory consumption per thread.
* Important metric for GPU kernel performance optimization.

### **OpenMP**

* Backend used by Numba to implement CPU parallelism when `parallel=True`.
* Handles thread scheduling, work distribution, and parallel reductions.
* Industry-standard API for shared-memory multiprocessing.

---

## **P**

### **Parallel Execution**

* Ability to use multiple CPU cores simultaneously for a single computation.
* Enabled via:
  * `@njit(parallel=True)` for automatic parallelization
  * Explicit `prange()` for parallel loops
* Requires nopython mode.
* Releases the GIL, allowing true concurrent execution.

### **`prange` (Parallel Range)**

* A parallel version of `range` that distributes iterations across multiple CPU threads.
* Must be used with `parallel=True` in the decorator.
* Best for embarrassingly parallel loops where iterations are independent.

  ```python
  from numba import njit, prange

  @njit(parallel=True)
  def parallel_sum_of_squares(x):
      total = 0.0
      for i in prange(len(x)):
          total += x[i] * x[i]
      return total
  ```

### **Python Object Prohibition**

* In nopython mode, Python objects (lists, dicts, generic classes, strings in many contexts) cannot be manipulated.
* Only supported types: numeric scalars, NumPy arrays, typed lists/dicts, tuples, and a restricted subset of Python constructs.
* This restriction enables the performance gains of nopython mode.

---

## **R**

### **Reduction Operations**

* Operations that combine many values into one (e.g., sum, mean, max, min).
* Numba can automatically parallelize reductions when `parallel=True` is enabled.
* Requires careful handling in parallel contexts to avoid race conditions.

---

## **S**

### **SIMD (Single Instruction, Multiple Data)**

* Vectorization technique where one CPU instruction operates on multiple data elements simultaneously.
* Modern CPUs provide SIMD instruction sets (SSE, AVX, AVX-512).
* Numba relies on LLVM's auto-vectorizer to emit SIMD instructions for supported loops.
* Most effective on contiguous array operations with simple, uniform computations.

### **SIMT (Single Instruction, Multiple Threads)**

* GPU execution model where groups of 32 threads (a **warp**) execute the same instruction simultaneously.
* Different from SIMD: threads can diverge (though with performance penalty).
* Fundamental to CUDA programming and GPU architecture.

### **Specialization**

* Numba creates distinct compiled versions of a function for each unique input type signature.
* Example: `foo(int64, int64)` and `foo(float64, float64)` compile separately.
* Enables type-specific optimizations but increases memory usage if many signatures are used.

### **Streaming Multiprocessor (SM)**

* Basic processing unit on NVIDIA GPUs containing multiple CUDA cores.
* Executes thread blocks independently.
* Modern GPUs have dozens to over 100 SMs depending on model.
* See GPU architecture documentation for detailed hierarchy.

---

## **T**

### **Target**

* In Numba's `@vectorize` decorator, specifies the execution platform:
  * `target='cpu'`: Single-threaded CPU execution
  * `target='parallel'`: Multi-threaded CPU execution
  * `target='cuda'`: GPU execution
* Allows the same Python function to target different hardware with minimal code changes.

### **Thread Block**

* In CUDA programming, a group of threads that execute a kernel together.
* Threads within a block can synchronize and share memory.
* Maximum size typically 1024 threads per block.
* See **CUDA Programming Model** for full hierarchy.

### **Typed Containers**

* Numba provides typed versions of Python `list` and `dict` that can be used in nopython mode:
  * `numba.typed.List`
  * `numba.typed.Dict`
* All elements must share a known, consistent type.
* Enable use of dynamic-size collections in compiled code.

### **Type Inference**

* Numba's ability to automatically determine the types of variables and expressions from context.
* Happens at compile time based on input types and operations.
* Enables writing code without explicit type annotations while still generating statically-typed machine code.

---

## **U**

### **UFuncs (Universal Functions)**

* NumPy-style element-wise functions that operate on arrays.
* Numba can compile custom UFuncs using `@vectorize`.
* Can target CPU or GPU backends with appropriate `target` parameter.
* Automatically handle broadcasting and output array creation.

  ```python
  from numba import vectorize

  @vectorize(['float64(float64)'], target='cpu')
  def square(x):
      return x * x
  ```

---

## **V**

### **Vectorization**

* Automatic use of CPU SIMD instructions (AVX, SSE) by LLVM.
* Triggered when Numba determines loop independence and favorable data access patterns.
* Significantly improves performance for array operations.
* Also refers to the `@vectorize` decorator for creating UFuncs.

---

## **W**

### **Warm-up Call**

* A preliminary function call to trigger JIT compilation before performance measurement.
* Essential for accurate benchmarking, as first call includes compilation overhead.
* Pattern: `_ = my_function(data)` followed by the timed call.

### **Warp**

* In NVIDIA GPUs, a group of 32 threads that execute in lockstep (SIMT).
* Basic scheduling unit; all threads in a warp execute the same instruction simultaneously.
* **Warp divergence** occurs when threads take different branches, forcing serialized execution of each path.

### **Warp Divergence**

* Performance penalty when threads in the same warp take different execution paths (e.g., different branches of an `if` statement).
* Forces the warp to serialize execution of each branch, reducing parallelism.
* Minimize by ensuring uniform control flow within warps or restructuring algorithms to avoid divergence.

---

## **GPU / CUDA Section**

### **CUDA Target / GPU Compilation**

* Numba supports GPU kernels via `numba.cuda` for NVIDIA CUDA-capable GPUs.
* Allows launching custom kernels using CUDA-like syntax in Python:

  ```python
  from numba import cuda

  @cuda.jit
  def vector_add_kernel(a, b, out):
      idx = cuda.grid(1)
      if idx < out.size:
          out[idx] = a[idx] + b[idx]
  ```

* Completely separate from CPU JIT; different restrictions and APIs.
* Numba also supports creating GPU UFuncs via `@vectorize` with `target='cuda'`.

### **`cuda.grid()`**

* Helper function to compute a thread's global index in the grid.
* `idx = cuda.grid(1)` computes 1D global index: `blockIdx.x * blockDim.x + threadIdx.x`
* Essential for mapping threads to data elements in GPU kernels.

### **Global Memory (GPU)**

* Main GPU memory (VRAM), accessible by all threads.
* Largest GPU memory space (gigabytes) but highest latency (hundreds of cycles).
* All data must start in global memory before being accessed by kernels.
* Efficient access requires **memory coalescing**.

### **Grid**

* In CUDA, the complete collection of thread blocks launched for a kernel.
* Organized as 1D, 2D, or 3D structure depending on problem dimensionality.
* Total threads = blocks in grid × threads per block.

### **Shared Memory (GPU)**

* Fast, on-chip memory local to a thread block.
* Much lower latency than global memory (~10x faster).
* Limited size (typically 48-96 KB per SM).
* Used for thread collaboration and reducing global memory traffic.
* Must be explicitly allocated and managed in custom kernels.

### **Thread Indexing (CUDA)**

* Built-in variables for identifying thread position:
  * `threadIdx`: Thread's index within its block (x, y, z)
  * `blockIdx`: Block's index within the grid (x, y, z)
  * `blockDim`: Number of threads per block (x, y, z)
  * `gridDim`: Number of blocks in the grid (x, y, z)
* Used to compute global index and map threads to data.

---

## **Cross-Reference**

### **Numba vs CuPy**

* **Numba**: JIT compiler for writing custom kernels (CPU and GPU).
* **CuPy**: NumPy-compatible library with GPU arrays and high-level operations.
* **Use together**: CuPy for GPU array management and high-level ops; Numba for custom compiled kernels.
* **Interoperability**: Numba CPU functions can process data before/after CuPy GPU operations.

### **Recommended Workflow**

1. Start with NumPy on CPU for prototyping and correctness.
2. Apply `@njit` to CPU bottlenecks for immediate speedup.
3. Use `parallel=True` and `prange` for multi-core CPU parallelism.
4. For very large data, consider CuPy for GPU acceleration of standard operations.
5. For custom GPU algorithms, write Numba CUDA kernels.

---
