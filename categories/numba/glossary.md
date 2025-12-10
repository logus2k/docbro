# **Numba Glossary: Key Concepts and Terms**


## **A**

### **Ahead-of-Time (AOT) Compilation**

* Compiles Numba-decorated functions into a **shared library (.so/.pyd)** *before* runtime.
* Implemented via `numba.pycc.CC`.
* Eliminates JIT warm-up cost and is useful for deployment.
* Less flexible than JIT: **signatures must be explicitly provided** and runtime specialization is not possible.
* Does **not** support all Python features; same restrictions as JIT in nopython mode.

### **Auto-Parallelization**

* Numba’s ability to automatically parallelize supported operations (e.g., reductions, array expressions) when `parallel=True` is set.
* Controlled by the underlying LLVM/OpenMP toolchain.



## **B**

### **Blocking Operations**

* Operations that **pause execution** (e.g., I/O, disk access, network calls).
* Numba cannot JIT-compile I/O code; such sections fall back to **object mode** or fail to compile.
* Numba is designed for CPU-bound numeric operations, not for asynchronous or I/O workloads.



## **C**

### **Cache / Cached Compilation**

* Optional caching of compiled machine code to disk via `@njit(cache=True)`.
* Subsequent runs load the cached binary, avoiding recompilation unless source or environment changes.

### **CPU-Bound Tasks**

* Workloads limited by CPU computation (e.g., numerical loops, vector math).
* Numba is highly optimized for these via LLVM and optional parallelization.

### **Cython**

* A separate technology for compiling Python-like syntax to C extensions.
* Requires manual type annotations and explicit C-level control.
* Numba is simpler for numeric kernels but Cython is more flexible for full-module extension development.



## **D**

### **Dynamic Typing**

* Python’s runtime typing model.
* Numba replaces it with **static typing** inferred at compile time.
* Highly dynamic constructs (changing types, polymorphic containers) often **cannot** be compiled.



## **E**

### **Embarrassingly Parallel**

* Workloads where each iteration is independent.
* Ideal for Numba's `prange`, which distributes loop iterations across threads.



## **F**

### **Function Signature**

* A static type specification for a Numba-compiled function (e.g., `"float64(float64, float64)"`).
* Supplying explicit signatures forces immediate compilation and avoids type inference overhead.



## **G**

### **Global Interpreter Lock (GIL)**

* CPython’s lock preventing concurrent execution of Python bytecode.
* Numba **releases the GIL in nopython mode**, enabling true multithreaded execution for numeric kernels.



## **I**

### **Intermediate Representation (IR)**

* Numba lowers Python bytecode to LLVM IR before generating machine code.
* Enables optimizations such as dead-code elimination, loop unrolling, vectorization, etc.



## **J**

### **Just-In-Time (JIT) Compilation**

* Compiles Python functions at runtime on first call or upon explicit request.
* Default mechanism used via `@jit` or `@njit`.
* Creates specialized machine code **per input type signature**.
* First call incurs compilation overhead; subsequent calls run at native speed.



## **L**

### **Loop Unrolling**

* Optimization where short loops are expanded into repeated statements.
* Applied by LLVM when iteration count is known and small, improving instruction-level efficiency.



## **M**

### **Machine Code**

* CPU-native binary instructions executed without interpretation.
* Numba ultimately emits LLVM-generated machine code for supported functions.



## **N**

### **nopython Mode**

* The mode where Numba generates **pure machine code with no Python object operations**.
* Enabled explicitly by `@njit` or `@jit(nopython=True)`.
* Fastest execution mode; Python operations not supported result in compilation failure.

### **NumPy Integration**

* Numba supports a large subset of NumPy, translating array operations directly to optimized native loops.
* Not all NumPy features are supported (e.g., object arrays, advanced indexing semantics).



## **O**

### **Object Mode**

* Fallback mode where Numba executes unsupported operations using Python’s interpreter.
* Greatly slower than nopython mode.
* Usually avoided by using `@njit` (which forbids object mode).

### **OpenMP**

* Backend used by Numba to implement CPU parallelism when `parallel=True`.
* Handles thread scheduling and reductions.



## **P**

### **Parallel Execution**

* Ability to use multiple CPU cores simultaneously.
* Enabled via:

  * `@njit(parallel=True)`
  * Explicit `prange()` for parallel loops
* Requires nopython mode.

### **`prange` (Parallel Range)**

* A parallel version of `range` that distributes iterations across threads.

  ```python
  from numba import njit, prange

  @njit(parallel=True)
  def double(arr):
      for i in prange(len(arr)):
          arr[i] *= 2
  ```

### **Python Object Prohibition**

* In nopython mode, Python objects (lists, dicts, generic classes) cannot be manipulated.
* Only supported types: scalars, arrays, typed lists/dicts, and a restricted subset of Python constructs.



## **S**

### **SIMD (Vectorization)**

* Single Instruction, Multiple Data execution.
* Numba relies on LLVM’s vectorizer to automatically emit SIMD machine instructions for supported loops.

### **Specialization**

* Numba creates distinct compiled versions of a function for each unique input type signature.
* Example: `foo(int64, int64)` and `foo(float64, float64)` compile separately.



## **T**

### **Typed Containers**

* Numba provides typed versions of Python `list` and `dict` that can be used in nopython mode:

  * `numba.typed.List`
  * `numba.typed.Dict`
* All elements must share a known type.



## **U**

### **UFuncs (Universal Functions)**

* Numba can compile element-wise NumPy-style universal functions using `@vectorize`.
* Can target CPU or GPU backends.

  ```python
  from numba import vectorize

  @vectorize(['float64(float64)'])
  def square(x):
      return x * x
  ```



## **V**

### **Vectorization**

* Automatic use of CPU SIMD instructions (AVX, SSE) by LLVM.
* Triggered when Numba determines loop independence and data alignment.



## **C (GPU Section)**

### **CUDA Target / GPU Compilation**

* Numba supports GPU kernels via `numba.cuda`.

* Allows launching custom kernels using CUDA syntax:

  ```python
  from numba import cuda

  @cuda.jit
  def kernel(a, b, out):
      i = cuda.grid(1)
      out[i] = a[i] + b[i]
  ```

* Completely separate from CPU JIT; different restrictions and APIs.

* Numba also supports creating GPU UFuncs and device functions.


