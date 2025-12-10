# **CuPy Glossary: Key Concepts and Terms**

---

## **A**

### **Array API Compatibility**

CuPy provides a GPU-accelerated array object (`cupy.ndarray`) with an API closely matching NumPy.
Most NumPy code can be converted to CuPy by simply replacing `numpy` with `cupy`, enabling drop-in GPU acceleration.

---

## **C**

### **CUDA**

A parallel computing platform and API model from NVIDIA used by CuPy to execute array operations directly on the GPU.
Operations run as compiled CUDA kernels.

### **cupy.ndarray**

CuPy’s GPU array type, analogous to `numpy.ndarray` but stored in device memory.
Supports:

* vectorized operations
* broadcasting
* slicing
* reductions
* advanced indexing (subset)

Most operations dispatch to GPU kernels under the hood.

### **CUDA Kernel**

A function executed in parallel on the GPU.
CuPy uses both:

* **pre-built kernels** for standard operations
* **JIT-compiled kernels** (RawKernel, ElementwiseKernel, ReductionKernel) for custom GPU functions

---

## **D**

### **Device Memory**

Memory allocated on the GPU (VRAM), distinct from host RAM.
CuPy arrays reside in device memory; moving data between host and device requires explicit transfer (`cupy.asarray`, `cupy.asnumpy`).

### **Device Synchronization**

Most CuPy operations are asynchronous with respect to the CPU.
A synchronization occurs when:

* transferring data back to host
* calling `cupy.cuda.Stream.null.synchronize()`
* invoking certain blocking operations

---

## **E**

### **ElementwiseKernel**

A CuPy utility to JIT-compile custom element-wise GPU kernels using a concise DSL.
Example usage:

```python
square = cupy.ElementwiseKernel(
    'float32 x',
    'float32 y',
    'y = x * x;',
    'square_kernel'
)
```

Generates a CUDA kernel at runtime.

---

## **F**

### **Fusion**

Automatic kernel fusion combines multiple elementwise operations into a single GPU kernel to reduce memory reads/writes and dispatcher overhead.
Used implicitly in some CuPy operations, and explicitly via `@cupy.fuse` decorator.

---

## **G**

### **GPU Stream**

A CUDA stream represents an execution queue on the GPU.
CuPy exposes `cupy.cuda.Stream` for controlling asynchronous kernel launches, overlapping compute and memory transfers.

---

## **J**

### **JIT Compilation (CuPy Kernels)**

CuPy uses NVRTC (NVIDIA Runtime Compilation) to compile CUDA C/C++ source code into a kernel at runtime.
Used by:

* `RawKernel`
* `RawModule`
* `ElementwiseKernel`
* `ReductionKernel`

JIT kernels are cached so that subsequent runs do not recompile.

---

## **M**

### **Memory Pool**

CuPy implements a caching memory allocator that reuses GPU memory blocks to reduce the overhead of repeated `cudaMalloc`/`cudaFree` calls.
Benefits:

* Faster execution
* Lower GPU fragmentation
  Controllable via `cupy.cuda.set_allocator`.

### **Multi-GPU Support**

CuPy supports multiple CUDA devices via:

* `cupy.cuda.Device(device_id)`
* `with cupy.cuda.Device(1): ...`
  Arrays remain bound to the device where they were created.

---

## **N**

### **NumPy Interoperability**

CuPy supports easy conversion:

* Host → GPU: `cupy.asarray(ndarray)`
* GPU → Host: `cupy.asnumpy(cparray)`
  Many NumPy functions have equivalent CuPy implementations.

### **NVRTC**

NVIDIA’s runtime CUDA compiler.
Used by CuPy to compile raw CUDA kernels from Python at runtime.

---

## **P**

### **Pinned Memory**

Page-locked host memory that allows faster GPU transfer speeds and asynchronous memory copies.
Allocatable via:

```python
cupy.cuda.alloc_pinned_memory(size)
```

---

## **R**

### **RawKernel**

A JIT-compiled kernel defined from CUDA C source code, giving full control over GPU programming:

```python
mod = cupy.RawModule(code='''
extern "C" __global__
void add(const float* a, const float* b, float* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    out[i] = a[i] + b[i];
}
''')
add = mod.get_function('add')
```

Used for high-performance or specialized GPU kernels.

### **ReductionKernel**

Specialized utility for reduction operations like sum, max, min, etc.

---

## **S**

### **Streams (CUDA Streams)**

Used for scheduling asynchronous GPU operations.
CuPy uses the default stream if none is specified, but custom streams allow overlapping transfers and computation.

### **Synchronization**

Because kernel launches are asynchronous, explicit synchronization is required when:

* timing GPU operations
* accessing results on the CPU
* coordinating multiple streams

---

## **T**

### **Thrust Backend**

Some CuPy operations (sorting, scanning, reduction) rely on NVIDIA Thrust, a high-level C++ template library built on CUDA.

---

## **U**

### **Universal Functions (UFuncs)**

CuPy implements GPU-accelerated versions of most NumPy ufuncs (e.g., `cupy.sin`, `cupy.exp`).
Many are fused or implemented as custom CUDA kernels for performance.

---

## **Z**

### **Zero-Copy Data Transfer**

Under certain conditions (especially with pinned memory), CuPy supports asynchronous or zero-copy transfers, improving host-device data communication.

---

## **General Summary**

CuPy provides a highly efficient, NumPy-compatible API that offloads array operations to the GPU using CUDA.
Core features include:

* GPU arrays (`cupy.ndarray`)
* JIT-compiled CUDA kernels
* memory pools & pinned memory
* streams and asynchronous execution
* NumPy API compatibility

---
