# GPU Programming and CUDA: Fundamentals and Basic Concepts

Graphics Processing Units (GPUs) emerged originally as specialised hardware for accelerating the rendering pipeline in computer graphics. Over the last two decades, however, they have evolved into some of the most powerful general-purpose parallel processors available. With hundreds to thousands of arithmetic units, high memory bandwidth, and a throughput-oriented execution model, GPUs have become indispensable in scientific computing, high-end simulations, signal processing, data analytics, and deep learning.

The central idea that underpins GPU computing is the recognition that many computational tasks exhibit a high degree of _data parallelism_. When an operation can be performed independently on large arrays, vectors, images, or volumes of data, a GPU is capable of executing tens of thousands of threads concurrently, dramatically increasing computational throughput compared to CPUs.

We will study the conceptual foundations of GPU programming from three complementary perspectives:
1. **The architecture of modern GPUs**
2. **The CUDA programming model developed by NVIDIA**
3. **The integration of GPU programming into the Python ecosystem**
    
### 2\. CPU and GPU Architectures: A Conceptual Comparison

To understand GPU programming, one must first appreciate the fundamental architectural differences between CPUs and GPUs.

A modern CPU is engineered for **low-latency, high-flexibility execution**. Its design invests significant transistor area into large caches, aggressive branch predictors, speculative execution, out-of-order pipelines, and mechanisms that prioritise single-thread performance. CPUs are excellent for workloads that are inherently sequential, involve complex control flow, or require tight interaction with the operating system.

GPUs, by contrast, adopt a **throughput-oriented** architecture. Instead of a small number of powerful cores, they integrate hundreds or thousands of lightweight execution units, each capable of performing arithmetic operations with minimal overhead. Rather than prioritising latency or single-thread speed, GPUs are built to perform vast numbers of operations in parallel, provided that the workload contains sufficient regularity to exploit this massively parallel model.

When programming a GPU, groups of **32 threads** are organised in _**Warps**_. Threads in a warp can only execute one instruction at a time. This model is known as **SIMT: Single Instruction, Multiple Threads**. Because many scientific and multimedia operations apply identical transformations across large sets of data (such as pixels in an image or elements in a vector), the SIMT model enables extraordinary computational throughput.

<img src="https://logus2k.com/docbro/categories/cuda/images/warps-sm.png" width=600/>

However, **control flows**, such as `if` statements, can be a problem because they introduce branch divergence. This is where some threads want to execute one branch of the `if` statement, and the other threads want to work on the other branch. When this happens in a warp, each branch is executed serially. This wastes GPU resources because some threads are stalled when either branch is executed.

<img src="https://logus2k.com/docbro/categories/cuda/images/branch_divergence.png" width=600/>

_The left shows the pseudo-code of an `if` statement. The right shows threads in a warp. For illustration purposes, 8 threads in a warp are shown as arrows. Each branch of the `if` statement is executed serially, causing a branch divergence. Threads in one branch execute their commands first before threads in the other branch can._

As a result, pleasingly parallel tasks are more suited to GPUs. In contrast, problems with lots of complex conditionals and boundaries may struggle to make the most of a GPU.

One of the key consequences of this architecture is that branching and divergent control flow should be minimised. If threads within the same warp diverge (for example, due to an `if` statement), the warp must serialise the execution of each branch, significantly reducing performance. Thus, GPU programs perform best when the computation for each thread is as uniform as possible.

### 3\. Memory Hierarchy in GPUs

Like CPUs, GPUs possess a hierarchy of memory types, each with different access levels, speeds, and scopes. However, the GPU memory hierarchy is structured to support large-scale parallel execution rather than general-purpose tasks.

<img src="https://logus2k.com/docbro/categories/cuda/images//memory.jpg" width=600/>

At the top of the hierarchy is **global memory**, typically several gigabytes in size, but with relatively high latency. All threads on the device can access global memory, but the latency of these accesses can be hundreds of clock cycles. Efficient GPU programming therefore requires reducing global memory access whenever possible.

The second level is **shared memory**, which is on-chip and can be considered analogous to a manually managed cache. Shared memory has much lower latency and can be accessed with high throughput. It is local to a _block_ of threads, which encourages programmers to design algorithms in which threads within the same block collaborate by exchanging data via shared memory. Many high-performance GPU kernels (for example, tiled matrix multiplication) rely heavily on shared memory to reduce global memory traffic.

Next come **registers**, the fastest memory available but also the most limited. Registers are private to each thread, making them ideal for storing intermediate values. However, high register usage per thread limits the number of concurrently resident threads on a Streaming Multiprocessor (SM), reducing occupancy and therefore the GPU’s ability to hide latency.

Finally, GPUs also provide specialised memory spaces such as **constant memory**, which is cached and efficient for broadcasting the same data to many threads, and **texture memory**, which includes hardware support for spatially coherent access patterns common in image processing.

Effective GPU programming involves understanding the trade-offs between these memory levels and structuring algorithms to maximise data locality, minimise global memory traffic, and organise threads to access memory in a coalesced manner.

### 4\. The CUDA Programming Model

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform and programming model. CUDA provides a mental model for organising computations on the GPU in terms of _kernels_, _threads_, _thread blocks_, and _grids_.

A **kernel** is a function executed on the GPU. When a kernel is launched, it is not executed once: instead, it is executed by a large number of parallel threads. These threads are organised in a three-level hierarchy:

1. **Threads**: A **thread** is a aingle execution unit that runs GPU function (kernel) on GPU. In other words, a thread is a single instance of the kernel executed on a GPU device.
2. **Thread Blocks:** A **block** contains a fixed number of threads (up to $1024$ in most architectures). Threads within the same block can synchronise with one another and share data via shared memory. Blocks, however, are independent and may execute in any order.
3. **Kernel Grids:**  A **grid** consists of many blocks. A full kernel launch is thus the execution of the kernel by every thread in the grid.

<img src="https://logus2k.com/docbro/categories/cuda/images/cuda_programming_model.png" width=700/>

Reading the image above, each:
1. **CUDA Thread** is executed by a **CUDA Core**.
2. **Thread Block** (which consists on many Threads) is executed by a **Streaming Multiprocessor (SM)** (which consists on many CUDA Cores).
3. **Kernel grid** (which consists on many Thread Blocks) is executed by the **GPU unit** (which consists on many SMs).

#### 4.1 GPU Definitions and their CPU equivalent
The table below attempts to reduce the potential sources of confusion. It lists and defines the terms that apply to the various levels of parallelism in a GPU, and gives their rough equivalents in CPU terminology.

| GPU term | Quick definition for a GPU | CPU equivalent |
| --- | --- | --- |
| **_thread_** | The stream of instructions and data that is assigned to one CUDA core; note, a Single Instruction applies to Multiple Threads, acting on multiple data (SIMT) | N/A |
| **_CUDA core_** | Unit that processes one data item after another, to execute its portion of a SIMT instruction stream | vector lane |
| warp | Group of 32 threads that executes the same stream of instructions together, on different data | vector |
| kernel | Function that runs on the device; a kernel may be subdivided into _thread blocks_ | **_thread(s)_** |
| SM, streaming multiprocessor | Unit capable of executing a thread block of a kernel; multiple SMs may work together on a kernel | **_core_** |

#### 4.2 Thread position and indexing
To identify their position within the computational domain, threads use three pre-defined indices:
- `threadIdx` — the thread’s index within its block
- `blockIdx` — the block’s index within the grid
- `blockDim` — the number of threads in the block
- `gridDim` — the number of blocks in the grid
    
The global thread ID (`index`) is often computed as:
```C
global_id = blockIdx.x * blockDim.x + threadIdx.x
```
The thread marked by **orange color** is part of a grid of threads size $4096$. The threads are grouped in blocks of size $256$. The “orange” thread has index $3$ in the block $2$ and the global calculated index $515$.

<img src="https://logus2k.com/docbro/categories/cuda/images/Indexing.png" width=500/>

CUDA thereby exposes a structured but flexible model for mapping data-parallel computations onto GPU hardware. Each thread typically handles a different data element, and an entire array or image can be transformed efficiently by launching a kernel with enough threads.

#### 4.3 Understanding Thread Indexing in a Simple CUDA Kernel (`vector_add`)

To understand how CUDA threads know which part of the data they should process, it helps to look at one of the simplest GPU kernels: **vector addition**. In this operation, each thread is responsible for computing exactly **one element** of the result vector:

$$
C[i] = A[i] + B[i]
$$

The challenge is that CUDA does **not** number its threads from `0` to `N−1`. Instead, threads are organised hierarchically:

- Threads are grouped into **blocks**
- Blocks are grouped into a **grid**

<img src="https://logus2k.com/docbro/categories/cuda/images/CUDA_thread_hierarchy.png" width=700/>

This means that a thread’s identity is expressed in two parts:
- `blockIdx.x`: which block it belongs to  
- `threadIdx.x`: its index inside that block  

But the data we wish to manipulate — an array — is **flat**, indexed from `0` to `N−1`.  
Thus, every thread needs a single, deterministic **global index** that maps it into this 1D data space.

##### Computing the Global Index

CUDA provides `blockDim.x`, the number of threads per block. With this value, the following formula converts the 2-level thread structure into a single linear index:

$$
\text{global\_id} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

This formula assigns each thread in the entire grid a **unique** index, regardless of which block it belongs to. It effectively “flattens” the grid of blocks into a continuous sequence of thread identifiers:

```text
Block 0 → threads 0 ... blockDim.x−1
Block 1 → threads blockDim.x ... 2×blockDim.x−1
Block 2 → threads 2×blockDim.x ... 3×blockDim.x−1
```
Now each thread can identify the exact data element for which it is responsible.

##### How This Works in the `vector_add` Kernel

Consider the classic CUDA implementation of vector addition:

```cpp
__global__
void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

The computation unfolds as follows:

1. **Step 1 — Compute `idx`**: The thread uses the flattening formula to determine its unique global position in the vector.
2. **Step 2 — Access the correct elements**: Using its global index:
    - Thread `idx` reads `A[idx]` and `B[idx]`
    - Computes their sum
    - Stores it in `C[idx]`
        - No two threads compute the same index, so no synchronisation is required and no race conditions occur.
3. **Step 3 — Parallel execution across the entire array**: If the array has 4096 elements, we simply launch enough blocks and threads so that the total number of threads ≥ 4096. Then every element is processed in parallel, each by a different thread.

### 5\. Performance Considerations in GPU Programming
Effective GPU programming requires awareness of both architectural and algorithmic bottlenecks. Among the most important are:

1. **Warp Divergence**: Because threads within a warp execute in lockstep, conditional branches that lead to different execution paths degrade performance. GPU kernels should minimise divergent control flow within warps.
2. **Memory Coalescing**: When threads in a warp access consecutive memory addresses, the GPU can combine (coalesce) these accesses into a single memory transaction. Non-coalesced accesses may result in many smaller transactions, drastically increasing memory latency.
3. **Occupancy**: Occupancy measures how well the GPU’s computational resources are utilised. High occupancy allows the GPU to better hide memory latency, but excessive register or shared memory usage can reduce occupancy by limiting the number of active warps.
4. **Efficient Use of Shared Memory**: Shared memory is a powerful optimisation tool when many threads operate on overlapping data. By loading data once into shared memory, threads can reuse it many times without repeatedly accessing global memory.
5. **Computation vs. Memory Bound Workloads**: Some tasks are limited by compute throughput (e.g., dense matrix multiplication), while others are primarily constrained by memory bandwidth (e.g., vector addition). Understanding where a kernel lies on this spectrum guides the optimisation strategies.

### 6\. Python and GPU Programming

Although CUDA was designed for C and C++, the Python ecosystem has developed extensive support for GPU computing. These tools abstract the complexities of the CUDA runtime while retaining high performance. Three major approaches stand out:

<img src="https://logus2k.com/docbro/categories/cuda/images/pygpuchart.png" width=600/>

#### CuPy
CuPy provides a **NumPy-compatible** interface such that:
- Arrays are allocated in GPU memory,
- Operations such as addition, multiplication, and matrix multiplication are executed on the GPU,
- Many SciPy-like functions have GPU implementations.
    
CuPy is ideal when the workload can be expressed in vectorised, NumPy-like operations.

#### Numba
Numba provides a **CUDA backend** that allows writing **custom GPU kernels** in Python. With Numba:
- Python functions can be JIT-compiled into CUDA kernels,
- Thread/block indexing is directly exposed,
- Shared memory, synchronisation, and other CUDA features are available.
    
Numba strikes a middle ground between ease of use and low-level control, making it highly suitable for educational purposes and custom GPU algorithm development.

#### PyCUDA
PyCUDA offers a thin wrapper around the CUDA C APIs. Kernels are written in CUDA C (not Python) and compiled at runtime. This approach provides maximal control but is less beginner-friendly. We will not explore this library.
