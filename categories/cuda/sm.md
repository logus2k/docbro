# Streaming Multiprocessor

SMs are fundamental to CUDA architecture by enabling GPUs to achieve high parallel throughput for data-intensive operations like neural network training and inference.

The SM is a unified graphics and computing multiprocessor that executes vertex, geometry, pixel-fragment shader and parallel computing programs.

![CUDA Compute Unified Architecture](https://logus2k.com/docbro/categories/cuda/images/cuda_architecture.png)

_CUDA compute unified device architecture (G80). Note the absence of distinct processor types - all meaningful computation occurs in the identical Streaming Multiprocessors in the center of the diagram, fed with instructions for vertex, geometry, and pixel threads._

---

**Streaming Multiprocessor**

- The processing unit of a GPU that executes CUDA threads
- Contains multiple CUDA cores, shared memory, and control units
- Each SM can execute thousands of threads simultaneously through time-slicing

**SM components**

- Multiple CUDA cores (the actual arithmetic units)
- Shared memory (fast memory shared by threads in the same block)
- Control units and scheduling hardware
- Registers and cache memory

**Thread execution**

- SMs execute warps (groups of 32 threads) from different blocks
- Multiple blocks can run on the same SM simultaneously
- SMs use time-slicing to hide memory latency (switching between warps while others wait for memory)

**For neural networks**

- The number of SMs determines how many blocks can run in parallel
- Occupancy (how well SMs are utilized) affects performance
- Understanding SM limitations helps optimize block sizes and memory usage

**Key metrics**

- Each GPU has a specific number of SMs (e.g., 80 on some RTX cards)
- Each SM has limits on threads, blocks, and shared memory usage
- Optimizing for SM utilization is crucial for achieving peak performance

---
