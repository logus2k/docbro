# CUDA Programming Glossary

This **CUDA Programming Glossary** serves as a reference guide to the essential terminology, concepts, and APIs used in CUDA, NVIDIA’s parallel computing platform and programming model. Whether you are new to GPU-accelerated computing or an experienced developer, this glossary provides definitions for key terms such as kernels, threads, blocks, grids, memory hierarchies, and streaming multiprocessors.

## Core Execution Model

- **Kernel**: A function written in CUDA C/C++ that runs on the GPU
- **Thread**: The smallest execution unit that runs kernel code
- **Warp**: A group of 32 threads that execute in lockstep (NVIDIA GPUs)
- **Block**: A group of threads that can share data through shared memory and synchronize
- **Grid**: The collection of all blocks for a single kernel launch

## Memory Hierarchy

- **Global Memory**: Large, high-latency memory accessible by all threads (GPU DRAM)
- **Shared Memory**: Fast, on-chip memory shared by threads within a block
- **Local Memory**: Per-thread memory, typically stored in global memory
- **Constant Memory**: Read-only memory with cache for uniform access patterns
- **Texture Memory**: Cached memory optimized for spatial locality access patterns
- **Registers**: Fastest memory, private to each thread

## Memory Management

- **Unified Memory**: Automatic memory management between CPU and GPU
- **Memory Coalescing**: Efficient memory access pattern where threads access contiguous memory
- **Page-locked Memory**: Host memory that can be directly accessed by GPU (pinned memory)

## Performance Concepts

- **Occupancy**: Ratio of active warps to maximum possible active warps on an SM
- **Warp Divergence**: When threads in the same warp take different execution paths
- **Latency Hiding**: Keeping GPU busy with other threads while waiting for memory operations
- **Throughput**: Amount of work completed per unit time

## Hardware Concepts

- **Streaming Multiprocessor (SM)**: Processing unit containing multiple CUDA cores
- **CUDA Cores**: Individual processing units within an SM
- **Occupancy**: How well the GPU is utilized with active threads

## Synchronization

- **__syncthreads()**: Function to synchronize all threads within a block
- **Barriers**: Points where threads wait for each other to reach the same execution point

## Streams and Events

- **Stream**: Sequence of operations that execute in order on GPU
- **Event**: Marker in a stream for timing or synchronization
- **Concurrent Kernels**: Multiple kernels running simultaneously on different SMs

## Advanced Features

- **Cooperative Groups**: APIs for thread collaboration beyond block boundaries
- **Dynamic Parallelism**: Ability for kernels to launch other kernels
- **Multi-GPU Programming**: Using multiple GPUs for computation
