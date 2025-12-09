# NVIDIA Tesla Architecture

## A Unified Approach to Graphics and Computing

This video explores the unified graphics and computing architecture that enables flexible, high-performance parallel processing, programmable in C via CUDA.

<video width="900" controls>
    <source src="https://logus2k.com/docbro/categories/cuda/videos/nvidia_tesla.mp4" type="video/mp4">
</video>

Launched in 2006, the NVIDIA Tesla architecture is a scalable unified graphics and parallel computing platform designed to enable flexible, programmable graphics and high-performance computing. It unifies the traditional vertex and pixel processors, allowing for dynamic load balancing of varying workloads. The architecture is built on a scalable processor array, where the Streaming Multiprocessor (SM) is the core unified multiprocessor. This design executes both graphics shaders and parallel computing programs, supporting up to 768 concurrent threads in hardware with zero scheduling overhead.

High-performance applications are programmable in C or C++ using the Compute Unified Device Architecture (CUDA) parallel programming model. This computing architecture transparently scales application performance based on the number of SMs and streaming processor cores, meaning a program executes on any size GPU without recompiling.

---
