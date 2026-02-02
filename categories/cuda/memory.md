# CUDA Memory Types

CUDA provides several distinct memory types. Each memory type has different latency, bandwidth, and capacity characteristics, making them optimized and suitable for specific access patterns and use cases:

**1. Global Memory:**
- Largest capacity (GBs), slowest access
- Accessible by all threads and host CPU
- Cached on newer GPUs (L1/L2 cache)
- Used for main data storage

**2. Shared Memory:**
- Small (typically 48KB-164KB per block)
- Fastest on-chip memory after registers
- Shared among threads in the same block only
- Manually managed by programmer
- Used for data sharing and cooperation between threads

**3. Registers:**
- Fastest memory type
- Private to each thread
- Limited quantity (typically 64K 32-bit registers per block)
- Automatically managed by compiler
- Holds local variables and intermediate calculations

**4. Local Memory:**
- Actually global memory, but appears "local" to threads
- Used when registers overflow or for large arrays
- Slower than registers, cached on newer GPUs

**5. Constant Memory:**
- Small (64KB), cached memory
- Read-only from device side
- Optimized for broadcast access (all threads reading same location)
- Cached to improve performance for read-only data

**6. Texture Memory:**
- Cached read-only memory
- Optimized for spatial locality and 2D access patterns
- Provides interpolation capabilities
- Useful for image processing applications

**7. Unified Memory (Unified Addressing):**
- Single memory space accessible from both CPU and GPU
- Managed automatically by system
- Simplifies programming but may have performance implications

---
