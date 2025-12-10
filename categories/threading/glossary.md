# **Threading and Multiprocessing Glossary: Key Concepts and Terms**

---

## **C**

### **Concurrency**

The ability to **deal with many things at once** by overlapping tasks in time.
Multiple tasks make progress during overlapping time periods, but may not execute simultaneously.

* Achieved through **time slicing** on single-core CPUs.
* Does not require multiple physical cores.
* Example: A single-core CPU running multiple applications by rapidly switching between them.
* Contrast with **Parallelism**.

### **Context Switching**

The process of saving and restoring the state of a thread or process so that execution can be resumed later.

* Overhead cost when switching between threads/processes.
* Managed by the operating system scheduler.
* Frequent switching can reduce performance due to cache misses and scheduler overhead.
* More expensive for processes than threads (due to separate address spaces).

### **CPU-Bound Task**

A workload where execution time is primarily limited by CPU computation rather than I/O operations.

* Examples: mathematical calculations, prime number testing, image processing, sorting algorithms.
* **With GIL**: Multi-threading provides no speedup for CPU-bound Python code.
* **Without GIL**: Multi-threading can provide speedup if work is properly parallelized.
* Best parallelized using **multiprocessing** (standard Python) or **free-threaded mode** (Python 3.13+).

---

## **D**

### **Data Parallelism**

Parallelization strategy where the **same operation** is applied to **different portions of data** simultaneously.

* Each thread/process works on a distinct subset of the data.
* Example: Thread 1 processes pixels 0-1000, Thread 2 processes pixels 1001-2000.
* Ideal for array operations, image processing, large dataset transformations.
* Contrast with **Task Parallelism**.

### **Deadlock**

A situation where two or more threads/processes are permanently blocked, each waiting for a resource held by another.

* Classic example: Thread A holds Lock 1 and waits for Lock 2; Thread B holds Lock 2 and waits for Lock 1.
* Prevention strategies:
  * Lock ordering (always acquire locks in the same order)
  * Timeouts on lock acquisition
  * Avoiding nested locks when possible
* More common in multi-threaded programs with complex synchronization.

---

## **F**

### **Free-Threaded Mode**

Experimental mode in Python 3.13+ where the **GIL is disabled**, allowing true parallel execution of Python bytecode.

* Enabled by building Python with `--disable-gil` flag.
* Separate executable: `python3.13t` or `python3.14t`.
* Benefits:
  * True multi-threaded parallelism for CPU-bound tasks
  * Multiple threads can execute Python code simultaneously
* Trade-offs:
  * Single-threaded performance may be slower
  * Some C extensions not yet compatible
  * Requires more careful thread synchronization

---

## **G**

### **GIL (Global Interpreter Lock)**

A mutual exclusion lock in CPython that allows only **one thread to execute Python bytecode at a time** within a process.

* Simplifies memory management (reference counting) and C extension integration.
* **Consequences:**
  * CPU-bound multi-threaded Python code does not achieve parallel speedup
  * Only one thread executes at a time, even on multi-core systems
  * I/O-bound code can still benefit (GIL released during I/O)
* **Workarounds:**
  * Use `multiprocessing` (separate processes have separate GILs)
  * Use free-threaded Python 3.13+ (`--disable-gil`)
  * Offload work to C extensions, NumPy, or GPU libraries (these release GIL)

---

## **I**

### **I/O-Bound Task**

A workload where execution time is primarily limited by waiting for input/output operations.

* Examples: network requests, disk reads/writes, database queries, file downloads.
* **Multi-threading effective**: Threads release GIL during I/O operations.
* While one thread waits for I/O, other threads can execute.
* **Multiprocessing unnecessary**: Overhead not justified since threads work well.

### **Inter-Process Communication (IPC)**

Mechanisms for processes to exchange data and synchronize since they have separate address spaces.

* Methods:
  * **Pipes**: One-way or two-way data channels
  * **Queues**: Thread-safe, process-safe message passing (`multiprocessing.Queue`)
  * **Shared Memory**: Direct memory sharing (`multiprocessing.Value`, `Array`, `Manager`)
  * **Sockets**: Network-based communication
* Required because processes cannot directly access each other's memory.
* More overhead than thread communication (which uses shared memory natively).

---

## **J**

### **`join()` Method**

Blocks the calling thread until the target thread/process completes execution.

```python
thread.start()
thread.join()  # Wait until thread finishes
print("Thread has completed")
```

* Essential for coordinating execution order.
* Ensures parent thread waits for child threads to complete.
* Without `join()`, parent may exit while children still running (can cause incomplete work or resource leaks).

---

## **L**

### **Lock**

A synchronization primitive that ensures only one thread can access a shared resource at a time.

```python
import threading

lock = threading.Lock()

with lock:  # Acquire lock
    # Critical section - only one thread executes this at a time
    shared_variable += 1
# Lock automatically released
```

* Prevents **race conditions** when multiple threads access shared data.
* Types:
  * **Lock**: Basic mutual exclusion
  * **RLock** (Reentrant Lock): Can be acquired multiple times by same thread
  * **Semaphore**: Allows N threads to access resource
* Must be used carefully to avoid **deadlocks**.

---

## **M**

### **Multiprocessing**

Python's approach to parallelism using **separate processes**, each with its own Python interpreter and memory space.

* Module: `multiprocessing`
* Each process has its own **separate GIL**.
* Enables true parallel execution of CPU-bound Python code.
* **Advantages:**
  * Bypasses GIL limitations
  * True parallelism on multi-core systems
  * Process isolation (crashes don't affect others)
* **Disadvantages:**
  * Higher memory overhead (separate interpreters)
  * Slower startup time
  * Communication requires serialization (pickling)
  * More complex shared state management

### **`multiprocessing.Pool`**

A pool of worker processes for distributing tasks across multiple CPU cores.

```python
from multiprocessing import Pool

def worker(x):
    return x * x

with Pool(processes=4) as pool:
    results = pool.map(worker, [1, 2, 3, 4, 5])
```

* **Key methods:**
  * `map(func, iterable)`: Apply function to each item, return ordered results
  * `apply(func, args)`: Execute single function call in pool
  * `apply_async()`: Non-blocking version
  * `starmap()`: Like map but unpacks argument tuples
* Automatically manages process lifecycle.
* Use `with` statement for automatic cleanup.

### **`multiprocessing.Process`**

Low-level primitive representing a single process.

```python
from multiprocessing import Process

def worker(name):
    print(f"Process {name}")

p = Process(target=worker, args=("Worker-1",))
p.start()
p.join()
```

* Similar API to `threading.Thread`.
* More control than `Pool` but requires manual management.
* Each process completely independent.

---

## **P**

### **Parallelism**

The ability to **do many things at the same time** by executing multiple tasks simultaneously on different processing units.

* Requires multiple physical cores or processors.
* Example: Four CPU cores each executing different threads simultaneously.
* **True parallelism** vs **concurrency**: Parallelism requires actual simultaneous execution.
* In Python:
  * **With GIL**: Limited to I/O operations or C extensions for threads
  * **Multiprocessing**: Achieves true parallelism for CPU-bound work
  * **Free-threaded mode**: Enables parallel threading for CPU-bound work

### **Pickle / Pickling**

Python's serialization mechanism for converting objects to byte streams.

* Required for `multiprocessing` to send data between processes.
* Limitations:
  * Not all objects can be pickled (e.g., lambda functions, local classes)
  * Overhead for large data structures
  * Can be a bottleneck in multiprocessing workflows
* Workarounds:
  * Use shared memory (`multiprocessing.Value`, `Array`)
  * Design workers to load data independently
  * Use `multiprocessing.Manager` for complex shared objects

### **Process**

An independent execution unit with its own **address space** (separate memory).

* Contains one or more threads.
* Managed by the operating system.
* **Advantages:**
  * Complete isolation (crashes don't propagate)
  * No shared memory race conditions (by default)
  * Each has separate GIL in Python
* **Disadvantages:**
  * Higher creation and memory overhead
  * Communication requires IPC mechanisms
  * Context switching more expensive than threads

---

## **R**

### **Race Condition**

A bug that occurs when multiple threads access shared data concurrently, and the outcome depends on the timing of their execution.

```python
# Race condition example
counter = 0

def increment():
    global counter
    temp = counter
    temp += 1
    counter = temp

# Multiple threads calling increment() can produce incorrect results
```

* Results are **non-deterministic** (vary between runs).
* Prevented by proper **synchronization** (locks, atomic operations).
* Common in multi-threaded programs with shared state.
* Testing difficult because race conditions may not appear consistently.

---

## **S**

### **Shared Memory**

Memory that can be accessed by multiple threads or processes.

**Threads:**
* Naturally share memory (same address space).
* Direct access to all process variables.
* Requires locks to prevent race conditions.

**Processes:**
* Separate memory by default.
* Shared memory requires explicit setup:
  ```python
  from multiprocessing import Value, Array
  
  shared_value = Value('i', 0)  # Shared integer
  shared_array = Array('d', [1.0, 2.0, 3.0])  # Shared array
  ```

### **Speedup**

The ratio of execution time for single-threaded/process vs multi-threaded/process execution.

$$\text{Speedup} = \frac{T_{\text{single}}}{T_{\text{multi}}}$$

* **Ideal speedup**: Linear with number of cores (2x with 2 cores, 4x with 4 cores).
* **Actual speedup**: Usually less due to:
  * Overhead (thread/process creation, synchronization)
  * GIL limitations (CPU-bound threading)
  * Non-parallelizable portions (Amdahl's Law)
  * Communication costs between threads/processes
* **Negative speedup** (slowdown): Multi-threaded slower than single-threaded due to overhead.

---

## **T**

### **Task Duplication**

Anti-pattern where multiple threads/processes perform the **same work on the same data**.

```python
# WRONG: Both threads do the entire sum
def worker(data):
    return sum(data)

t1 = Thread(target=worker, args=(data,))
t2 = Thread(target=worker, args=(data,))  # Duplicate work!
```

* No performance improvement (work is duplicated, not parallelized).
* Contrast with **Data Parallelism** (work is divided).
* Common beginner mistake when first learning threading/multiprocessing.

### **Task Granularity**

The size or computational cost of individual parallel tasks.

* **Fine-grained**: Small, frequent tasks (high overhead potential).
* **Coarse-grained**: Large, infrequent tasks (better overhead amortization).

**Trade-offs:**
* Too fine: Overhead dominates (thread creation, synchronization, GIL contention).
* Too coarse: Load imbalance (some threads finish early, others work longer).
* Optimal granularity depends on:
  * Overhead costs
  * Number of available cores
  * Total problem size

### **Task Parallelism**

Parallelization strategy where **different operations** are performed simultaneously.

* Each thread/process performs a different task.
* Example: Thread 1 applies horizontal flip, Thread 2 applies vertical flip to same image.
* Useful when multiple independent operations needed on same data.
* Contrast with **Data Parallelism**.

### **Thread**

A lightweight execution context within a process that shares the same address space.

* Multiple threads in a process share:
  * Memory (heap, global variables)
  * File handles
  * Process resources
* Each thread has its own:
  * Program counter
  * Stack
  * Local variables

**Python specifics:**
* Module: `threading`
* Subject to GIL for CPU-bound work
* Effective for I/O-bound work
* Lower overhead than processes

### **`threading.Thread`**

Python's class representing a single thread of execution.

```python
import threading

def worker(name):
    print(f"Thread {name}")

thread = threading.Thread(target=worker, args=("Worker-1",), name="T1")
thread.start()  # Begin execution
thread.join()   # Wait for completion
```

**Key methods:**
* `start()`: Begin thread execution
* `join()`: Wait for thread to complete
* `is_alive()`: Check if thread is still running

**Key attributes:**
* `name`: Thread name (for debugging)
* `daemon`: If True, thread terminates when main program exits

### **Thread Pool**

A collection of pre-created threads ready to execute tasks.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(worker_function, data_list)
```

* Avoids repeated thread creation overhead.
* Reuses threads for multiple tasks.
* Higher-level abstraction than `threading.Thread`.
* Module: `concurrent.futures.ThreadPoolExecutor`

### **Thread Safety**

Property of code that functions correctly when executed by multiple threads concurrently.

* Requires proper synchronization for shared data.
* Techniques:
  * **Locks**: Mutual exclusion
  * **Atomic operations**: Indivisible operations
  * **Immutable data**: Cannot be modified (inherently safe)
  * **Thread-local storage**: Separate copy per thread
* Python's GIL provides some implicit thread safety for certain operations, but explicit locks still needed for complex operations.

### **Throughput-Oriented vs Latency-Oriented**

Design philosophies for processing architectures.

**Latency-Oriented (CPU):**
* Minimize time for single task completion.
* Few powerful cores with large caches.
* Complex control flow and branch prediction.
* Optimized for sequential execution.

**Throughput-Oriented (GPU):**
* Maximize total work completed per unit time.
* Many lightweight cores.
* Simple control flow.
* Optimized for massive parallelism.

**Relevance to threading:**
* Multi-threading on CPU aims to improve throughput while managing latency.
* Effective when many independent tasks can be executed concurrently.

### **Time Slicing**

Operating system technique for sharing CPU time among multiple threads/processes.

* Scheduler rapidly switches between threads (context switching).
* Gives illusion of simultaneous execution on single core.
* Each thread gets a **time slice** (quantum) of CPU time.
* Enables **concurrency** without **parallelism**.

---

## **W**

### **Worker Function**

A function executed by threads or processes to perform parallel work.

```python
def worker(task_id, data):
    """Process assigned portion of work"""
    result = process(data)
    return result

# Used with threads or processes
thread = threading.Thread(target=worker, args=(1, data_chunk))
```

* Should be **stateless** when possible (easier to parallelize).
* For multiprocessing: Must be **picklable** (top-level function, not lambda).
* Often wrapped to handle shared state or result collection.

---

## **Cross-Reference**

### **Threading vs Multiprocessing Decision Matrix**

| Criterion | Threading | Multiprocessing |
|-----------|-----------|-----------------|
| **Workload type** | I/O-bound | CPU-bound |
| **GIL impact** | Limited by GIL | Bypasses GIL |
| **Memory overhead** | Low (shared memory) | High (separate memory) |
| **Startup cost** | Fast | Slow |
| **Communication** | Direct (shared memory) | IPC required (pickling) |
| **Best for** | Network I/O, file I/O, concurrent tasks | Computation, data processing, parallel algorithms |

### **When to Use Each Approach**

**Use Threading when:**
* Task is I/O-bound (network, disk, database)
* Need low overhead and fast startup
* Shared state is essential and manageable
* Running in free-threaded Python for CPU-bound work

**Use Multiprocessing when:**
* Task is CPU-bound
* Need true parallelism with standard Python
* Process isolation desirable
* Have sufficient memory for multiple interpreters

**Use GPU (CuPy/Numba CUDA) when:**
* Very large data arrays
* Highly parallelizable operations
* Data-parallel algorithms
* Computational intensity justifies transfer overhead

---

## **Common Patterns**

### **Producer-Consumer Pattern**

```python
from queue import Queue
import threading

queue = Queue()

def producer():
    for i in range(10):
        queue.put(i)
    queue.put(None)  # Sentinel

def consumer():
    while True:
        item = queue.get()
        if item is None:
            break
        process(item)

threading.Thread(target=producer).start()
threading.Thread(target=consumer).start()
```

### **Map-Reduce Pattern**

```python
from multiprocessing import Pool

def map_function(x):
    return x * x

def reduce_function(results):
    return sum(results)

with Pool(4) as pool:
    mapped = pool.map(map_function, range(100))
    reduced = reduce_function(mapped)
```

---

## **Best Practices**

1. **Always use `with` statement** for `Pool` or `ThreadPoolExecutor` (automatic cleanup).
2. **Minimize shared state** between threads/processes when possible.
3. **Use locks consistently** to protect shared data.
4. **Profile before parallelizing** - measure actual bottlenecks.
5. **Consider task granularity** - ensure work justifies parallelization overhead.
6. **Test with different core counts** - optimal thread/process count varies by workload.
7. **For Jupyter/notebooks** - run multiprocessing code in `.py` files (avoid kernel issues).
8. **Guard multiprocessing code** with `if __name__ == "__main__":` (prevents recursive spawning).

---

## **General Summary**

Threading and multiprocessing in Python provide different approaches to concurrent and parallel execution:

**Threading:**
* Lightweight, shared memory
* Effective for I/O-bound tasks
* Limited by GIL for CPU-bound work (unless free-threaded mode)
* Lower overhead, faster startup

**Multiprocessing:**
* Separate processes, isolated memory
* True parallelism for CPU-bound tasks
* Bypasses GIL limitations
* Higher overhead, requires IPC

**Key Insight:** Choose based on workload characteristics (I/O-bound vs CPU-bound) and GIL constraints. For CPU-intensive Python code, multiprocessing or GPU acceleration typically outperforms threading in standard Python. With free-threaded Python 3.13+, threading becomes viable for CPU-bound work.

---
