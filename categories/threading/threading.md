# Threading in Python: Concurrency, Parallelism and the Global Interpreter Lock (GIL)

## 1. Conceptual Foundations

### 1.1 Concurrency vs. Parallelism

At a high level:
- **Concurrency** is about **dealing with many things at once**. Multiple tasks _overlap_ in time, but not necessarily execute at the exact same instant.
- **Parallelism** is about **doing many things at the same time**. Multiple tasks are physically executed simultaneously on different processing units (e.g., cores).

In modern operating systems, even a single-core CPU can run multiple applications “at the same time” via **time slicing**: the scheduler rapidly switches between tasks. This is **concurrency without parallelism**.

On a multi-core CPU, we can have both **concurrency and true parallelism**: multiple threads of execution can run at the same time on different cores.

#### Visual intuition

You can think of concurrency vs parallelism as different ways to allocate work:

<img src="https://logus2k.com/docbro/categories/cgad/images/parallel_concurrent.png" width="600"/>

### 1.2 Threads and Processes

Drawing from the slides from past class:

-   A **process**:
    -   Has its **own address space** (independent memory view).
    -   Contains one or more **threads**.
    -   Communication between processes requires **inter-process communication (IPC)** (pipes, queues, shared memory, sockets).

-   A **thread**:
    -   Is a **lightweight execution context** within a process.
    -   Shares the **same address space** as its sibling threads.
    -   Can access all objects of the parent process (and thus risk race conditions).

High-level trade-offs (adapted from the second PDF):

| Aspect | Threads | Processes |
| --- | --- | --- |
| Memory model | Shared address space | Separate address spaces |
| Overhead | Low creation & switching overhead | Higher overhead (OS-level creation, IPC) |
| Best for | I/O-bound, high concurrency | CPU-bound, true parallelism |
| GIL impact (CPython) | Restricted for CPU-bound work | Bypasses GIL (each process has its own GIL) |

We will focus on **Python threads** and their interaction with the **GIL**.

## 2. The Global Interpreter Lock (GIL)

### 2.1 What is the GIL?

The dominant implementation of Python (CPython) uses a **Global Interpreter Lock**:
-   The GIL is a **mutual exclusion lock** that ensures that at most **one OS thread executes Python bytecode at a time** within a given process.
-   Historically, it simplified:
    -   memory management (reference counting),
    -   interaction with C extensions,
    -   and thread-safety of the interpreter core.

As a consequence:

-   **CPU-bound multi-threaded code** (doing intensive pure-Python computation) **does not achieve true parallel speed-up** in CPython. Threads contend for the GIL and effectively run one at a time.
-   **I/O-bound multi-threaded code** can still benefit because threads **release the GIL during blocking I/O** (e.g., network, disk).

### 2.2 Python 3.13 and Free-Threaded Mode

Recent work in Python 3.13 introduced an **experimental “free-threaded mode”** where the GIL can be disabled (using `--disable-gil` at build time) and a separate `python3.13t` executable.

In this mode:
-   Threads can execute Python code **truly in parallel** across cores.
-   Additional fine-grained locking and changes to memory management are required, so:
    -   Single-threaded performance is sometimes **slower**.
    -   Some existing C extensions are **not yet compatible**.

## 3. Python’s `threading` Module

### 3.1 Overview

The `threading` module provides a **high-level interface** to OS threads:
-   `Thread` – class representing a single thread of control.
-   `Lock`, `RLock`, `Semaphore`, `Event`, `Condition` – synchronisation primitives.
-   `Timer` – a thread that waits and then executes a function.

> Despite the name, using `threading` does **not** guarantee parallel execution of Python code on multiple cores under the GIL; it **guarantees concurrency** and potential parallelism when I/O or C extensions release the GIL.

