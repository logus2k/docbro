# Numba Glossary: Key Concepts and Terms

---

## **A**
### **Ahead-of-Time (AOT) Compilation**
- Compiling Numba functions to machine code **before runtime** (e.g., for deployment).
- Contrasts with **Just-In-Time (JIT)** compilation, which happens during execution.
- Useful for **reducing startup time** in production environments.

---

## **B**
### **Blocking Operations**
- Operations that **pause execution** until completion (e.g., I/O, sleep).
- Numba is **not designed** for blocking operations—use `asyncio` or `threading` instead.

---

## **C**
### **CPU-Bound Tasks**
- Tasks limited by **CPU speed** (e.g., mathematical computations, loops).
- Numba **excels** at accelerating CPU-bound tasks via compilation and parallelization.

### **Cython**
- A tool to write C extensions for Python, often used as an alternative to Numba.
- Numba is **easier to use** for simple numerical acceleration but less flexible than Cython for complex cases.

---

## **D**
### **Dynamic Typing**
- Python’s default behavior, where variable types are determined at runtime.
- Numba **infers types** during compilation for performance but may struggle with highly dynamic code.

---

## **E**
### **Embarrassingly Parallel**
- Tasks that can be **easily divided** into independent subtasks (e.g., applying a function to each element of an array).
- Numba’s `prange` is ideal for such tasks.

---

## **G**
### **Global Interpreter Lock (GIL)**
- A mutex in CPython that **limits thread execution** to one at a time for Python bytecode.
- Numba **bypasses the GIL** for compiled code, enabling true parallelism in CPU-bound loops.

---

## **J**
### **Just-In-Time (JIT) Compilation**
- Compiling functions to machine code **during runtime** (e.g., using `@njit`).
- Numba’s default mode, balancing **flexibility** and **performance**.

---

## **L**
### **Loop Unrolling**
- A compiler optimization where loops are **expanded into repeated statements** to reduce overhead.
- Numba applies this to **small, fixed-size loops** for speedups.

---

## **M**
### **Machine Code**
- Low-level code executed directly by the CPU.
- Numba compiles Python functions to **optimized machine code** for faster execution.

---

## **N**
### **NumPy**
- A library for numerical computing in Python.
- Numba **integrates seamlessly** with NumPy, accelerating operations on NumPy arrays.

---

## **O**
### **OpenMP**
- A multi-platform API for **multi-threading**.
- Numba uses OpenMP to **parallelize loops** when `parallel=True` is enabled.

---

## **P**
### **Parallel Execution**
- Running multiple operations **simultaneously** (e.g., using `prange`).
- Numba supports parallel loops via `@njit(parallel=True)`.

### **`prange` (Parallel Range)**
- A Numba-specific function to **parallelize loops** across threads.
- Example:
  ```python
  from numba import njit, prange

  @njit(parallel=True)
  def parallel_loop(arr):
      for i in prange(arr.shape[0]):
          arr[i] *= 2

---
