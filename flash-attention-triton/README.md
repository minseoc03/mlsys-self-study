# Flash Attention (CUDA / Triton)

This project is a hands-on implementation of **FlashAttention-style attention kernels**, built while following and studying the implementation walkthrough in the following talk:

- Reference video: https://www.youtube.com/watch?v=zy8ChVd_oTM

The purpose of this project is not merely to reproduce FlashAttention, but to **understand and implement the core systems ideas** behind it: tiling strategies, memory efficiency, numerical stability, and the decomposition of forward and backward passes.

---

## Overview

This project explores how attention can be computed efficiently on GPUs **without materializing the full attention matrix**, using block-wise computation and an online softmax formulation.

Two implementations are provided:

- **CUDA**: low-level kernel implementations to understand memory access patterns and GPU execution directly
- **Triton**: a higher-level, compiler-assisted implementation focusing on correctness, structure, and readability

---

## Directory structure
```
flash-attention-triton/
├── cuda/
│ ├── Makefile
│ ├── cuda_common.cuh
│ ├── vector_add_simple.cu
│ ├── vector_add.cu
│ └── matrix_add.cu
│
├── triton/
│ ├── flash_attention.py
│ └── requirements.txt
```

---

## CUDA implementation (`cuda/`)

The `cuda/` directory contains small CUDA programs used to build intuition for GPU programming before moving to full attention kernels.

### Files

- **`cuda_common.cuh`**  
  Common CUDA utilities and helper macros shared across kernels.

- **`vector_add_simple.cu`**  
  A minimal vector addition kernel, focusing on basic thread indexing and kernel launch structure.

- **`vector_add.cu`**  
  An extended vector addition example, used to reason about memory access, block/thread configuration, and performance implications.

- **`matrix_add.cu`**  
  A simple matrix addition kernel, serving as a stepping stone toward tiled matrix operations used in attention.

- **`Makefile`**  
  Build rules for compiling and running the CUDA examples with `nvcc`.

### Goal of CUDA examples

These kernels are **not FlashAttention themselves**, but are included to:
- Develop intuition for CUDA execution models
- Understand memory layout, indexing, and kernel launches
- Prepare for reasoning about tiled attention kernels

---

## Triton implementation (`triton/`)

The `triton/` directory contains a full **FlashAttention-style attention implementation** using Triton.

### Files

- **`flash_attention.py`**  
  Implements attention forward and backward passes using Triton kernels:
  - Block-wise computation over queries and keys
  - Numerically stable online softmax using running max and normalization
  - Separate kernels for forward pass, backward preprocessing, `dQ`, and `dK/dV`
  - Supports both causal and non-causal attention
  - Includes correctness tests against PyTorch reference implementation

- **`requirements.txt`**  
  Python dependencies required to run the Triton implementation.

---

## Key ideas explored

### Blocking / tiling
- Queries and keys are processed in blocks that fit into on-chip memory
- Query blocks are reused while streaming key/value blocks

### Online softmax
- Maintains a running maximum (`m_i`) and normalization factor (`l_i`)
- Ensures numerical stability while processing attention incrementally

### Memory efficiency
- Avoids materializing the full attention matrix
- Reduces memory bandwidth pressure by computing attention on the fly

### Backward pass decomposition
- Preprocessing step computes `D_i = sum(dO * O)`
- Separate kernels handle gradients for `dQ` and `dK/dV`
- Softmax probabilities are reconstructed using saved log-sum-exp values

---

## How to run (Triton version)

From the `triton/` directory:

```bash
pip install -r requirements.txt
python flash_attention.py
```

Successful execution prints:
```
PASSED
```

Be sure to modify configurations based on your environment, so computer doesn't exploding while running it!
