# mlsys-self-study

Hands-on self-study repository on ML systems, including LLVM, MLIR, CUDA & Triton kernels, and efficient LLM attention mechanisms.

## What this repo is

This is a consolidated workspace where I collect my learning notes and implementations across **ML compilers + GPU kernels + efficient inference**.  
Instead of spreading work across multiple small repos, everything lives here with a consistent structure and runnable code where applicable.

## Topics covered

- **ML Compilers**
  - LLVM fundamentals (IR, codegen, optimization mindset)
  - MLIR (dialects, ops/types/attributes, passes, rewriting, lowering)

- **GPU Kernels / Performance**
  - CUDA & Triton kernel programming
  - Memory/layout reasoning, tiling, parallelism, numerical stability

- **LLM Inference**
  - Efficient attention mechanisms (e.g., FlashAttention-style tiled attention)
  - Causal vs non-causal attention, forward/backward correctness checks

## Repository structure

- `flash-attention-triton/`  
  Triton-based attention kernels (FlashAttention-style). Includes forward/backward implementations and correctness tests against PyTorch reference.

- `kaleidoscope-llvm/`  
  LLVM Kaleidoscope tutorial work: building a small language frontend and generating LLVM IR/JIT execution to understand compiler fundamentals.

- `tutorial-mlir/`  
  MLIR tutorial implementations and experiments: dialect/ops, canonicalization/folding, pattern rewrites, lowering pipelines, and debugging notes.

For more details on each project, please refer to its individual README.

## Notes

This repository is primarily for personal learning and experimentation. If something looks incomplete, it’s likely mid-iteration.
