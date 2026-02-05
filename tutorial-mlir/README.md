# MLIR Tutorial (Getting Started)

This directory contains a hands-on implementation of an **MLIR tutorial**, built by following and studying the blog post:

- Reference: https://www.jeremykun.com/2023/08/10/mlir-getting-started/

The goal of this work is to build a concrete understanding of **MLIR fundamentals** by implementing a small but complete pipeline: defining IR, running transformations, and executing tools using the MLIR infrastructure.

---

## Overview

This project focuses on understanding MLIR from a **compiler engineer’s perspective**, including:

- How MLIR programs are structured
- How dialects, operations, and types are represented
- How passes and tooling are wired together
- How to build and test MLIR-based tools using Bazel

Rather than treating MLIR as a black box, this tutorial emphasizes **hands-on exploration of IR design and transformation mechanics**.

---

## Directory structure
```
tutorial-mlir/
├── bazel/
│ └── Bazel configuration and dependency setup for MLIR
│
├── lib/
│ └── MLIR-related libraries and implementation code
│
├── tests/
│ └── Tests for MLIR transformations and tooling
│
├── tools/
│ └── Command-line tools built on top of MLIR
│
├── BUILD
├── MODULE.bazel
├── extensions.bzl
├── ctlz.mlir
└── requirements.txt
```


---

## Key components

### Bazel build files

- **`BUILD`**, **`MODULE.bazel`**, and **`extensions.bzl`** define:
  - MLIR/LLVM dependencies
  - Build rules for libraries, tools, and tests
  - A reproducible build setup for MLIR-based development

This setup mirrors real-world MLIR projects that rely on Bazel for large-scale builds.

---

### `lib/`

Contains implementation code related to MLIR:
- IR manipulation
- Pass registration and execution
- Core logic shared across tools

---

### `tools/`

Command-line utilities built on top of MLIR:
- Parse MLIR files
- Apply transformations or analyses
- Emit transformed IR or diagnostics

These tools demonstrate how MLIR is embedded into custom compiler workflows.

---

### `tests/`

Tests validating:
- Correctness of transformations
- Expected IR output
- Integration of passes and tools

---

## Concepts explored

- **MLIR IR structure**
  - Operations, regions, and blocks
  - SSA values and type system

- **Dialect-driven design**
  - Separation of concerns via dialects
  - Extensibility of IR and operations

- **Pass infrastructure**
  - How MLIR passes are defined and composed
  - Running transformations over IR modules

- **Tooling integration**
  - Building MLIR-based command-line tools
  - Wiring parsing, passes, and output together

- **Bazel-based MLIR builds**
  - Managing LLVM/MLIR dependencies
  - Structuring real-world compiler projects

---

## How to build and run

This project uses **Bazel** for building.

From the `tutorial-mlir/` directory:
```bash
bazel build //...
```

To run tests:
```bash
bazel tests //...
```

To run tools:
```bash
bazel run //tools:<tool-name>
```


