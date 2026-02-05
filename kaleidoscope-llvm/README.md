
# 🔧 Kaleidoscope Tutorial (LLVM-Based Language Implementation Practice)

This project follows the official [LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/),  
a step-by-step guide to implementing a simple functional programming language.  
It aims to explore the fundamentals of compiler frontends using LLVM in C++.

---

## 📚 Progress Overview

| Chapter # | Topic | Note | Code |
|:--------:|:-------:|:--------:|:--------:|
| 1 | Implementing a Lexer (tokenization) | [Note](https://minseoc03.github.io/mlsys/llvm/kal_ch1/) | [Code](src/no_llvm.cpp)
| 2 | Building AST + Recursive Parser | [Note](https://minseoc03.github.io/mlsys/llvm/kal_ch2/) | [Code](src/no_llvm.cpp)
| 3 | Generating LLVM IR | [Note](https://minseoc03.github.io/mlsys/llvm/kal_ch3/) | [Code](src/llvm_codegen.cpp)
| 4 | Adding JIT and Optimizer | [Note](https://minseoc03.github.io/mlsys/llvm/kal_ch4/) | [Code](src/llvm_opt_jit.cpp) |
---

## 📁 Project Structure

```
kaleidoscope_tutorial/
├── src/                    # Source files (varying versions according to progress of chapters)
│   └── no_llvm.cpp             # Chapter 1 & 2
│   └── llvm_codgen.cpp         # Chapter 3
│   └── llvm_opt_jit.cpp        # Chapter 4
├── build/                  # Build directory (CMake outputs)
├── notes/                  # Markdown notes per chapter (written in Korean)
├── includes/
│   └── KaleidoscoprJIT.h   # headerfile for custom JIT
├── CMakeLists.txt          # CMake configuration
├── LICENSE                 # MIT License
└── README.md               # Project overview
```

---

## 🚀 Build & Run (macOS + LLVM via Homebrew)

### 1. Install LLVM

```bash
brew install llvm
```

### 2. Add LLVM to your PATH

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

### 3. Build and run the project

```bash
cd kaleidoscope_tutorial
mkdir build && cd build
cmake ..
make
./kaleidoscope
```

---

## 📄 References

- [LLVM Kaleidoscope Official Tutorial](https://llvm.org/docs/tutorial/)
