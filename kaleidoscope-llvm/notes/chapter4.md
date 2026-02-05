
# 🧠 Kaleidoscope Tutorial Chapter 4 - 최적화 및 JIT 기능 추가

---

## ✅ Chapter 4의 핵심 변화 요약

| 구분 | 내용 | 도입 이유 |
|------|------|------------|
| ✅ JIT 추가 | `KaleidoscopeJIT` 클래스 도입 | 코드를 JIT 컴파일 후 즉시 실행하기 위해 |
| ✅ 최적화 지원 | `FunctionPassManager` 도입, `PassBuilder` 활용 | 코드 성능 개선 |
| ✅ 함수별 모듈 분리 | `ThreadSafeModule` 사용 | 모듈을 독립적으로 JIT에 등록 가능하게 하기 위해 |
| ✅ 심볼 탐색 및 실행 | `lookup("__anon_expr")` 사용 | 익명 함수 실행 지원 |

---

## 📌 Chapter 3 → Chapter 4 달라진 주요 코드들과 이유

### 1. `KaleidoscopeJIT` 관련

#### [추가]
```cpp
#include "KaleidoscopeJIT.h"
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
```
- 📌 **이유**: LLVM의 Orc JIT 기능을 사용하기 위한 사용자 정의 래퍼 클래스
- 👉 정의된 함수/표현식을 JIT으로 실행하고, 결과를 바로 얻기 위해 필요

---

### 2. 함수 평가 로직 변화 (TopLevelExpr)

#### Chapter 3:
```cpp
if (auto *FnIR = FnAST->codegen()) {
  FnIR->print(errs());
  FnIR->eraseFromParent();
}
```

#### Chapter 4:
```cpp
if (auto FnAST = ParseTopLevelExpr()) {
  if (FnAST->codegen()) {
    auto RT = TheJIT->getMainJITDylib().createResourceTracker();
    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
    InitializeModuleAndPassManager();

    auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));
    double (*FP)() = ExprSymbol.getAddress().toPtr<double (*)()>();
    fprintf(stderr, "Evaluated to %f
", FP());

    ExitOnErr(RT->remove());
  }
}
```

- 📌 **이유**:
  - IR을 단순히 출력하고 제거하던 3장에서 → 실제로 실행(JIT)하도록 변경
  - `ThreadSafeModule` + `addModule()`을 통해 모듈을 등록하고,
  - `lookup()`을 통해 익명 함수 실행 가능하게 함

---

### 3. 최적화 패스 매니저 도입

#### [추가]
```cpp
FunctionPassManager TheFPM;
PassBuilder PB;
PB.registerModuleAnalyses(...);
PB.registerFunctionAnalyses(...);
PB.crossRegisterProxies(...);

TheFPM.addPass(InstCombinePass());
TheFPM.addPass(ReassociatePass());
...
```
- 📌 **이유**: IR 수준에서 `a + 0`, `x * 1` 등을 제거하는 최적화 적용
- → 튜토리얼에서는 실질적으로 생성된 IR의 질 향상을 위해

---

### 4. 모듈 초기화 방식 변화

#### Chapter 3:
```cpp
TheContext = std::make_unique<LLVMContext>();
TheModule = std::make_unique<Module>("my cool jit", *TheContext);
Builder = std::make_unique<IRBuilder<>>(*TheContext);
```

#### Chapter 4:
```cpp
TheContext = std::make_unique<LLVMContext>();
TheModule = std::make_unique<Module>("KaleidoscopeJIT", *TheContext);
TheModule->setDataLayout(TheJIT->getDataLayout());
```
- 📌 **이유**: JIT이 예상하는 데이터 레이아웃으로 모듈을 구성하기 위해
- `setDataLayout()`은 JIT이 IR을 올바르게 해석하게 해줌

---

## 💡 함수당 하나의 모듈 사용 전략

### 📌 왜 필요한가?
- LLVM JIT은 **모듈 단위로 메모리/코드 관리**
- 하나의 모듈에 여러 함수가 있다면 재정의가 어렵고 제거도 어려움
- → 각 함수/표현식을 **독립적인 모듈에 넣으면** 실행 후 제거 용이

---

### ⚙️ 어떻게 실현했는가?
1. `codegen()` 완료 후:
```cpp
auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
```
2. 모듈을 JIT에 등록:
```cpp
TheJIT->addModule(std::move(TSM), RT);
```
3. 다음 입력 처리를 위해 새 모듈 초기화:
```cpp
InitializeModuleAndPassManager();
```

---

### 🧩 이 구조의 문제점과 해결 방법

| 문제 | 해결 방법 |
|------|------------|
| 새로운 모듈에서는 이전 함수(`foo`)를 못 찾음 | `FunctionProtos` + `getFunction()`으로 선언만 복원해서 호출 가능하게 함 |
| 모듈마다 중복 선언 방지 필요 | `getFunction()`이 현재 모듈에 없을 경우에만 복원 |

#### 관련 코드:
```cpp
if (auto *F = TheModule->getFunction(Name)) return F;
if (FunctionProtos[Name]) return FunctionProtos[Name]->codegen();
```

---

## 📚 최종 요약

Chapter 4에서는 JIT과 최적화를 통해 Kaleidoscope 언어가 **실행 가능한 인터프리터에 가까운 구조**로 발전함. 
함수당 모듈 전략은 실행과 메모리 관리에 유리하지만, `FunctionProtos`와 `getFunction()`으로 함수 선언을 복원하는 구조적 장치가 꼭 필요함.