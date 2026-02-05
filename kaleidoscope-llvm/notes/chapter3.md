
# 🧠 Kaleidoscope Tutorial Chapter 3 - LLVM IR 생성

이 챕터는 본격적으로 **LLVM IR 생성기를 구현하는 단계**이다.  
앞서 만든 AST를 기반으로 LLVM의 IRBuilder를 이용해 실제 LLVM IR을 생성하며,  
각 노드별로 `codegen()` 메서드를 작성해 구조적으로 IR을 만들어간다.

---

## 🎯 핵심 목표

- `ExprAST`를 상속받는 각 클래스에 `codegen()` 메서드 구현
- LLVM이 제공하는 `IRBuilder<>`를 사용하여 IR 명령어 생성
- 연산자 처리, 함수 호출, 비교 연산 등의 IR 변환 이해
- 타입 불일치, 조건 연산 처리 등에서 발생할 수 있는 예외 고려

---

## 🔩 1. 코드 생성 도우미: IRBuilder

```cpp
llvm::IRBuilder<> Builder(TheContext);
```

- **LLVM IR 명령어를 생성하고, 현재 기본 블록 끝에 삽입**함
- 주요 메서드:
  - `CreateFAdd()`, `CreateFSub()`, `CreateFMul()`: 부동소수 연산
  - `CreateRet()`: 반환
  - `CreateCall()`: 함수 호출
  - `CreateFCmpULT()`: 부동소수 비교 (less-than)
  - `CreateUIToFP()`: 정수 → 실수 형 변환 (i1 → double)

---

## 🌲 2. AST 노드별 `codegen()` 메서드

### `NumberExprAST::codegen()`

```cpp
return ConstantFP::get(TheContext, APFloat(Val));
```

- `double` 값 하나를 부동소수 LLVM 상수로 변환

---

### `VariableExprAST::codegen()`

```cpp
Value *V = NamedValues[Name];
if (!V) return LogErrorV("Unknown variable name");
return V;
```

- 이름을 기반으로 `NamedValues`에서 값을 찾음 (이 값은 함수 인자에서 설정됨)

---

### `BinaryExprAST::codegen()`

```cpp
Value *L = LHS->codegen();
Value *R = RHS->codegen();
switch (Op) {
  case '+': return Builder.CreateFAdd(L, R, "addtmp");
  case '-': return Builder.CreateFSub(L, R, "subtmp");
  case '*': return Builder.CreateFMul(L, R, "multmp");
  case '<':
    L = Builder.CreateFCmpULT(L, R, "cmptmp");
    return Builder.CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
}
```

- `<` 연산은 `i1` (1비트 정수)로 나오기 때문에, `UIToFP`로 `double`로 변환 필요
- ❗ **중요 문법 포인트**:
  - `CreateFCmpULT(...)`: unordered less-than
  - `CreateUIToFP(...)`: unsigned integer → floating-point

---

### `CallExprAST::codegen()`

```cpp
Function *CalleeF = TheModule->getFunction(Callee);
std::vector<Value *> ArgValues;
for (...) {
  ArgValues.push_back(Arg->codegen());
}
return Builder.CreateCall(CalleeF, ArgValues, "calltmp");
```

- 함수 이름을 `TheModule`에서 검색 → 존재하지 않으면 에러
- 인자 순서대로 `codegen()` → `CreateCall(...)`로 호출

---

## 📦 3. 함수 시그니처 및 본문 생성

### `PrototypeAST::codegen()`

```cpp
std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
FunctionType *FT = FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);
Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());
```

- 함수의 타입은 `double(double, double, ...)` 형식
- `ExternalLinkage`: 외부에서 호출 가능
- 각 인자의 이름은 `F->args()` 순회하며 `setName()`으로 지정

---

### `FunctionAST::codegen()`

```cpp
Function *TheFunction = TheModule->getFunction(Proto->getName());
if (!TheFunction) TheFunction = Proto->codegen();
...
BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
Builder->SetInsertPoint(BB);
...
Builder->CreateRet(RetVal);
verifyFunction(*TheFunction);
```

- 함수가 이전에 선언된 경우(extern), 그 함수 가져오기
- 없으면 새로 codegen()
- `BasicBlock` 만들고 `IRBuilder`로 명령어 삽입 위치 설정
- 본문 `ExprAST->codegen()` 실행
- 마지막에 `ret` 명령어 붙이고, `verifyFunction()`으로 일관성 검증

---

## ⚠️ 실수하기 쉬운 포인트

| 개념 | 설명 |
|------|------|
| `i1` → `double` 변환 | 비교 연산은 `CreateFCmpULT()`로 생성되며, 결과는 `i1` 타입. 이를 `CreateUIToFP()`로 float으로 변환해야 Kaleidoscope 언어 규칙에 맞음 |
| 함수 시그니처 생성 | `FunctionType::get(...)`에서 반환 타입, 인자 타입 배열, `false` (가변 인자 아님) 필요 |
| 변수 참조 | 함수 인자 외에는 별도 스코프 처리 없음 (NamedValues 맵이 전역 변수처럼 작동) |

---

## 🧪 예제 입력 → IR 출력 예시

입력:
```
def foo(x y) x + y * 2;
```

IR 출력:
```llvm
define double @foo(double %x, double %y) {
entry:
  %multmp = fmul double %y, 2.0
  %addtmp = fadd double %x, %multmp
  ret double %addtmp
}
```

---

## ✅ 요약

| 항목 | 설명 |
|------|------|
| `codegen()` | 각 AST 노드를 LLVM IR로 변환 |
| `IRBuilder<>` | 명령어 생성 및 삽입 도우미 |
| 주요 변환 | 숫자 상수, 연산자, 함수 호출, 비교 연산 |
| 트릭 포인트 | `fcmp` 결과 변환, extern 중복, verifyFunction() |

---
