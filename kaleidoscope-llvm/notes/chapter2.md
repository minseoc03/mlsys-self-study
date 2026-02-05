
# 🌿 Kaleidoscope Tutorial Chapter 2 - AST & 파서 구현

이 챕터에서는 토큰으로 분리된 입력을 **추상 구문 트리(Abstract Syntax Tree, AST)**로 구조화하는 파서를 구현한다.  
이는 이후 LLVM IR로 변환될 내부 표현이며, 모든 표현식, 함수 호출, 연산 등을 트리 형태로 객체화하는 것이 핵심이다.

---

## 🎯 핵심 목표

- 사용자 입력을 구조적으로 표현하는 **AST 클래스 계층 설계**
- 파서를 구현하여 토큰 스트림 → AST 변환
- 연산자 우선순위 처리 및 재귀적 표현식 트리 구성
- 함수 정의 / 호출 / 괄호 / 이항 연산까지 처리

---

## 🌲 1. AST 클래스 계층 구조

모든 AST 노드는 `ExprAST`를 상속한다.

### 📦 `ExprAST` (base class)

```cpp
class ExprAST {
public:
  virtual ~ExprAST() = default;
};
```

---

### 🔹 숫자 리터럴: `NumberExprAST`

```cpp
class NumberExprAST : public ExprAST {
  double Val;
};
```

예: `42.0`

---

### 🔹 변수 참조: `VariableExprAST`

```cpp
class VariableExprAST : public ExprAST {
  std::string Name;
};
```

예: `x`

---

### 🔹 이항 연산자: `BinaryExprAST`

```cpp
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;
};
```

예: `x + y * 2`

---

### 🔹 함수 호출: `CallExprAST`

```cpp
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;
};
```

예: `foo(1, 2)`

---

### 🔹 함수 시그니처 및 정의

```cpp
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;
};

class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;
};
```

---

## 🧠 2. 파서 구조

### 🔧 핵심 변수

```cpp
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }
```

### 🧩 연산자 우선순위 테이블

```cpp
std::map<char, int> BinopPrecedence = {
  {'<', 10},
  {'+', 20}, {'-', 20},
  {'*', 40}
};
```

---

## 🔄 3. 주요 파서 함수

| 함수 이름 | 역할 | 반환 AST |
|-----------|------|-----------|
| `ParseExpression()` | 최상위 수식 파싱 | `ExprAST` |
| `ParsePrimary()` | 숫자/변수/괄호/함수 호출 | `ExprAST` |
| `ParseNumberExpr()` | 숫자 리터럴 | `NumberExprAST` |
| `ParseIdentifierExpr()` | 변수 or 함수 호출 | `VariableExprAST` or `CallExprAST` |
| `ParseParenExpr()` | 괄호로 감싼 식 | `ExprAST` |
| `ParseBinOpRHS()` | 이항 연산자 처리 | `BinaryExprAST` |
| `ParsePrototype()` | 함수 시그니처 파싱 | `PrototypeAST` |
| `ParseDefinition()` | `def` 파싱 | `FunctionAST` |
| `ParseExtern()` | `extern` 파싱 | `PrototypeAST` |
| `ParseTopLevelExpr()` | 익명 함수 wrapping | `FunctionAST` |

---

## 🔍 4. 연산자 우선순위 처리 핵심

```cpp
while (true) {
  int TokPrec = GetTokPrecedence();
  if (TokPrec < ExprPrec)
    return LHS;
  ...
  RHS = ParsePrimary();
  ...
  LHS = std::make_unique<BinaryExprAST>(Op, std::move(LHS), std::move(RHS));
}
```

- 현재 연산자 우선순위(`TokPrec`)를 기준으로 재귀 파싱
- 트리 구조에서 올바른 결합 방향을 유지함

---

## 🧪 5. 입력 → AST 예시

입력:
```
def foo(x y) (x + y) * 2;
```

AST 구조:

```
FunctionAST
├── PrototypeAST("foo", ["x", "y"])
└── BinaryExprAST('*')
    ├── BinaryExprAST('+')
    │   ├── VariableExprAST("x")
    │   └── VariableExprAST("y")
    └── NumberExprAST(2.0)
```

---

## ✅ 요약

| 항목 | 설명 |
|------|------|
| AST | 표현식, 함수, 변수, 연산자 등 모든 구문 표현 |
| 파서 | 토큰 → AST로 변환, 재귀적 구조 |
| 우선순위 | `ParseBinOpRHS()`가 처리 |
| 함수 정의 | 시그니처(`Prototype`) + 본문(`ExprAST`) 분리 |
| 복잡도 | 연산자 트리 구성과 재귀 파싱에서 핵심 로직 발생 |

---
