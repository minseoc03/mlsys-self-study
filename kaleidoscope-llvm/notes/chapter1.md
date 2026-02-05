
# 🌟 Kaleidoscope Tutorial: Chapter 1 - 언어 정의 및 토큰화

이 챕터에서는 LLVM 기반 toy language인 **Kaleidoscope**의 기초를 다진다.  
언어의 핵심 구조를 정의하고, **토큰화(lexing)** 과정을 구현하는 것이 주요 목표이다.

---

## 🔧 1. 언어 구조 설계

Kaleidoscope는 아래와 같은 표현들을 지원하는 단순한 함수형 언어이다:

- 숫자 리터럴: `123.0`
- 변수 참조: `x`
- 이항 연산자: `+`, `-`, `*`, `<`
- 괄호 연산자: `(1 + 2) * 3`
- 함수 호출: `foo(1, 2)`
- 함수 정의: `def foo(x y) x + y;`
- 외부 함수 선언: `extern sin(x);`

---

## 🔍 2. Lexer: 토큰화 구현

### 🔹 반환되는 토큰 종류

| Token 이름     | 의미 |
|----------------|------|
| `tok_eof`      | 파일 끝 |
| `tok_def`      | 함수 정의 (`def`) |
| `tok_extern`   | 외부 선언 (`extern`) |
| `tok_identifier` | 변수명 또는 함수명 |
| `tok_number`   | 숫자 리터럴 |

### 🔹 주요 전역 변수

- `std::string IdentifierStr` – 현재 토큰이 identifier일 경우 이름 저장
- `double NumVal` – 현재 토큰이 숫자일 경우 값 저장

---

## ✏️ 3. `gettok()` 함수의 동작

소스 코드를 한 글자씩 읽으며, 다음을 수행:

1. 공백 스킵 (`isspace`)
2. 알파벳 시작 → identifier or keyword
3. 숫자 or `.` 시작 → 부동소수점 파싱
4. `#` 시작 → 주석 (줄 끝까지 무시)
5. 나머지 → 아스키 값 그대로 반환 (e.g. `+`, `*`, `(` 등)

### 예시:

```cpp
if (isalpha(LastChar)) {
    IdentifierStr = LastChar;
    ...
    if (IdentifierStr == "def") return tok_def;
}
```

---

## 🧪 4. 테스트

소스 코드를 직접 입력하면, `gettok()`이 토큰을 하나씩 반환하며 출력된다.

### 예시 입력:
```
def foo(x y) x + y * 2;
```

### 출력 (토큰 스트림 예시):
```
tok_def, tok_identifier(foo), '(', tok_identifier(x), tok_identifier(y), ')', ...
```

---

## 🧠 요점 정리

- Kaleidoscope는 **수식 중심의 간단한 언어**
- Chapter 1에서는 **문자열 입력을 토큰으로 변환하는 lexer 구현**에 집중
- 핵심 함수: `gettok()`
- 토큰 종류를 enum으로 구분하며 전역변수에 부가 정보 저장

---
