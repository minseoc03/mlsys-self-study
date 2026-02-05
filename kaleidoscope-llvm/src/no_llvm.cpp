#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//---------------------------------------
// Lexer
//---------------------------------------

//enum of pre-defined tokens
//undefined characters will be tokenzied into ascii
enum Token {
    tok_eof = -1,
    tok_def = -2,
    tok_extern = -3,
    tok_identifier = -4,
    tok_number = -5
};

static std::string IdentifierStr; //will store identifier (string itself, not token)
static double NumVal; //will store number value (value itself, not token)

//return a token from standard input
static int gettok() {
    static int LastChar = ' ';

    //ignore whitespace
    while (isspace(LastChar)) {
        LastChar = getchar();
    }

    //identifier lexer
    //first character : [a-zA-Z]
    //after that : [a-zA-Z0-9]
    if (isalpha(LastChar)) {
        IdentifierStr = LastChar;
        while (isalnum(LastChar = getchar())) {
            IdentifierStr += LastChar;
        }

        //return pre-defined tokens
        if (IdentifierStr == "def") {
            return tok_def;
        }
        if (IdentifierStr == "extern") {
            return tok_extern;
        }
        return tok_identifier;
    }

    //number lexer
    //include all numbers and dot in case of float number
    if (isdigit(LastChar) || LastChar == '.') {
        std::string NumStr;
        do {
            NumStr += LastChar;
            LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.');

        NumVal = strtod(NumStr.c_str(), nullptr);
        return tok_number;
    }

    //comment lexer
    if (LastChar == '#') {
        do {
            LastChar = getchar();
        } while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if (LastChar != EOF) {
            return gettok();
        }
    }

    //Check for end of file
    if (LastChar == EOF) {
        return tok_eof;
    }

    //Otherwise return character in ascii value
    int ThisChar = LastChar;
    //for lookahead buffer
    LastChar = getchar();
    return ThisChar;
}

//---------------------------------------
// Abstract Syntax Tree
//---------------------------------------
namespace {
    /// ExprAST - base class for expression nodes
    class ExprAST {
        public : 
            // destructor
            virtual ~ExprAST() = default;
    };

    /// NumberExprAST - Expression class for numeric literal
    class NumberExprAST : public ExprAST {
        double Val;
        
        public:
            NumberExprAST(double Val) : Val(Val) {} //instance variable for storing number value
    };

    /// VariableExprAST - Expression class for variable
    class VariableExprAST : public ExprAST {
        std::string Name;

        public:
            VariableExprAST(const std::string &Name) : Name(Name) {} //instance variable for storing string value
    };

    /// BinaryExprAST - Expression class for binary operation
    class BinaryExprAST : public ExprAST {
        char Op; //operation (e.g. '+')
        std::unique_ptr<ExprAST> LHS, RHS; //smart pointer (cause it's dynamic)

        public :
            BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS) 
                : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    };

    /// CallExprAST - Expression class for function calls
    class CallExprAST : public ExprAST {
        std::string Callee;
        std::vector<std::unique_ptr<ExprAST>> Args; //smart pointer (cause it's dynamic)

        public:
            CallExprAST(const std::string &Callee, std::vector<std::unique_ptr<ExprAST>> Args)
                : Callee(Callee), Args(std::move(Args)) {}
    };

    /// PrototypeAST - This class represents the "prototype" for a function,
    /// which captures its name, and its argument names (thus implicitly the number
    /// of arguments the function takes).
    class PrototypeAST {
        std::string Name;
        std::vector<std::string> Args;

        public :
            PrototypeAST(const std::string &Name, std::vector<std::string> Args)
                : Name(Name), Args(std::move(Args)) {}
            
            const std::string &getName() const { return Name; }
    };

    /// FunctionAST - This class represents a function definition itself.
    class FunctionAST {
        std::unique_ptr<PrototypeAST> Proto;
        std::unique_ptr<ExprAST> Body;
    
        public :
            FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                        std::unique_ptr<ExprAST> Body)
                : Proto(std::move(Proto)), Body(std::move(Body)) {}
    };
}

//---------------------------------------
// Parser
//---------------------------------------

/// CurTok/getNextToken - Provide a simple token buffer.
/// CurTok : current token the parser is looking at.
/// getNextToken() : reads another token based on lexer above and
/// update CurTok with its result
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }

/// BinopPrecedence : holds the precdence for each binary operator in map structure
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence : Get the precedence of the current token.
static int GetTokPrecedence() {
    if (!isascii(CurTok)) {
        return -1;
    }

    // make sure binary operation are in the map
    int TokPrec = BinopPrecedence[CurTok];
    if (TokPrec <= 0) {
        return -1;
    return TokPrec;
    }
}

/// LogError - helper functions for error handling
std::unique_ptr<ExprAST> LogError(const char *Str) {
    fprintf(stderr, "Error: %s\n", Str);
    return nullptr;
}
std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
    LogError(Str);
    return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// Parsing numbers
static std::unique_ptr<ExprAST> ParseNumberExpr() {
    // create a node for number with its value using global variable, NumVal
    // NumVal contains current token's number value if current token is numeric literal
    auto Result = std::make_unique<NumberExprAST>(NumVal);
    getNextToken();
    return std::move(Result);
}

/// Parsing parenthesis
static std::unique_ptr<ExprAST> ParseParenExpr() {
    getNextToken(); //eat '('
    auto V = ParseExpression();
    if (!V) {
        // if this is the case (e.g. function calls)
        // ')' will be eaten at the top-level parsing
        return nullptr;
    }

    if (CurTok != ')') {
        return LogError("expected ')'");
    }
    getNextToken(); //eat ')'
    return V;
}

/// Parsing identifers
/// case 1 : simple variable declaration
/// case 2 : function call
static std::unique_ptr<ExprAST> ParseIdentiferExpr() {
    std::string IdName = IdentifierStr; //global variable containing identifer's data

    getNextToken(); //eat identifer

    if (CurTok != '(') { //case 1
        return std::make_unique<VariableExprAST>(IdName);
    }

    //case 2
    getNextToken(); //eat '('
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (CurTok != ')') {
        while (true) {
            //parse argument
            if (auto Arg = ParseExpression()) {
                Args.push_back(std::move(Arg));
            }
            else {
                return nullptr;
            }

            if (CurTok == ')') {
                break; //end of argument
            }

            if (CurTok != ',') {
                return LogError("Expected ')' or ',' in argument.");
            }
            getNextToken(); //eat ','
        }
    }
    getNextToken(); //eat ')'

    return std::make_unique<CallExprAST> (IdName, std::move(Args));
}

/// primary
///  ::= identifier
///  ::= number
///  ::= parenthesis
static std::unique_ptr<ExprAST> ParsePrimary() {
    switch (CurTok) {
        default:
            return LogError("Unknown token when expecting an expression");
        case tok_identifier:
            return ParseIdentiferExpr();
        case tok_number:
            return ParseNumberExpr();
        case '(':
            return ParseParenExpr();
    }
}

/// binoprhs
/// parsing binary operations
///  ::= ('+', primary)
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS) {
    while (true) {
        //find current precedence of binary operation
        int TokPrec = GetTokPrecedence();

        //if current binop is less or equal to this binop, consume LHS
        //otherwise finish parsing here
        if (TokPrec < ExprPrec) {
            return LHS;
        }

        //store current binop and eat it
        int BinOp = CurTok;
        getNextToken();

        //parsing right primary data
        auto RHS = ParsePrimary();
        if (!RHS) {
            return nullptr;
        }

        //if next binop has higher precedence than current one,
        //merge current RHS with LHS and take new primary as RHS
        int NextPrec = GetTokPrecedence();
        if (TokPrec < NextPrec) {
            RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
            if (!RHS) {
                return nullptr;
            }
        }
        // merge LHS & RHS
        LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
    }
}

/// Expression
static std::unique_ptr<ExprAST> ParseExpression() {
    auto LHS = ParsePrimary();
    if (!LHS) {
        return nullptr;
    }
    //0 is entered to finish parsing if next token is not binop
    return ParseBinOpRHS(0, std::move(LHS));
}

/// Prototype
///  ::= id'('id')'
static std::unique_ptr<PrototypeAST> ParsePrototype() {
    if (CurTok != tok_identifier) {
        return LogErrorP("Expected function name in prototype");
    }
    std::string FnName = IdentifierStr;
    getNextToken(); //eat function name

    if (CurTok != '(') {
        return LogErrorP("Expected '(' in prototype");
    }

    std::vector<std::string> ArgNames;
    while (getNextToken() == tok_identifier) {
        ArgNames.push_back(IdentifierStr);
    }
    if (CurTok != ')') {
        return LogErrorP("Expected ')' in prototype");
    }

    //success
    getNextToken(); //eat ')'

    return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
    getNextToken(); //eat def
    auto Proto = ParsePrototype();
    if (!Proto) {
        return nullptr;
    }
    if (auto E = ParseExpression()) {
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
    if (auto E = ParseExpression()) {
        //make anonymous proto
        auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                    std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

/// external
static std::unique_ptr<PrototypeAST> ParseExtern() {
    getNextToken(); //eat Extern
    return ParsePrototype();
}

//---------------------------------------
// Top-Level Parsing
//---------------------------------------

static void HandleDefinition() {
    if (ParseDefinition()) {
      fprintf(stderr, "Parsed a function definition.\n");
    } else {
      // Skip token for error recovery.
      getNextToken();
    }
  }
  
  static void HandleExtern() {
    if (ParseExtern()) {
      fprintf(stderr, "Parsed an extern\n");
    } else {
      // Skip token for error recovery.
      getNextToken();
    }
  }
  
  static void HandleTopLevelExpression() {
    // Evaluate a top-level expression into an anonymous function.
    if (ParseTopLevelExpr()) {
      fprintf(stderr, "Parsed a top-level expr\n");
    } else {
      // Skip token for error recovery.
      getNextToken();
    }
  }
  
  /// top ::= definition | external | expression | ';'
  static void MainLoop() {
    while (true) {
      fprintf(stderr, "ready> ");
      switch (CurTok) {
      case tok_eof:
        return;
      case ';': // ignore top-level semicolons.
        getNextToken();
        break;
      case tok_def:
        HandleDefinition();
        break;
      case tok_extern:
        HandleExtern();
        break;
      default:
        HandleTopLevelExpression();
        break;
      }
    }
  }
  
//---------------------------------------
// Main Driver Code
//---------------------------------------
  
  int main() {
    // Install standard binary operators.
    // 1 is lowest precedence.
    BinopPrecedence['<'] = 10;
    BinopPrecedence['+'] = 20;
    BinopPrecedence['-'] = 20;
    BinopPrecedence['*'] = 40; // highest.
  
    // Prime the first token.
    fprintf(stderr, "ready> ");
    getNextToken();
  
    // Run the main "interpreter loop" now.
    MainLoop();
  
    return 0;
  }