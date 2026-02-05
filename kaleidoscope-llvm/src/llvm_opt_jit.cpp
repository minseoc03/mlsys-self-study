#include "../include/KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

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

            //llvm codegen 
            virtual Value *codegen()  = 0; // abstract (= 0)
    };

    /// NumberExprAST - Expression class for numeric literal
    class NumberExprAST : public ExprAST {
        double Val;
        
        public:
            NumberExprAST(double Val) : Val(Val) {} //instance variable for storing number value
            Value *codegen() override;
    };

    /// VariableExprAST - Expression class for variable
    class VariableExprAST : public ExprAST {
        std::string Name;

        public:
            VariableExprAST(const std::string &Name) : Name(Name) {} //instance variable for storing string value
            Value *codegen() override;
        };

    /// BinaryExprAST - Expression class for binary operation
    class BinaryExprAST : public ExprAST {
        char Op; //operation (e.g. '+')
        std::unique_ptr<ExprAST> LHS, RHS; //smart pointer (cause it's dynamic)

        public :
            BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS) 
                : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
            Value *codegen() override;
    };

    /// CallExprAST - Expression class for function calls
    class CallExprAST : public ExprAST {
        std::string Callee;
        std::vector<std::unique_ptr<ExprAST>> Args; //smart pointer (cause it's dynamic)

        public:
            CallExprAST(const std::string &Callee, std::vector<std::unique_ptr<ExprAST>> Args)
                : Callee(Callee), Args(std::move(Args)) {}
            Value *codegen() override;
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
            
            Function *codegen();
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
            Function *codegen();
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
    }
    return TokPrec;
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
// Code Generation
//---------------------------------------

static std::unique_ptr<LLVMContext> TheContext; // core libraries of llvm / backbone of the compiler
static std::unique_ptr<Module> TheModule; // contains the modules of the code
static std::unique_ptr<IRBuilder<>> Builder; // containes helper functions to generate IR
static std::map<std::string, Value *> NamedValues; // contains variables' names and values of the code
static std::unique_ptr<KaleidoscopeJIT> TheJIT; // user-defined JIT / compile IR into machine code

// PM (PassManager) -> what directly mutate IR to optimize
// AM (AnalysisManager) -> what looks over the code and suggest way to optimize to PM
static std::unique_ptr<FunctionPassManager> TheFPM; // optimizes IR in function level (e.g. dead code elimination)
static std::unique_ptr<LoopAnalysisManager> TheLAM; // analyze IR in loop level (e.g. help for loop unrolling)
static std::unique_ptr<FunctionAnalysisManager> TheFAM; // analyze IR in function level (e.g. detect no argument function)
static std::unique_ptr<CGSCCAnalysisManager> TheCGAM; // analyze IR in inter-function level (e.g. look over function callings)
static std::unique_ptr<ModuleAnalysisManager> TheMAM; // analyze IR in module level (e.g. number of global variables)
static std::unique_ptr<PassInstrumentationCallbacks> ThePIC; // callbacks when opt pass is on (for debugging)
static std::unique_ptr<StandardInstrumentations> TheSI; // debugging tools

// global map that stores name of function and its prototype
// will be used for calling functions from separate modules
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr; // for error handling

// Error handling in IR code generation
Value *LogErrorV(const char *Str) {
    LogError(Str);
    return nullptr;
}

// custom getFunction method to call function from other modules
Function *getFunction(std::string Name) {
    // check if the function has been added to the current module
    if (auto *F = TheModule->getFunction(Name)) {
        return F;
    }
    // check if we can codegen the declaration using existing prototype
    auto FI = FunctionProtos.find(Name);
    if (FI != FunctionProtos.end()) {
        return FI->second->codegen();
    }

    // if function is not in FunctionProtos, return null as an invalid funtion name
    return nullptr;
}

// Assign number to SSA register
Value *NumberExprAST::codegen() {
    return ConstantFP::get(*TheContext, APFloat(Val)); //APFloat is a floating number with stability
}

// Find value of variable from NamedValues
Value *VariableExprAST::codegen() {
    Value *V = NamedValues[Name];
    if (!V) {
        return LogErrorV("Unknown variable name");
    }
    return V;
}

// Binary Operations into IR
Value *BinaryExprAST::codegen() {
    // Assign LHS and RHS to SSA register
    Value *L = LHS->codegen();
    Value *R = RHS->codegen();
    if (!L || !R) {
        return nullptr;
    }

    // Assign computations from IRBuilder
    switch (Op) {
        case '+':
            return Builder->CreateFAdd(L, R, "addtmp");
        case '-':
            return Builder->CreateFSub(L, R, "subtmp");
        case '*':
            return Builder->CreateFMul(L, R, "multmp");
        case '<':
            return Builder->CreateFCmpULT(L, R, "cmptmp");
            //Convert Bool int value to float
            return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
        default:
            return LogErrorV("invalid binary operator");
    }
}

// Calling Functions
Value *CallExprAST::codegen() {
    // Lookup function name in the global module table
    Function *CalleeF = getFunction(Callee);
    if (!CalleeF) {
        return LogErrorV("Unknown function referenced");
    }
    // if argument mismatch error
    if (CalleeF->arg_size() != Args.size()) {
        return LogErrorV("Incorrect # arguments passed");
    }

    // Assign SSA register to each arguments
    std::vector<Value *> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
        ArgsV.push_back(Args[i]->codegen());
        if (!ArgsV.back()) {
            return nullptr;
        }
    }
    // build function call in IR
    return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

// Prototype
// returns Function pointer (not Value pointer)
Function *PrototypeAST::codegen() {
    // Make the function type: double(double, double) etc.
    std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
    FunctionType *FT = FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);

    // Create function in IR
    Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

    // Set names for all arguments (optional)
    unsigned Idx = 0;
    for (auto &Arg : F->args()) {
        Arg.setName(Args[Idx++]);
    }

    return F;
}

// Function (prototype + body)
Function *FunctionAST::codegen() {
    auto &P = *Proto; // Proto : PrototypeAST instance
    FunctionProtos[Proto->getName()] = std::move(Proto); //transfer ownership of Proto node to FunctionProtos global map
    Function *TheFunction = getFunction(P.getName()); // make reference for below use
    if (!TheFunction) {
        return nullptr;
    }

    // Create a basic block for a function
    // basic block is a node for CFG (Control Flow Graph) of LLVM
    // it only has one entrance and one exit, so control flow can be recognized easily and directly
    BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map
    NamedValues.clear(); //clear because at this stage variables are only defined as arguments of the function
    for (auto &Arg : TheFunction->args()) {
        NamedValues[std::string(Arg.getName())] = &Arg;
    }

    if (Value *RetVal = Body->codegen()) {
        // Finish the function
        Builder->CreateRet(RetVal);

        // Validate the function
        verifyFunction(*TheFunction);

        // Run the optimizer on the function.
        TheFPM->run(*TheFunction, *TheFAM);

        return TheFunction;
    }

    // Error handling by erasing function
    TheFunction->eraseFromParent();
    return nullptr;
}

//---------------------------------------
// Top-Level Parsing and JIT Driver
//---------------------------------------

static void InitializeModuleAndManagers() {
    // Open a new context and module.
    TheContext = std::make_unique<LLVMContext>();
    TheModule = std::make_unique<Module>("KaleidoscopeJIT", *TheContext);
    TheModule->setDataLayout(TheJIT->getDataLayout());
  
    // Create a new builder for the module.
    Builder = std::make_unique<IRBuilder<>>(*TheContext);

    // Create new pass and analysis managers
    TheFPM = std::make_unique<FunctionPassManager>();
    TheLAM = std::make_unique<LoopAnalysisManager>();
    TheFAM = std::make_unique<FunctionAnalysisManager>();
    TheCGAM = std::make_unique<CGSCCAnalysisManager>();
    TheMAM = std::make_unique<ModuleAnalysisManager>();
    ThePIC = std::make_unique<PassInstrumentationCallbacks>();
    TheSI = std::make_unique<StandardInstrumentations>(*TheContext, true /*debuglogging*/);

    // register all the debugging tools into PIC
    // automaticaly runs debug during IR pass around the module (TheMAM.get())
    TheSI->registerCallbacks(*ThePIC, TheMAM.get());

    // Add optimization passes
    // Do simple optimzations
    TheFPM->addPass(InstCombinePass());
    // Reassociate expressions
    TheFPM->addPass(ReassociatePass());
    // Eliminate common subexpressions
    TheFPM->addPass(GVNPass());
    // Simplify the control flow graph
    TheFPM->addPass(SimplifyCFGPass());
    //
    //
    //... add more passes

    // Connect analysis passes to transform passes
    // thus PM can mutate IR based on analysis from AM
    PassBuilder PB;
    PB.registerModuleAnalyses(*TheMAM);
    PB.registerFunctionAnalyses(*TheFAM);
    PB.crossRegisterProxies(*TheLAM, *TheFAM, *TheCGAM, *TheMAM);

  }
  
  static void HandleDefinition() {
    if (auto FnAST = ParseDefinition()) {
      if (auto *FnIR = FnAST->codegen()) {
        fprintf(stderr, "Read function definition:");
        FnIR->print(errs());
        fprintf(stderr, "\n");
        // Add currnet module onto JIT engine
        // calling functions are now possible around different modules
        ExitOnErr(TheJIT->addModule(ThreadSafeModule(std::move(TheModule), std::move(TheContext))));
        InitializeModuleAndManagers();
      }
    } else {
      // Skip token for error recovery.
      getNextToken();
    }
  }
  
  static void HandleExtern() {
    if (auto ProtoAST = ParseExtern()) {
      if (auto *FnIR = ProtoAST->codegen()) {
        fprintf(stderr, "Read extern: ");
        FnIR->print(errs());
        fprintf(stderr, "\n");
        // Add prototype of external function into FunctionProtos global map
        FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
      }
    } else {
      // Skip token for error recovery.
      getNextToken();
    }
  }
  
  static void HandleTopLevelExpression() {
    // Evaluate a top-level expression into an anonymous function.
    if (auto FnAST = ParseTopLevelExpr()) {
      if (FnAST->codegen()) {
        // Create a ResourceTracker to track JIT'd memory allocated to our
        // anonymous expression -- that way we can free it after executing.
        auto RT = TheJIT->getMainJITDylib().createResourceTracker();
  
        auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
        ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
        InitializeModuleAndManagers();
  
        // Search the JIT for the __anon_expr symbol.
        auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));
  
        // Get the symbol's address and cast it to the right type (takes no
        // arguments, returns a double) so we can call it as a native function.
        double (*FP)() = ExprSymbol.toPtr<double (*)()>();
        fprintf(stderr, "Evaluated to %f\n", FP());
  
        // Delete the anonymous expression module from the JIT.
        ExitOnErr(RT->remove());
      }
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
// "Library" functions that can be "extern'd" from user code.
//---------------------------------------

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}

//---------------------------------------
// Main Driver Code
//---------------------------------------
  
int main() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
    // Install standard binary operators.
    // 1 is lowest precedence.
    BinopPrecedence['<'] = 10;
    BinopPrecedence['+'] = 20;
    BinopPrecedence['-'] = 20;
    BinopPrecedence['*'] = 40; // highest.
  
    // Prime the first token.
    fprintf(stderr, "ready> ");
    getNextToken();
  
    TheJIT = ExitOnErr(KaleidoscopeJIT::Create());

    // Make the module, which holds all the code.
    InitializeModuleAndManagers();
  
    // Run the main "interpreter loop" now.
    MainLoop();
  
    return 0;
}