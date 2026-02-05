#include "lib/Transform/Arith/MulToAdd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

    struct PowerOfTwoExpand : public OpRewritePattern<arith::MulIOp> {
        PowerOfTwoExpand(mlir::MLIRContext *context) : OpRewritePattern<arith::MulIOp>(context, 2) {}
        
        LogicalResult matchAndRewrite(arith::MulIOp op, PatternRewriter &rewriter) const override {
            Value lhs = op.getOperand(0);
            Value rhs = op.getOperand(1);

            // check rhs is ConstantIntOp type
            auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
            if (!rhsDefiningOp) {
                return failure();
            }

            // bring rhs value and if not multiple of 2, return failure
            int64_t value = rhsDefiningOp.value();
            bool is_power_of_two = (value & (value - 1)) == 0;
            if (!is_power_of_two) {
                return failure();
            }

            // create a new constant with x/2 value
            arith::ConstantOp newConstant = rewriter.create<arith::ConstantOp> (rhsDefiningOp.getLoc(), rewriter.getIntegerAttr(rhs.getType(), value/2));

            arith::MulIOp newMul = rewriter.create<arith::MulIOp> (op.getLoc(), lhs, newConstant);
            arith::AddIOp newAdd = rewriter.create<arith::AddIOp> (op.getLoc(), newMul, newMul);

            rewriter.replaceOp(op, newAdd);
            rewriter.eraseOp(rhsDefiningOp);

            return success();

        };
    };

    struct PeelFromMul : public OpRewritePattern<arith::MulIOp> {
        PeelFromMul(mlir::MLIRContext *context) : OpRewritePattern<arith::MulIOp>(context, 1) {}

        LogicalResult matchAndRewrite(arith::MulIOp op, PatternRewriter &rewriter) const override {
            Value lhs = op.getOperand(0);
            Value rhs = op.getOperand(1);

            // check whether rhs is constant
            auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
            if (!rhsDefiningOp) {
                return failure();
            }

            int64_t value = rhsDefiningOp.value();
            arith::ConstantOp newConstant = rewriter.create<arith::ConstantOp> (rhsDefiningOp.getLoc(), rewriter.getIntegerAttr(rhs.getType(), value-1));

            arith::MulIOp newMul = rewriter.create<arith::MulIOp> (op.getLoc(), lhs, newConstant);
            arith::AddIOp newAdd = rewriter.create<arith::AddIOp> (op.getLoc(), newMul, lhs);

            rewriter.replaceOp(op, newAdd);
            rewriter.eraseOp(rhsDefiningOp);

            return success();
        }
    };

    void MulToAddPass::runOnOperation() {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<PowerOfTwoExpand>(&getContext());
        patterns.add<PeelFromMul>(&getContext());
        (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
}
}