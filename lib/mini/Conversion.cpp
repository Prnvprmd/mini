#include "mini/Ops.h"
#include "mini/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mini;

namespace {

struct AddLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto opType = op.getType();

    if (isa<FloatType>(opType)) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getLhs(), op.getRhs());
      return success();
    }

    if (isa<IntegerType>(opType)) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, op.getLhs(), op.getRhs());
      return success();
    }

    auto tensorType = llvm::dyn_cast<RankedTensorType>(opType);
    if (!tensorType)
      return failure();

    auto elemType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    Value result = rewriter.create<tensor::EmptyOp>(loc, shape, elemType);
    int64_t rank = tensorType.getRank();

    if (rank == 1) {
      int64_t N = shape[0];
      for (int64_t i = 0; i < N; ++i) {
        Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);

        Value lhs = rewriter.create<tensor::ExtractOp>(loc, op.getLhs(),
                                                       ValueRange{iVal});
        Value rhs = rewriter.create<tensor::ExtractOp>(loc, op.getRhs(),
                                                       ValueRange{iVal});

        Value sum;
        if (isa<FloatType>(elemType)) {
          sum = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
        } else {
          sum = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
        }
        result = rewriter.create<tensor::InsertOp>(loc, sum, result,
                                                   ValueRange{iVal});
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    if (rank == 2) {
      int64_t M = shape[0];
      int64_t N = shape[1];
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value jVal = rewriter.create<arith::ConstantIndexOp>(loc, j);

          Value lhs = rewriter.create<tensor::ExtractOp>(
              loc, op.getLhs(), ValueRange{iVal, jVal});

          Value rhs = rewriter.create<tensor::ExtractOp>(
              loc, op.getRhs(), ValueRange{iVal, jVal});

          Value sum;
          if (isa<FloatType>(elemType)) {
            sum = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
          } else {
            sum = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
          }
          result = rewriter.create<tensor::InsertOp>(loc, sum, result,
                                                     ValueRange{iVal, jVal});
        }
      }
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
};

struct MatmulLowering : public OpRewritePattern<MatmulOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto lhsType = llvm::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = llvm::dyn_cast<RankedTensorType>(op.getRhs().getType());

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
      return failure();
    }
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];

    auto elemType = lhsType.getElementType();

    auto resultType = RankedTensorType::get({M, N}, elemType);

    Value result = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{M, N}, elemType);

    Value zero;

    if (isa<FloatType>(elemType)) {
      zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemType, 0.0));
    } else {
      zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elemType, 0));
    }

    // Fill tensor with zeros.
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
        Value jVal = rewriter.create<arith::ConstantIndexOp>(loc, j);

        result = rewriter.create<tensor::InsertOp>(loc, zero, result,
                                                   ValueRange{iVal, jVal});
      }
    }

    // Accumulate products.
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        Value acc;

        Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
        Value jVal = rewriter.create<arith::ConstantIndexOp>(loc, j);

        acc = rewriter.create<tensor::ExtractOp>(loc, result,
                                                 ValueRange{iVal, jVal});

        for (int64_t k = 0; k < K; ++k) {
          Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, k);

          Value lhs = rewriter.create<tensor::ExtractOp>(
              loc, op.getLhs(), ValueRange{iVal, kVal});
          Value rhs = rewriter.create<tensor::ExtractOp>(
              loc, op.getRhs(), ValueRange{kVal, jVal});

          Value mul;

          if (isa<FloatType>(elemType)) {
            mul = rewriter.create<arith::MulFOp>(loc, lhs, rhs);
            acc = rewriter.create<arith::AddFOp>(loc, acc, mul);
          } else {
            mul = rewriter.create<arith::MulIOp>(loc, lhs, rhs);
            acc = rewriter.create<arith::AddIOp>(loc, acc, mul);
          }
        }
        result = rewriter.create<tensor::InsertOp>(loc, acc, result,
                                                   ValueRange{iVal, jVal});
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

namespace {

struct LowerMiniToAffinePass
    : public PassWrapper<LowerMiniToAffinePass, OperationPass<func::FuncOp>> {

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<AddLowering>(context);
    patterns.add<MatmulLowering>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                tensor::TensorDialect, memref::MemRefDialect>();
  }

  llvm::StringRef getArgument() const final { return "lower-mini-to-affine"; }

  llvm::StringRef getDescription() const final {
    return "Lower mini dialect to affine/arithmetic dialects";
  }
};

} // namespace

std::unique_ptr<Pass> mini::createLowerMiniToAffinePass() {
  return std::make_unique<LowerMiniToAffinePass>();
}

void mini::registerMiniPasses() { PassRegistration<LowerMiniToAffinePass>(); }
