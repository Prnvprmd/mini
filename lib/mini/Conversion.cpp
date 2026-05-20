#include "mini/Ops.h"
#include "mini/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
    auto resultType =
        llvm::dyn_cast<RankedTensorType>(op.getResult().getType());

    //
    // Restrict initially to rank-2
    //

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
      return failure();
    }
    auto elemType = resultType.getElementType();

    //
    // Create output tensor
    //

    SmallVector<Value> dynSizes;
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

    //
    // Fill output tensor with zeros
    //
    Value zero = arith::ConstantOp::create(rewriter, loc, elemType,
                                           rewriter.getZeroAttr(elemType));
    Value filled =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    //
    // Matmul indexing maps
    //
    AffineExpr i, j, k;
    bindDims(rewriter.getContext(), i, j, k);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(3, 0, {i, k}, rewriter.getContext()),
        AffineMap::get(3, 0, {k, j}, rewriter.getContext()),
        AffineMap::get(3, 0, {i, j}, rewriter.getContext())};

    //
    // Iterator types
    //
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};

    //
    // Create linalg.generic
    //
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{filled}, indexingMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value lhs = blockArgs[0];
          Value rhs = blockArgs[1];
          Value acc = blockArgs[2];

          Value mul;
          Value sum;

          if (isa<FloatType>(elemType)) {
            mul = nestedBuilder.create<arith::MulFOp>(nestedLoc, lhs, rhs);
            sum = nestedBuilder.create<arith::AddFOp>(nestedLoc, acc, mul);
          } else {
            mul = nestedBuilder.create<arith::MulIOp>(nestedLoc, lhs, rhs);
            sum = nestedBuilder.create<arith::AddIOp>(nestedLoc, acc, mul);
          }
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResults());
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
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    memref::MemRefDialect, linalg::LinalgDialect>();
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
