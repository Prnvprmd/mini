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

struct AddLoweringToScalarizedIR : public OpRewritePattern<AddOp> {
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

struct MatmulLoweringToScalarizedIR : public OpRewritePattern<MatmulOp> {

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

template<typename miniOp, typename arithIOp, typename arithFOp>
struct BinaryOpLowering : public OpRewritePattern<miniOp> {
  using OpRewritePattern<miniOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miniOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType = op.getType();

    // Scalar integer
    if (isa<IntegerType>(resultType)) {
      rewriter.replaceOpWithNewOp<arithIOp>(op, op.getLhs(), op.getRhs());
      return success();
    }

    // Scalar float
    if (isa<FloatType>(resultType)) {
      rewriter.replaceOpWithNewOp<arithFOp>(op, op.getLhs(), op.getRhs());
      return success();
    }

    // Tensor
    auto tensorType =
        llvm::dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!tensorType)
      return failure();

    auto rank = tensorType.getRank();
    auto elemType = tensorType.getElementType();

    // Create output tensor
    SmallVector<Value> dynSizes;
    for (auto [idx, dim] : llvm::enumerate(tensorType.getShape())) {
      if (dim == ShapedType::kDynamic) {
        dynSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, op.getLhs(), idx));
      }
    }
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), elemType, dynSizes);

    // Identity indexing maps
    SmallVector<AffineMap> indexingMaps;
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    indexingMaps.push_back(identityMap);
    indexingMaps.push_back(identityMap);
    indexingMaps.push_back(identityMap);

    // All iterators are parallel
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Create linalg.generic
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, tensorType, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{initTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value lhs = blockArgs[0];
          Value rhs = blockArgs[1];
          Value sum;
          if (isa<FloatType>(elemType)) {
            sum = nestedBuilder.create<arithFOp>(nestedLoc, lhs, rhs);
          } else {
            sum = nestedBuilder.create<arithIOp>(nestedLoc, lhs, rhs);
          }
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResults());

    return success();
  }
};

using AddLowering = BinaryOpLowering<AddOp, arith::AddIOp, arith::AddFOp>;
using MulLowering = BinaryOpLowering<MulOp, arith::MulIOp, arith::MulFOp>;

struct MatmulLowering : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto lhsType = llvm::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = llvm::dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resultType =
        llvm::dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!lhsType || !rhsType || !resultType)
      return failure();

    int64_t rank = lhsType.getRank();
    if (rank < 2)
      return failure();

    auto elemType = resultType.getElementType();

    // Create output tensor
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

    // Fill with zeros
    Value zero;
    if (isa<FloatType>(elemType)) {
      zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemType, 0.0));
    } else {
      zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elemType, 0));
    }
    Value filled =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    // Build affine expressions
    int64_t numLoops = rank + 1;

    // Iteration space: [batch dims..., i, j, k]
    SmallVector<AffineExpr> dims;
    for (int64_t idx = 0; idx < numLoops; ++idx) {
      dims.push_back(rewriter.getAffineDimExpr(idx));
    }
    int64_t batchRank = rank - 2;

    //
    // Naming convention:
    //
    // batch dims:
    //   dims[0 ... batchRank-1]
    //
    // i:
    //   dims[batchRank]
    //
    // j:
    //   dims[batchRank + 1]
    //
    // k:
    //   dims[batchRank + 2]
    //
    AffineExpr iExpr = dims[batchRank];
    AffineExpr jExpr = dims[batchRank + 1];
    AffineExpr kExpr = dims[batchRank + 2];

    // Build lhs indexing map: [batch..., i, k]
    SmallVector<AffineExpr> lhsExprs;
    // Build rhs indexing map: [batch..., k, j]
    SmallVector<AffineExpr> rhsExprs;
    // Build output indexing map: [batch..., i, j]
    SmallVector<AffineExpr> outExprs;
    for (int64_t b = 0; b < batchRank; ++b) {
      lhsExprs.push_back(dims[b]);
      rhsExprs.push_back(dims[b]);
      outExprs.push_back(dims[b]);
    }
    lhsExprs.push_back(iExpr);
    lhsExprs.push_back(kExpr);
    rhsExprs.push_back(kExpr);
    rhsExprs.push_back(jExpr);
    outExprs.push_back(iExpr);
    outExprs.push_back(jExpr);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(numLoops, 0, lhsExprs, rewriter.getContext()),
        AffineMap::get(numLoops, 0, rhsExprs, rewriter.getContext()),
        AffineMap::get(numLoops, 0, outExprs, rewriter.getContext())};

    // Iterator types
    SmallVector<utils::IteratorType> iteratorTypes;
    for (int64_t b = 0; b < batchRank; ++b) {
      iteratorTypes.push_back(utils::IteratorType::parallel);
    }

    iteratorTypes.push_back(utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);

    // Create linalg.generic
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getLhs(), op.getRhs()},
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
    patterns.add<MulLowering>(context);
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
