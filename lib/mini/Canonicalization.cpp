#include "mini/Ops.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mini;

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());

  if (!lhs || !rhs)
    return {};

  APInt result = lhs.getValue() + rhs.getValue();

  return IntegerAttr::get(lhs.getType(), result);
}

namespace {

struct AddZeroFoldPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {

    APInt value;

    if (matchPattern(op.getRhs(), m_ConstantInt(&value)) && value.isZero()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    if (matchPattern(op.getLhs(), m_ConstantInt(&value)) && value.isZero()) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    return failure();
  }
};

} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {

  results.add<AddZeroFoldPattern>(context);
}