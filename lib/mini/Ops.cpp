#include "mini/Ops.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"


using namespace mlir;
using namespace mini;

#define GET_OP_CLASSES
#include "mini/Ops.cpp.inc"


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder,
                  mlir::OperationState &state,
                  mlir::Value lhs,
                  mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType());
}

LogicalResult AddOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  // Operand types must match
  if (lhsType != rhsType)
    return emitOpError("operand types must match");

  // Result type must match operands
  if (lhsType != resultType)
    return emitOpError("result type must match operand type");

  // Scalars
  if (lhsType.isIntOrFloat())
    return success();

  // Tensors
  if (auto tensorType = llvm::dyn_cast<TensorType>(lhsType)) {
    Type elemType = tensorType.getElementType();

    if (!elemType.isIntOrFloat())
      return emitOpError("tensor element type must be int or float");

    return success();
  }

  return emitOpError("unsupported type");
}



//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder,
                  mlir::OperationState &state,
                  mlir::Value lhs,
                  mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType());
}


//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//


static bool isCompatibleBatchDims(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i])
      return false;
  }

  return true;
}


LogicalResult MatmulOp::verify() {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());

  if (!lhsType || !rhsType || !resultType)
    return emitOpError("requires ranked tensor operands/results");

  if (lhsType.getRank() < 2)
    return emitOpError("lhs must have rank >= 2");

  if (rhsType.getRank() < 2)
    return emitOpError("rhs must have rank >= 2");

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto resultShape = resultType.getShape();

  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();

  if (lhsRank != rhsRank)
    return emitOpError("lhs/rhs rank mismatch");

  int64_t lhsK = lhsShape[lhsRank - 1];
  int64_t rhsK = rhsShape[rhsRank - 2];

  if (lhsK != rhsK)
    return emitOpError("contracting dimensions mismatch");

  ArrayRef<int64_t> lhsBatch =
      lhsShape.drop_back(2);

  ArrayRef<int64_t> rhsBatch =
      rhsShape.drop_back(2);

  if (!isCompatibleBatchDims(lhsBatch, rhsBatch))
    return emitOpError("batch dimensions mismatch");

  SmallVector<int64_t> expectedResultShape;

  expectedResultShape.append(lhsBatch.begin(),
                             lhsBatch.end());

  expectedResultShape.push_back(lhsShape[lhsRank - 2]);
  expectedResultShape.push_back(rhsShape[rhsRank - 1]);

  if (expectedResultShape != resultShape)
    return emitOpError("incorrect result shape");

  if (lhsType.getElementType() != rhsType.getElementType())
    return emitOpError("element types must match");

  if (lhsType.getElementType() !=
      resultType.getElementType())
    return emitOpError("result element type mismatch");

  return success();
}

