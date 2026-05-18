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

// LogicalResult AddOp::inferReturnTypes(
//     MLIRContext *context,
//     std::optional<Location> location,
//     ValueRange operands,
//     DictionaryAttr attributes,
//     OpaqueProperties properties,
//     RegionRange regions,
//     SmallVectorImpl<Type> &inferredReturnTypes)  {

//   if (operands.size() != 2)
//     return failure();

//   Type lhsType = operands[0].getType();
//   Type rhsType = operands[1].getType();

//   if (lhsType != rhsType)
//     return failure();

//   inferredReturnTypes.push_back(lhsType);

//   return success();
// }

// LogicalResult AddOp::verify() {
//   Type type = getResult().getType();

//   if (type.isIntOrFloat())
//     return success();

//   if (auto tensorType = llvm::dyn_cast<TensorType>(type)) {
//     if (!tensorType.getElementType().isIntOrFloat())
//       return emitOpError("tensor element type must be numeric");

//     return success();
//   }

//   return emitOpError("unsupported type");
// }
