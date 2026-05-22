include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"
include "mini/Dialect.td"


class Mini_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mini_Dialect, mnemonic, traits> {
}


def Mini_ScalarOrTensor :
    TypeConstraint<
      Or<[
        AnyFloat.predicate,
        AnyInteger.predicate,
        TensorOf<[AnyFloat, AnyInteger]>.predicate
      ]>,
      "scalar or tensor of int/float"
    >;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Mini_Op<"add", [Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "element-wise addition operation";

  let arguments = (ins Mini_ScalarOrTensor:$lhs,
                       Mini_ScalarOrTensor:$rhs);

  let results = (outs Mini_ScalarOrTensor:$result);

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;

//   let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
//   let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs,
                   "::mlir::Value":$rhs)>
  ];
}


//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Mini_Op<"mul", [Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "element-wise multiplication operation";

  let arguments = (ins Mini_ScalarOrTensor:$lhs,
                       Mini_ScalarOrTensor:$rhs);

  let results = (outs Mini_ScalarOrTensor:$result);

  let hasVerifier = 1;
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs,
                   "::mlir::Value":$rhs)>
  ];
}


//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : Mini_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication operation";

  let arguments = (ins TensorOf<[AnyFloat, AnyInteger]>:$lhs,
                       TensorOf<[AnyFloat, AnyInteger]>:$rhs);

  let results = (outs TensorOf<[AnyFloat, AnyInteger]>:$result);

  let hasVerifier = 1;

  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

