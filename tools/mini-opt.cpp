#include "mini/Dialect.h"
#include "mini/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  registry.insert<mini::MiniDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect>();

  registerTransformsPasses();

  mini::registerMiniPasses();

  return failed(MlirOptMain(argc, argv, "mini MLIR optimizer\n", registry));
}