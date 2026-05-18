#include "mini/Dialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  registry.insert<mini::MiniDialect,
                  mlir::func::FuncDialect,
                  mlir::arith::ArithDialect>();

  registerTransformsPasses();

  return failed(MlirOptMain(
      argc,
      argv,
      "mini MLIR optimizer\n",
      registry));
}