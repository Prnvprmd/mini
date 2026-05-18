
include "mlir/IR/DialectBase.td"

def Mini_Dialect : Dialect {
  let name = "mini";
  let cppNamespace = "::mini";
}
