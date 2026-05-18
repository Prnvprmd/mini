add_mlir_dialect_library(MLIRMini

    Ops.cpp
    Dialect.cpp
    Canonicalization.cpp

    DEPENDS
    MiniDialectIncGen
    MiniOpsIncGen
    # MiniShapeInferenceInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRCastInterfaces
)