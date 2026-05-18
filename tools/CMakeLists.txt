add_llvm_executable(mini-opt
  mini-opt.cpp
)

llvm_update_compile_flags(mini-opt)

target_link_libraries(mini-opt
  PRIVATE

  MLIRMini

  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms

  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect

  MLIROptLib
)

mlir_check_all_link_libraries(mini-opt)