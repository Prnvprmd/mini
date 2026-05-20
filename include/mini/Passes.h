#pragma once

#include "mlir/Pass/Pass.h"

namespace mini {

std::unique_ptr<mlir::Pass> createLowerMiniToAffinePass();

void registerMiniPasses();

} // namespace mini
