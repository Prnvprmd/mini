#include "mini/Ops.h"
#include "mini/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mini;


namespace {

/// Fuse:
///
/// %0 = linalg.generic ... { add }
/// %1 = linalg.generic ins(%0, ...) { mul }
///
/// into:
///
/// %fused = linalg.generic ... {
///   %a = add
///   %b = mul(%a, ...)
/// }
///
struct FuseElementwiseGenericOps : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {

    // // Only tensor semantics for simplicity.
    // if (!bufferization::hasTensorSemantics(consumer))
    //   return failure();

    // Require exactly one output/result.
    if (consumer.getNumDpsInits() != 1 || consumer->getNumResults() != 1)
      return failure();

    // Look for producer among consumer inputs.
    linalg::GenericOp producer;
    OpOperand *producerUse = nullptr;
    for (OpOperand *input : consumer.getDpsInputOperands()) {
      producer = input->get().getDefiningOp<linalg::GenericOp>();
      if (producer) {
        producerUse = input;
        break;
      }
    }

    if (!producer)
      return failure();

    // if (!bufferization::hasTensorSemantics(producer))
    //   return failure();

    if (producer.getNumDpsInits() != 1 || producer->getNumResults() != 1)
      return failure();

    // Only fuse elementwise ops.
    if (!isElementwise(producer) || !isElementwise(consumer))
      return failure();

    // Producer result must only be used by consumer.
    if (!producer.getResult(0).hasOneUse())
      return failure();

    // Iterators and maps must match exactly for this simple example.
    if (producer.getIndexingMapsArray() != consumer.getIndexingMapsArray())
      return failure();

    if (producer.getIteratorTypesArray() != consumer.getIteratorTypesArray())
      return failure();

    Location loc = consumer.getLoc();

    SmallVector<Value> fusedInputs;

    // Consumer inputs except producer result.
    for (OpOperand *input : consumer.getDpsInputOperands()) {
      if (input == producerUse)
        continue;
      fusedInputs.push_back(input->get());
    }

    // Add producer original inputs.
    for (Value v : producer.getDpsInputs())
      fusedInputs.push_back(v);

    SmallVector<Value> outputs(consumer.getDpsInits());

    // Rebuild indexing maps:
    //
    // consumer(non-produced-inputs)
    // + producer(inputs)
    // + output map
    //
    SmallVector<AffineMap> fusedMaps;

    auto consumerMaps = consumer.getIndexingMapsArray();
    auto producerMaps = producer.getIndexingMapsArray();

    // Consumer input maps excluding producer result.
    for (unsigned i = 0; i < consumer.getNumDpsInputs(); ++i) {
      if (&consumer->getOpOperand(i) == producerUse)
        continue;
      fusedMaps.push_back(consumerMaps[i]);
    }

    // Producer input maps.
    for (unsigned i = 0; i < producer.getNumDpsInputs(); ++i)
      fusedMaps.push_back(producerMaps[i]);

    // Output map.
    fusedMaps.push_back(consumerMaps.back());

    SmallVector<utils::IteratorType> iteratorTypes =
        consumer.getIteratorTypesArray();

    auto fusedOp = linalg::GenericOp::create(
        rewriter, loc, consumer.getResultTypes(), fusedInputs, outputs,
        fusedMaps, iteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          //
          // Argument layout:
          //
          // [consumer_non_producer_inputs,
          //  producer_inputs]
          //

          unsigned numConsumerExtraInputs = consumer.getNumDpsInputs() - 1;
          unsigned producerNumInputs = producer.getNumDpsInputs();
          SmallVector<Value> producerArgs(
              args.begin() + numConsumerExtraInputs,
              args.begin() + numConsumerExtraInputs + producerNumInputs);

          // Clone producer body.
          Block &producerBlock = producer.getRegion().front();

          IRMapping producerMap;

          for (auto [oldArg, newArg] :
               llvm::zip(producerBlock.getArguments(), producerArgs)) {
            producerMap.map(oldArg, newArg);
          }

          for (Operation &op : producerBlock.without_terminator()) {
            b.clone(op, producerMap);
          }

          Value produced =
              producerMap.lookup(producerBlock.getTerminator()->getOperand(0));

          // Build consumer args.
          SmallVector<Value> consumerArgs;

          unsigned consumerArgIdx = 0;

          for (unsigned i = 0; i < consumer.getNumDpsInputs(); ++i) {
            if (&consumer->getOpOperand(i) == producerUse) {
              consumerArgs.push_back(produced);
            } else {
              consumerArgs.push_back(args[consumerArgIdx++]);
            }
          }

          // Clone consumer body.
          Block &consumerBlock = consumer.getRegion().front();

          IRMapping consumerMap;

          for (auto [oldArg, newArg] :
               llvm::zip(consumerBlock.getArguments(), consumerArgs)) {
            consumerMap.map(oldArg, newArg);
          }

          for (Operation &op : consumerBlock.without_terminator()) {
            b.clone(op, consumerMap);
          }

          Value finalValue =
              consumerMap.lookup(consumerBlock.getTerminator()->getOperand(0));

          linalg::YieldOp::create(b, nestedLoc, finalValue);
        });

    rewriter.replaceOp(consumer, fusedOp.getResults());

    // Cleanup producer if dead.
    if (producer->use_empty())
      rewriter.eraseOp(producer);

    return success();
  }
};

} // namespace

namespace {

struct MiniFusionPass
    : public PassWrapper<MiniFusionPass, OperationPass<func::FuncOp>> {

  void runOnOperation() final {

    RewritePatternSet patterns(&getContext());

    patterns.add<FuseElementwiseGenericOps>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {

      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    memref::MemRefDialect, linalg::LinalgDialect>();
  }

  llvm::StringRef getArgument() const final { return "mini-fuse"; }

  llvm::StringRef getDescription() const final {
    return "Fuse linalg.generic ops";
  }
};

} // namespace

void mini::registerMiniFusionPasses() { PassRegistration<MiniFusionPass>(); }
