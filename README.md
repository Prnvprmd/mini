# MiniDialect

And experimental custom MLIR dialect to understand basic concepts of MLIR, similar to ToyDialect from the MLIR docs.

## Build Commands

```sh
cmake -B build -S . -DMLIR_DIR=${...} -DLLVM_DIR=${...}
cd build
make
```

## Run tests

Run after adding AddOp or MatmulOp. Tests input shapes and dtypes
```sh
${PROJECT_SOURCE_DIR}/build/tools/mini-opt basic.mlir
```

Run after adding Canonicalization Patterns(folding a + 0 = a)
```sh
${PROJECT_SOURCE_DIR}/build/tools/mini-opt --canonicalize basic.mlir
```

Run after adding Lowering functions for AddOp and MatmulOp
```sh
${PROJECT_SOURCE_DIR}/build/tools/mini-opt --lower-mini-to-affine basic.mlir

```

## Rough Plan

Each commit covers this plan

- create custom dialect "mini", and register it with tool named mini-opt.
- add AddOp. Verify dialect and ops with tests.
- add constant folding optimization via canoncalization pass.
- add MatmulOp and MulOp
- create lowering passes for all operations, to lower mini ops to other dialects like tensor, arith, affine etc
    - initially, lower mini ops(with tensor inputs) to Low-Level Scalarized IR
    - lower to linalg.generic instead of previous.
    - High Level dialect
        - linalg.generic or
        - affine loops
- LIT tests
- affine transforms
- op fusions
- Tiling/SCF transforms
- Polyhedral transforms