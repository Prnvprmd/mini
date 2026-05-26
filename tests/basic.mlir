// // // works
// // func.func @test1(%a : f64, %b : f64) -> f64 {
// //   %0 = mini.add %a, %b : (f64,f64) -> f64
// //   func.return %0 : f64
// // }

// // works
// func.func @test5(%a : i32, %b : i32) -> i32 {
//   %0 = mini.add %a, %b : i32
//   func.return %0 : i32
// }

// // doesn't work
// func.func @test2(%a : f64, %b : f64) -> f64 {
//   %0 = mini.add %a, %b : f64
//   func.return %0 : f64
// }

// doesn't work
func.func @test3(%a: tensor<4x4xf64>, %b: tensor<4x4xf64>, %c: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = mini.add %a, %b : tensor<4x4xf64>
  %1 = mini.mul %0, %c : tensor<4x4xf64>
  return %1 : tensor<4x4xf64>
}


// // doesn't work
// func.func @test6() -> i32 {
//   %c1 = arith.constant 1 : i32
//   %c2 = arith.constant 2 : i32
//   %0 = mini.add %c1, %c2 : i32
//   return %0 : i32
// }

// func.func @test7(%arg0 : i32) -> i32 {
//   %c0 = arith.constant 0 : i32
//   %0 = mini.add %arg0, %c0 : i32
//   return %0 : i32
// }

// func.func @test8(%arg0 : f32) -> f32 {
//   %c0 = arith.constant 0.0 : f32
//   %0 = mini.add %arg0, %c0 : f32
//   return %0 : f32
// }

// // // works
// // func.func @test4(%a : tensor<4xf64>,%b : tensor<4xf64>) -> tensor<4xf64> {
// //   %0 = mini.add %a, %b : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
// //   func.return %0 : tensor<4xf64>
// // }

// func.func @matmul_f32(
//     %a : tensor<2x3xf32>,
//     %b : tensor<3x4xf32>)
//     -> tensor<2x4xf32> {

//   %0 = mini.matmul %a, %b
//       : tensor<2x3xf32>,
//         tensor<3x4xf32>
//         -> tensor<2x4xf32>

//   return %0 : tensor<2x4xf32>
// }

// func.func @matmul_i32(
//     %a : tensor<8x16xi32>,
//     %b : tensor<16x32xi32>)
//     -> tensor<8x32xi32> {

//   %0 = mini.matmul %a, %b
//       : tensor<8x16xi32>,
//         tensor<16x32xi32>
//         -> tensor<8x32xi32>

//   return %0 : tensor<8x32xi32>
// }

// func.func @batched(
//     %a : tensor<5x2x3xf32>,
//     %b : tensor<5x3x4xf32>)
//     -> tensor<5x2x4xf32> {

//   %0 = mini.matmul %a, %b
//       : tensor<5x2x3xf32>,
//         tensor<5x3x4xf32>
//         -> tensor<5x2x4xf32>

//   return %0 : tensor<5x2x4xf32>
// }

// func.func @multi_batch(
//     %a : tensor<2x5x2x3xf32>,
//     %b : tensor<2x5x3x4xf32>)
//     -> tensor<2x5x2x4xf32> {

//   %0 = mini.matmul %a, %b
//       : tensor<2x5x2x3xf32>,
//         tensor<2x5x3x4xf32>
//         -> tensor<2x5x2x4xf32>

//   return %0 : tensor<2x5x2x4xf32>
// }

// func.func @bad_k(
//     %a : tensor<2x3xf32>,
//     %b : tensor<5x4xf32>)
//     -> tensor<2x4xf32> {

//   %0 = mini.matmul %a, %b
//       : tensor<2x3xf32>,
//         tensor<5x4xf32>
//         -> tensor<2x4xf32>

//   return %0 : tensor<2x4xf32>
// }


// func.func @bad_batch(
//     %a : tensor<5x2x3xf32>,
//     %b : tensor<7x3x4xf32>)
//     -> tensor<5x2x4xf32> {

//   %0 = mini.matmul %a, %b
//       : tensor<5x2x3xf32>,
//         tensor<7x3x4xf32>
//         -> tensor<5x2x4xf32>

//   return %0 : tensor<5x2x4xf32>
// }

// #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// func.func @matmul_f32_output(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
//   %cst = arith.constant 0.000000e+00 : f32
//   %0 = tensor.empty() : tensor<2x4xf32>
//   %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
//   %2 = linalg.generic {
//     indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]
//   } ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<2x4xf32>) {
//   ^bb0(%in: f32, %in_0: f32, %out: f32):
//     %3 = arith.mulf %in, %in_0 : f32
//     %4 = arith.addf %out, %3 : f32
//     linalg.yield %4 : f32
//   } -> tensor<2x4xf32>
//   return %2 : tensor<2x4xf32>
// }




#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_fusion(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>, %arg2: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = tensor.empty() : tensor<4x4xf64>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xf64>, tensor<4x4xf64>) outs(%0 : tensor<4x4xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %4 = arith.addf %in, %in_0 : f64
    linalg.yield %4 : f64
  } -> tensor<4x4xf64>
  %2 = tensor.empty() : tensor<4x4xf64>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %arg2 : tensor<4x4xf64>, tensor<4x4xf64>) outs(%2 : tensor<4x4xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %4 = arith.mulf %in, %in_0 : f64
    linalg.yield %4 : f64
  } -> tensor<4x4xf64>
  %5 = tensor.empty() : tensor<4x4xf64>
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %arg1 : tensor<4x4xf64>, tensor<4x4xf64>) outs(%0 : tensor<4x4xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %4 = arith.addf %in, %in_0 : f64
    linalg.yield %4 : f64
  } -> tensor<4x4xf64>
  return %6 : tensor<4x4xf64>
}


func.func @fuse_add_mul(
    %a : tensor<4xf32>,
    %b : tensor<4xf32>,
    %c : tensor<4xf32>,
    %out1 : tensor<4xf32>,
    %out2 : tensor<4xf32>)
    -> tensor<4xf32> {

  %0 = linalg.generic
    {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
    outs(%out1 : tensor<4xf32>) {
    ^bb0(%x : f32, %y : f32, %o : f32):
      %add = arith.addf %x, %y : f32
      linalg.yield %add : f32
    } -> tensor<4xf32>

  %1 = linalg.generic
    {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%0, %c : tensor<4xf32>, tensor<4xf32>)
    outs(%out2 : tensor<4xf32>) {
    ^bb0(%x : f32, %y : f32, %o : f32):
      %mul = arith.mulf %x, %y : f32
      linalg.yield %mul : f32
    } -> tensor<4xf32>

  return %1 : tensor<4xf32>
}