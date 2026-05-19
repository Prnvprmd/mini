// // works
// func.func @test1(%a : f64, %b : f64) -> f64 {
//   %0 = mini.add %a, %b : (f64,f64) -> f64
//   func.return %0 : f64
// }

// // works
// func.func @test5(%a : i32, %b : i32) -> i32 {
//   %0 = mini.add %a, %b : (i32,i32) -> i32
//   func.return %0 : i32
// }

// doesn't work
func.func @test2(%a : f64, %b : f64) -> f64 {
  %0 = mini.add %a, %b : f64
  func.return %0 : f64
}

// doesn't work
func.func @test3(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4xf64> {
  %0 = mini.add %a, %b : tensor<4xf64>
  return %0 : tensor<4xf64>
}


// doesn't work
func.func @test6() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %0 = mini.add %c1, %c2 : i32
  return %0 : i32
}

func.func @test7(%arg0 : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = mini.add %arg0, %c0 : i32
  return %0 : i32
}

func.func @test8(%arg0 : f32) -> f32 {
  %c0 = arith.constant 0.0 : f32
  %0 = mini.add %arg0, %c0 : f32
  return %0 : f32
}

// // works
// func.func @test4(%a : tensor<4xf64>,%b : tensor<4xf64>) -> tensor<4xf64> {
//   %0 = mini.add %a, %b : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
//   func.return %0 : tensor<4xf64>
// }

func.func @matmul_f32(
    %a : tensor<2x3xf32>,
    %b : tensor<3x4xf32>)
    -> tensor<2x4xf32> {

  %0 = mini.matmul %a, %b
      : tensor<2x3xf32>,
        tensor<3x4xf32>
        -> tensor<2x4xf32>

  return %0 : tensor<2x4xf32>
}

func.func @matmul_i32(
    %a : tensor<8x16xi32>,
    %b : tensor<16x32xi32>)
    -> tensor<8x32xi32> {

  %0 = mini.matmul %a, %b
      : tensor<8x16xi32>,
        tensor<16x32xi32>
        -> tensor<8x32xi32>

  return %0 : tensor<8x32xi32>
}

func.func @batched(
    %a : tensor<5x2x3xf32>,
    %b : tensor<5x3x4xf32>)
    -> tensor<5x2x4xf32> {

  %0 = mini.matmul %a, %b
      : tensor<5x2x3xf32>,
        tensor<5x3x4xf32>
        -> tensor<5x2x4xf32>

  return %0 : tensor<5x2x4xf32>
}

func.func @multi_batch(
    %a : tensor<2x5x2x3xf32>,
    %b : tensor<2x5x3x4xf32>)
    -> tensor<2x5x2x4xf32> {

  %0 = mini.matmul %a, %b
      : tensor<2x5x2x3xf32>,
        tensor<2x5x3x4xf32>
        -> tensor<2x5x2x4xf32>

  return %0 : tensor<2x5x2x4xf32>
}

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
