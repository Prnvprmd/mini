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
