// works
func.func @test1(%a : f64, %b : f64) -> f64 {
  %0 = mini.add %a, %b : (f64,f64) -> f64
  func.return %0 : f64
}

// // doesn't work
// func.func @test2(%a : f64, %b : f64) -> f64 {
//   %0 = mini.add %a, %b : f64
//   func.return %0 : f64
// }

// // doesn't work
// func.func @test3(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4xf64> {
//   %0 = mini.add %a, %b : tensor<4xf64>
//   return %0 : tensor<4xf64>
// }

// // works
// func.func @test4(%a : tensor<4xf64>,%b : tensor<4xf64>) -> tensor<4xf64> {
//   %0 = mini.add %a, %b : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
//   func.return %0 : tensor<4xf64>
// }

// works
func.func @test5(%a : i32, %b : i32) -> i32 {
  %0 = mini.add %a, %b : (i32,i32) -> i32
  func.return %0 : i32
}