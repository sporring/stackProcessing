/// <summary>
/// Expecto tests for the perpixel arithmetic helper functions
/// (add, sub, mul, div, pow) in Image.fs.
/// The structure follows <c>indexingTests.fs</c> so that the same tests
/// run automatically over a broad range of numeric pixel types.
/// </summary>

module Image.Tests.ArithmeticTests

open Expecto
open Image                           // the Image<'T> type & arithmetic helpers
open ImageFunctions

// ---------- helpers ----------

// tiny images with a handchecked pattern (identical to indexingTests.fs)

let inline make2D< ^T
                  when ^T : (static member op_Explicit : uint8 -> ^T)
                  and  ^T : (static member One : ^T)
                  and  ^T : equality > () : Image< ^T > =
    let raw = array2D [ [ 1uy; 2uy ]
                        [ 3uy; 4uy ] ]
    let cast = raw |> Array2D.map (fun v -> (^T : (static member op_Explicit : uint8 -> ^T) v))
    Image< ^T >.ofArray2D cast

let inline make3D< ^T
                  when ^T : (static member op_Explicit : uint8 -> ^T)
                  and  ^T : (static member One : ^T)
                  and  ^T : equality > () : Image< ^T > =
    // 2×2×2 voxels numbered 1‥8
    let arr = Array3D.init 2 2 2 (fun x y z ->
        let v = 1 + x + 2*y + 4*z   // 1,2,…,8
        (^T : (static member op_Explicit : uint8 -> ^T) (uint8 v)) )
    Image< ^T >.ofArray3D arr

let inline make4D< ^T
                  when ^T : (static member op_Explicit : uint8 -> ^T)
                  and  ^T : (static member One : ^T)
                  and  ^T : equality > () : Image< ^T > =
    // 2×2×1×2 = 8 voxels, same numbering as 3D for simplicity
    let arr = Array4D.init 2 2 1 2 (fun x y _ t ->
        let v = 1 + x + 2*y + 4*t
        (^T : (static member op_Explicit : uint8 -> ^T) (uint8 v)) )
    Image< ^T >.ofArray4D arr

/// integer power that works for any numeric ^T supporting (*) and One
let inline powT baseVal exp =
    let mutable acc = LanguagePrimitives.GenericOne
    for _ = 1 to exp do
        acc <- acc * baseVal
    acc

// ---------- generic test generator ----------

let inline arithmeticTests< ^T
                           when ^T : (static member op_Explicit : uint8  -> ^T)
                           and  ^T : (static member op_Explicit : ^T    -> float)
                           and  ^T : (static member One : ^T)
                           and  ^T : equality
                           and  ^T : (static member (+)  : ^T * ^T -> ^T)
                           and  ^T : (static member (-)  : ^T * ^T -> ^T)
                           and  ^T : (static member ( * ): ^T * ^T -> ^T)
                           and  ^T : (static member ( / ): ^T * ^T -> ^T) >
                           (name : string) =

  // handy literals
  let inline u n = (^T : (static member op_Explicit : uint8 -> ^T) n)

  testList name [

    // ---- 2D -----------------------------------------------------------
    testCase "2D arithmetic" <| fun _ ->
      let img = make2D< ^T >()
      let sAdd, sSub, sMul, sDiv, sPow = u 2uy, u 5uy, u 3uy, u 2uy, u 3uy

      // add --------------------------------------------------------------
      let add1 = imageAddScalar  img sAdd
      let add2 = scalarAddImage sAdd img
      Expect.equal add1 add2 "add commutative"
      Expect.equal add1.[1,0] (img.[1,0] + sAdd) "add pixel check"

      // sub --------------------------------------------------------------
      let sub1 = imageSubScalar  img sSub
      let sub2 = scalarSubImage sSub img
      Expect.equal sub1.[0,1] (img.[0,1] - sSub) "image − scalar"
      Expect.equal sub2.[0,1] (sSub - img.[0,1]) "scalar − image"

      // mul --------------------------------------------------------------
      let mul1 = imageMulScalar  img sMul
      let mul2 = scalarMulImage sMul img
      Expect.equal mul1 mul2          "mul commutative"
      Expect.equal mul1.[1,1] (img.[1,1] * sMul) "mul pixel check"

      // div --------------------------------------------------------------
      let div1 = imageDivScalar  img sDiv
      let div2 = scalarDivImage sDiv img
      Expect.equal div1.[0,0] (img.[0,0] / sDiv) "image ÷ scalar"
      Expect.equal div2.[0,0] (sDiv / img.[0,0]) "scalar ÷ image"

      // pow --------------------------------------------------------------
      let powImg = imagePowScalar  (img, sPow)     // img ^ scalar
      let powSc  = scalarPowImage (sPow, img)      // scalar ^ img
      Expect.equal powImg.[1,1] (powT img.[1,1] 3) "img ^ scalar pixel"
      Expect.equal powSc.[0,0]  (powT sPow (int (float img.[0,0]))) "scalar ^ img pixel"

    // ---- 3D -----------------------------------------------------------
    testCase "3D arithmetic (spotcheck)" <| fun _ ->
      let img = make3D< ^T >()
      let s = u 2uy
      let res = imageAddScalar img s
      Expect.equal res.[1,1,1] (img.[1,1,1] + s) "3D add pixel ok"

    // ---- 4D -----------------------------------------------------------
(* 4D is only supported in simple itk to a very limited extent

    testCase "4D arithmetic (spotcheck)" <| fun _ ->
      let img = make4D< ^T >()
      let s = u 2uy
      let res = imageMulScalar img s
      Expect.equal res.[1,1,0,1] (img.[1,1,0,1] * s) "4D mul pixel ok"
*)
  ]

// ---------- full suite over many concrete pixel types ----------

[<Tests>]
let ArithmeticTests =
  testList "Arithmetic operations on Image<'T>'" [
    arithmeticTests<uint8>   "uint8 arithmetic"
    arithmeticTests<int8>    "int8 arithmetic"
    arithmeticTests<uint16>  "uint16 arithmetic"
    arithmeticTests<int16>   "int16 arithmetic"
    arithmeticTests<uint32>  "uint32 arithmetic"
    arithmeticTests<int32>   "int32 arithmetic"
    arithmeticTests<uint64>  "uint64 arithmetic"
    arithmeticTests<int64>   "int64 arithmetic"
    arithmeticTests<float32> "float32 arithmetic"
    arithmeticTests<float>   "float arithmetic"
  ]
