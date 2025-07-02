module Tests.Arithmetic

open Expecto
open Image

// ──────────────────────────────────────────────────────────────────────
//  Safe uint8 → 'T conversion (removes FS0077 warning)
// ──────────────────────────────────────────────────────────────────────
let inline private fromUint8< ^T when ^T : equality> (v : uint8) : ^T =
    let t = typeof< ^T >
    if      t = typeof<uint8>   then unbox (box v)
    elif    t = typeof<int8>    then unbox (box (int8   v))
    elif    t = typeof<uint16>  then unbox (box (uint16 v))
    elif    t = typeof<int16>   then unbox (box (int16  v))
    elif    t = typeof<uint32>  then unbox (box (uint32 v))
    elif    t = typeof<int32>   then unbox (box (int32  v))
    elif    t = typeof<uint64>  then unbox (box (uint64 v))
    elif    t = typeof<int64>   then unbox (box (int64  v))
    elif    t = typeof<float32> then unbox (box (float32 v))
    elif    t = typeof<float>   then unbox (box (float   v))
    else failwithf "Unsupported conversion from uint8 to %A" t

// ──────────────────────────────────────────────────────────────────────
//  Tiny test images (2‑D / 3‑D / 4‑D)
// ──────────────────────────────────────────────────────────────────────
let inline make2D< ^T when ^T : equality> () : Image< ^T > =
    let raw  = array2D [ [ 1uy; 2uy ]
                         [ 3uy; 4uy ] ]
    let cast = raw |> Array2D.map fromUint8< ^T >
    Image< ^T >.ofArray2D cast

let inline make3D< ^T when ^T : equality> () : Image< ^T > =
    let arr = Array3D.init 2 2 2 (fun x y z ->
        let v = 1 + x + 2*y + 4*z      // 1 … 8
        fromUint8< ^T > (uint8 v) )
    Image< ^T >.ofArray3D arr

let inline make4D< ^T when ^T : equality> () : Image< ^T > =
    let arr = Array4D.init 2 2 1 2 (fun x y _ t ->
        let v = 1 + x + 2*y + 4*t
        fromUint8< ^T > (uint8 v) )
    Image< ^T >.ofArray4D arr          // simpleITK 4‑D support is limited

// ──────────────────────────────────────────────────────────────────────
//  Generic helpers
// ──────────────────────────────────────────────────────────────────────
let inline u< ^T when ^T : equality> (n : uint8) : ^T =
    fromUint8< ^T > n                  // handy literal converter

let inline powT baseVal exp =
    let mutable acc = LanguagePrimitives.GenericOne
    for _ = 1 to exp do
        acc <- acc * baseVal
    acc

// ──────────────────────────────────────────────────────────────────────
//  Generic test generator for one pixel type
// ──────────────────────────────────────────────────────────────────────
let inline arithmeticTests< ^T
                           when ^T : equality
                           and  ^T : (static member op_Explicit : ^T -> float)
                           and  ^T : (static member One : ^T)
                           and  ^T : (static member (+)  : ^T * ^T -> ^T)
                           and  ^T : (static member (-)  : ^T * ^T -> ^T)
                           and  ^T : (static member ( * ): ^T * ^T -> ^T)
                           and  ^T : (static member ( / ): ^T * ^T -> ^T) >
                           (name : string) =

  testList name [

    // ── 2‑D ───────────────────────────────────────────────────────────
    testCase "2‑D arithmetic" <| fun _ ->
      let img = make2D< ^T >()
      let sAdd, sSub, sMul, sDiv, sPow =
          u 2uy, u 5uy, u 3uy, u 2uy, u 3uy

      // add
      let add1 = ImageFunctions.imageAddScalar  img sAdd
      let add2 = ImageFunctions.scalarAddImage sAdd img
      Expect.equal add1 add2                              "add commutative"
      Expect.equal add1.[1,0] (img.[1,0] + sAdd)          "add pixel"

      // sub
      let sub1 = ImageFunctions.imageSubScalar  img sSub
      let sub2 = ImageFunctions.scalarSubImage sSub img
      Expect.equal sub1.[0,1] (img.[0,1] - sSub)          "image − scalar"
      Expect.equal sub2.[0,1] (sSub - img.[0,1])          "scalar − image"

      // mul
      let mul1 = ImageFunctions.imageMulScalar  img sMul
      let mul2 = ImageFunctions.scalarMulImage sMul img
      Expect.equal mul1 mul2                              "mul commutative"
      Expect.equal mul1.[1,1] (img.[1,1] * sMul)          "mul pixel"

      // div
      let div1 = ImageFunctions.imageDivScalar  img sDiv
      let div2 = ImageFunctions.scalarDivImage sDiv img
      Expect.equal div1.[0,0] (img.[0,0] / sDiv)          "image ÷ scalar"
      Expect.equal div2.[0,0] (sDiv / img.[0,0])          "scalar ÷ image"

      // pow
      let powImg = ImageFunctions.imagePowScalar  (img,  sPow)           // img ^ scalar
      let powSc  = ImageFunctions.scalarPowImage (sPow, img)             // scalar ^ img
      Expect.equal powImg.[1,1] (powT img.[1,1] 3)        "img ^ scalar"
      Expect.equal powSc.[0,0]  (powT sPow (int (float img.[0,0])))
                                                            "scalar ^ img"

    // ── 3‑D (spot‑check) ──────────────────────────────────────────────
    testCase "3‑D arithmetic (spot‑check)" <| fun _ ->
      let img = make3D< ^T >()
      let s   = u 2uy
      let res = ImageFunctions.imageAddScalar img s
      Expect.equal res.[1,1,1] (img.[1,1,1] + s)          "3‑D add pixel"

    // ── 4‑D (optional) ────────────────────────────────────────────────
(* simpleITK 4‑D support is limited; enable if available

    testCase "4‑D arithmetic (spot‑check)" <| fun _ ->
      let img = make4D< ^T >()
      let s   = u 2uy
      let res = imageMulScalar img s
      Expect.equal res.[1,1,0,1] (img.[1,1,0,1] * s)      "4‑D mul pixel"
*)
  ]

// ──────────────────────────────────────────────────────────────────────
//  Full suite over common numeric pixel types
// ──────────────────────────────────────────────────────────────────────
[<Tests>]
let ArithmeticTests =
  testList "Arithmetic on Image<'T>'" [
    arithmeticTests<uint8>   "uint8"
    arithmeticTests<int8>    "int8"
    arithmeticTests<uint16>  "uint16"
    arithmeticTests<int16>   "int16"
    arithmeticTests<uint32>  "uint32"
    arithmeticTests<int32>   "int32"
    arithmeticTests<uint64>  "uint64"
    arithmeticTests<int64>   "int64"
    arithmeticTests<float32> "float32"
    arithmeticTests<float>   "float"
  ]
