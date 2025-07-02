module Tests.indexing

open Expecto
open Image   // open the namespace that defines Image<'T>

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


let inline u< ^T when ^T : equality> (n : uint8) : ^T =
    fromUint8< ^T > n                  // handy literal converter


let inline indexerTests<^T when ^T : (static member op_Explicit : uint8 -> ^T)
                           and  ^T: (static member One: ^T)
                           and  ^T : equality    // needed for Expect.equal
                           and  ^T : (static member (+) : ^T * ^T -> ^T)   // used when mutating values
                          > (name : string) =
  testList name [

    testCase "2D get/set" <| fun _ ->
      let img = make2D<^T>()
      Expect.equal img.[1,0] (u 3uy) "Get 2D index"
      img.[1,0] <- u 7uy
      Expect.equal img.[1,0] (u 7uy) "Set 2D index"

    testCase "3D get/set" <| fun _ ->
      let img = make3D<^T>()
      Expect.equal img.[1,1,0] (u 4uy) "Get 3D index"
      img.[1,1,0] <- u 8uy
      Expect.equal img.[1,1,0] (u 8uy) "Set 3D index"

    testCase "4D get/set" <| fun _ ->
      let img = make4D<^T>()
      Expect.equal img.[1,1,0,1] (u 8uy) "Get 4D index"
      img.[1,1,0,1] <- u 12uy
      Expect.equal img.[1,1,0,1] (u 12uy) "Set 4D index"
  ]

[<Tests>]
let indexerSuite =
  testList "Arithmetic operations on Image<'T>" [
    indexerTests<uint8>  "uint8 indexer"
    indexerTests<int8>  "int8 indexer"
    indexerTests<uint16>  "uint16 indexer"
    indexerTests<int16>  "int16 indexer"
    indexerTests<uint>  "uint indexer"
    indexerTests<int>  "int indexer"
    indexerTests<uint64>  "uint64 indexer"
    indexerTests<int64>  "int64 indexer"
    indexerTests<float32>  "float32 indexer"
    indexerTests<float>  "float indexer"
  ]
