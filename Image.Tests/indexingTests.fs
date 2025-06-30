/// <summary>
/// Expecto tests for the Image<'T>.Item indexer (get & set) overloads
/// covering 1D, 2D, 3D and 4D images.
/// </summary>

module Image.Tests.ItemIndexer

open Expecto
open Image   // open the namespace that defines Image<'T>

// ---------- helpers ----------

/// Creates a very small test image for a given dimension using an easytoverify pattern.
/// For 1D the data is [|1;2;3|].  For higher rank we just count upward.
let inline make2D<^T when ^T : (static member op_Explicit : uint8 -> ^T) 
                     and  ^T : (static member One : ^T) 
                     and  ^T: equality> () : Image<^T> =
    let raw = array2D [ [ 1uy; 2uy ]; [ 3uy; 4uy ] ]
    let casted = raw |> Array2D.map (fun v -> (^T : (static member op_Explicit : uint8 -> ^T) v))
    Image<^T>.ofArray2D casted

let inline make3D<^T when ^T : (static member op_Explicit : uint8 -> ^T) 
                     and  ^T : (static member One : ^T) 
                     and  ^T: equality> () : Image<^T> =
    // 2×2×2 = 8 voxels numbered 1 .. 8
    let arr = Array3D.init 2 2 2 (fun x y z -> 1 + x + 2*y + 4*z |> uint8 |> fun v -> (^T : (static member op_Explicit : uint8 -> ^T) v))
    Image<^T>.ofArray3D arr

let inline make4D<^T when ^T : (static member op_Explicit : uint8 -> ^T) 
                     and  ^T : (static member One : ^T) 
                     and  ^T: equality> () : Image<^T> =
    // 2×2×1×2 = 8 voxels
    let arr = Array4D.init 2 2 1 2 (fun x y z t -> 1 + x + 2*y + 4*t |> uint8 |> fun v -> (^T : (static member op_Explicit : uint8 -> ^T) v))
    Image<^T>.ofArray4D arr

// ---------- generic test generator ----------

let inline indexerTests<^T when ^T : (static member op_Explicit : uint8 -> ^T)
                           and  ^T: (static member One: ^T)
                           and  ^T : equality    // needed for Expect.equal
                           and  ^T : (static member (+) : ^T * ^T -> ^T)   // used when mutating values
                          > (name : string) =
  testList name [

    testCase "2D get/set" <| fun _ ->
      let img = make2D<^T>()
      Expect.equal img.[1,0] ((^T : (static member op_Explicit : uint8 -> ^T) 3uy)) "Get 2D index"
      img.[1,0] <- (^T : (static member op_Explicit : uint8 -> ^T) 7uy)
      Expect.equal img.[1,0] ((^T : (static member op_Explicit : uint8 -> ^T) 7uy)) "Set 2D index"

    testCase "3D get/set" <| fun _ ->
      let img = make3D<^T>()
      Expect.equal img.[1,1,0] ((^T : (static member op_Explicit : uint8 -> ^T) 4uy)) "Get 3D index"
      img.[1,1,0] <- (^T : (static member op_Explicit : uint8 -> ^T) 8uy)
      Expect.equal img.[1,1,0] ((^T : (static member op_Explicit : uint8 -> ^T) 8uy)) "Set 3D index"

    testCase "4D get/set" <| fun _ ->
      let img = make4D<^T>()
      Expect.equal img.[1,1,0,1] ((^T : (static member op_Explicit : uint8 -> ^T) 8uy)) "Get 4D index"
      img.[1,1,0,1] <- (^T : (static member op_Explicit : uint8 -> ^T) 12uy)
      Expect.equal img.[1,1,0,1] ((^T : (static member op_Explicit : uint8 -> ^T) 12uy)) "Set 4D index"
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
