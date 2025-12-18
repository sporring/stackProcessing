module Tests.Iteration

open Expecto
open System
open Image
open Image.InternalHelpers

// Same types and representative values as in PixelBoxingTests.fs
let pixelSamplesMap = [
    (fromType<uint8>     , box 3uy)
    (fromType<int8>      , box 3y)
    (fromType<uint16>    , box 3us)
    (fromType<int16>     , box 3s)
    (fromType<uint32>    , box 3u)
    (fromType<int32>     , box 3)
    (fromType<uint64>    , box 3UL)
    (fromType<int64>     , box 3L)
    (fromType<float32>   , box 3.0f)
    (fromType<float>     , box 3.0)
]

// Test helper: create a 2×2 image filled with value
let createImage<'T when 'T : equality> (v: 'T) : Image<'T> =
    let img = Image<'T>([2u; 2u])
    img.Set [0u;0u] v
    img.Set [1u;0u] v
    img.Set [0u;1u] v
    img.Set [1u;1u] v
    img

let inline testOps<^T when ^T: equality
                      and  ^T: (static member op_Explicit: ^T -> float)
                      and  ^T : (static member One  : ^T)
                      and  ^T : (static member ( + ) : ^T * ^T -> ^T)
                      >
    (name: string) =

    testList name [
        let t = fromType<'T>
        let sample = List.find (fun (a,b) -> a = fromType<'T>) pixelSamplesMap |> snd
        let img = createImage<'T> (unbox sample)

        // forAll: all elements equal the value
        let allEqual = img.forAll (fun x -> box x = sample)
        Expect.isTrue allEqual $"forAll equal (x)"

        // fold: count pixels
        let count = img |> Image<_>.fold (fun acc _ -> acc + 1) 0
        Expect.equal count 4 "fold count is 4"

        // foldi: collect indices
        let indices = img |> Image<_>.foldi (fun i acc _ -> i :: acc) []
        Expect.equal (List.length indices) 4 "foldi index count is 4"

        // map: double each value
        let doubled = img |> Image<_>.map (fun v -> v+(unbox sample))
        let first = doubled.Get [0u; 0u]
        Expect.notEqual (box first) sample "map doubles value"

        // mapi: double each value
        let doubled = img |> Image<_>.mapi (fun i v -> v+(unbox sample))
        let first = doubled.Get [0u; 0u]
        Expect.notEqual (box first) sample "map doubles value"

        // iter: side-effect counter
        let mutable seen = 0
        img |> Image<_>.iter (fun _ -> seen <- seen + 1)
        Expect.equal seen 4 "iter ran 4 times"

        // iteri: side-effect counter
        let mutable seen = 0
        img |> Image<_>.iteri (fun _ _ -> seen <- seen + 1)
        Expect.equal seen 4 "iter ran 4 times"

        // zip
        let zipped = Image<_>.zip [img; doubled]
        let unzipLst = Image<_>.unzip zipped
        Expect.isTrue (Image<_>.eq (unzipLst[0], img)) "unzip left equals original"
        Expect.isTrue (Image<_>.eq (unzipLst[1], doubled)) "unzip right equals mapped"
    ]

[<Tests>]
let SetGetTests =
  testList "Arithmetic operations on Image<'T>" [
    testOps<uint8>  "uint8 set/get"
    testOps<int8>  "int8 set/get"
    testOps<uint16>  "uint16 set/get"
    testOps<int16>  "int16 set/get"
    testOps<uint>  "uint set/get"
    testOps<int>  "int set/get"
    testOps<uint64>  "uint64 set/get"
    testOps<int64>  "int64 set/get"
    testOps<float32>  "float32 set/get"
    testOps<float>  "float set/get"
  ]

