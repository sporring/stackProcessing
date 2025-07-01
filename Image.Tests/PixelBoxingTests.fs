module Image.Tests.PixelBoxing

open Expecto
open System
open itk.simple
open Image
open Image.InternalHelpers

// Sample values for each type to test roundtrip
let pixelSamplesMap = [
    (fromType<uint8>     , box 42uy)
    (fromType<int8>      , box -42y)
    (fromType<uint16>    , box 1234us)
    (fromType<int16>     , box -1234s)
    (fromType<uint32>    , box 123456u)
    (fromType<int32>     , box -123456)
    (fromType<uint64>    , box 12345678901234UL)
    (fromType<int64>     , box -12345678901234L)
    (fromType<float32>   , box 3.14f)
    (fromType<float>     , box 2.718)
    (fromType<System.Numerics.Complex>, box (System.Numerics.Complex(1.0, -2.0)))
    (fromType<uint8 list>     , box [ 1uy; 2uy ])
    (fromType<int8 list>      , box [ -1y; -2y ])
    (fromType<uint16 list>    , box [ 100us; 200us ])
    (fromType<int16 list>     , box [ -100s; -200s ])
    (fromType<uint32 list>    , box [ 10u; 20u ])
    (fromType<int32 list>     , box [ -10; -20 ])
    (fromType<uint64 list>    , box [ 1234UL; 5678UL ])
    (fromType<int64 list>     , box [ -1234L; -5678L ])
    (fromType<float32 list>   , box [ 1.1f; 2.2f ])
    (fromType<float list>     , box [ 1.1; 2.2 ])
]

// Convert uint list to VectorUInt32 (SimpleITK index)
let pos = [0u; 0u] |> toVectorUInt32
let makeImage (itkId: itk.simple.PixelIDValueEnum) =
    let sz = [1u;1u]
    new itk.simple.Image(sz |> toVectorUInt32, itkId)

let inline testOps<^T when ^T: equality
                      and  ^T: (static member op_Explicit: ^T -> float)
                      and  ^T : (static member One  : ^T)
                      and  ^T : (static member ( + ) : ^T * ^T -> ^T)
                      >
    (name: string) =

    testList name [
        let t = fromType<'T>
        let img = makeImage t
        let sample = List.find (fun (a,b) -> a = fromType<'T>) pixelSamplesMap |> snd
        setPixelAs img t pos sample
        let result = getPixelBoxed img t pos 
        Expect.equal result sample $"Mismatch for type {t}"
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
