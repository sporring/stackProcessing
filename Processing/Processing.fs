module Processing

open System
open System.IO
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open AsyncSeqExtensions
open StackPipeline
open Core
open Core.Helpers
open Slice

// --- Processing Utilities ---
let private explodeSlice (slices: Slice<'T>) : AsyncSeq<Slice<'T>> =
    let baseIndex = slices.Index
    let volume = slices.Image
    let size = volume.GetSize()
    let width, height, depth = size.[0], size.[1], size.[2]
    Seq.init (int depth) (fun z -> extractSlice (baseIndex+(uint z)) slices)
    |> AsyncSeq.ofSeq

let private reduce (label: string) (profile: MemoryProfile) (reducer: AsyncSeq<'In> -> Async<'Out>) : Pipe<'In, 'Out> =
    {
        Name = label
        Profile = profile
        Apply = fun input ->
            reducer input |> ofAsync
    }

let fold (label: string) (profile: MemoryProfile)  (folder: 'State -> 'In -> 'State) (state0: 'State) : Pipe<'In, 'State> =
    reduce label profile (fun stream ->
        async {
            let! result = stream |> AsyncSeq.fold folder state0
            return result
        })

let map (label: string) (profile: MemoryProfile) (f: 'S -> 'T) : Pipe<'S,'T> =
    {
        Name = "map"; 
        Profile = profile
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun slice -> f slice)
    }

let mapWindowed (label: string) (depth: uint) (stride: uint) (f: 'S list -> 'T) : Pipe<'S,'T> =
    {
        Name = "mapWindowed"; 
        Profile = Sliding depth
        Apply = fun input ->
            AsyncSeqExtensions.windowed depth stride input
            |> AsyncSeq.map (fun window -> window |> f)
    }

let mapChunked (label: string) (chunkSize: uint) (baseIndex: uint) (f: Slice<'T> -> Slice<'T>) : Pipe<Slice<'T>,Slice<'T>> =
    { // Due to stack, this is Pipe<Slice<'T>,Slice<'T>>
        Name = "mapChunked"; 
        Profile = Sliding chunkSize
        Apply = fun input ->
            AsyncSeqExtensions.chunkBySize (int chunkSize) input
            |> AsyncSeq.collect (fun chunk ->
                    let volume = stack chunk
                    let result = f { volume with Index = baseIndex }
                    explodeSlice result)
    }

let addFloat (value: float) : Pipe<Slice<float>, Slice<float>> =
    printfn "[addFloat]"
    map "addFloat" Streaming (swap addFloat value)

let addInt (value: int) : Pipe<Slice<int>, Slice<int>> =
    printfn "[addInt]"
    map "addInt" Streaming (swap addInt value)

let addUInt8 (value: uint8) : Pipe<Slice<uint8>, Slice<uint8>> =
    printfn "[addUInt8]"
    map "addUInt8" Streaming (swap addUInt8 value)

let add (image: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[add]"
    map "Add" Streaming (add image)

let subFloat (value: float) : Pipe<Slice<float>, Slice<float>> =
    printfn "[subFloat]"
    map "subFloat" Streaming (swap subFloat value)

let subInt (value: int) : Pipe<Slice<int>, Slice<int>> =
    printfn "[subInt]"
    map "subInt" Streaming (swap subInt value)

let sub (image: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[sub]"
    map "sub" Streaming (sub image)

let mulFloat (value: float) : Pipe<Slice<float>, Slice<float>> =
    printfn "[mulFloat]"
    map "mulFloat" Streaming (swap mulFloat value)

let mulInt (value: int) : Pipe<Slice<int>, Slice<int>> =
    printfn "[mulInt]"
    map "mulInt" Streaming (swap mulInt value)

let mulUInt8 (value: uint8) : Pipe<Slice<uint8>, Slice<uint8>> =
    printfn "[mulUInt8]"
    map "mulUInt8" Streaming (swap mulUInt8 value)

let mul (image: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[mul]"
    map "mul" Streaming (mul image)

let divFloat (value: float) : Pipe<Slice<float>, Slice<float>> =
    printfn "[divFloat]"
    map "divFloat" Streaming (swap divFloat value)

let divInt (value: int) : Pipe<Slice<int>, Slice<int>> =
    printfn "[divInt]"
    map "divInt" Streaming (swap divInt value)

let div (image: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[div]"
    map "div" Streaming (div image)

(*
let modulus (value: uint) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[modulus]"
    map "Modulus" Streaming (modulusConst value)

let modulusWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[modulusWithImage]"
    map "ModulusImage" Streaming (modulusImage image)

let pow (exponent: float) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[pow]"
    map "Power" Streaming (powConst exponent)

let powWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[powWithImage]"
    map "PowerImage" Streaming (powImage image)

let greaterEqual (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[greaterEqual]"
    map "GreaterEqual" Streaming (greaterEqualConst value)

let greaterEqualWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[greaterEqualWithImage]"
    map "GreaterEqualImage" Streaming (greaterEqualImage image)

let greater (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[greater]"
    map "Greater" Streaming (greaterConst value)

let greaterWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[greaterWithImage]"
    map "GreaterImage" Streaming (greaterImage image)

let equal (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[equal]"
    map "Equal" Streaming (equalConst value)

let equalWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[equalWithImage]"
    map "EqualImage" Streaming (equalImage image)

let notEqual (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[notEqual]"
    map "NotEqual" Streaming (notEqualConst value)

let notEqualWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[notEqualWithImage]"
    map "NotEqualImage" Streaming (notEqualImage image)

let less (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[less]"
    map "Less" Streaming (lessConst value)

let lessWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[lessWithImage]"
    map "LessImage" Streaming (lessImage image)

let lessEqual (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[lessEqual]"
    map "LessEqual" Streaming (lessEqualConst value)

let lessEqualWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[lessEqualWithImage]"
    map "LessEqualImage" Streaming (lessEqualImage image)

let bitwiseAnd (value: int) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseAnd]"
    map "BitwiseAnd" Streaming (bitwiseAndConst value)

let bitwiseAndWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseAndWithImage]"
    map "BitwiseAndImage" Streaming (bitwiseAndImage image)

let bitwiseOr (value: int) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseOr]"
    map "BitwiseOr" Streaming (bitwiseOrConst value)

let bitwiseOrWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseOrWithImage]"
    map "BitwiseOrImage" Streaming (bitwiseOrImage image)

let bitwiseXor (value: int) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseXor]"
    map "BitwiseXor" Streaming (bitwiseXorConst value)

let bitwiseXorWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseXorWithImage]"
    map "BitwiseXorImage" Streaming (bitwiseXorImage image)

let bitwiseNot (maximum: int) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseNot]"
    map "BitwiseNot" Streaming (bitwiseNot maximum)

let bitwiseLeftShift (shiftBits: int) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseLeftShift]"
    map "BitwiseLeftShift" Streaming (bitwiseLeftShift shiftBits)

let bitwiseRightShift (shiftBits: int) : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseRightShift]"
    map "BitwiseRightShift" Streaming (bitwiseRightShift shiftBits)
*)
let absProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[abs]"
    map "Abs" Streaming absSlice<'T>

let sqrtProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[sqrt]"
    map "Sqrt" Streaming sqrtSlice<'T>

let logProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[log]"
    map "Log" Streaming logSlice<'T>

let expProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    printfn "[exp]"
    map "Exp" Streaming expSlice<'T>

let histogram<'T when 'T: comparison> : Pipe<Slice<'T>, Map<'T, uint64>> =
    printfn "[histogram]"
    let histogramReducer (slices: AsyncSeq<Slice<'T>>) =
        slices
        |> AsyncSeq.map Slice.histogram
        |> AsyncSeqExtensions.fold Slice.addHistogram (Map<'T,uint64> [])
    reduce "Histogram" StreamingConstant histogramReducer

let map2pairs<'T, 'S when 'T: comparison> : Pipe<Map<'T, 'S>,('T * 'S) list> =
    printfn "[map2pairs]"
    map "map2pairs" Streaming Slice.map2pairs

let inline pairs2floats<^T, ^S when ^T : (static member op_Explicit : ^T -> float)
                                 and ^S : (static member op_Explicit : ^S -> float)> : Pipe<('T * 'S) list,(float * float) list> =
    printfn "[pairs2floats]"
    map "pairs2floats" Streaming Slice.pairs2floats

let inline pairs2int<^T, ^S when ^T : (static member op_Explicit : ^T -> int)
                                 and ^S : (static member op_Explicit : ^S -> int)> : Pipe<('T * 'S) list,(int * int) list> =
    printfn "[pairs2int]"
    map "pairs2int" Streaming Slice.pairs2ints

let addNormalNoise (mean: float) (stddev: float) : Pipe<Slice<'T> ,Slice<'T>> =
    printfn "[addNormalNoise]"
    map "addNormalNoise" Streaming (addNormalNoise mean stddev)

let threshold (lower: float) (upper: float) : Pipe<Slice<'T> ,Slice<'T>> =
    printfn "[threshold]"
    map "Threshold" Streaming (threshold lower upper)

let discreteGaussian (sigma: float) : Pipe<Slice<'T> ,Slice<'T>> =
    printfn "[discreteGaussian]"
    let depth = 1u + 2u * uint (0.5 + sigma)
    let stride = 1u
    mapWindowed "discreteGaussian" depth stride (stack >> discreteGaussian sigma)

type FileInfo = Slice.FileInfo
let getFileInfo(fname: string): FileInfo = Slice.getFileInfo(fname)
let getVolumeSize (inputDir: string) (suffix: string) = Slice.getVolumeSize inputDir suffix

type ImageStats = Slice.ImageStats
let computeStats<'T when 'T : equality> : Pipe<Slice<'T>, ImageStats> =
    printfn "[computeStats]"
    let computeStatsReducer (slices: AsyncSeq<Slice<'T>>) =
        let zeroStats: ImageStats = { 
            NumPixels = 0u
            Mean = 0.0
            Std = 0.0
            Min = infinity
            Max = -infinity
            Sum = 0.0
            Var = 0.0}
        slices
        |> AsyncSeq.map Slice.computeStats
        |> AsyncSeqExtensions.fold Slice.addComputeStats zeroStats
    reduce "Compute Statistics" StreamingConstant computeStatsReducer

