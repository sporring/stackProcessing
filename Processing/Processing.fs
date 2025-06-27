module Processing

open System
open System.IO
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open AsyncSeqExtensions
open StackPipeline
open Slice

//
let private fromReducer (name: string) (profile: MemoryProfile) (reducer: AsyncSeq<'In> -> Async<'Out>) : StackProcessor<'In, 'Out> =
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            reducer input |> ofAsync
    }

// --- Processing Utilities ---
let private unstack (slices: Slice<'T>) : AsyncSeq<Slice<'T>> =
    let baseIndex = slices.Index
    let volume = slices.Image
    let size = volume.GetSize()
    let width, height, depth = size.[0], size.[1], size.[2]
    Seq.init (int depth) (fun z -> extractSlice (baseIndex+(uint z)) slices)
    |> AsyncSeq.ofSeq

(*
let addTo (other: AsyncSeq<Slice<'T>>) : StackProcessor<Slice<'T> ,Slice<'T>> =
    {   
        Name = "AddTo"
        Profile = Streaming
        Apply = fun input ->
            zipJoin add input other (Some "[addTo]")
    }

let multiplyWith (other: AsyncSeq<Slice<'T>>) : StackProcessor<Slice<'T> ,Slice<'T>> =
    printfn "[multiplyWith]"
    {
        Name = "multiplyWith"
        Profile = Streaming
        Apply = fun input ->
            printfn "[multiplyWith]"
            zipJoin mul input other (Some "[multiplyWith]")
    }
*)
let mapSlices (label: string) (profile: MemoryProfile) (f: 'S -> 'T) : StackProcessor<'S,'T> =
    {
        Name = "mapSlices"; 
        Profile = profile
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun slice -> f slice)
    }

let mapSlicesWindowed (label: string) (depth: uint) (f: Slice<'T> -> Slice<'T>) : StackProcessor<Slice<'T>,Slice<'T>> =
    { // Due to stack, this is StackProcessor<Slice<'T>,Slice<'T>>
        Name = "mapSlicesWindowed"; 
        Profile = Sliding depth
        Apply = fun input ->
            AsyncSeqExtensions.windowed (int depth) input
            |> AsyncSeq.map (fun window -> window |> stack |> f)
    }

let mapSlicesChunked (label: string) (chunkSize: uint) (baseIndex: uint) (f: Slice<'T> -> Slice<'T>) : StackProcessor<Slice<'T>,Slice<'T>> =
    { // Due to stack, this is StackProcessor<Slice<'T>,Slice<'T>>
        Name = "mapSlicesChunked"; 
        Profile = Sliding chunkSize
        Apply = fun input ->
            AsyncSeqExtensions.chunkBySize (int chunkSize) input
            |> AsyncSeq.collect (fun chunk ->
                    let volume = stack chunk
                    let result = f { volume with Index = baseIndex }
                    unstack result)
    }

let addFloat (value: float) : StackProcessor<Slice<float>, Slice<float>> =
    printfn "[addFloat]"
    mapSlices "addFloat" Streaming (swap addFloat value)

let addInt (value: int) : StackProcessor<Slice<int>, Slice<int>> =
    printfn "[addInt]"
    mapSlices "addInt" Streaming (swap addInt value)

let addUInt8 (value: uint8) : StackProcessor<Slice<uint8>, Slice<uint8>> =
    printfn "[addUInt8]"
    mapSlices "addUInt8" Streaming (swap addUInt8 value)

let add (image: Slice<'T>) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[add]"
    mapSlices "Add" Streaming (add image)

let subFloat (value: float) : StackProcessor<Slice<float>, Slice<float>> =
    printfn "[subFloat]"
    mapSlices "subFloat" Streaming (swap subFloat value)

let subInt (value: int) : StackProcessor<Slice<int>, Slice<int>> =
    printfn "[subInt]"
    mapSlices "subInt" Streaming (swap subInt value)

let sub (image: Slice<'T>) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[sub]"
    mapSlices "sub" Streaming (sub image)

let mulFloat (value: float) : StackProcessor<Slice<float>, Slice<float>> =
    printfn "[mulFloat]"
    mapSlices "mulFloat" Streaming (swap mulFloat value)

let mulInt (value: int) : StackProcessor<Slice<int>, Slice<int>> =
    printfn "[mulInt]"
    mapSlices "mulInt" Streaming (swap mulInt value)

let mulUInt8 (value: uint8) : StackProcessor<Slice<uint8>, Slice<uint8>> =
    printfn "[mulUInt8]"
    mapSlices "mulUInt8" Streaming (swap mulUInt8 value)

let mul (image: Slice<'T>) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[mul]"
    mapSlices "mul" Streaming (mul image)

let divFloat (value: float) : StackProcessor<Slice<float>, Slice<float>> =
    printfn "[divFloat]"
    mapSlices "divFloat" Streaming (swap divFloat value)

let divInt (value: int) : StackProcessor<Slice<int>, Slice<int>> =
    printfn "[divInt]"
    mapSlices "divInt" Streaming (swap divInt value)

let div (image: Slice<'T>) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[div]"
    mapSlices "div" Streaming (div image)

(*
let modulus (value: uint) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[modulus]"
    mapSlices "Modulus" Streaming (modulusConst value)

let modulusWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[modulusWithImage]"
    mapSlices "ModulusImage" Streaming (modulusImage image)

let pow (exponent: float) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[pow]"
    mapSlices "Power" Streaming (powConst exponent)

let powWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[powWithImage]"
    mapSlices "PowerImage" Streaming (powImage image)

let greaterEqual (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[greaterEqual]"
    mapSlices "GreaterEqual" Streaming (greaterEqualConst value)

let greaterEqualWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[greaterEqualWithImage]"
    mapSlices "GreaterEqualImage" Streaming (greaterEqualImage image)

let greater (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[greater]"
    mapSlices "Greater" Streaming (greaterConst value)

let greaterWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[greaterWithImage]"
    mapSlices "GreaterImage" Streaming (greaterImage image)

let equal (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[equal]"
    mapSlices "Equal" Streaming (equalConst value)

let equalWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[equalWithImage]"
    mapSlices "EqualImage" Streaming (equalImage image)

let notEqual (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[notEqual]"
    mapSlices "NotEqual" Streaming (notEqualConst value)

let notEqualWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[notEqualWithImage]"
    mapSlices "NotEqualImage" Streaming (notEqualImage image)

let less (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[less]"
    mapSlices "Less" Streaming (lessConst value)

let lessWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[lessWithImage]"
    mapSlices "LessImage" Streaming (lessImage image)

let lessEqual (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[lessEqual]"
    mapSlices "LessEqual" Streaming (lessEqualConst value)

let lessEqualWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[lessEqualWithImage]"
    mapSlices "LessEqualImage" Streaming (lessEqualImage image)

let bitwiseAnd (value: int) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseAnd]"
    mapSlices "BitwiseAnd" Streaming (bitwiseAndConst value)

let bitwiseAndWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseAndWithImage]"
    mapSlices "BitwiseAndImage" Streaming (bitwiseAndImage image)

let bitwiseOr (value: int) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseOr]"
    mapSlices "BitwiseOr" Streaming (bitwiseOrConst value)

let bitwiseOrWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseOrWithImage]"
    mapSlices "BitwiseOrImage" Streaming (bitwiseOrImage image)

let bitwiseXor (value: int) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseXor]"
    mapSlices "BitwiseXor" Streaming (bitwiseXorConst value)

let bitwiseXorWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseXorWithImage]"
    mapSlices "BitwiseXorImage" Streaming (bitwiseXorImage image)

let bitwiseNot (maximum: int) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseNot]"
    mapSlices "BitwiseNot" Streaming (bitwiseNot maximum)

let bitwiseLeftShift (shiftBits: int) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseLeftShift]"
    mapSlices "BitwiseLeftShift" Streaming (bitwiseLeftShift shiftBits)

let bitwiseRightShift (shiftBits: int) : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[bitwiseRightShift]"
    mapSlices "BitwiseRightShift" Streaming (bitwiseRightShift shiftBits)
*)
let absProcess<'T when 'T: equality> : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[abs]"
    mapSlices "Abs" Streaming absSlice<'T>

let sqrtProcess<'T when 'T: equality> : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[sqrt]"
    mapSlices "Sqrt" Streaming sqrtSlice<'T>

let logProcess<'T when 'T: equality> : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[log]"
    mapSlices "Log" Streaming logSlice<'T>

let expProcess<'T when 'T: equality> : StackProcessor<Slice<'T>, Slice<'T>> =
    printfn "[exp]"
    mapSlices "Exp" Streaming expSlice<'T>

// let histogram (img: Image<'T>) : Map<'T, uint64> =
(*let histogramPerSlice<'T when 'T: comparison> : StackProcessor<Slice<'T>, Map<'T, uint64>> =
    printfn "[histogramPerSlice]"
    mapSlices "HistogramPerSlice" Streaming Slice.histogram

let histogramReducer<'T when 'T: comparison> : AsyncSeq<Map<'T, uint64>> -> Async<Map<'T, uint64>> =
    fun hist ->
        AsyncSeqExtensions.fold Slice.addHistogram Map.empty hist

let histogram<'T when 'T: comparison> : StackProcessor<Map<'T, uint64>, Map<'T, uint64>> =
    printfn "[histogramReduced]"
    fromReducer "HistogramReduced" Streaming histogramReducer
*)
let histogram<'T when 'T: comparison> : StackProcessor<Slice<'T>, Map<'T, uint64>> =
    printfn "[histogram]"
    let histogramReducer (slices: AsyncSeq<Slice<'T>>) =
        slices
        |> AsyncSeq.map Slice.histogram
        |> AsyncSeqExtensions.fold Slice.addHistogram (Map<'T,uint64> [])
    fromReducer "Histogram" Streaming histogramReducer

let map2pairs<'T, 'S when 'T: comparison> : StackProcessor<Map<'T, 'S>,('T * 'S) list> =
    printfn "[map2pairs]"
    mapSlices "map2pairs" Streaming Slice.map2pairs

let inline pairs2floats<^T, ^S when ^T : (static member op_Explicit : ^T -> float)
                                 and ^S : (static member op_Explicit : ^S -> float)> : StackProcessor<('T * 'S) list,(float * float) list> =
    printfn "[pairs2floats]"
    mapSlices "pairs2floats" Streaming Slice.pairs2floats

let inline pairs2int<^T, ^S when ^T : (static member op_Explicit : ^T -> int)
                                 and ^S : (static member op_Explicit : ^S -> int)> : StackProcessor<('T * 'S) list,(int * int) list> =
    printfn "[pairs2int]"
    mapSlices "pairs2int" Streaming Slice.pairs2ints

let create<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) : StackProcessor<unit, Slice<'T>> =
    printfn "[create]"
    {
        Name = "[create]"
        Profile = Streaming
        Apply = fun _ ->
            AsyncSeq.init (int depth) (fun i -> printfn "[create %d]" i; Slice.create<'T> width height 1u (uint i))
    }

let readSlices<'T when 'T: equality> (inputDir: string) (suffix: string) : StackProcessor<unit, Slice<'T>> =
    printfn "[readSlices]"
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = filenames.Length
    {
        Name = $"[readSlices {inputDir}]"
        Profile = Streaming
        Apply = fun _ ->
            AsyncSeq.init (int depth) (fun i -> 
                let fileName = filenames[int i]; 
                printfn "[readSlices] Reading slice %d to %s" (uint i) fileName
                Slice.readSlice<'T> (uint i) fileName)
    }

let readSliceN<'T when 'T: equality> (idx: uint) (inputDir: string) (suffix: string) : StackProcessor<unit, Slice<'T>> =
    printfn "[readSliceN]"
    let fileNames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    if fileNames.Length <= (int idx) then
        failwith "[readSliceN] Index out of bounds"
    else
    let fileName = fileNames[int idx]
    {
        Name = $"[readSliceN {fileName}]"
        Profile = Streaming
        Apply = fun _ ->
            AsyncSeq.init 1 (fun i -> 
                printfn "[readSliceN] Reading slice %d to %s" (uint idx) fileName
                Slice.readSlice<'T> (uint idx) fileName)
    }

let readRandomSlices<'T when 'T: equality> (count: uint) (inputDir: string) (suffix: string) :StackProcessor<unit, Slice<'T>> =
    printfn "[readRandomSlices]"
    let fileNames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices (int count)
    {
        Name = $"[readRandomSlices {inputDir}]"
        Profile = Streaming
        Apply = fun _ ->
            AsyncSeq.init (int count) (fun i -> 
                let fileName = fileNames[int i]; 
                printfn "[readRandomSlices] Reading slice %d to %s" (uint i) fileName
                Slice.readSlice<'T> (uint i) fileName)
    }

let addNormalNoise (mean: float) (stddev: float) : StackProcessor<Slice<'T> ,Slice<'T>> =
    printfn "[addNormalNoise]"
    mapSlices "addNormalNoise" Streaming (addNormalNoise mean stddev)

let threshold (lower: float) (upper: float) : StackProcessor<Slice<'T> ,Slice<'T>> =
    printfn "[threshold]"
    mapSlices "Threshold" Streaming (threshold lower upper)

let discreteGaussian (sigma: float) : StackProcessor<Slice<'T> ,Slice<'T>> =
    printfn "[discreteGaussian]"
    let depth = 1u + 2u * uint (0.5 + sigma)
    mapSlicesWindowed "discreteGaussian" depth (discreteGaussian sigma)

type FileInfo = Slice.FileInfo
let getFileInfo(fname: string): FileInfo = Slice.getFileInfo(fname)
let getVolumeSize (inputDir: string) (suffix: string) = Slice.getVolumeSize inputDir suffix

type ImageStats = Slice.ImageStats
let computeStats<'T when 'T : equality> : StackProcessor<Slice<'T>, ImageStats> =
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
    fromReducer "Compute Statistics" Streaming computeStatsReducer
