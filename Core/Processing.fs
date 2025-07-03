module Processing

open System
open System.IO
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open AsyncSeqExtensions
open SourceSink
open Core
open Core.Helpers
open Routing
open Slice
open Image

// --- Processing Utilities ---
let private explodeSlice (slices: Slice<'T>) : AsyncSeq<Slice<'T>> =
    let baseIndex = slices.Index
    let volume = slices |> Slice.toImage
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
        Name = label; 
        Profile = profile
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun slice -> f slice)
    }

/// mapWindowed keeps a running window along the slice direction of depth images
/// and processes them by f. The stepping size of the running window is stride.
/// So if depth is 3 and stride is 1 then first image 0,1,2 is sent to f, then 1, 2, 3
/// and so on. If depth is 3 and stride is 3, then it'll be image 0, 1, 2 followed by
/// 3, 4, 5. It is also possible to use this for sampling, e.g., setting depth to 1
/// and stride to 2 sends every second image to f.  
let mapWindowed (label: string) (depth: uint) (stride: uint) (f: 'S list -> 'T list) : Pipe<'S,'T> =
    {
        Name = label; 
        Profile = Sliding depth
        Apply = fun input ->
            printfn "[Runtine analysis: Windowed analysis size]"
            AsyncSeqExtensions.windowed depth stride input
                |> AsyncSeq.collect (f  >> AsyncSeq.ofSeq)
    }

let inline cast<'S,'T when 'S: equality and 'T: equality> label fct : Pipe<Slice<'S>, Slice<'T>> =
    {
        Name = label
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun slice -> fct slice)
    }

let castUInt8ToInt8 = cast "[castUInt8ToInt8]" Slice.castUInt8ToInt8
let castUInt8ToUInt16 = cast "[castUInt8ToUInt16]" Slice.castUInt8ToUInt16
let castUInt8ToInt16 = cast "[castUInt8ToInt16]" Slice.castUInt8ToInt16
let castUInt8ToUInt = cast "[castUInt8ToUInt]" Slice.castUInt8ToUInt
let castUInt8ToInt = cast "[castUInt8ToInt]" Slice.castUInt8ToInt
let castUInt8ToUInt64 = cast "[castUInt8ToUInt64]" Slice.castUInt8ToUInt64
let castUInt8ToInt64 = cast "[castUInt8ToInt64]" Slice.castUInt8ToInt64
let castUInt8ToFloat32 = cast "[castUInt8ToFloat32]" Slice.castUInt8ToFloat32
let castUInt8ToFloat = cast "[castUInt8ToFloat]" Slice.castUInt8ToFloat
let castInt8ToUInt8 = cast "[castInt8ToUInt8]" Slice.castInt8ToUInt8
let castInt8ToUInt16 = cast "[castInt8ToUInt16]" Slice.castInt8ToUInt16
let castInt8ToInt16 = cast "[castInt8ToInt16]" Slice.castInt8ToInt16
let castInt8ToUInt = cast "[castInt8ToUInt]" Slice.castInt8ToUInt
let castInt8ToInt = cast "[castInt8ToInt]" Slice.castInt8ToInt
let castInt8ToUInt64 = cast "[castInt8ToUInt64]" Slice.castInt8ToUInt64
let castInt8ToInt64 = cast "[castInt8ToInt64]" Slice.castInt8ToInt64
let castInt8ToFloat32 = cast "[castInt8ToFloat32]" Slice.castInt8ToFloat32
let castInt8ToFloat = cast "[castInt8ToFloat]" Slice.castInt8ToFloat

let castUInt16ToUInt8 = cast "[castUInt16ToUInt8]" Slice.castUInt16ToUInt8
let castUInt16ToInt8 = cast "[castUInt16ToInt8]" Slice.castUInt16ToInt8
let castUInt16ToInt16 = cast "[castUInt16ToInt16]" Slice.castUInt16ToInt16
let castUInt16ToUInt = cast "[castUInt16ToUInt]" Slice.castUInt16ToUInt
let castUInt16ToInt = cast "[castUInt16ToInt]" Slice.castUInt16ToInt
let castUInt16ToUInt64 = cast "[castUInt16ToUInt64]" Slice.castUInt16ToUInt64
let castUInt16ToInt64 = cast "[castUInt16ToInt64]" Slice.castUInt16ToInt64
let castUInt16ToFloat32 = cast "[castUInt16ToFloat32]" Slice.castUInt16ToFloat32
let castUInt16ToFloat = cast "[castUInt16ToFloat]" Slice.castUInt16ToFloat
let castInt16ToUInt8 = cast "[castInt16ToUInt8]" Slice.castInt16ToUInt8
let castInt16ToInt8 = cast "[castInt16ToInt8]" Slice.castInt16ToInt8
let castInt16ToUInt16 = cast "[castInt16ToUInt16]" Slice.castInt16ToUInt16
let castInt16ToUInt = cast "[castInt16ToUInt]" Slice.castInt16ToUInt
let castInt16ToInt = cast "[castInt16ToInt]" Slice.castInt16ToInt
let castInt16ToUInt64 = cast "[castInt16ToUInt64]" Slice.castInt16ToUInt64
let castInt16ToInt64 = cast "[castInt16ToInt64]" Slice.castInt16ToInt64
let castInt16ToFloat32 = cast "[castInt16ToFloat32]" Slice.castInt16ToFloat32
let castInt16ToFloat = cast "[castInt16ToFloat]" Slice.castInt16ToFloat

let castUIntToUInt8 = cast "[castUIntToUInt8]" Slice.castUIntToUInt8
let castUIntToInt8 = cast "[castUIntToInt8]" Slice.castUIntToInt8
let castUIntToUInt16 = cast "[castUIntToUInt16]" Slice.castUIntToUInt16
let castUIntToInt16 = cast "[castUIntToInt16]" Slice.castUIntToInt16
let castUIntToInt = cast "[castUIntToInt]" Slice.castUIntToInt
let castUIntToUInt64 = cast "[castUIntToUInt64]" Slice.castUIntToUInt64
let castUIntToInt64 = cast "[castUIntToInt64]" Slice.castUIntToInt64
let castUIntToFloat32 = cast "[castUIntToFloat32]" Slice.castUIntToFloat32
let castUIntToFloat = cast "[castUIntToFloat]" Slice.castUIntToFloat
let castIntToUInt8 = cast "[castIntToUInt8]" Slice.castIntToUInt8
let castIntToInt8 = cast "[castIntToInt8]" Slice.castIntToInt8
let castIntToUInt16 = cast "[castIntToUInt16]" Slice.castIntToUInt16
let castIntToInt16 = cast "[castIntToInt16]" Slice.castIntToInt16
let castIntToUInt = cast "[castIntToUInt]" Slice.castIntToUInt
let castIntToUInt64 = cast "[castIntToUInt64]" Slice.castIntToUInt64
let castIntToInt64 = cast "[castIntToInt64]" Slice.castIntToInt64
let castIntToFloat32 = cast "[castIntToFloat32]" Slice.castIntToFloat32
let castIntToFloat = cast "[castIntToFloat]" Slice.castIntToFloat

let castUInt64ToUInt8 = cast "[castUInt64ToUInt8]" Slice.castUInt64ToUInt8
let castUInt64ToInt8 = cast "[castUInt64ToInt8]" Slice.castUInt64ToInt8
let castUInt64ToUInt16 = cast "[castUInt64ToUInt16]" Slice.castUInt64ToUInt16
let castUInt64ToInt16 = cast "[castUInt64ToInt16]" Slice.castUInt64ToInt16
let castUInt64ToUInt = cast "[castUInt64ToUInt]" Slice.castUInt64ToUInt
let castUInt64ToInt = cast "[castUInt64ToInt]" Slice.castUInt64ToInt
let castUInt64ToInt64 = cast "[castUInt64ToInt64]" Slice.castUInt64ToInt64
let castUInt64ToFloat32 = cast "[castUInt64ToFloat32]" Slice.castUInt64ToFloat32
let castUInt64ToFloat = cast "[castUInt64ToFloat]" Slice.castUInt64ToFloat
let castInt64ToUInt8 = cast "[castInt64ToUInt8]" Slice.castInt64ToUInt8
let castInt64ToInt8 = cast "[castInt64ToInt8]" Slice.castInt64ToInt8
let castInt64ToUInt16 = cast "[castInt64ToUInt16]" Slice.castInt64ToUInt16
let castInt64ToInt16 = cast "[castInt64ToInt16]" Slice.castInt64ToInt16
let castInt64ToUInt = cast "[castInt64ToUInt]" Slice.castInt64ToUInt
let castInt64ToInt = cast "[castInt64ToInt]" Slice.castInt64ToInt
let castInt64ToUInt64 = cast "[castInt64ToUInt64]" Slice.castInt64ToUInt64
let castInt64ToFloat32 = cast "[castInt64ToFloat32]" Slice.castInt64ToFloat32
let castInt64ToFloat = cast "[castInt64ToFloat]" Slice.castInt64ToFloat

let castFloat32ToUInt8 = cast "[castFloat32ToUInt8]" Slice.castFloat32ToUInt8
let castFloat32ToInt8 = cast "[castFloat32ToInt8]" Slice.castFloat32ToInt8
let castFloat32ToUInt16 = cast "[castFloat32ToUInt16]" Slice.castFloat32ToUInt16
let castFloat32ToInt16 = cast "[castFloat32ToInt16]" Slice.castFloat32ToInt16
let castFloat32ToUInt = cast "[castFloat32ToUInt]" Slice.castFloat32ToUInt
let castFloat32ToInt = cast "[castFloat32ToInt]" Slice.castFloat32ToInt
let castFloat32ToUInt64 = cast "[castFloat32ToUInt64]" Slice.castFloat32ToUInt64
let castFloat32ToInt64 = cast "[castFloat32ToInt64]" Slice.castFloat32ToInt64
let castFloat32ToFloat = cast "[castFloat32ToFloat]" Slice.castFloat32ToFloat
let castFloatToUInt8 = cast "[castFloatToUInt8]" Slice.castFloatToUInt8
let castFloatToInt8 = cast "[castFloatToInt8]" Slice.castFloatToInt8
let castFloatToUInt16 = cast "[castFloatToUInt16]" Slice.castFloatToUInt16
let castFloatToInt16 = cast "[castFloatToInt16]" Slice.castFloatToInt16
let castFloatToUInt = cast "[castFloatToUInt]" Slice.castFloatToUInt
let castFloatToInt = cast "[castFloatToInt]" Slice.castFloatToInt
let castFloatToUIn64 = cast "[castFloatToUIn64]" Slice.castFloatToUIn64
let castFloatToInt64 = cast "[castFloatToInt64]" Slice.castFloatToInt64
let castFloatToFloat32 = cast "[castFloatToFloat32]" Slice.castFloatToFloat32

let add (slice: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    map "Add" Streaming (add slice)

let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "scalarAddSlice" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.scalarAddSlice<^T> i s))
    }

let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "sliceAddScalar" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.sliceAddScalar<^T> s i))
    }

let sub (slice: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    map "sub" Streaming (sub slice)

let inline scalarSubSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "scalarSubSlice" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.scalarSubSlice<^T> i s))
    }

let inline sliceSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "sliceSubScalar" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.sliceSubScalar<^T> s i))
    }

let mul (slice: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    map "mul" Streaming (mul slice)

let inline scalarMulSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "scalarMulSlice" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.scalarMulSlice<^T> i s))
    }

let inline sliceMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "sliceMulScalar" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.sliceMulScalar<^T> s i))
    }

let div (slice: Slice<'T>) : Pipe<Slice<'T>, Slice<'T>> =
    map "div" Streaming (div slice)

let inline scalarDivSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "scalarDivSlice" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.scalarDivSlice<^T> i s))
    }

let inline sliceDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) : Pipe<Slice<^T>, Slice<^T>> =
    {
        Name = "sliceDivScalar" 
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun s -> (Slice.sliceDivScalar<^T> s i))
    }


(*
let modulus (value: uint) : Pipe<Slice<'T>, Slice<'T>> =
    map "Modulus" Streaming (modulusConst value)

let modulusWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "ModulusImage" Streaming (modulusImage image)

let pow (exponent: float) : Pipe<Slice<'T>, Slice<'T>> =
    map "Power" Streaming (powConst exponent)

let powWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "PowerImage" Streaming (powImage image)

let greaterEqual (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    map "GreaterEqual" Streaming (greaterEqualConst value)

let greaterEqualWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "GreaterEqualImage" Streaming (greaterEqualImage image)

let greater (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    map "Greater" Streaming (greaterConst value)

let greaterWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "GreaterImage" Streaming (greaterImage image)

let equal (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    map "Equal" Streaming (equalConst value)

let equalWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "EqualImage" Streaming (equalImage image)

let notEqual (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    map "NotEqual" Streaming (notEqualConst value)

let notEqualWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "NotEqualImage" Streaming (notEqualImage image)

let less (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    map "Less" Streaming (lessConst value)

let lessWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "LessImage" Streaming (lessImage image)

let lessEqual (value: float) : Pipe<Slice<'T>, Slice<'T>> =
    map "LessEqual" Streaming (lessEqualConst value)

let lessEqualWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "LessEqualImage" Streaming (lessEqualImage image)

let bitwiseAnd (value: int) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseAnd" Streaming (bitwiseAndConst value)

let bitwiseAndWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseAndImage" Streaming (bitwiseAndImage image)

let bitwiseOr (value: int) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseOr" Streaming (bitwiseOrConst value)

let bitwiseOrWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseOrImage" Streaming (bitwiseOrImage image)

let bitwiseXor (value: int) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseXor" Streaming (bitwiseXorConst value)

let bitwiseXorWithImage (image: Image) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseXorImage" Streaming (bitwiseXorImage image)

let bitwiseNot (maximum: int) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseNot" Streaming (bitwiseNot maximum)

let bitwiseLeftShift (shiftBits: int) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseLeftShift" Streaming (bitwiseLeftShift shiftBits)

let bitwiseRightShift (shiftBits: int) : Pipe<Slice<'T>, Slice<'T>> =
    map "BitwiseRightShift" Streaming (bitwiseRightShift shiftBits)
*)
let absProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    map "Abs" Streaming absSlice<'T>

let sqrtProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    map "Sqrt" Streaming sqrtSlice<'T>

let logProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    map "Log" Streaming logSlice<'T>

let expProcess<'T when 'T: equality> : Pipe<Slice<'T>, Slice<'T>> =
    map "Exp" Streaming expSlice<'T>

let histogram<'T when 'T: comparison> : Pipe<Slice<'T>, Map<'T, uint64>> =
    let histogramReducer (slices: AsyncSeq<Slice<'T>>) =
        slices
        |> AsyncSeq.map Slice.histogram
        |> AsyncSeqExtensions.fold Slice.addHistogram (Map<'T,uint64> [])
    reduce "Histogram" StreamingConstant histogramReducer

let map2pairs<'T, 'S when 'T: comparison> : Pipe<Map<'T, 'S>,('T * 'S) list> =
    map "map2pairs" Streaming Slice.map2pairs

let inline pairs2floats<^T, ^S when ^T : (static member op_Explicit : ^T -> float)
                                 and ^S : (static member op_Explicit : ^S -> float)> : Pipe<('T * 'S) list,(float * float) list> =
    map "pairs2floats" Streaming Slice.pairs2floats

let inline pairs2int<^T, ^S when ^T : (static member op_Explicit : ^T -> int)
                                 and ^S : (static member op_Explicit : ^S -> int)> : Pipe<('T * 'S) list,(int * int) list> =
    map "pairs2int" Streaming Slice.pairs2ints

let addNormalNoise (mean: float) (stddev: float) : Pipe<Slice<'T> ,Slice<'T>> =
    map "addNormalNoise" Streaming (addNormalNoise mean stddev)

let threshold (lower: float) (upper: float) : Pipe<Slice<'T> ,Slice<'T>> =
    map "Threshold" Streaming (threshold lower upper)

// Chained type definitions do expose the originals
let zeroPad = ImageFunctions.ZeroPad
let periodicPad = ImageFunctions.PerodicPad
let zeroFluxNeumannPad = ImageFunctions.ZeroFluxNeumannPad
let valid = ImageFunctions.Valid
let same = ImageFunctions.Same

let convolve (kern: Slice<'T>) (boundaryCondition: BoundaryCondition option) (windowSize: uint option) (stride: uint option): Pipe<Slice<'T> ,Slice<'T>> =
    let ksz = kern |> GetDepth |>  max 1u
    let dpth = Option.defaultValue ksz windowSize |> max ksz
    let strd = Option.defaultValue (1u+dpth-ksz) stride |> max 1u
    mapWindowed "discreteGaussian" dpth strd (stack >> convolve (Some valid) boundaryCondition kern >> unstack)

let conv (kern: Slice<'T>): Pipe<Slice<'T> ,Slice<'T>> =
    convolve kern None None None

let convolveStreams
    (kernelSrc : Pipe<'S, Slice<'T>>)
    (imageSrc  : Pipe<'S, Slice<'T>>) : Pipe<'S, Slice<'T>> =

    zipWith
        (fun kernel image ->
            let convPipe = conv kernel
            image
            |> AsyncSeq.singleton     // wrap image slice as stream
            |> convPipe.Apply         // apply convolution
            |> Helpers.singletonPipe $"conv({kernel.Index},{image.Index})"
        )
        kernelSrc
        imageSrc
    |> Helpers.bindPipe

let discreteGaussian (sigma: float) (kernelSize: uint option) (boundaryCondition: BoundaryCondition option) (windowSize: uint option) (stride: uint option): Pipe<Slice<'T> ,Slice<'T>> =
    let ksz = Option.defaultValue (1u + 2u * uint (0.5 + sigma)) kernelSize
    let dpth = Option.defaultValue ksz windowSize |> max ksz
    let minStride = 1u
    let strd = Option.defaultValue (1u+dpth-ksz) stride |> max minStride
    mapWindowed "convGauss/discreteGaussian" dpth strd (stack >> discreteGaussian 3u sigma (Some ksz) (Some valid) boundaryCondition >> unstack)

let convGauss (sigma: float) (boundaryCondition: BoundaryCondition option) : Pipe<Slice<'T> ,Slice<'T>> =
    discreteGaussian sigma None boundaryCondition None None

let skipFirstLast (n: int) (lst: 'a list) : 'a list =
    let m = lst.Length - 2*n;
    if m <= 0 then []
    else lst |> List.skip n |> List.take m 

let private binaryMathMorph (name: string) f (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<uint8> ,Slice<uint8>> =
    let ksz = 1u+2u*radius
    let dpth = Option.defaultValue ksz windowSize |> max ksz
    let strd = Option.defaultValue (1u+dpth-ksz) stride |> max 1u
    mapWindowed $"{name}" dpth strd (stack >> f radius  >> unstack >> skipFirstLast (int radius))

let binaryErode (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<uint8> ,Slice<uint8>> =
    binaryMathMorph "binaryErode" binaryErode radius windowSize stride

let binaryDilate (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<uint8> ,Slice<uint8>> =
    binaryMathMorph "binaryDilate" binaryDilate radius windowSize stride

let binaryOpening (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<uint8> ,Slice<uint8>> =
    binaryMathMorph "binaryOpening" binaryOpening radius windowSize stride

let binaryClosing (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<uint8> ,Slice<uint8>> =
    binaryMathMorph "binaryClosing" binaryClosing radius windowSize stride

let piecewiseConnectedComponents (windowSize: uint option) : Pipe<Slice<uint8> ,Slice<uint64>> =
    let dpth = Option.defaultValue 1u windowSize |> max 1u
    mapWindowed "connectedComponents" dpth dpth (stack >> connectedComponents >> unstack)

type ImageStats = Slice.ImageStats
let computeStats<'T when 'T : equality> : Pipe<Slice<'T>, ImageStats> =
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


/////////////////////////////////////////////////////////////////////
// new experiement with Operator type and more
let liftUnaryOp name (f: Slice<'T> -> Slice<'T>) : Operation<Slice<'T>,Slice<'T>> =
    { 
        Name = name
        Transition = transition Streaming Streaming
        Pipe = // This looks like overloading. Only new information is Apply
        { 
            Name = name
            Profile = Streaming
            Apply = fun input -> input |> AsyncSeq.map f 
        } 
    }

(*
let liftImageScalarOpUInt8 (name : string) (scalar : uint8) (core : Slice<uint8> -> uint8 -> Slice<uint8>) : Operation<Slice<uint8>,Slice<uint8>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpUInt16 (name : string) (scalar : uint16) (core : Slice<uint16> -> uint16 -> Slice<uint16>) : Operation<Slice<uint16>,Slice<uint16>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpUInt (name : string) (scalar : uint) (core : Slice<uint> -> uint -> Slice<uint>) : Operation<Slice<uint>,Slice<uint>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpUInt64 (name : string) (scalar : uint64) (core : Slice<uint64> -> uint64 -> Slice<uint64>) : Operation<Slice<uint64>,Slice<uint64>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpInt8 (name : string) (scalar : int8) (core : Slice<int8> -> int8 -> Slice<int8>) : Operation<Slice<int8>,Slice<int8>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpInt16 (name : string) (scalar : int16) (core : Slice<int16> -> int16 -> Slice<int16>) : Operation<Slice<int16>,Slice<int16>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpInt (name : string) (scalar : int) (core : Slice<int> -> int -> Slice<int>) : Operation<Slice<int>,Slice<int>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpInt64 (name : string) (scalar : int64) (core : Slice<int64> -> int64 -> Slice<int64>) : Operation<Slice<int64>,Slice<int64>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpFloat32 (name : string) (scalar : float32) (core : Slice<float32> -> float32 -> Slice<float32>) : Operation<Slice<float32>,Slice<float32>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftImageScalarOpFloat (name : string) (scalar : float) (core : Slice<float> -> float -> Slice<float>) : Operation<Slice<float>,Slice<float>> =
    liftUnaryOp name (fun s -> core s scalar)

let liftScalarImageOpUInt8 (name : string) (scalar : uint8) (core : uint8 -> Slice<uint8> -> Slice<uint8>) : Operation<Slice<uint8>,Slice<uint8>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpUInt16 (name : string) (scalar : uint16) (core : uint16 -> Slice<uint16> -> Slice<uint16>) : Operation<Slice<uint16>,Slice<uint16>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpUInt (name : string) (scalar : uint) (core : uint -> Slice<uint> -> Slice<uint>) : Operation<Slice<uint>,Slice<uint>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpUInt64 (name : string) (scalar : uint64) (core : uint64 -> Slice<uint64> -> Slice<uint64>) : Operation<Slice<uint64>,Slice<uint64>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpInt8 (name : string) (scalar : int8) (core : int8 -> Slice<int8> -> Slice<int8>) : Operation<Slice<int8>,Slice<int8>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpInt16 (name : string) (scalar : int16) (core : int16 -> Slice<int16> -> Slice<int16>) : Operation<Slice<int16>,Slice<int16>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpInt (name : string) (scalar : int) (core : int -> Slice<int> -> Slice<int>) : Operation<Slice<int>,Slice<int>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpInt64 (name : string) (scalar : int64) (core : int64 -> Slice<int64> -> Slice<int64>) : Operation<Slice<int64>,Slice<int64>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpFloat32 (name : string) (scalar : float32) (core : float32 -> Slice<float32> -> Slice<float32>) : Operation<Slice<float32>,Slice<float32>> =
    liftUnaryOp name (fun s -> core s scalar)
let liftScalarImageOpFloat (name : string) (scalar : float) (core : float -> Slice<float> -> Slice<float>) : Operation<Slice<float>,Slice<float>> =
    liftUnaryOp name (fun s -> core s scalar)
*)

let liftWindowedOp (name: string) (window: uint) (stride: uint) (f: Slice<'S> -> Slice<'T>) : Operation<Slice<'S>, Slice<'T>> =
    {
        Name = name
        Transition = transition (Sliding window) Streaming
        Pipe = mapWindowed name window stride (stack >> f >> unstack)
    }

let liftWindowedTrimOp (name: string) (window: uint) (stride: uint) (trim: uint) (f: Slice<'S> -> Slice<'T>)
    : Operation<Slice<'S>, Slice<'T>> =
    {
        Name = name
        Transition = transition (Sliding window) Streaming
        Pipe =
            mapWindowed name window stride (fun windowSlices ->
                windowSlices
                |> stack
                |> f
                |> unstack
                |> skipFirstLast (int trim)
            )
    }


let liftUnaryOpInt (name: string) (f: Slice<int> -> Slice<int>) =
    liftUnaryOp name f

let liftUnaryOpFloat32 (name: string) (f: Slice<float32> -> Slice<float32>) =
    liftUnaryOp name f

let liftUnaryOpFloat (name: string) (f: Slice<float> -> Slice<float>) =
    liftUnaryOp name f

let liftBinaryOp (name: string) (f: Slice<'T> -> Slice<'T> -> Slice<'T>) : Operation<Slice<'T> * Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Streaming Streaming
        Pipe =
            {
                Name = name
                Profile = Streaming
                Apply = fun input ->
                    input
                    |> AsyncSeq.map (fun (a, b) -> f a b)
            }
    }

let liftBinaryOpFloat (name: string) (f: Slice<float> -> Slice<float> -> Slice<float>) =
    liftBinaryOp name f

let liftBinaryZipOp (name: string) (f: Slice<'T> -> Slice<'T> -> Slice<'T>) (p1: Pipe<'In, Slice<'T>>) (p2: Pipe<'In, Slice<'T>>) : Pipe<'In, Slice<'T>> =
    zipWith f p1 p2

let liftFullOp
    (name: string)
    (f: Slice<'T> -> Slice<'T>)
    : Operation<Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe =
            {
                Name = name
                Profile = Full
                Apply = fun input ->
                    asyncSeq {
                        let! slices = input |> AsyncSeq.toListAsync
                        let stack = Slice.stack slices
                        let result = f stack
                        yield! Slice.unstack result |> AsyncSeq.ofSeq
                    }
            }
    }

let liftFullParamOp
    (name: string)
    (f: 'P -> Slice<'T> -> Slice<'T>)
    (param: 'P)
    : Operation<Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe =
            {
                Name = name
                Profile = Full
                Apply = fun input ->
                    asyncSeq {
                        let! slices = input |> AsyncSeq.toListAsync
                        let stack = Slice.stack slices
                        let result = f param stack
                        yield! Slice.unstack result |> AsyncSeq.ofSeq
                    }
            }
    }

let liftFullParam2Op
    (name: string)
    (f: 'P -> 'Q -> Slice<'T> -> Slice<'T>)
    (param1: 'P)
    (param2: 'Q)
    : Operation<Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe =
            {
                Name = name
                Profile = Full
                Apply = fun input ->
                    asyncSeq {
                        let! slices = input |> AsyncSeq.toListAsync
                        let stack = Slice.stack slices
                        let result = f param1 param2 stack
                        yield! Slice.unstack result |> AsyncSeq.ofSeq
                    }
            }
    }

let liftMapOp<'T, 'U when 'T: equality and 'T: comparison> (name: string) (f: Slice<'T> -> 'U) : Operation<Slice<'T>, 'U> =
    {
        Name = name
        Transition = transition Streaming Streaming
        Pipe =
            {
                Name = name
                Profile = Streaming
                Apply = fun input -> input |> AsyncSeq.map f
            }
    }


let absIntOp       name = liftUnaryOpInt name absSlice
let absFloat32Op   name = liftUnaryOpFloat32 name absSlice
let absFloatOp     name = liftUnaryOpFloat name absSlice
let logFloat32Op   name = liftUnaryOpFloat32 name logSlice
let logFloatOp     name = liftUnaryOpFloat name logSlice
let log10Float32Op name = liftUnaryOpFloat32 name log10Slice
let log10FloatOp   name = liftUnaryOpFloat name log10Slice
let expFloat32Op   name = liftUnaryOpFloat32 name expSlice
let expFloatOp     name = liftUnaryOpFloat name expSlice
let sqrtIntOp      name = liftUnaryOpInt name sqrtSlice
let sqrtFloat32Op  name = liftUnaryOpFloat32 name sqrtSlice
let sqrtFloatOp    name = liftUnaryOpFloat name sqrtSlice
let squareIntOp    name = liftUnaryOpInt name squareSlice
let squareFloat32Op name = liftUnaryOpFloat32 name squareSlice
let squareFloatOp  name = liftUnaryOpFloat name squareSlice
let sinFloat32Op   name = liftUnaryOpFloat32 name sinSlice
let sinFloatOp     name = liftUnaryOpFloat name sinSlice
let cosFloat32Op   name = liftUnaryOpFloat32 name cosSlice
let cosFloatOp     name = liftUnaryOpFloat name cosSlice
let tanFloat32Op   name = liftUnaryOpFloat32 name tanSlice
let tanFloatOp     name = liftUnaryOpFloat name tanSlice
let asinFloat32Op  name = liftUnaryOpFloat32 name asinSlice
let asinFloatOp    name = liftUnaryOpFloat name asinSlice
let acosFloat32Op  name = liftUnaryOpFloat32 name acosSlice
let acosFloatOp    name = liftUnaryOpFloat name acosSlice
let atanFloat32Op  name = liftUnaryOpFloat32 name atanSlice
let atanFloatOp    name = liftUnaryOpFloat name atanSlice
let roundFloat32Op name = liftUnaryOpFloat32 name roundSlice
let roundFloatOp   name = liftUnaryOpFloat name roundSlice

let roundFloatToUint v = uint (v+0.5)

let discreteGaussianOp (name:string) (sigma:float) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option): Operation<Slice<float>, Slice<float>> =
    let ksz = 2.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> min ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    liftWindowedOp name win stride (fun slices -> Slice.discreteGaussian 3u sigma (ksz |> Some) (Some valid) bc slices)

let windowFromSlices (a: Slice<'T>) (b: Slice<'T>) : uint =
    min (a |> Slice.GetDepth) (a |> Slice.GetDepth)

let windowFromKernel (k: Slice<'T>) : uint =
    max 3u (k |> Slice.GetDepth)

// stride calculation example
// ker = 3, win = 7
// Image position:  2 1 0 1 2 3 4 5 6 7 8 9 
// First window         * * * * * * *
// Kern position1   * * *            
//                    * * *         
//                      * * * √        
//                        * * * √      
//                          * * * √   
//                            * * * √    
//                              * * * √   
//                                * * *
//                                  * * *
//                                    * * *
// Next window                    * * * * * * *
// Kern                       * * *
//                              * * *         
//                                * * * √  
//                                  * * * √   
//                                    * * * √
//.                                     * * * √
//                                        * * * √
//                                          * * *
//                                            * * *

let convolveOp (name: string) (kernel: Slice<'T>) (bc: BoundaryCondition option) (winSz: uint option): Operation<Slice<'T>, Slice<'T>> =
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win-ksz+1u
    liftWindowedOp name win stride (fun slices -> Slice.convolve (Some valid) bc slices kernel)

let private makeMorphOp (name:string) (radius:uint) (winSz: uint option) (core: uint -> Slice<'T> -> Slice<'T>) : Operation<Slice<'T>,Slice<'T>> when 'T: equality =
    let winFromRadius (r:uint) = 2u * r + 1u
    let ksz   = winFromRadius radius
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win - ksz + 1u
    liftWindowedTrimOp name win stride radius (fun slices -> core radius slices)

// Only uint8
let binaryErodeOp     name radius winSz = makeMorphOp name radius winSz Slice.binaryErode
let binaryDilateOp    name radius winSz = makeMorphOp name radius winSz Slice.binaryDilate
let binaryOpeningOp   name radius winSz = makeMorphOp name radius winSz Slice.binaryOpening
let binaryClosingOp   name radius winSz = makeMorphOp name radius winSz Slice.binaryClosing
let binaryFillHolesOp name = liftFullOp name Slice.binaryFillHoles
let connectedComponentsOp (name: string) : Operation<Slice<uint8>, Slice<uint64>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe =
            {
                Name = name
                Profile = Full
                Apply = fun input ->
                    asyncSeq {
                        let! slices = input |> AsyncSeq.toListAsync
                        let stack = Slice.stack slices
                        let result = Slice.connectedComponents stack
                        yield! Slice.unstack result |> AsyncSeq.ofSeq
                    }
            }
    }

(* Assymetry could be handled as
let inline subFromScalarIntOp (name:string) (scalar:int) =
    liftUnaryOp name (fun img -> subInt img scalar |> fun s -> s )
let inline subFromScalarFloatOp (name:string) (scalar:float) =
    liftUnaryOp name (fun img -> subFloat img scalar |> fun s -> s )
*)
(*
let addUInt8Op   name scalar = liftImageScalarOpUInt8    name scalar Slice.sliceAddScalar
let addInt8Op    name scalar = liftImageScalarOpInt8     name scalar Slice.sliceAddScalar
let addUInt16Op  name scalar = liftImageScalarOpUInt16   name scalar Slice.sliceAddScalar
let addInt16Op   name scalar = liftImageScalarOpInt16    name scalar Slice.sliceAddScalar
let addUIntOp    name scalar = liftImageScalarOpUInt     name scalar Slice.sliceAddScalar
let addIntOp     name scalar = liftImageScalarOpInt      name scalar Slice.sliceAddScalar
let addUInt64Op  name scalar = liftImageScalarOpUInt64   name scalar Slice.sliceAddScalar
let addInt64Op   name scalar = liftImageScalarOpInt64    name scalar Slice.sliceAddScalar
let addFloat32Op name scalar = liftImageScalarOpFloat32  name scalar Slice.sliceAddScalar
let addFloatOp   name scalar = liftImageScalarOpFloat    name scalar Slice.sliceMulScalar

let subUInt8Op   name scalar = liftImageScalarOpUInt8    name scalar Slice.sliceSubScalar
let subInt8Op    name scalar = liftImageScalarOpInt8     name scalar Slice.sliceSubScalar
let subUInt16Op  name scalar = liftImageScalarOpUInt16   name scalar Slice.sliceSubScalar
let subInt16Op   name scalar = liftImageScalarOpInt16    name scalar Slice.sliceSubScalar
let subUIntOp    name scalar = liftImageScalarOpUInt     name scalar Slice.sliceSubScalar
let subIntOp     name scalar = liftImageScalarOpInt      name scalar Slice.sliceSubScalar
let subUInt64Op  name scalar = liftImageScalarOpUInt64   name scalar Slice.sliceSubScalar
let subInt64Op   name scalar = liftImageScalarOpInt64    name scalar Slice.sliceSubScalar
let subFloat32Op name scalar = liftImageScalarOpFloat32  name scalar Slice.sliceSubScalar
let subFloatOp   name scalar = liftImageScalarOpFloat    name scalar Slice.sliceSubScalar

let mulUInt8Op   name scalar = liftImageScalarOpUInt8    name scalar Slice.sliceMulScalar
let mulInt8Op    name scalar = liftImageScalarOpInt8     name scalar Slice.sliceMulScalar
let mulUInt16Op  name scalar = liftImageScalarOpUInt16   name scalar Slice.sliceMulScalar
let mulInt16Op   name scalar = liftImageScalarOpInt16    name scalar Slice.sliceMulScalar
let mulUIntOp    name scalar = liftImageScalarOpUInt     name scalar Slice.sliceMulScalar
let mulIntOp     name scalar = liftImageScalarOpInt      name scalar Slice.sliceMulScalar
let mulUInt64Op  name scalar = liftImageScalarOpUInt64   name scalar Slice.sliceMulScalar
let mulInt64Op   name scalar = liftImageScalarOpInt64    name scalar Slice.sliceMulScalar
let mulFloat32Op name scalar = liftImageScalarOpFloat32  name scalar Slice.sliceMulScalar
let mulFloatOp   name scalar = liftImageScalarOpFloat    name scalar Slice.sliceMulScalar

let divUInt8Op   name scalar = liftImageScalarOpUInt8    name scalar Slice.sliceDivScalar
let divInt8Op    name scalar = liftImageScalarOpInt8     name scalar Slice.sliceDivScalar
let divUInt16Op  name scalar = liftImageScalarOpUInt16   name scalar Slice.sliceDivScalar
let divInt16Op   name scalar = liftImageScalarOpInt16    name scalar Slice.sliceDivScalar
let divUIntOp    name scalar = liftImageScalarOpUInt     name scalar Slice.sliceDivScalar
let divIntOp     name scalar = liftImageScalarOpInt      name scalar Slice.sliceDivScalar
let divUInt64Op  name scalar = liftImageScalarOpUInt64   name scalar Slice.sliceDivScalar
let divInt64Op   name scalar = liftImageScalarOpInt64    name scalar Slice.sliceDivScalar
let divFloat32Op name scalar = liftImageScalarOpFloat32  name scalar Slice.sliceDivScalar
let divFloatOp   name scalar = liftImageScalarOpFloat    name scalar Slice.sliceDivScalar
*)
// ---------------------------------------------------------------------------
// NOTE on image ⊕ image variants
// ---------------------------------------------------------------------------
// For image‑image versions (add, sub, mul, div that take two Slice<'T>),
// simply create a `Pipe<'S,'T>` for each input image and use `zipWith`:
//
//    let addImgPipe  = asPipe (addOp  "addImg" )   // where addOp returns Operation<(Slice<'T>*Slice<'T>),Slice<'T>>
//    let pipeline = zipWith add addSrcPipe bSrcPipe  // Routing.zipWith
//
// Those are not generated here because they require two distinct input streams.
//
// Alternatively
let addOpFloat = liftBinaryOpFloat "add" Slice.add // src1 >=> addOp >=> next
// Same goes for all Streaming pixelwise binary operators such as isGreaterEqual and xOr

let sNotOp name = liftUnaryOpInt name Slice.sNot


// Not Pipes nor Operators
type FileInfo = Slice.FileInfo
let getStackDepth = Slice.getStackDepth
let getStackInfo  = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth
let getStackHeight = Slice.getStackHeight


let otsuThresholdOp name = liftFullOp name (Slice.otsuThreshold: Slice<'T> -> Slice<'T>) 
let otsuMultiThresholdOp name n = liftFullParamOp name Slice.otsuMultiThreshold n
let momentsThresholdOp name = liftFullOp name Slice.momentsThreshold
let signedDistanceMapOp name : Operation<Slice<uint8>, Slice<float>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe =
            {
                Name = name
                Profile = Full
                Apply = fun input ->
                    asyncSeq {
                        let! slices = input |> AsyncSeq.toListAsync
                        let stack = Slice.stack slices
                        let result = Slice.signedDistanceMap 0uy 1uy stack
                        yield! Slice.unstack result |> AsyncSeq.ofSeq
                    }
            }
    }

let watershedOp name a = liftFullParamOp name Slice.watershed a
let thresholdOp name a b = liftFullParam2Op name Slice.threshold a b
let addNormalNoiseOp name a b = liftFullParam2Op name Slice.addNormalNoise a b
let relabelComponentsOp name a = liftFullParamOp name Slice.relabelComponents a

let histogramOp<'T when 'T: equality and 'T: comparison> (name: string) : Operation<Slice<'T>, Map<'T, uint64>> =
    liftMapOp name Slice.histogram
let computeStatsOp<'T when 'T: equality and 'T: comparison> (name: string) : Operation<Slice<'T>, ImageStats> =
    liftMapOp name Slice.computeStats

let constantPad2DOp<'T when 'T : equality> (name: string) (padLower : uint list) (padUpper : uint list) (c : double) =
    liftUnaryOp name (Slice.constantPad2D padLower padUpper c)
