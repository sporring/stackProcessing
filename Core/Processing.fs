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
            AsyncSeqExtensions.windowed depth stride input
                |> AsyncSeq.collect (f  >> AsyncSeq.ofSeq)
    }

let castUInt8ToFloat : Pipe<Slice<uint8>, Slice<float>> =
    printfn "[castUInt8ToFloat]"
    {
        Name = "castUInt8ToFloat"
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun slice -> Slice.castUInt8ToFloat slice)
    }

let castFloatToUInt8 : Pipe<Slice<float>, Slice<uint8>> =
    printfn "[castFloatToUInt8]"
    {
        Name = "castFloatToUInt8"
        Profile = Streaming
        Apply = fun input ->
            input
            |> AsyncSeq.map (fun slice -> Slice.castFloatToUInt8 slice)
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

// Chained type definitions do expose the originals
let zeroPad = ImageFunctions.ZeroPad
let periodicPad = ImageFunctions.PerodicPad
let zeroFluxNeumannPad = ImageFunctions.ZeroFluxNeumannPad
let valid = ImageFunctions.Valid
let same = ImageFunctions.Same

let convolve (kern: Slice<'T>) (boundaryCondition: BoundaryCondition option) (windowSize: uint option) (stride: uint option): Pipe<Slice<'T> ,Slice<'T>> =
    printfn "[conv/convolve]"
    let ksz = kern |> GetDepth |>  max 1u
    let dpth = Option.defaultValue ksz windowSize |> max ksz
    let strd = Option.defaultValue (1u+dpth-ksz) stride |> max 1u
    printfn $"convolve: {ksz} {dpth} {strd}"
    mapWindowed "discreteGaussian" dpth strd (stack >> convolve boundaryCondition kern >> unstack)

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
    printfn "[convGauss/discreteGaussian]"
    let ksz = Option.defaultValue (1u + 2u * uint (0.5 + sigma)) kernelSize
    let dpth = Option.defaultValue ksz windowSize |> max ksz
    let minStride = 1u
    let strd = Option.defaultValue (1u+dpth-ksz) stride |> max minStride
    mapWindowed "convGauss/discreteGaussian" dpth strd (stack >> discreteGaussian sigma (Some ksz) boundaryCondition >> unstack)

let convGauss (sigma: float) (boundaryCondition: BoundaryCondition option) : Pipe<Slice<'T> ,Slice<'T>> =
    discreteGaussian sigma None boundaryCondition None None

let skipFirstLast (n: int) (lst: 'a list) : 'a list =
    let m = lst.Length - 2*n;
    if m <= 0 then []
    else lst |> List.skip n |> List.take m 

let private binaryMathMorph (name: string) f (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<'T> ,Slice<'T>> =
    printfn $"[{name}]"
    let ksz = 1u+2u*radius
    let dpth = Option.defaultValue ksz windowSize |> max ksz
    let strd = Option.defaultValue (1u+dpth-ksz) stride |> max 1u
    mapWindowed $"{name}" dpth strd (stack >> f radius  >> unstack >> skipFirstLast (int radius))

let binaryErode (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<'T> ,Slice<'T>> =
    binaryMathMorph "binaryErode" binaryErode radius windowSize stride

let binaryDilate (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<'T> ,Slice<'T>> =
    binaryMathMorph "binaryDilate" binaryDilate radius windowSize stride

let binaryOpening (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<'T> ,Slice<'T>> =
    binaryMathMorph "binaryOpening" binaryOpening radius windowSize stride

let binaryClosing (radius: uint) (windowSize: uint option) (stride: uint option) : Pipe<Slice<'T> ,Slice<'T>> =
    binaryMathMorph "binaryClosing" binaryClosing radius windowSize stride

let piecewiseConnectedComponents (windowSize: uint option) : Pipe<Slice<'T> ,Slice<'T>> =
    printfn "[connectedComponents]"
    let dpth = Option.defaultValue 1u windowSize |> max 1u
    mapWindowed "connectedComponents" dpth dpth (stack >> connectedComponents >> unstack)

type FileInfo = Slice.FileInfo
let getStackDepth (inputDir: string) (suffix: string) : uint = Slice.getStackDepth inputDir suffix
let getStackInfo (inputDir: string) (suffix: string): FileInfo = Slice.getStackInfo inputDir suffix
let getStackSize (inputDir: string) (suffix: string) = Slice.getStackSize inputDir suffix
let getStackWidth (inputDir: string) (suffix: string): uint64 = Slice.getStackWidth inputDir suffix
let getStackHeigth (inputDir: string) (suffix: string): uint64 = Slice.getStackHeight inputDir suffix

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

let liftWindowedOp (name: string) (window: uint) (stride: uint) (f: Slice<'S> -> Slice<'T>) : Operation<Slice<'S>, Slice<'T>> =
    {
        Name = name
        Transition = transition (Sliding window) Streaming
        Pipe = mapWindowed name window stride (stack >> f >> unstack)
    }

let liftUnaryOpInt (name: string) (f: Slice<int> -> Slice<int>) =
    liftUnaryOp name f

let liftUnaryOpFloat32 (name: string) (f: Slice<float32> -> Slice<float32>) =
    liftUnaryOp name f

let liftUnaryOpFloat (name: string) (f: Slice<float> -> Slice<float>) =
    liftUnaryOp name f

let roundFloatToUint v = uint (v+0.5)

let discreteGaussianOp (name:string) (sigma:float) (bc: ImageFunctions.BoundaryCondition option) : Operation<Slice<float>, Slice<float>> =
    let ksz = 2.0 * sigma + 1.0
    let win = max 7.0 ksz // max should be found by memory availability
    let stride = 1.0 + (win - ksz) |> roundFloatToUint
    liftWindowedOp name (win|> uint) stride (fun slices -> Slice.discreteGaussian sigma (ksz |> uint |> Some) bc slices)

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
