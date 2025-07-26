module Processing

open System
open System.IO
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open AsyncSeqExtensions
open SourceSink
open Core
open Routing
open Slice
open Image

// --- Processing Utilities ---
(* // Not used. Needed?
let internal explodeSlice (slices: Slice<'T>) 
    : AsyncSeq<Slice<'T>> =
    let baseIndex = slices.Index
    let volume = slices |> Slice.toImage
    let size = volume.GetSize()
    let width, height, depth = size.[0], size.[1], size.[2]
    Seq.init (int depth) (fun z -> extractSlice (baseIndex+(uint z)) slices)
    |> AsyncSeq.ofSeq
*) 

let skipFirstLast (n: int) (lst: 'a list) 
    : 'a list =
    let m = lst.Length - 2*n;
    if m <= 0 then []
    else lst |> List.skip n |> List.take m 

/// mapWindowed keeps a running window along the slice direction of depth images
/// and processes them by f. The stepping size of the running window is stride.
/// So if depth is 3 and stride is 1 then first image 0,1,2 is sent to f, then 1, 2, 3
/// and so on. If depth is 3 and stride is 3, then it'll be image 0, 1, 2 followed by
/// 3, 4, 5. It is also possible to use this for sampling, e.g., setting depth to 1
/// and stride to 2 sends every second image to f.  
let internal mapWindowed (label: string) (depth: uint) (stride: uint) (f: 'S list -> 'T list) 
    : Pipe<'S,'T> =
    {
        Name = label; 
        Profile = Sliding depth
        Apply = fun input ->
            printfn $"[Runtime analysis: Windowed analysis size {depth} {stride}]"
            AsyncSeqExtensions.windowed depth stride input
                |> AsyncSeq.collect (f  >> AsyncSeq.ofSeq)
    }

let internal liftWindowedOp (name: string) (window: uint) (stride: uint) (f: Slice<'S> -> Slice<'T>) 
    : Operation<Slice<'S>, Slice<'T>> =
    {
        Name = name
        Transition = transition (Sliding window) Streaming
        Pipe = mapWindowed name window stride (stack >> f >> unstack)
    }

let internal liftWindowedTrimOp (name: string) (window: uint) (stride: uint) (trim: uint) (f: Slice<'S> -> Slice<'T>)
    : Operation<Slice<'S>, Slice<'T>> =
    {
        Name = name
        Transition = transition (Sliding window) Streaming
        Pipe =
            mapWindowed name window stride (
                stack >> f >> unstack >> (skipFirstLast (int trim)))
    }

/// quick constructor for Streaming→Streaming unary ops
let internal liftUnaryOpInt (name: string) (f: Slice<int> -> Slice<int>)
    : Operation<Slice<int>,Slice<int>> =
    liftUnaryOp name f

let internal liftUnaryOpFloat32 (name: string) (f: Slice<float32> -> Slice<float32>) 
    : Operation<Slice<float32>,Slice<float32>> =
    liftUnaryOp name f

let internal liftUnaryOpFloat (name: string) (f: Slice<float> -> Slice<float>) 
    : Operation<Slice<float>,Slice<float>> =
    liftUnaryOp name f

let internal liftBinaryOp (name: string) (f: Slice<'T> -> Slice<'T> -> Slice<'T>)
    : Operation<Slice<'T> * Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Streaming Streaming
        Pipe = {
            Name = name
            Profile = Streaming
            Apply = fun input ->
                input
                |> AsyncSeq.map (fun (a, b) -> f a b) }
    }

let internal liftBinaryOpFloat (name: string) (f: Slice<float> -> Slice<float> -> Slice<float>)
    : Operation<Slice<float> * Slice<float>, Slice<float>> =
    liftBinaryOp name f

(* zipWithOld no longer available. is this function used?
let internal liftBinaryZipOp (name: string) (f: Slice<'T> -> Slice<'T> -> Slice<'T>) (p1: Pipe<'In, Slice<'T>>) (p2: Pipe<'In, Slice<'T>>) 
    : Pipe<'In, Slice<'T>> =
    zipWithOld f p1 p2
*)
let internal liftFullOp (name: string) (f: Slice<'T> -> Slice<'T>)
    : Operation<Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe = {
            Name = name
            Profile = Full
            Apply = fun input ->
                asyncSeq {
                    let! slices = input |> AsyncSeq.toListAsync
                    let stack = Slice.stack slices
                    let result = f stack
                    yield! Slice.unstack result |> AsyncSeq.ofSeq } }
    }

let internal liftFullParamOp (name: string) (f: 'P -> Slice<'T> -> Slice<'T>) (param: 'P)
    : Operation<Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe = {
            Name = name
            Profile = Full
            Apply = fun input ->
                asyncSeq {
                    let! slices = input |> AsyncSeq.toListAsync
                    let stack = Slice.stack slices
                    let result = f param stack
                    yield! Slice.unstack result |> AsyncSeq.ofSeq } }
    }

let internal liftFullParam2Op (name: string) (f: 'P -> 'Q -> Slice<'T> -> Slice<'T>) (param1: 'P) (param2: 'Q)
    : Operation<Slice<'T>, Slice<'T>> =
    {
        Name = name
        Transition = transition Full Streaming
        Pipe = {
            Name = name
            Profile = Full
            Apply = fun input ->
                asyncSeq {
                    let! slices = input |> AsyncSeq.toListAsync
                    let stack = Slice.stack slices
                    let result = f param1 param2 stack
                    yield! Slice.unstack result |> AsyncSeq.ofSeq } }
    }

let internal liftMapOp<'T, 'U when 'T: equality and 'T: comparison> (name: string) (f: Slice<'T> -> 'U) 
    : Operation<Slice<'T>, 'U> =
    {
        Name = name
        Transition = transition Streaming Streaming
        Pipe = {
            Name = name
            Profile = Streaming
            Apply = fun input -> input |> AsyncSeq.map f }
    }

/////////////////////////////////////////////////////////////////////////////////////
let inline castOp<'S,'T when 'S: equality and 'T: equality> name f 
    : Operation<Slice<'S>,Slice<'T>> =
    { 
        Name = name
        Transition = transition Streaming Streaming
        Pipe = { 
            Name = name
            Profile = Streaming
            Apply = fun input -> input |> AsyncSeq.map f } 
    }

let castUInt8ToInt8Op name = castOp name Slice.castUInt8ToInt8
let castUInt8ToUInt16Op name = castOp name Slice.castUInt8ToUInt16
let castUInt8ToInt16Op name = castOp name Slice.castUInt8ToInt16
let castUInt8ToUIntOp name = castOp name Slice.castUInt8ToUInt
let castUInt8ToIntOp name = castOp name Slice.castUInt8ToInt
let castUInt8ToUInt64Op name = castOp name Slice.castUInt8ToUInt64
let castUInt8ToInt64Op name = castOp name Slice.castUInt8ToInt64
let castUInt8ToFloat32Op name = castOp name Slice.castUInt8ToFloat32
let castUInt8ToFloatOp name = castOp name Slice.castUInt8ToFloat
let castInt8ToUInt8Op name = castOp name Slice.castInt8ToUInt8
let castInt8ToUInt16Op name = castOp name Slice.castInt8ToUInt16
let castInt8ToInt16Op name = castOp name Slice.castInt8ToInt16
let castInt8ToUIntOp name = castOp name Slice.castInt8ToUInt
let castInt8ToIntOp name = castOp name Slice.castInt8ToInt
let castInt8ToUInt64Op name = castOp name Slice.castInt8ToUInt64
let castInt8ToInt64Op name = castOp name Slice.castInt8ToInt64
let castInt8ToFloat32Op name = castOp name Slice.castInt8ToFloat32
let castInt8ToFloatOp name = castOp name Slice.castInt8ToFloat

let castUInt16ToUInt8Op name = castOp name Slice.castUInt16ToUInt8
let castUInt16ToInt8Op name = castOp name Slice.castUInt16ToInt8
let castUInt16ToInt16Op name = castOp name Slice.castUInt16ToInt16
let castUInt16ToUIntOp name = castOp name Slice.castUInt16ToUInt
let castUInt16ToIntOp name = castOp name Slice.castUInt16ToInt
let castUInt16ToUInt64Op name = castOp name Slice.castUInt16ToUInt64
let castUInt16ToInt64Op name = castOp name Slice.castUInt16ToInt64
let castUInt16ToFloat32Op name = castOp name Slice.castUInt16ToFloat32
let castUInt16ToFloatOp name = castOp name Slice.castUInt16ToFloat
let castInt16ToUInt8Op name = castOp name Slice.castInt16ToUInt8
let castInt16ToInt8Op name = castOp name Slice.castInt16ToInt8
let castInt16ToUInt16Op name = castOp name Slice.castInt16ToUInt16
let castInt16ToUIntOp name = castOp name Slice.castInt16ToUInt
let castInt16ToIntOp name = castOp name Slice.castInt16ToInt
let castInt16ToUInt64Op name = castOp name Slice.castInt16ToUInt64
let castInt16ToInt64Op name = castOp name Slice.castInt16ToInt64
let castInt16ToFloat32Op name = castOp name Slice.castInt16ToFloat32
let castInt16ToFloatOp name = castOp name Slice.castInt16ToFloat

let castUIntToUInt8Op name = castOp name Slice.castUIntToUInt8
let castUIntToInt8Op name = castOp name Slice.castUIntToInt8
let castUIntToUInt16Op name = castOp name Slice.castUIntToUInt16
let castUIntToInt16Op name = castOp name Slice.castUIntToInt16
let castUIntToIntOp name = castOp name Slice.castUIntToInt
let castUIntToUInt64Op name = castOp name Slice.castUIntToUInt64
let castUIntToInt64Op name = castOp name Slice.castUIntToInt64
let castUIntToFloat32Op name = castOp name Slice.castUIntToFloat32
let castUIntToFloatOp name = castOp name Slice.castUIntToFloat
let castIntToUInt8Op name = castOp name Slice.castIntToUInt8
let castIntToInt8Op name = castOp name Slice.castIntToInt8
let castIntToUInt16Op name = castOp name Slice.castIntToUInt16
let castIntToInt16Op name = castOp name Slice.castIntToInt16
let castIntToUIntOp name = castOp name Slice.castIntToUInt
let castIntToUInt64Op name = castOp name Slice.castIntToUInt64
let castIntToInt64Op name = castOp name Slice.castIntToInt64
let castIntToFloat32Op name = castOp name Slice.castIntToFloat32
let castIntToFloatOp name = castOp name Slice.castIntToFloat

let castUInt64ToUInt8Op name = castOp name Slice.castUInt64ToUInt8
let castUInt64ToInt8Op name = castOp name Slice.castUInt64ToInt8
let castUInt64ToUInt16Op name = castOp name Slice.castUInt64ToUInt16
let castUInt64ToInt16Op name = castOp name Slice.castUInt64ToInt16
let castUInt64ToUIntOp name = castOp name Slice.castUInt64ToUInt
let castUInt64ToIntOp name = castOp name Slice.castUInt64ToInt
let castUInt64ToInt64Op name = castOp name Slice.castUInt64ToInt64
let castUInt64ToFloat32Op name = castOp name Slice.castUInt64ToFloat32
let castUInt64ToFloatOp name = castOp name Slice.castUInt64ToFloat
let castInt64ToUInt8Op name = castOp name Slice.castInt64ToUInt8
let castInt64ToInt8Op name = castOp name Slice.castInt64ToInt8
let castInt64ToUInt16Op name = castOp name Slice.castInt64ToUInt16
let castInt64ToInt16Op name = castOp name Slice.castInt64ToInt16
let castInt64ToUIntOp name = castOp name Slice.castInt64ToUInt
let castInt64ToIntOp name = castOp name Slice.castInt64ToInt
let castInt64ToUInt64Op name = castOp name Slice.castInt64ToUInt64
let castInt64ToFloat32Op name = castOp name Slice.castInt64ToFloat32
let castInt64ToFloatOp name = castOp name Slice.castInt64ToFloat

let castFloat32ToUInt8Op name = castOp name Slice.castFloat32ToUInt8
let castFloat32ToInt8Op name = castOp name Slice.castFloat32ToInt8
let castFloat32ToUInt16Op name = castOp name Slice.castFloat32ToUInt16
let castFloat32ToInt16Op name = castOp name Slice.castFloat32ToInt16
let castFloat32ToUIntOp name = castOp name Slice.castFloat32ToUInt
let castFloat32ToIntOp name = castOp name Slice.castFloat32ToInt
let castFloat32ToUInt64Op name = castOp name Slice.castFloat32ToUInt64
let castFloat32ToInt64Op name = castOp name Slice.castFloat32ToInt64
let castFloat32ToFloatOp name = castOp name Slice.castFloat32ToFloat
let castFloatToUInt8Op name = castOp name Slice.castFloatToUInt8
let castFloatToInt8Op name = castOp name Slice.castFloatToInt8
let castFloatToUInt16Op name = castOp name Slice.castFloatToUInt16
let castFloatToInt16Op name = castOp name Slice.castFloatToInt16
let castFloatToUIntOp name = castOp name Slice.castFloatToUInt
let castFloatToIntOp name = castOp name Slice.castFloatToInt
let castFloatToUIn64Op name = castOp name Slice.castFloatToUIn64
let castFloatToInt64Op name = castOp name Slice.castFloatToInt64
let castFloatToFloat32Op name = castOp name Slice.castFloatToFloat32

/// Basic arithmetic
let addOp name slice = liftUnaryOp name (Slice.add slice)
let inline scalarAddSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.scalarAddSlice<^T> i s)
let inline sliceAddScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.sliceAddScalar<^T> s i)

let subOp name slice = liftUnaryOp name (Slice.sub slice)
let inline scalarSubSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.scalarSubSlice<^T> i s)
let inline sliceSubScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.sliceSubScalar<^T> s i)

let mulOp name slice = liftUnaryOp name (Slice.mul slice)
let inline scalarMulSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.scalarMulSlice<^T> i s)
let inline sliceMulScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.sliceMulScalar<^T> s i)

let divOp name slice = liftUnaryOp name (Slice.div slice)
let inline scalarDivSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.scalarDivSlice<^T> i s)
let inline sliceDivScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    liftUnaryOp name (fun (s:Slice<'T>)->Slice.sliceDivScalar<^T> s i)

/// Simple functions
let absFloat32Op   name = liftUnaryOpFloat32 name absSlice
let absFloatOp     name = liftUnaryOpFloat name absSlice
let absIntOp       name = liftUnaryOpInt name absSlice
let acosFloat32Op  name = liftUnaryOpFloat32 name acosSlice
let acosFloatOp    name = liftUnaryOpFloat name acosSlice
let asinFloat32Op  name = liftUnaryOpFloat32 name asinSlice
let asinFloatOp    name = liftUnaryOpFloat name asinSlice
let atanFloat32Op  name = liftUnaryOpFloat32 name atanSlice
let atanFloatOp    name = liftUnaryOpFloat name atanSlice
let cosFloat32Op   name = liftUnaryOpFloat32 name cosSlice
let cosFloatOp     name = liftUnaryOpFloat name cosSlice
let expFloat32Op   name = liftUnaryOpFloat32 name expSlice
let expFloatOp     name = liftUnaryOpFloat name expSlice
let log10Float32Op name = liftUnaryOpFloat32 name log10Slice
let log10FloatOp   name = liftUnaryOpFloat name log10Slice
let logFloat32Op   name = liftUnaryOpFloat32 name logSlice
let logFloatOp     name = liftUnaryOpFloat name logSlice
let roundFloat32Op name = liftUnaryOpFloat32 name roundSlice
let roundFloatOp   name = liftUnaryOpFloat name roundSlice
let sinFloat32Op   name = liftUnaryOpFloat32 name sinSlice
let sinFloatOp     name = liftUnaryOpFloat name sinSlice
let sqrtFloat32Op  name = liftUnaryOpFloat32 name sqrtSlice
let sqrtFloatOp    name = liftUnaryOpFloat name sqrtSlice
let sqrtIntOp      name = liftUnaryOpInt name sqrtSlice
let squareFloat32Op name = liftUnaryOpFloat32 name squareSlice
let squareFloatOp  name = liftUnaryOpFloat name squareSlice
let squareIntOp    name = liftUnaryOpInt name squareSlice
let tanFloat32Op   name = liftUnaryOpFloat32 name tanSlice
let tanFloatOp     name = liftUnaryOpFloat name tanSlice

/// Histogram related functions
let histogramOp<'T when 'T: comparison> name
    : Operation<Slice<'T>, Map<'T, uint64>>  =
    let histogramReducer (slices: AsyncSeq<Slice<'T>>) =
        slices
        |> AsyncSeq.map Slice.histogram
        |> AsyncSeqExtensions.fold Slice.addHistogram (Map<'T, uint64> [])
    {
        Name = name
        Transition = transition Streaming Constant
        Pipe = reduce name Streaming histogramReducer
    }

let map2pairsOp<'T, 'S when 'T: comparison> name
    : Operation<Map<'T, 'S>, ('T * 'S) list> =
    {
        Name = name
        Transition = transition Streaming Streaming
        Pipe = map name Streaming Slice.map2pairs
    }
let inline pairs2floatsOp< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > name
    : Operation<(^T * ^S) list, (float * float) list> =
    {
        Name = name
        Transition = transition Streaming Streaming
        Pipe = map name Streaming Slice.pairs2floats
    }
let inline pairs2intsOp< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > name
    : Operation<(^T * ^S) list, (int * int) list> =
    {
        Name = name
        Transition = transition Streaming Streaming
        Pipe = map name Streaming Slice.pairs2ints
    }

type ImageStats = Slice.ImageStats
let computeStatsOp<'T when 'T : equality> name : Operation<Slice<'T>, ImageStats> =
    let computeStatsReducer (slices: AsyncSeq<Slice<'T>>) =
        let zeroStats: ImageStats = { 
            NumPixels = 0u
            Mean = 0.0
            Std = 0.0
            Min = infinity
            Max = -infinity
            Sum = 0.0
            Var = 0.0
        }
        slices
        |> AsyncSeq.map Slice.computeStats
        |> AsyncSeqExtensions.fold Slice.addComputeStats zeroStats
    {
        Name = name
        Transition = transition Streaming Constant
        Pipe = reduce name Streaming computeStatsReducer
    }

/// Convolution like operators
// Chained type definitions do expose the originals
let zeroPad = ImageFunctions.ZeroPad
let periodicPad = ImageFunctions.PerodicPad
let zeroFluxNeumannPad = ImageFunctions.ZeroFluxNeumannPad
let valid = ImageFunctions.Valid
let same = ImageFunctions.Same

let discreteGaussianOp (name:string) (sigma:float) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option): Operation<Slice<float>, Slice<float>> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 2.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> min ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    liftWindowedOp name win stride (fun slices -> Slice.discreteGaussian 3u sigma (ksz |> Some) (Some valid) bc slices)


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
    let windowFromKernel (k: Slice<'T>) : uint =
        max 1u (k |> Slice.GetDepth)
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win-ksz+1u
    liftWindowedOp name win stride (fun slices -> Slice.convolve (Some valid) bc slices kernel)

let private makeMorphOp (name:string) (radius:uint) (winSz: uint option) (core: uint -> Slice<'T> -> Slice<'T>) : Operation<Slice<'T>,Slice<'T>> when 'T: equality =
    let ksz   = 2u * radius + 1u
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
    { // fsharp gets confused about the change of units, so we make the record by hand
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

let piecewiseConnectedComponentsOp (name:string) (windowSize: uint option): Operation<Slice<uint8>, Slice<uint64>> =
    let dpth = Option.defaultValue 1u windowSize |> max 1u
    liftWindowedOp name dpth dpth (fun slices -> Slice.connectedComponents slices)

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
let thresholdOp name a b = liftUnaryOp name (Slice.threshold a b)
let addNormalNoiseOp name a b = liftUnaryOp name (Slice.addNormalNoise a b)
let relabelComponentsOp name a = liftFullParamOp name Slice.relabelComponents a

let constantPad2DOp<'T when 'T : equality> (name: string) (padLower : uint list) (padUpper : uint list) (c : double) =
    liftUnaryOp name (Slice.constantPad2D padLower padUpper c)

// Not Pipes nor Operators
type FileInfo = Slice.FileInfo
let getStackDepth = Slice.getStackDepth
let getStackInfo  = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth
let getStackHeight = Slice.getStackHeight
