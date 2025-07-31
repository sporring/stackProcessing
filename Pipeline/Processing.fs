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
open type ImageFunctions.OutputRegionMode // weird notation for exposing the discriminated union

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

let skipNTakeM (n: uint) (m: uint) (lst: 'a list) : 'a list =
    let m = uint lst.Length - 2u*n;
    if m = 0u then []
    else lst |> List.skip (int n) |> List.take (int m) 

let internal liftWindowedOp (name: string) (window: uint) (pad: uint) (zeroMaker: Slice<'S>->Slice<'S>) (stride: uint) (emitStart: uint) (emitCount: uint) (f: Slice<'S> -> Slice<'T>) 
    : Stage<Slice<'S>, Slice<'T>,'Shape> =
    {
        Name = name
        Pipe = Pipe.mapWindowed name window updateId pad zeroMaker stride emitStart emitCount (stack >> f >> unstack)
        Transition = Stage.transition (Sliding (window,stride,emitStart,emitCount)) Streaming
        ShapeUpdate = id
    }

let internal liftWindowedTrimOp (name: string) (window: uint) (pad: uint) (zeroMaker: Slice<'S>->Slice<'S>) (stride: uint) (emitStart: uint) (emitCount: uint) (trim: uint) (f: Slice<'S> -> Slice<'T>)
    : Stage<Slice<'S>, Slice<'T>,'Shape> =
    {
        Name = name
        Transition = Stage.transition (Sliding (window,stride,emitStart,emitCount)) Streaming
        Pipe =
            Pipe.mapWindowed name window updateId pad zeroMaker stride emitStart emitCount (
                stack >> f >> unstack >> fun lst -> let m = uint lst.Length - 2u*trim in skipNTakeM trim m lst)
        ShapeUpdate = id
    }

/// quick constructor for Streaming→Streaming unary ops
let internal liftUnaryOpInt (name: string) (f: Slice<int> -> Slice<int>) : Stage<Slice<int>,Slice<int>, 'Shape> =
    Stage.liftUnary name f

let internal liftUnaryOpFloat32 (name: string) (f: Slice<float32> -> Slice<float32>) : Stage<Slice<float32>,Slice<float32>, 'Shape> =
    Stage.liftUnary name f

let internal liftUnaryOpFloat (name: string) (f: Slice<float> -> Slice<float>) : Stage<Slice<float>,Slice<float>, 'Shape> =
    Stage.liftUnary name f

let internal liftBinaryOp (name: string) (f: Slice<'T> -> Slice<'T> -> Slice<'T>) : Stage<Slice<'T> * Slice<'T>, Slice<'T>, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = {
            Name = name
            Profile = Streaming
            Apply = fun input ->
                input
                |> AsyncSeq.map (fun (a, b) -> f a b) }
        ShapeUpdate = id
    }

let internal liftBinaryOpFloat (name: string) (f: Slice<float> -> Slice<float> -> Slice<float>) : Stage<Slice<float> * Slice<float>, Slice<float>, 'Shape> =
    liftBinaryOp name f

let internal liftFullOp (name: string) (f: Slice<'T> -> Slice<'T>) : Stage<Slice<'T>, Slice<'T>, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Full Streaming
        Pipe = {
            Name = name
            Profile = Full
            Apply = fun input ->
                asyncSeq {
                    let! slices = input |> AsyncSeq.toListAsync
                    let stack = Slice.stack slices
                    let result = f stack
                    yield! Slice.unstack result |> AsyncSeq.ofSeq } }
        ShapeUpdate = id
    }

let internal liftFullParamOp (name: string) (f: 'P -> Slice<'T> -> Slice<'T>) (param: 'P) : Stage<Slice<'T>, Slice<'T>, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Full Streaming
        Pipe = {
            Name = name
            Profile = Full
            Apply = fun input ->
                asyncSeq {
                    let! slices = input |> AsyncSeq.toListAsync
                    let stack = Slice.stack slices
                    let result = f param stack
                    yield! Slice.unstack result |> AsyncSeq.ofSeq } }
        ShapeUpdate = id
    }

let internal liftFullParam2Op (name: string) (f: 'P -> 'Q -> Slice<'T> -> Slice<'T>) (param1: 'P) (param2: 'Q) : Stage<Slice<'T>, Slice<'T>, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Full Streaming
        Pipe = {
            Name = name
            Profile = Full
            Apply = fun input ->
                asyncSeq {
                    let! slices = input |> AsyncSeq.toListAsync
                    let stack = Slice.stack slices
                    let result = f param1 param2 stack
                    yield! Slice.unstack result |> AsyncSeq.ofSeq } }
        ShapeUpdate = id
    }

let internal liftMapOp<'T, 'U when 'T: equality and 'T: comparison> (name: string) (f: Slice<'T> -> 'U) : Stage<Slice<'T>, 'U, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = {
            Name = name
            Profile = Streaming
            Apply = fun input -> input |> AsyncSeq.map f }
        ShapeUpdate = id
    }

/////////////////////////////////////////////////////////////////////////////////////
let inline castOp<'S,'T when 'S: equality and 'T: equality> name f : Stage<Slice<'S>,Slice<'T>, 'Shape> =
    { 
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = { 
            Name = name
            Profile = Streaming
            Apply = fun input -> input |> AsyncSeq.map f } 
        ShapeUpdate = id
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
let addOp name slice = Stage.liftUnary name (Slice.add slice)
let inline scalarAddSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.scalarAddSlice<^T> i s)
let inline sliceAddScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.sliceAddScalar<^T> s i)

let subOp name slice = Stage.liftUnary name (Slice.sub slice)
let inline scalarSubSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.scalarSubSlice<^T> i s)
let inline sliceSubScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.sliceSubScalar<^T> s i)

let mulOp name slice = Stage.liftUnary name (Slice.mul slice)
let inline scalarMulSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.scalarMulSlice<^T> i s)
let inline sliceMulScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.sliceMulScalar<^T> s i)

let divOp name slice = Stage.liftUnary name (Slice.div slice)
let inline scalarDivSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.scalarDivSlice<^T> i s)
let inline sliceDivScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<'T>)->Slice.sliceDivScalar<^T> s i)

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
let histogramOp<'T when 'T: comparison> name : Stage<Slice<'T>, Map<'T, uint64>, 'Shape>  =
    let histogramReducer (slices: AsyncSeq<Slice<'T>>) =
        slices
        |> AsyncSeq.map Slice.histogram
        |> AsyncSeqExtensions.fold Slice.addHistogram (Map<'T, uint64> [])
    {
        Name = name
        Transition = Stage.transition Streaming Constant
        Pipe = Pipe.reduce name Streaming histogramReducer
        ShapeUpdate = id
    }

let map2pairsOp<'T, 'S when 'T: comparison> name : Stage<Map<'T, 'S>, ('T * 'S) list, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = Pipe.map name Streaming Slice.map2pairs
        ShapeUpdate = id
    }
let inline pairs2floatsOp< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > name : Stage<(^T * ^S) list, (float * float) list, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = Pipe.map name Streaming Slice.pairs2floats
        ShapeUpdate = id
    }
let inline pairs2intsOp< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > name : Stage<(^T * ^S) list, (int * int) list, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = Pipe.map name Streaming Slice.pairs2ints
        ShapeUpdate = id
    }

type ImageStats = Slice.ImageStats
let computeStatsOp<'T when 'T : equality> name : Stage<Slice<'T>, ImageStats, 'Shape> =
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
        Transition = Stage.transition Streaming Constant
        Pipe = Pipe.reduce name Streaming computeStatsReducer
        ShapeUpdate = id
    }

/// Convolution like operators
// Chained type definitions do expose the originals
let zeroPad = ImageFunctions.ZeroPad
let periodicPad = ImageFunctions.PerodicPad
let zeroFluxNeumannPad = ImageFunctions.ZeroFluxNeumannPad
let valid = ImageFunctions.Valid
let same = ImageFunctions.Same
let zeroMaker<'S when 'S: equality> (ex:Slice<'S>) : Slice<'S> = Slice.create<'S> (GetWidth ex) (GetHeight ex) 1u 0u

let discreteGaussianOp (name:string) (sigma:float) (outputRegionMode: OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Slice<float>, Slice<float>, 'Shape> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 2.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> min ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    liftWindowedOp name win pad zeroMaker<float> stride (stride - 1u) stride (fun slices -> Slice.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition slices)

let convGaussOp name sigma = discreteGaussianOp name sigma None None None

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

let convolveOp (name: string) (kernel: Slice<'T>) (outputRegionMode: OutputRegionMode option) (bc: BoundaryCondition option) (winSz: uint option): Stage<Slice<'T>, Slice<'T>, 'Shape> =
    let windowFromKernel (k: Slice<'T>) : uint =
        max 1u (k |> Slice.GetDepth)
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win-ksz+1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    liftWindowedOp name win pad zeroMaker<'T> stride (stride - 1u) stride (fun slices -> Slice.convolve outputRegionMode bc slices kernel)

let convOp name kernel = convolveOp name kernel None None

let private makeMorphOp (name:string) (radius:uint) (winSz: uint option) (core: uint -> Slice<'T> -> Slice<'T>) : Stage<Slice<'T>,Slice<'T>, 'Shape> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win - ksz + 1u
    liftWindowedTrimOp name win 0u zeroMaker<'T> stride (stride - 1u) stride radius (fun slices -> core radius slices)

// Only uint8
let binaryErodeOp     name radius winSz = makeMorphOp name radius winSz Slice.binaryErode
let binaryDilateOp    name radius winSz = makeMorphOp name radius winSz Slice.binaryDilate
let binaryOpeningOp   name radius winSz = makeMorphOp name radius winSz Slice.binaryOpening
let binaryClosingOp   name radius winSz = makeMorphOp name radius winSz Slice.binaryClosing
let binaryFillHolesOp name = liftFullOp name Slice.binaryFillHoles
let connectedComponentsOp (name: string) : Stage<Slice<uint8>, Slice<uint64>, 'Shape> =
    { // fsharp gets confused about the change of units, so we make the record by hand
        Name = name
        Transition = Stage.transition Full Streaming
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
        ShapeUpdate = id
    }

let piecewiseConnectedComponentsOp (name:string) (windowSize: uint option): Stage<Slice<uint8>, Slice<uint64>, 'Shape> =
    let dpth = Option.defaultValue 1u windowSize |> max 1u
    liftWindowedOp name dpth 0u zeroMaker<uint8> dpth 0u dpth (fun slices -> Slice.connectedComponents slices)

let otsuThresholdOp name = liftFullOp name (Slice.otsuThreshold: Slice<'T> -> Slice<'T>) 
let otsuMultiThresholdOp name n = liftFullParamOp name Slice.otsuMultiThreshold n
let momentsThresholdOp name = liftFullOp name Slice.momentsThreshold
let signedDistanceMapOp name : Stage<Slice<uint8>, Slice<float>, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Full Streaming
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
        ShapeUpdate = id
    }

let watershedOp name a = liftFullParamOp name Slice.watershed a
let thresholdOp name a b = Stage.liftUnary name (Slice.threshold a b)
let addNormalNoiseOp name a b = Stage.liftUnary name (Slice.addNormalNoise a b)
let relabelComponentsOp name a = liftFullParamOp name Slice.relabelComponents a

let constantPad2DOp<'T when 'T : equality> (name: string) (padLower : uint list) (padUpper : uint list) (c : double) =
    Stage.liftUnary name (Slice.constantPad2D padLower padUpper c)

// Not Pipes nor Operators
type FileInfo = Slice.FileInfo
let getStackDepth = Slice.getStackDepth
let getStackInfo  = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth
let getStackHeight = Slice.getStackHeight
