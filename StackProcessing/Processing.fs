module Processing

open System
open System.IO
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open AsyncSeqExtensions
open SlimPipeline
//open Slice
//open Image
open type ImageFunctions.OutputRegionMode // weird notation for exposing the discriminated union

let liftImageSource (name: string) (img: Slice.Slice<'T>) : Pipe<unit, Slice.Slice<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> Slice.unstack |> AsyncSeq.ofSeq
    }

let skipNTakeM (n: uint) (m: uint) (lst: 'a list) : 'a list =
    let m = uint lst.Length - 2u*n;
    if m = 0u then []
    else lst |> List.skip (int n) |> List.take (int m) 

let internal liftWindowedOp (name: string) (window: uint) (pad: uint) (zeroMaker: Slice.Slice<'S>->Slice.Slice<'S>) (stride: uint) (emitStart: uint) (emitCount: uint) (f: Slice.Slice<'S> -> Slice.Slice<'T>) 
    : Stage<Slice.Slice<'S>, Slice.Slice<'T>,'Shape> =
    {
        Name = name
        Pipe = Pipe.mapWindowed name window Slice.updateId pad zeroMaker stride emitStart emitCount (Slice.stack >> f >> Slice.unstack)
        Transition = MemoryTransition.create (Sliding (window,stride,emitStart,emitCount)) Streaming
        ShapeUpdate = id
    }

let internal liftWindowedTrimOp (name: string) (window: uint) (pad: uint) (zeroMaker: Slice.Slice<'S>->Slice.Slice<'S>) (stride: uint) (emitStart: uint) (emitCount: uint) (trim: uint) (f: Slice.Slice<'S> -> Slice.Slice<'T>)
    : Stage<Slice.Slice<'S>, Slice.Slice<'T>,'Shape> =
    {
        Name = name
        Transition = MemoryTransition.create (Sliding (window,stride,emitStart,emitCount)) Streaming
        Pipe =
            Pipe.mapWindowed name window Slice.updateId pad zeroMaker stride emitStart emitCount (
                Slice.stack >> f >> Slice.unstack >> fun lst -> let m = uint lst.Length - 2u*trim in skipNTakeM trim m lst)
        ShapeUpdate = id
    }

/// quick constructor for Streaming→Streaming unary ops
let internal liftFullOp (name: string) (f: Slice.Slice<'T> -> Slice.Slice<'T>) : Stage<Slice.Slice<'T>, Slice.Slice<'T>, 'Shape> =
    {
        Name = name
        Transition = MemoryTransition.create Full Streaming
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

let internal liftFullParamOp (name: string) (f: 'P -> Slice.Slice<'T> -> Slice.Slice<'T>) (param: 'P) : Stage<Slice.Slice<'T>, Slice.Slice<'T>, 'Shape> =
    {
        Name = name
        Transition = MemoryTransition.create Full Streaming
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

let internal liftFullParam2Op (name: string) (f: 'P -> 'Q -> Slice.Slice<'T> -> Slice.Slice<'T>) (param1: 'P) (param2: 'Q) : Stage<Slice.Slice<'T>, Slice.Slice<'T>, 'Shape> =
    {
        Name = name
        Transition = MemoryTransition.create Full Streaming
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

/// Histogram related functions
type ImageStats = Slice.ImageStats
let computeStatsOp<'T,'Shape when 'T : equality> name : Stage<Slice.Slice<'T>, ImageStats, 'Shape> =
    let computeStatsReducer (slices: AsyncSeq<Slice.Slice<'T>>) =
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
        Transition = MemoryTransition.create Streaming Constant
        Pipe = Pipe.reduce name computeStatsReducer Streaming
        ShapeUpdate = id
    }

/// Convolution like operators
// Chained type definitions do expose the originals
let zeroPad = ImageFunctions.ZeroPad
let periodicPad = ImageFunctions.PerodicPad
let zeroFluxNeumannPad = ImageFunctions.ZeroFluxNeumannPad
let valid = ImageFunctions.Valid
let same = ImageFunctions.Same
let zeroMaker<'S when 'S: equality> (ex:Slice.Slice<'S>) : Slice.Slice<'S> = Slice.create<'S> (Slice.GetWidth ex) (Slice.GetHeight ex) 1u 0u

let discreteGaussianOp<'Shape> (name:string) (sigma:float) (outputRegionMode: Slice.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Slice.Slice<float>, Slice.Slice<float>, 'Shape> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 2.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> min ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    liftWindowedOp name win pad zeroMaker<float> stride (stride - 1u) stride (fun slices -> Slice.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition slices)

let convGaussOp<'Shape> name sigma = discreteGaussianOp<'Shape> name sigma None None None

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

let convolveOp (name: string) (kernel: Slice.Slice<'T>) (outputRegionMode: Slice.OutputRegionMode option) (bc: Slice.BoundaryCondition option) (winSz: uint option): Stage<Slice.Slice<'T>, Slice.Slice<'T>, 'Shape> =
    let windowFromKernel (k: Slice.Slice<'T>) : uint =
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

let private makeMorphOp (name:string) (radius:uint) (winSz: uint option) (core: uint -> Slice.Slice<'T> -> Slice.Slice<'T>) : Stage<Slice.Slice<'T>,Slice.Slice<'T>, 'Shape> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win - ksz + 1u
    liftWindowedTrimOp name win 0u zeroMaker<'T> stride (stride - 1u) stride radius (fun slices -> core radius slices)

// Only uint8
let binaryErodeOp     name radius winSz = makeMorphOp name radius winSz Slice.binaryErode
let binaryDilateOp    name radius winSz = makeMorphOp name radius winSz Slice.binaryDilate
let binaryOpeningOp   name radius winSz = makeMorphOp name radius winSz Slice.binaryOpening
let binaryClosingOp   name radius winSz = makeMorphOp name radius winSz Slice.binaryClosing
let binaryFillHolesOp<'Shape> name = liftFullOp name Slice.binaryFillHoles
let connectedComponentsOp<'Shape> (name: string) : Stage<Slice.Slice<uint8>, Slice.Slice<uint64>, 'Shape> =
    { // fsharp gets confused about the change of units, so we make the record by hand
        Name = name
        Transition = MemoryTransition.create Full Streaming
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

let piecewiseConnectedComponentsOp (name:string) (windowSize: uint option): Stage<Slice.Slice<uint8>, Slice.Slice<uint64>, 'Shape> =
    let dpth = Option.defaultValue 1u windowSize |> max 1u
    liftWindowedOp name dpth 0u zeroMaker<uint8> dpth 0u dpth (fun slices -> Slice.connectedComponents slices)

let otsuThresholdOp name = liftFullOp name (Slice.otsuThreshold: Slice.Slice<'T> -> Slice.Slice<'T>) 
let otsuMultiThresholdOp name n = liftFullParamOp name Slice.otsuMultiThreshold n
let momentsThresholdOp name = liftFullOp name Slice.momentsThreshold
let signedDistanceMapOp<'Shape> name : Stage<Slice.Slice<uint8>, Slice.Slice<float>, 'Shape> =
    {
        Name = name
        Transition = MemoryTransition.create Full Streaming
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

