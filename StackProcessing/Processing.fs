module Processing

open System
open System.IO
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open AsyncSeqExtensions
open SlimPipeline
open Slice
open Image
open type ImageFunctions.OutputRegionMode // weird notation for exposing the discriminated union


module internal InternalHelpers =
    // https://plotly.net/#For-applications-and-libraries
    let plotListAsync (plt: (float list)->(float list)->unit) (vectorSeq: AsyncSeq<(float*float) list>) =
        vectorSeq
        |> AsyncSeq.iterAsync (fun points ->
            async {
                let x,y = points |> List.unzip
                plt x y
            })

    let showSliceAsync (plt: (Slice<'T>->unit)) (slices : AsyncSeq<Slice<'T>>) =
        slices
        |> AsyncSeq.iterAsync (fun slice ->
            async {
                let width = slice |> GetWidth |> int
                let height = slice |>GetHeight |> int
                plt slice
            })

    let printAsync (slices: AsyncSeq<'T>) =
        slices
        |> AsyncSeq.iterAsync (fun data ->
            async {
                printfn "[Print] %A" data
            })

    let writeSlicesAsync (outputDir: string) (suffix: string) (slices: AsyncSeq<Slice<'T>>) =
        if not (Directory.Exists(outputDir)) then
            Directory.CreateDirectory(outputDir) |> ignore
        slices
        |> AsyncSeq.iterAsync (fun slice ->
            async {
                let fileName = Path.Combine(outputDir, sprintf "slice_%03d%s" slice.Index suffix)
                slice.Image.toFile(fileName)
                printfn "[Write] Saved slice %d to %s" slice.Index fileName
            })

open InternalHelpers

/// Sink parts
let writeOp (path: string) (suffix: string) : Stage<Slice<'a>, unit, 'Shape> =
    let writeReducer stream = async { do! writeSlicesAsync path suffix stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = $"write:{path}"
        Pipe = Pipe.consumeWith "write" writeReducer Streaming
        Transition = Stage.transition Streaming Constant
        ShapeUpdate = shapeUpdate
    }

let showOp (plt: Slice.Slice<'T> -> unit) : Stage<Slice<'T>, unit, 'Shape> =
    let showReducer stream = async {do! showSliceAsync plt stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = "show"
        Pipe = Pipe.consumeWith "show" showReducer Streaming
        Transition = Stage.transition Streaming Constant
        ShapeUpdate = shapeUpdate
    }

let plotOp (plt: float list -> float list -> unit) : Stage<(float * float) list, unit, 'Shape> =
    let plotReducer stream = async { do! plotListAsync plt stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = "plot"
        Pipe = Pipe.consumeWith "plot" plotReducer Streaming
        Transition = Stage.transition Streaming Streaming
        ShapeUpdate = shapeUpdate
    }

let printOp () : Stage<'T, unit,'Shape> =
    let printReducer stream = async { do! printAsync stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = "print"
        Pipe = Pipe.consumeWith "print" printReducer Streaming
        Transition = Stage.transition Streaming Streaming
        ShapeUpdate = shapeUpdate
    }

let liftImageSource (name: string) (img: Slice<'T>) : Pipe<unit, Slice<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> unstack |> AsyncSeq.ofSeq
    }

(*
/// Yet to be moved into Pipeline-Operator version
let readSliceN<'T when 'T: equality> (idx: uint) (inputDir: string) (suffix: string) transform : Pipe<unit, Slice<'T>> =
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
    |> transform

let ignore<'T> : Pipe<'T, unit> = // Is this needed?
    printfn "[ignore]"
    Pipe.consumeWith "ignore" Streaming (fun stream ->
        async {
            do! stream |> AsyncSeq.iterAsync (fun _ -> async.Return())
        })
*)


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

/////////////////////////////////////////////////////////////////////////////////////
let inline castOp<'S,'T,'Shape when 'S: equality and 'T: equality> name f : Stage<Slice<'S>,Slice<'T>, 'Shape> =
    { 
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = { 
            Name = name
            Profile = Streaming
            Apply = fun input -> input |> AsyncSeq.map f } 
        ShapeUpdate = id
    }

/// Basic arithmetic
let addOp name slice = Stage.liftUnary name (Slice.add slice)
let inline scalarAddSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<^T>)->Slice.scalarAddSlice<^T> i s)
let inline sliceAddScalarOp<^T,^Shape when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary<Slice<^T>,^Shape> name (fun (s:Slice<^T>)->Slice.sliceAddScalar<^T> s i)

let subOp name slice = Stage.liftUnary name (Slice.sub slice)
let inline scalarSubSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<^T>)->Slice.scalarSubSlice<^T> i s)
let inline sliceSubScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<^T>)->Slice.sliceSubScalar<^T> s i)

let mulOp name slice = Stage.liftUnary name (Slice.mul slice)
let inline scalarMulSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<^T>)->Slice.scalarMulSlice<^T> i s)
let inline sliceMulScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<^T>)->Slice.sliceMulScalar<^T> s i)

let divOp name slice = Stage.liftUnary name (Slice.div slice)
let inline scalarDivSliceOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<^T>)->Slice.scalarDivSlice<^T> i s)
let inline sliceDivScalarOp<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (name:string) (i: ^T) = 
    Stage.liftUnary name (fun (s:Slice<^T>)->Slice.sliceDivScalar<^T> s i)

/// Histogram related functions
let histogramOp<'T,'Shape when 'T: comparison> name : Stage<Slice<'T>, Map<'T, uint64>, 'Shape>  =
    let histogramReducer (slices: AsyncSeq<Slice<'T>>) =
        slices
        |> AsyncSeq.map Slice.histogram
        |> AsyncSeqExtensions.fold Slice.addHistogram (Map<'T, uint64> [])
    {
        Name = name
        Transition = Stage.transition Streaming Constant
        Pipe = Pipe.reduce name histogramReducer Streaming 
        ShapeUpdate = id
    }

let map2pairsOp<'T,'S,'Shape when 'T: comparison> name : Stage<Map<'T, 'S>, ('T * 'S) list, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = Pipe.lift name Streaming Slice.map2pairs
        ShapeUpdate = id
    }
let inline pairs2floatsOp<^T,^S,^Shape when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > name : Stage<(^T * ^S) list, (float * float) list, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = Pipe.lift name Streaming Slice.pairs2floats
        ShapeUpdate = id
    }
let inline pairs2intsOp<^T,^S,^Shape when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > name : Stage<(^T * ^S) list, (int * int) list, 'Shape> =
    {
        Name = name
        Transition = Stage.transition Streaming Streaming
        Pipe = Pipe.lift name Streaming Slice.pairs2ints
        ShapeUpdate = id
    }

type ImageStats = Slice.ImageStats
let computeStatsOp<'T,'Shape when 'T : equality> name : Stage<Slice<'T>, ImageStats, 'Shape> =
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
let zeroMaker<'S when 'S: equality> (ex:Slice<'S>) : Slice<'S> = Slice.create<'S> (GetWidth ex) (GetHeight ex) 1u 0u

let discreteGaussianOp<'Shape> (name:string) (sigma:float) (outputRegionMode: OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Slice<float>, Slice<float>, 'Shape> =
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
let binaryFillHolesOp<'Shape> name = liftFullOp name Slice.binaryFillHoles
let connectedComponentsOp<'Shape> (name: string) : Stage<Slice<uint8>, Slice<uint64>, 'Shape> =
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
let signedDistanceMapOp<'Shape> name : Stage<Slice<uint8>, Slice<float>, 'Shape> =
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

