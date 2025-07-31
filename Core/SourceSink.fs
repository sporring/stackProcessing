module SourceSink

open FSharp.Control
open AsyncSeqExtensions
open System.IO
open Slice
open Core
open Routing

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

    let readSlicesAsync<'T when 'T: equality> (inputDir: string) (suffix: string) : AsyncSeq<Slice<'T>> = // Not used
        Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
        |> Array.mapi (fun i fileName ->
            async {
                printfn "[Read] Reading slice %d to %s" (uint i) fileName
                return Slice.readSlice (uint i) fileName
            })
        |> Seq.ofArray
        |> AsyncSeq.ofSeqAsync

    let readSlices<'T when 'T: equality> (inputDir: string) (suffix: string) : Pipe<unit, Slice<'T>> = // Not used
        printfn "[readSlices]"
        let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
        let depth = filenames.Length
        {
            Name = $"[readSlices {inputDir}]"
            Profile = Streaming
            Apply = fun _ ->
                AsyncSeq.init (int depth) (fun i -> 
                    let fileName = filenames[int i]; 
                    let slice = Slice.readSlice<'T> (uint i) fileName
                    printfn "[readSlices] Reading slice %d from %s" (uint i) fileName
                    slice)
        }

    let readRandomSlices<'T when 'T: equality> (count: uint) (inputDir: string) (suffix: string) :Pipe<unit, Slice<'T>> = // Not used
        let fileNames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices (int count)
        {
            Name = $"[readRandomSlices {inputDir}]"
            Profile = Streaming
            Apply = fun _ ->
                AsyncSeq.init (int count) (fun i -> 
                    let fileName = fileNames[int i]; 
                    printfn "[readRandomSlices] Reading slice %d from %s" (uint i) fileName
                    Slice.readSlice<'T> (uint i) fileName)
        }

open InternalHelpers

/// Source parts
let createOp<'T,'Shape when 'T: equality> (width: uint) (height: uint) (depth: uint) (pl : Pipeline<unit, unit, 'Shape>) : Pipeline<unit, Slice<'T>,'Shape> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    let mapper (i: uint) : Slice<'T> = 
        let slice = Slice.create<'T> width height 1u i
        printfn "[create] Created slice %A" i
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = fun (s:'Shape) -> s
    let stage = Stage.create "create" depth mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Some [width;height]
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    Pipeline.create flow pl.mem pl.shape context

let readOp<'T when 'T: equality>
    (inputDir : string)
    (suffix : string)
    (pl : Pipeline<unit, unit,'Shape>) : Pipeline<unit, Slice<'T>,'Shape> =
    // much should be deferred to outside Core!!!
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = filenames.Length
    let mapper (i: uint) : Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        printfn "[readSlices] Reading slice %A from %s" i fileName
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = fun (s:'Shape) -> s
    let stage = Stage.create $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Some [width;height]
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem shape context

let readRandomOp<'T when 'T: equality>
    (count: uint) 
    (inputDir : string) 
    (suffix : string)
    (pl : Pipeline<unit, unit,'Shape>) : Pipeline<unit, Slice<'T>,'Shape> =

    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices (int count)
    let depth = filenames.Length
    let mapper (i: uint) : Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        printfn "[readRandomSlices] Reading slice %A from %s" i fileName
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = fun (s:'Shape) -> s
    let stage = Stage.create $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Some [width;height]
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem shape context

let writeOp (path: string) (suffix: string) : Stage<Slice<'a>, unit, 'Shape> =
    let writeReducer stream = async { do! writeSlicesAsync path suffix stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = $"write:{path}"
        Pipe = Pipe.consumeWith "write" Streaming writeReducer
        Transition = Stage.transition Streaming Constant
        ShapeUpdate = shapeUpdate
    }

let showOp (plt: Slice.Slice<'T> -> unit) : Stage<Slice<'T>, unit, 'Shape> =
    let showReducer stream = async {do! showSliceAsync plt stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = "show"
        Pipe = Pipe.consumeWith "show" Streaming showReducer
        Transition = Stage.transition Streaming Constant
        ShapeUpdate = shapeUpdate
    }

let plotOp (plt: float list -> float list -> unit) : Stage<(float * float) list, unit, 'Shape> =
    let plotReducer stream = async { do! plotListAsync plt stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = "plot"
        Pipe = Pipe.consumeWith "plot" Streaming plotReducer
        Transition = Stage.transition Streaming Streaming
        ShapeUpdate = shapeUpdate
    }

let printOp () : Stage<'T, unit,'Shape> =
    let printReducer stream = async { do! printAsync stream }
    let shapeUpdate = fun (s:'Shape) -> s
    {
        Name = "print"
        Pipe = Pipe.consumeWith "print" Streaming printReducer
        Transition = Stage.transition Streaming Streaming
        ShapeUpdate = shapeUpdate
    }

let liftImageSource (name: string) (img: Slice<'T>) : Pipe<unit, Slice<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> unstack |> AsyncSeq.ofSeq
    }

let axisSourceOp 
    (axis: int) 
    (size: int list)
    (pl : Pipeline<unit, unit,'Shape>) 
    : Pipeline<unit, Slice<uint>,'Shape> =
    let img = Slice.generateCoordinateAxis axis size
    let sz = GetSize img
    let shapeUpdate = fun (s:'Shape) -> s
    let op : Stage<unit, Slice<uint>,'Shape> =
        {
            Name = "axisSource"
            Pipe = img |> liftImageSource "axisSource"
            Transition = Stage.transition Constant Streaming
            ShapeUpdate = shapeUpdate
        }
    let width, height, depth = sz[0], sz[1], sz[2]
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = MemFlow.returnM op
        mem = pl.mem
        shape = Some [width;height]
        context = context
    }

let finiteDiffFilter3DOp 
    (direction: uint) 
    (order: uint)
    (pl : Pipeline<unit, unit,'Shape>) 
    : Pipeline<unit, Slice<float>,'Shape> =
    let img = finiteDiffFilter3D direction order
    let sz = GetSize img
    let shapeUpdate = fun (s:'Shape) -> s
    let op : Stage<unit, Slice<float>, 'Shape> =
        {
            Name = "gaussSource"
            Pipe = img |> liftImageSource "gaussSource"
            Transition = Stage.transition Constant Streaming
            ShapeUpdate = shapeUpdate
        }
    let width, height, depth = sz[0], sz[1], sz[2]
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = MemFlow.returnM op
        mem = pl.mem
        shape = Some [width;height]
        context = context
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
