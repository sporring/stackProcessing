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

    let readSlicesAsync<'T when 'T: equality> (inputDir: string) (suffix: string) : AsyncSeq<Slice<'T>> =
        Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
        |> Array.mapi (fun i fileName ->
            async {
                printfn "[Read] Reading slice %d to %s" (uint i) fileName
                return Slice.readSlice (uint i) fileName
            })
        |> Seq.ofArray
        |> AsyncSeq.ofSeqAsync

    let readSlices<'T when 'T: equality> (inputDir: string) (suffix: string) : Pipe<unit, Slice<'T>> =
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

    let readRandomSlices<'T when 'T: equality> (count: uint) (inputDir: string) (suffix: string) :Pipe<unit, Slice<'T>> =
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
let createPipe<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) : Pipe<unit, Slice<'T>> =
    {
        Name = "[create]"
        Profile = Streaming
        Apply = fun _ ->
            AsyncSeq.init (int depth) (fun i -> 
                let slice = Slice.create<'T> width height 1u (uint i)
                printfn "[create] Created slice %d" i
                slice)
    }

let createOp<'T when 'T: equality> 
    (width: uint) 
    (height: uint) 
    (depth: uint) 
    (pl : Pipeline<unit, unit>) 
    : Pipeline<unit, Slice<'T>> =

    let op : Operation<unit, Slice<'T>> =
        {
            Name = "create"
            Transition = transition Constant Streaming
            Pipe = createPipe<'T> width height depth
        }
    {
        shape = Some [width;height;depth]
        mem = pl.mem
        flow = returnM op
    }

let readOp<'T when 'T: equality>
    (inputDir : string)
    (suffix : string)
    (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice<'T>> =

    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let op : Operation<unit, Slice<'T>> =
        {
            Name = $"read:{inputDir}"
            Transition = transition Constant Streaming
            Pipe = readSlices<'T> inputDir suffix
        }
    {
        shape = Some [width;height;depth]
        mem = pl.mem
        flow = returnM op
    }

let readRandomOp<'T when 'T: equality>
    (count: uint) 
    (inputDir : string) 
    (suffix : string)
    (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice<'T>> =

    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let op : Operation<unit, Slice<'T>> =
        {
            Name = $"readRandom:{inputDir}"
            Transition = transition Constant Streaming
            Pipe = readRandomSlices<'T> count inputDir suffix 
        }
    {
        shape = Some [width;height;count]
        mem = pl.mem
        flow = returnM op
    }

let writeOp (path: string) (suffix: string) : Operation<Slice<'a>, unit> =
    let writeReducer stream = async { do! writeSlicesAsync path suffix stream }
    {
        Name = $"write:{path}"
        Transition = transition Streaming Constant
        Pipe = consumeWith "write" Streaming writeReducer
    }

let showOp (plt: Slice.Slice<'T> -> unit) : Operation<Slice<'T>, unit> =
    let showReducer stream = async {do! showSliceAsync plt stream }
    {
        Name = "show"
        Transition = transition Streaming Constant
        Pipe = consumeWith "show" Streaming showReducer
    }

let plotOp (plt: float list -> float list -> unit) : Operation<(float * float) list, unit> =
    let plotReducer stream = async { do! plotListAsync plt stream }
    {
        Name = "plot"
        Transition = transition Streaming Streaming
        Pipe = consumeWith "plot" Streaming plotReducer
    }

let printOp () : Operation<'T, unit> =
    let printReducer stream = async { do! printAsync stream }
    {
        Name = "print"
        Transition = transition Streaming Streaming
        Pipe = consumeWith "print" Streaming printReducer
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
    (pl : Pipeline<unit, unit>) 
    : Pipeline<unit, Slice<uint>> =
    let img = Slice.generateCoordinateAxis axis size
    let sz = GetSize img
    let op : Operation<unit, Slice<uint>> =
        {
            Name = "axisSource"
            Transition = transition Constant Streaming
            Pipe = img |> liftImageSource "axisSource"
        }
    {
        shape = Some sz
        mem = pl.mem
        flow = returnM op
    }

let finiteDiffFilter3DOp 
    (direction: uint) 
    (order: uint)
    (pl : Pipeline<unit, unit>) 
    : Pipeline<unit, Slice<float>> =
    let img = finiteDiffFilter3D direction order
    let sz = GetSize img
    let op : Operation<unit, Slice<float>> =
        {
            Name = "gaussSource"
            Transition = transition Constant Streaming
            Pipe = img |> liftImageSource "gaussSource"
        }
    {
        shape = Some sz
        mem = pl.mem
        flow = returnM op
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
    consumeWith "ignore" Streaming (fun stream ->
        async {
            do! stream |> AsyncSeq.iterAsync (fun _ -> async.Return())
        })
*)
