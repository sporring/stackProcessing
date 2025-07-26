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

open InternalHelpers

(*
let sourceLst<'T>
    (availableMemory: uint64)
    (processors: Pipe<unit,'T> list) 
    : Pipe<unit,'T> list =
    processors |>
    List.map (fun p ->
        pipeline availableMemory {return p}
    )

let source<'T>
    (availableMemory: uint64)
    (p: Pipe<unit,'T>) 
    : Pipe<unit,'T> =
    let lst = sourceLst<'T> availableMemory [p]
    List.head lst

let sinkLst (processors: Pipe<unit, unit> list) : unit =
    if processors.Length > 1 then
        printfn "[Compile time analysis: sinkList parallel]"
    processors
    |> List.map (fun p -> run p |> AsyncSeq.iterAsync (fun () -> async.Return()))
    |> Async.Parallel
    |> Async.Ignore
    |> Async.RunSynchronously

let sink (p: Pipe<unit, unit>) : unit = 
    sinkLst [p]
*)

let internal readSlices<'T when 'T: equality> (inputDir: string) (suffix: string) : Pipe<unit, Slice<'T>> =
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

let read<'T when 'T : equality> (inputDir : string) (suffix : string) transform : Core.Pipe<unit,Slice<'T>> =
    printfn "[read]"
    readSlices<'T> inputDir suffix |> transform
    
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

let internal readRandomSlices<'T when 'T: equality> (count: uint) (inputDir: string) (suffix: string) :Pipe<unit, Slice<'T>> =
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

let readRandom<'T when 'T : equality> (count: uint) (inputDir : string) (suffix : string) transform : Core.Pipe<unit,Slice.Slice<'T>> =
    printfn "[readRandom]"
    readRandomSlices<'T> count inputDir suffix |> transform

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

let create<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) transform : Core.Pipe<unit,Slice.Slice<'T>> =
    printfn "[create]"
    createPipe<'T> width height depth  |> transform

let liftImageSource (name: string) (img: Slice<'T>) : Pipe<unit, Slice<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> unstack |> AsyncSeq.ofSeq
    }

let gaussSource sigma kernelSize transform =
    let img = Slice.gauss 3u sigma kernelSize
    liftImageSource "gauss" img |> transform

let gaussSourceOp // 20250726: This is not working, it seems
    (sigma: float) 
    (kernelSize: uint option)
    (pl : Builder.Pipeline<unit, unit>) 
    : Builder.Pipeline<unit, Slice<float>> =
    let sz = Option.defaultValue (1u + 2u*2u * uint sigma) kernelSize
    let op : Operation<unit, Slice<'T>> =
        {
            Name = $"gaussSource"
            Transition = transition Constant Streaming
            Pipe = Slice.gauss 3u sigma (Some sz) |> liftImageSource "gaussSource"
        }
    {
        shape = Some [sz;sz;sz]
        mem = pl.mem
        flow = Builder.returnM op
    }

let axisSource axis size transform =
    let img = Slice.generateCoordinateAxis axis size
    liftImageSource "axis" img |> transform

let finiteDiffFilter2D (direction: uint) (order: uint) transform : Pipe<unit, Slice<float>> =
        let img = finiteDiffFilter2D direction order
        liftImageSource "finiteDiffFilter2D" img |> transform

let finiteDiffFilter3D (direction: uint) (order: uint) transform : Pipe<unit, Slice<float>> =
    let img = finiteDiffFilter3D direction order
    liftImageSource "finiteDiffFilter3D" img |> transform

let finiteDiffFilter3DOp 
    (direction: uint) 
    (order: uint)
    (pl : Builder.Pipeline<unit, unit>) 
    : Builder.Pipeline<unit, Slice<float>> =
    let img = Slice.finiteDiffFilter3D direction order
    let sz = GetSize img
    let op : Operation<unit, Slice<'T>> =
        {
            Name = $"gaussSource"
            Transition = transition Constant Streaming
            Pipe = img |> liftImageSource "gaussSource"
        }
    {
        shape = Some sz
        mem = pl.mem
        flow = Builder.returnM op
    }

/// Sink parts
let print<'T> : Pipe<'T, unit> =
    printfn "[print]"
    consumeWith "print" Streaming (fun stream ->
        async {
            do! printAsync stream 
        })

let plot (plt: float list -> float list -> unit) : Pipe<(float*float) list, unit> =
    printfn "[plot]"
    consumeWith "plot" Streaming (fun stream ->
        async {
            do! (plotListAsync plt) stream 
        })

let show (plt: Slice.Slice<'a> -> unit) : Pipe<Slice<'a>, unit> =
    printfn "[show]"
    consumeWith "show" Streaming (fun stream ->
        async {
            do! (showSliceAsync plt) stream
        })

let write (path: string) (suffix: string) : Pipe<Slice<'a>, unit> =
    printfn "[write]"
    consumeWith "write" Streaming (fun stream ->
        async {
            do! (writeSlicesAsync path suffix) stream
        })

let ignore<'T> : Pipe<'T, unit> =
    printfn "[ignore]"
    consumeWith "ignore" Streaming (fun stream ->
        async {
            do! stream |> AsyncSeq.iterAsync (fun _ -> async.Return())
        })

let readOp<'T when 'T: equality>
    (inputDir : string)
    (suffix : string)
    (pl : Builder.Pipeline<unit, unit>) : Builder.Pipeline<unit, Slice<'T>> =

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
        flow = Builder.returnM op
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

let createOp<'T when 'T: equality> 
    (width: uint) 
    (height: uint) 
    (depth: uint) 
    (pl : Builder.Pipeline<unit, unit>) 
    : Builder.Pipeline<unit, Slice<'T>> =

    let op : Operation<unit, Slice<'T>> =
        {
            Name = "create"
            Transition = transition Constant Streaming
            Pipe = createPipe<'T> width height depth
        }
    {
        shape = Some [width;height;depth]
        mem = pl.mem
        flow = Builder.returnM op
    }

let readRandomOp<'T when 'T: equality>
    (count: uint) 
    (inputDir : string) 
    (suffix : string)
    (pl : Builder.Pipeline<unit, unit>) : Builder.Pipeline<unit, Slice<'T>> =

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
        flow = Builder.returnM op
    }
