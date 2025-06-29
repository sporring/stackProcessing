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
                printfn "[Write] Saved slice %d to %s of size %A" slice.Index fileName (slice.Image.GetSize())
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
                printfn "[readSlices] Reading slice %d from %s got %A" (uint i) fileName (slice.Image.GetSize())
                slice)
    }

//let read<'T when 'T: equality> (inputDir: string) (suffix: string) p = 
//    readSlices<'T> inputDir suffix |> p
let read<'T when 'T : equality> (inputDir : string) (suffix : string) transform : Core.Pipe<unit,Slice.Slice<'T>> =
    readSlices<'T> inputDir suffix |> transform
    
let readSliceN<'T when 'T: equality> (idx: uint) (inputDir: string) (suffix: string) : Pipe<unit, Slice<'T>> =
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

let readRandomSlices<'T when 'T: equality> (count: uint) (inputDir: string) (suffix: string) :Pipe<unit, Slice<'T>> =
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


let liftImageSource (name: string) (img: Slice<'T>) : Pipe<unit, Slice<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> unstack |> AsyncSeq.ofSeq
    }

let gaussSource sigma kernelSize =
    let img = Slice.gauss sigma kernelSize
    liftImageSource "gauss" img

let axisSource axis size =
    let img = Slice.generateCoordinateAxis axis size
    liftImageSource "axis" img

let gauss (sigma: float) (kernelSize: uint option) : Pipe<unit, Slice<float>> =
    printfn "[gauss]"
    {
        Name = "[gauss]"
        Profile = Streaming
        Apply = fun _ ->
            let img = gauss sigma kernelSize
            printfn $"{img.Image.GetSize()}"
            let imgLst = img |> unstack
            printfn $"{imgLst.Length}"
            imgLst |> AsyncSeq.ofSeq
    }

let finiteDiffFilter1D (order: uint) : Pipe<unit, Slice<float>> =
    printfn "[finiteDiffFilter1D]"
    {
        Name = "[finiteDiffFilter1D]"
        Profile = Streaming
        Apply = fun _ ->
            finiteDiffFilter1D order
            |> unstack
            |> AsyncSeq.ofSeq
    }

let finiteDiffFilter2D (direction: uint) (order: uint) : Pipe<unit, Slice<float>> =
    printfn "[finiteDiffFilter2D]"
    {
        Name = "[finiteDiffFilter2D]"
        Profile = Streaming
        Apply = fun _ ->
            finiteDiffFilter2D direction order
            |> unstack
            |> AsyncSeq.ofSeq
    }

let finiteDiffFilter3D (direction: uint) (order: uint) : Pipe<unit, Slice<float>> =
    printfn "[finiteDiffFilter3D]"
    {
        Name = "[finiteDiffFilter3D]"
        Profile = Streaming
        Apply = fun _ ->
            finiteDiffFilter3D direction order
            |> unstack
            |> AsyncSeq.ofSeq
    }

/// Sink parts
let print<'T> : Pipe<'T, unit> =
    consumeWith "print" Streaming (fun stream ->
        async {
            printfn "[print]"
            do! printAsync stream 
        })

let plot (plt: float list -> float list -> unit) : Pipe<(float*float) list, unit> =
    consumeWith "plot" Streaming (fun stream ->
        async {
            printfn "[plot]"
            do! (plotListAsync plt) stream 
        })

let show (plt: Slice.Slice<'a> -> unit) : Pipe<Slice<'a>, unit> =
    consumeWith "show" Streaming (fun stream ->
        async {
            printfn "[show]"
            do! (showSliceAsync plt) stream
        })

let writeSlices (path: string) (suffix: string) : Pipe<Slice<'a>, unit> =
    consumeWith "write" Streaming (fun stream ->
        async {
            printfn "[write]"
            do! (writeSlicesAsync path suffix) stream
        })

let ignore<'T> : Pipe<'T, unit> =
    consumeWith "ignore" Streaming (fun stream ->
        async {
            printfn "[ignore]"
            do! stream |> AsyncSeq.iterAsync (fun _ -> async.Return())
        })
