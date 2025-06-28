module StackPipeline

open FSharp.Control
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

/// Source parts
let create<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) : Pipe<unit, Slice<'T>> =
    printfn "[create]"
    {
        Name = "[create]"
        Profile = Streaming
        Apply = fun _ ->
            AsyncSeq.init (int depth) (fun i -> printfn "[create %d]" i; Slice.create<'T> width height 1u (uint i))
    }

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
                printfn "[readSlices] Reading slice %d to %s" (uint i) fileName
                Slice.readSlice<'T> (uint i) fileName)
    }

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
            printfn "[show]"
            do! (writeSlicesAsync path suffix) stream
        })

let ignore<'T> : Pipe<'T, unit> =
    consumeWith "ignore" Streaming (fun stream ->
        async {
            printfn "[ignore]"
            do! stream |> AsyncSeq.iterAsync (fun _ -> async.Return())
        })
