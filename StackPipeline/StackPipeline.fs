module StackPipeline

open FSharp.Control
open System.IO
open Slice
open Core
open Routing

// https://plotly.net/#For-applications-and-libraries
let private plotListAsync (plt: (float list)->(float list)->unit) (vectorSeq: AsyncSeq<(float*float) list>) =
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

let private printAsync (slices: AsyncSeq<'T>) =
    slices
    |> AsyncSeq.iterAsync (fun data ->
        async {
            printfn "[Print] %A" data
        })

let private writeSlicesAsync (outputDir: string) (suffix: string) (slices: AsyncSeq<Slice<'T>>) =
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    slices
    |> AsyncSeq.iterAsync (fun slice ->
        async {
            let fileName = Path.Combine(outputDir, sprintf "slice_%03d%s" slice.Index suffix)
            slice.Image.toFile(fileName)
            printfn "[Write] Saved slice %d to %s" slice.Index fileName
        })

let private readSlices<'T when 'T: equality> (inputDir: string) (suffix: string) : AsyncSeq<Slice<'T>> =
    Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    |> Array.mapi (fun i fileName ->
        async {
            printfn "[Read] Reading slice %d to %s" (uint i) fileName
            return Slice.readSlice (uint i) fileName
        })
    |> Seq.ofArray
    |> AsyncSeq.ofSeqAsync


(*
let runNWriteSlices path suffix maker =
    printfn "[runNWriteSlices]"
    let stream = run maker
    writeSlicesAsync path suffix stream |> Async.RunSynchronously

let runNShowSlice<'T> maker =
    printfn "[runNShowSlice]"
    let stream = run maker
    showSliceAsync stream |> Async.RunSynchronously

let runNPrint maker =
    printfn "[runNPrint]"
    let stream = run maker
    printAsync stream |> Async.RunSynchronously

let runNPlotList plt maker =
    printfn "[runNPlotList]"
    let stream = run maker
    plotListAsync plt stream |> Async.RunSynchronously
*)

//let print p = printfn "[print]"; run p |> printAsync
//let plot plt p = printfn "[plot]"; run p |> plotListAsync plt
//let show plt p = printfn "[show]"; run p |> showSliceAsync plt

let print<'T> : Pipe<'T, unit> =
    consumeWith "print" Streaming (fun stream ->
        async {
            printfn "[print]"
            do! printAsync stream 
        })

let plot plt : Pipe<(float*float) list, unit> =
    consumeWith "plot" Streaming (fun stream ->
        async {
            printfn "[plot]"
            do! (plotListAsync plt) stream 
        })

let show plt : Pipe<Slice<'a>, unit> =
    consumeWith "show" Streaming (fun stream ->
        async {
            printfn "[show]"
            do! (showSliceAsync plt) stream
        })

let writeSlices path suffix : Pipe<Slice<'a>, unit> =
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
