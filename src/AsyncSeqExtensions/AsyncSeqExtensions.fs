module AsyncSeqExtensions

open FSharp.Control

let windowedWithPad
    (windowSize: uint)
    (stride: uint)
    (prePad: uint)
    (postPad: uint)
    (zeroMaker: int -> 'T -> 'T)
    (source: AsyncSeq<'T>)
    : AsyncSeq<'T list> =

    if windowSize = 0u then invalidArg "windowSize" "Must be > 0"
    if stride = 0u then invalidArg "stride" "Must be > 0"

    asyncSeq {
        let! firstItems, rest = AsyncSeq.splitAt 1 source

        match firstItems |> Array.tryHead with
        | None ->
            ()
        | Some first ->
            let mutable last = first
            let mutable nextId = 0

            let body =
                rest
                |> AsyncSeq.prependSeq firstItems
                |> AsyncSeq.map (fun item ->
                    last <- item
                    let id = nextId
                    nextId <- nextId + 1
                    item)

            let prePadding =
                seq {
                    for i in 0 .. int prePad - 1 do
                        yield zeroMaker (i - int prePad) first
                }

            let postPadding =
                seq {
                    for i in 0 .. int postPad - 1 do
                        yield zeroMaker (nextId + i) last
                }

            let trailingEmptySlots =
                seq {
                    for _ in 1 .. int windowSize - 1 do
                        yield None
                }

            let padded =
                body
                |> AsyncSeq.map Some
                |> AsyncSeq.prependSeq (prePadding |> Seq.map Some)
                |> AsyncSeq.appendSeq (
                    seq {
                        yield! postPadding |> Seq.map Some
                        yield! trailingEmptySlots
                    })

            let mutable emittedPartial = false

            let windows =
                padded
                |> AsyncSeq.windowed (int windowSize)
                |> AsyncSeq.indexed
                |> AsyncSeq.choose (fun (index, window) ->
                    if emittedPartial || index % int64 stride <> 0L then
                        None
                    else
                        let items = window |> Array.choose id |> Array.toList

                        if items.Length = int windowSize then
                            Some items
                        elif items.Length > 0 then
                            emittedPartial <- true
                            Some items
                        else
                            None)

            yield! windows
    }

/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
let ofAsync (computation: Async<'T>) : AsyncSeq<'T> =
    asyncSeq {
        let! result = computation
        yield result
    }

let zipConcurrent (s1: AsyncSeq<'U>) (s2: AsyncSeq<'V>) : AsyncSeq<'U * 'V> =
    AsyncSeq.zipWithParallel (fun left right -> left, right) s1 s2
