module AsyncSeqExtensions

open FSharp.Control

/// Windowed function with stride and optional padding.
let windowed
    (windowSize: uint)
    (stride: uint)
    (source: AsyncSeq<'T>)
    : AsyncSeq<'T list> =
    if windowSize = 0u then invalidArg "windowSize" "Must be > 0"
    if stride = 0u then invalidArg "stride" "Must be > 0"

    asyncSeq {
        let enum = (AsyncSeq.toAsyncEnum source).GetAsyncEnumerator()
        let buffer = ResizeArray<'T>()
        let mutable finished = false
        let mutable last: 'T option = None

        let rec fillBuffer (n: uint) = async {
            while uint buffer.Count < n && not finished do
                let! hasNext = enum.MoveNextAsync().AsTask() |> Async.AwaitTask
                if hasNext then
                    let current = enum.Current
                    buffer.Add(current)
                    last <- Some current
                else
                    finished <- true
        }

        let rec yieldWindows () = asyncSeq {
            do! fillBuffer windowSize

            if buffer.Count > 0 then
                if uint buffer.Count >= windowSize then
                    yield buffer |> Seq.take (int windowSize) |> Seq.toList

            // Drop stride elements
            let dropCount = min stride (uint buffer.Count)
            buffer.RemoveRange(0, int dropCount)
            if buffer.Count > 0 || not finished then
                yield! yieldWindows ()
        }

        yield! yieldWindows ()
    }

let windowedWithPad
    (windowSize: uint)
    (stride: uint)
    (prePad: uint)
    (postPad: uint)
    (zeroMaker: 'T->'T)
    (source: AsyncSeq<'T>)
    : AsyncSeq<'T list> =
    //printfn "windowedWithPad windowSize=%A stride=%A prePad=%A postPad=%A" windowSize stride prePad postPad

    if windowSize = 0u then invalidArg "windowSize" "Must be > 0"
    if stride = 0u then invalidArg "stride" "Must be > 0"

    asyncSeq {
        let enum = (AsyncSeq.toAsyncEnum source).GetAsyncEnumerator()
        let buffer: ResizeArray<'T> = ResizeArray<'T>()

        let tryGetNext () = async {
            let! hasNext = enum.MoveNextAsync().AsTask() |> Async.AwaitTask
            let next = if hasNext then Some enum.Current else None
            return next
        }

        let tryGetCurrent id prePad postPad zero current = async {
            if prePad > 0u then
                //printfn "yieldWindows prePad"
                return (zero, prePad - 1u, postPad, current)
            elif current <> None then
                let! next = tryGetNext ()
                return (current, prePad, postPad, next)
            elif postPad > 0u then
                //printfn "yieldWindows postPad"
                return (zero, prePad, postPad - 1u, current)
            else
                return (zero, prePad, postPad, current)
        }

        let rec yieldWindows (prePad: uint) (postPad: uint) (zero: 'T option) (step: uint) (current: 'T option)= asyncSeq {
            //printfn "yieldWindows prePad=%A postPad=%A step=%A buffer.length=%A" prePad postPad step buffer.Count
            if postPad = 0u && current = None then 
                //printfn "yieldWindows done"
                ()
            else
                let! curr, nPrePad, nPostPad, next = tryGetCurrent id prePad postPad zero current
                Option.iter buffer.Add curr
                if step > 0u then
                    if buffer.Count > 0 then buffer.RemoveAt 0
                    //printfn "yieldWindows stepping"
                    yield! yieldWindows nPrePad nPostPad zero (step - 1u) next
                elif uint buffer.Count >= windowSize then
                    //printfn "yieldWindows release window buffer.length=%A" buffer.Count
                    yield (buffer |> Seq.take (int windowSize) |> Seq.toList)
                    let dropCount = min stride (uint buffer.Count)
                    buffer.RemoveRange(0, int dropCount)
                    //printfn "yieldWindows continuing"
                    yield! yieldWindows nPrePad nPostPad zero (stride-dropCount) next
                else
                    //printfn "yieldWindows continuing"
                    yield! yieldWindows nPrePad nPostPad zero step next
        }
        let! first = tryGetNext()
        let zero = Option.map zeroMaker first
        yield! yieldWindows prePad postPad zero 0u first
    }

/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
let ofAsync (computation: Async<'T>) : AsyncSeq<'T> =
    asyncSeq {
        let! result = computation
        yield result
    }
