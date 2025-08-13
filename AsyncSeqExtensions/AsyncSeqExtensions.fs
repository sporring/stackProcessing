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
    (zeroMaker: int->'T->'T)
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

        let tryGetCurrent id prePad postPad (zeroMaker: int->'T->'T) prev current = async {
            //printfn $"id {id}, prePad {prePad}, postPad {postPad}, buffer.Count {buffer.Count}, windowSize {windowSize}"
            if prePad > 0u then
                //printfn "tryGetCurrent prePad"
                let zero = Option.map (zeroMaker id) current
                return (zero, prePad - 1u, postPad, current)
            elif current <> None then
                //printfn "tryGetCurrent current"
                let! next = tryGetNext ()
                return (current, prePad, postPad, next)
            else //elif postPad > 0u then
                //printfn "tryGetCurrent postPad"
                let zero = Option.map (zeroMaker id) prev
                return (zero, prePad, postPad - 1u, current)
//            else
//                printfn "tryGetCurrent default"
//                let zero = Option.map (zeroMaker id) current
//                return (zero, prePad, postPad, current)
        }
        // kernesize 5
        // index:     -2-1 0 1 2 3 4 5 6 7
        // pad:        * *             * *
        // winsize 7:  * * * * * * *
        // valid:          * * *
        // stride 3, ws 7:   * * * * * * *
        // valid:                * * *
        // stride 3, ws 7:         * * * * * * *
        // valid:                      
        // stride = windowSize-2*p
        // m+2*pad = n*stride+windowSize
        // n = (m+2*pad-windowSize)/stride

        let rec yieldWindows (id: int) (prePad: uint) (postPad: uint) (zeroMaker: int->'T->'T) (step: uint) (current: 'T option) (next: 'T option) = asyncSeq {
            //printfn "yieldWindows prePad=%A postPad=%A step=%A buffer.length=%A" prePad postPad step buffer.Count
            if postPad = 0u && next = None then // when postPad goes to zero, then we contiue padding until buffer is emptied a last time
                ()
            else
                //printfn $"Before {buffer.Count}"
                let! curr, nPrePad, nPostPad, nxt = tryGetCurrent id prePad postPad zeroMaker current next
                Option.iter buffer.Add curr
                //printfn $"After {curr}, {nxt}, {buffer.Count}"
                if step > 0u then
                    if buffer.Count > 0 then buffer.RemoveAt 0
                    //printfn "yieldWindows stepping"
                    yield! yieldWindows (id+1) nPrePad nPostPad zeroMaker (step - 1u) curr nxt
                elif uint buffer.Count >= windowSize then
                    //printfn "yieldWindows release window buffer.length=%A" buffer.Count
                    yield (buffer |> Seq.take (int windowSize) |> Seq.toList)
                    let dropCount = min stride (uint buffer.Count)
                    buffer.RemoveRange(0, int dropCount)
                    //printfn "yieldWindows continuing"
                    yield! yieldWindows (id+1) nPrePad nPostPad zeroMaker (stride-dropCount) curr nxt
                else
                    //printfn "yieldWindows continuing"
                    yield! yieldWindows (id+1) nPrePad nPostPad zeroMaker step curr nxt
        }
        let! first = tryGetNext()
        yield! yieldWindows (-(int prePad)) prePad postPad zeroMaker 0u None first
    }

/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
let ofAsync (computation: Async<'T>) : AsyncSeq<'T> =
    asyncSeq {
        let! result = computation
        yield result
    }

let zipConcurrent (s1: AsyncSeq<'U>) (s2: AsyncSeq<'V>) : AsyncSeq<'U * 'V> =
    asyncSeq {
        let e1 = (AsyncSeq.toAsyncEnum s1).GetAsyncEnumerator()
        let e2 = (AsyncSeq.toAsyncEnum s2).GetAsyncEnumerator()

        let rec loop () = asyncSeq {
            // Start both MoveNextAsync calls concurrently
            let! c1Child = e1.MoveNextAsync().AsTask() |> Async.AwaitTask |> Async.StartChild
            let! c2Child = e2.MoveNextAsync().AsTask() |> Async.AwaitTask |> Async.StartChild
            let! has1 = c1Child
            let! has2 = c2Child
            if has1 && has2 then
                yield (e1.Current, e2.Current)
                yield! loop ()
            else
                // end at the shorter stream
                ()
        }
        yield! loop ()
    }
