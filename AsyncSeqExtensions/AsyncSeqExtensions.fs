module AsyncSeqExtensions

open FSharp.Control

module Async =
    /// Transforms the result of an asynchronous computation using a pure function.
    let map f asyncVal = async {
        let! x = asyncVal
        return f x
    }

(*
/// Applies a folder function over an asynchronous sequence, threading an accumulator through the sequence like <c>List.fold</c>.
let fold (folder: 'State -> 'T -> 'State) (state: 'State) (source: AsyncSeq<'T>) : Async<'State> =
    let mutable acc = state
    source
    |> AsyncSeq.iterAsync (fun item ->
        async {
            acc <- folder acc item
        })
    |> Async.map (fun () -> acc)
*)

/// Attempts to retrieve the item at the specified index from an asynchronous sequence.
let tryItem (n: int) (source: AsyncSeq<'T>) : Async<'T option> =
    async {
        let mutable i = 0
        let mutable result = None
        use enumerator = source.GetEnumerator()
        let rec loop () =
            async {
                match! enumerator.MoveNext() with
                    | Some item ->
                        if i = n then
                            result <- Some item
                        else
                            i <- i + 1
                            return! loop ()
                    | None -> return ()
            }
        do! loop ()
        return result
    }

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

/// Splits an asynchronous sequence into fixed-size chunks.
let chunkBySize (chunkSize: int) (source: AsyncSeq<'T>) : AsyncSeq<'T list> =
    if chunkSize <= 0 then
        invalidArg "chunkSize" "Chunk size must be greater than 0"
    asyncSeq {
        let buffer = ResizeArray<'T>(chunkSize)
        for awaitElem in source do
            buffer.Add(awaitElem)
            if buffer.Count = chunkSize then
                yield buffer |> Seq.toList
                buffer.Clear()
        // yield remaining items
        if buffer.Count > 0 then
            yield buffer |> Seq.toList
    }


/// Combines two asynchronous sequences of image slices element-wise using a user-defined function.
let zipJoin (combine: 'S -> 'T -> 'U) 
            (a: AsyncSeq<'S>) 
            (b: AsyncSeq<'T>)
            (txt: string option) 
            : AsyncSeq<'U> =
    AsyncSeq.zip a b 
    |> AsyncSeq.map (
        fun (x, y) -> 
            match txt with 
                Some t -> printfn "%s" t 
                | None -> ()
            combine x y)

/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
let ofAsync (computation: Async<'T>) : AsyncSeq<'T> =
    asyncSeq {
        let! result = computation
        yield result
    }

/// Missing AsyncSeq.share: broadcasts each element to multiple readers, only caching one element at a time.
let share (input: AsyncSeq<'T>) : AsyncSeq<'T> =
    let agent = MailboxProcessor.Start(fun inbox ->
        async {
            let enumerator = (AsyncSeq.toAsyncEnum input).GetAsyncEnumerator()
            let mutable current: option<'T> = None
            let mutable waiting = ResizeArray<AsyncReplyChannel<option<'T>>>()
            let rec loop () = async {
                if current.IsNone then
                    let! hasNext = enumerator.MoveNextAsync().AsTask() |> Async.AwaitTask
                    current <- if hasNext then Some enumerator.Current else None
                while waiting.Count > 0 do
                    let ch = waiting.[0]
                    waiting.RemoveAt(0)
                    ch.Reply(current)
                if current.IsSome then
                    current <- None
                    return! loop ()
                else
                    ()
            }
            while true do
                let! reply = inbox.Receive()
                waiting.Add(reply)
                if current.IsNone then
                    do! loop ()
        })
    asyncSeq {
        let mutable done_ = false
        while not done_ do
            let! v = agent.PostAndAsyncReply(id)
            match v with
            | Some x -> yield x
            | None -> done_ <- true
    }
