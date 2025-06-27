module AsyncSeqExtensions

open FSharp.Control

module Async =
    /// Transforms the result of an asynchronous computation using a pure function.
    let map f asyncVal = async {
        let! x = asyncVal
        return f x
    }

/// Applies a folder function over an asynchronous sequence, threading an accumulator through the sequence like <c>List.fold</c>.
let fold (folder: 'State -> 'T -> 'State) (state: 'State) (source: AsyncSeq<'T>) : Async<'State> =
    let acc = ref state
    source
    |> AsyncSeq.iterAsync (fun item ->
        async {
            acc := folder !acc item
        })
    |> Async.map (fun () -> !acc)

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

/// Creates a sliding window over an asynchronous sequence, returning lists of elements of the specified window size.
let windowed (windowSize: int) (source: AsyncSeq<'T>) : AsyncSeq<'T list> =
    if windowSize <= 0 then
        invalidArg "windowSize" "Must be greater than 0"
    asyncSeq {
        let queue = System.Collections.Generic.Queue<'T>()
        for awaitElem in source do
            queue.Enqueue awaitElem
            if queue.Count = windowSize then
                yield queue |> Seq.toList
                queue.Dequeue() |> ignore
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
