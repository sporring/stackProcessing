module AsyncSeqExtensions

open FSharp.Control

module Async =
    /// <summary>
    /// Transforms the result of an asynchronous computation using a pure function.
    /// </summary>
    /// <param name="f">The function to apply to the result of the asynchronous computation.</param>
    /// <param name="asyncVal">The asynchronous computation to map over.</param>
    /// <returns>
    /// A new asynchronous computation whose result is the application of <c>f</c> to the result of <c>asyncVal</c>.
    /// </returns>
    /// <example>
    /// <code>
    /// let asyncResult = async { return 2 }
    /// let doubled = Async.map ((*) 2) asyncResult
    /// // doubled evaluates to async { return 4 }
    /// </code>
    /// </example>
    let map f asyncVal = async {
        let! x = asyncVal
        return f x
    }

/// <summary>
/// Applies a folder function over an asynchronous sequence, threading an accumulator through the sequence like <c>List.fold</c>.
/// </summary>
/// <param name="folder">A function that updates the state given the current state and the next element of the sequence.</param>
/// <param name="state">The initial state value.</param>
/// <param name="source">The asynchronous sequence to fold over.</param>
/// <returns>
/// An asynchronous computation that yields the final accumulated state after processing all elements of the sequence.
/// </returns>
/// <remarks>
/// This function evaluates elements of the sequence lazily and processes them in order as they become available.
/// </remarks>
let fold (folder: 'State -> 'T -> 'State) (state: 'State) (source: AsyncSeq<'T>) : Async<'State> =
    let acc = ref state
    source
    |> AsyncSeq.iterAsync (fun item ->
        async {
            acc := folder !acc item
        })
    |> Async.map (fun () -> !acc)

/// <summary>
/// Attempts to retrieve the item at the specified index from an asynchronous sequence.
/// </summary>
/// <param name="n">The zero-based index of the item to retrieve.</param>
/// <param name="source">The asynchronous sequence to retrieve the item from.</param>
/// <returns>
/// An asynchronous computation that yields <c>Some(item)</c> if found, or <c>None</c> if the index is out of bounds.
/// </returns>
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

/// <summary>
/// Creates a sliding window over an asynchronous sequence, returning lists of elements of the specified window size.
/// </summary>
/// <param name="windowSize">The number of elements in each window. Must be greater than 0.</param>
/// <param name="source">The asynchronous sequence to create windows from.</param>
/// <returns>
/// An asynchronous sequence of lists, where each list represents a window of elements from the source sequence.
/// </returns>
/// <exception cref="System.ArgumentException">
/// Thrown if <paramref name="windowSize"/> is less than or equal to 0.
/// </exception>
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

/// <summary>
/// Splits an asynchronous sequence into fixed-size chunks.
/// </summary>
/// <param name="chunkSize">The number of elements per chunk. Must be greater than 0.</param>
/// <param name="source">The asynchronous sequence to divide into chunks.</param>
/// <returns>
/// An asynchronous sequence of lists, where each list contains up to <paramref name="chunkSize"/> elements.
/// The last chunk may be smaller if the total number of elements is not a multiple of the chunk size.
/// </returns>
/// <exception cref="System.ArgumentException">
/// Thrown if <paramref name="chunkSize"/> is less than or equal to 0.
/// </exception>
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


/// <summary>
/// Combines two asynchronous sequences of image slices element-wise using a user-defined function.
/// </summary>
/// <param name="combine">A function that defines how to combine two <c>ImageSlice</c> values.</param>
/// <param name="a">The first asynchronous sequence of image slices.</param>
/// <param name="b">The second asynchronous sequence of image slices.</param>
/// <returns>
/// An asynchronous sequence where each element is the result of applying <paramref name="combine"/> to corresponding elements of <paramref name="a"/> and <paramref name="b"/>.
/// </returns>
/// <remarks>
/// The resulting sequence ends when the shorter of the two input sequences ends.
/// </remarks>
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

/// <summary>
/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
/// </summary>
/// <param name="computation">
/// An asynchronous computation that produces a single value of type <c>'T</c>.
/// </param>
/// <returns>
/// An <c>AsyncSeq&lt;'T&gt;</c> that yields the result of the computation as its only element.
/// </returns>
/// <remarks>
/// This function is useful when integrating reducers or standalone async computations into streaming pipelines
/// that expect an <c>AsyncSeq</c> output.
/// </remarks>
/// <example>
/// <code>
/// let singleItem = async { return 42 }
/// let stream = AsyncSeqUtil.ofAsync singleItem
/// </code>
/// </example>
let ofAsync (computation: Async<'T>) : AsyncSeq<'T> =
    asyncSeq {
        let! result = computation
        yield result
    }

let tee (source: AsyncSeq<'T>) : AsyncSeq<'T> * AsyncSeq<'T> =
    let buffer = new System.Collections.Concurrent.BlockingCollection<'T>()
    let copy =
        asyncSeq {
            for item in source do
                buffer.Add(item)
                yield item
            buffer.CompleteAdding()
        }
    let duplicate =
        asyncSeq {
            for item in buffer.GetConsumingEnumerable() do
                yield item
        }
    copy, duplicate

/// Simple version of AsyncSeq.share: broadcasts each element to multiple readers,
/// only caching one element at a time.
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
