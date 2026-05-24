module AsyncSeqExtensions

open FSharp.Control

/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
let ofAsync (computation: Async<'T>) : AsyncSeq<'T> =
    asyncSeq {
        let! result = computation
        yield result
    }

let zipConcurrent (s1: AsyncSeq<'U>) (s2: AsyncSeq<'V>) : AsyncSeq<'U * 'V> =
    AsyncSeq.zipWithParallel (fun left right -> left, right) s1 s2
