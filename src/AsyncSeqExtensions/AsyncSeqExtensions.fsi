namespace FSharp
module AsyncSeqExtensions
/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
val ofAsync: computation: Async<'T> -> FSharp.Control.AsyncSeq<'T>
val zipConcurrent:
  s1: FSharp.Control.AsyncSeq<'U> ->
    s2: FSharp.Control.AsyncSeq<'V> -> FSharp.Control.AsyncSeq<'U * 'V>
