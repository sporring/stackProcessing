namespace FSharp
module AsyncSeqExtensions
module Async =
    /// Transforms the result of an asynchronous computation using a pure function.
    val map: f: ('a -> 'b) -> asyncVal: Async<'a> -> Async<'b>
/// Applies a folder function over an asynchronous sequence, threading an accumulator through the sequence like <c>List.fold</c>.
val fold:
  folder: ('State -> 'T -> 'State) ->
    state: 'State -> source: FSharp.Control.AsyncSeq<'T> -> Async<'State>
/// Attempts to retrieve the item at the specified index from an asynchronous sequence.
val tryItem: n: int -> source: FSharp.Control.AsyncSeq<'T> -> Async<'T option>
/// Windowed function with stride and optional padding.
val windowed:
  windowSize: uint ->
    stride: uint ->
    source: FSharp.Control.AsyncSeq<'T> -> FSharp.Control.AsyncSeq<'T list>
val windowedWithPad:
  windowSize: uint ->
    stride: uint ->
    prePad: uint ->
    postPad: uint ->
    zeroMaker: ('T -> 'T) ->
    source: FSharp.Control.AsyncSeq<'T> -> FSharp.Control.AsyncSeq<'T list>
    when 'T: equality
/// Splits an asynchronous sequence into fixed-size chunks.
val chunkBySize:
  chunkSize: int ->
    source: FSharp.Control.AsyncSeq<'T> -> FSharp.Control.AsyncSeq<'T list>
/// Combines two asynchronous sequences of image slices element-wise using a user-defined function.
val zipJoin:
  combine: ('S -> 'T -> 'U) ->
    a: FSharp.Control.AsyncSeq<'S> ->
    b: FSharp.Control.AsyncSeq<'T> ->
    txt: string option -> FSharp.Control.AsyncSeq<'U>
/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
val ofAsync: computation: Async<'T> -> FSharp.Control.AsyncSeq<'T>
/// Missing AsyncSeq.share: broadcasts each element to multiple readers, only caching one element at a time.
val share: input: FSharp.Control.AsyncSeq<'T> -> FSharp.Control.AsyncSeq<'T>
