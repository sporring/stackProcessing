namespace FSharp
module AsyncSeqExtensions
val windowedWithPad:
  windowSize: uint ->
    stride: uint ->
    prePad: uint ->
    postPad: uint ->
    zeroMaker: (int -> 'T -> 'T) ->
    source: FSharp.Control.AsyncSeq<'T> -> FSharp.Control.AsyncSeq<'T list>
    when 'T: equality
/// Converts an asynchronous computation of a single value into an asynchronous sequence containing one item.
val ofAsync: computation: Async<'T> -> FSharp.Control.AsyncSeq<'T>
val zipConcurrent:
  s1: FSharp.Control.AsyncSeq<'U> ->
    s2: FSharp.Control.AsyncSeq<'V> -> FSharp.Control.AsyncSeq<'U * 'V>
