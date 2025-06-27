namespace FSharp
module StackPipeline
val private plotListAsync:
  plt: (float list -> float list -> unit) ->
    vectorSeq: FSharp.Control.AsyncSeq<(float * float) list> -> Async<unit>
val showSliceAsync:
  plt: (Slice.Slice<'T> -> unit) ->
    slices: FSharp.Control.AsyncSeq<Slice.Slice<'T>> -> Async<unit>
    when 'T: equality
val private printAsync: slices: FSharp.Control.AsyncSeq<'T> -> Async<unit>
val private writeSlicesAsync:
  outputDir: string ->
    suffix: string ->
    slices: FSharp.Control.AsyncSeq<Slice.Slice<'T>> -> Async<unit>
    when 'T: equality
val private readSlices:
  inputDir: string -> suffix: string -> FSharp.Control.AsyncSeq<Slice.Slice<'T>>
    when 'T: equality
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Streaming
    | Sliding of uint
    | Buffered
    member EstimateUsage: width: uint -> height: uint -> depth: uint -> uint64
    member
      RequiresBuffering: availableMemory: uint64 ->
                           width: uint -> height: uint -> depth: uint -> bool
/// A configurable image processing step that operates on image slices.
type StackProcessor<'S,'T> =
    {
      Name: string
      Profile: MemoryProfile
      Apply: (FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>)
    }
/// Pipeline computation expression
type PipelineBuilder =
    new: availableMemory: uint64 * width: uint * height: uint * depth: uint ->
           PipelineBuilder
    member
      Bind: p: StackProcessor<'S,'T> * f: ('T -> StackProcessor<'S,'U>) ->
              StackProcessor<'S,'U>
    /// Chain two <c>StackProcessor</c> instances, optionally inserting intermediate disk I/O
    member
      Bind: p: StackProcessor<'S,'T> *
            f: (StackProcessor<'S,'T> -> StackProcessor<'S,'T>) ->
              StackProcessor<'S,'T>
    /// Wraps a processor value for use in the pipeline computation expression.
    member Return: p: StackProcessor<'S,'T> -> StackProcessor<'S,'T>
    /// Allows returning a processor directly from another computation expression.
    member ReturnFrom: p: StackProcessor<'S,'T> -> StackProcessor<'S,'T>
    /// Provides a default identity processor using streaming as the memory profile.
    member Zero: unit -> StackProcessor<'a,'a>
/// Combine two <c>StackProcessor</c> instances into one by composing their memory profiles and transformation functions.
val (>>=>) :
  p1: StackProcessor<'S,'T> ->
    p2: StackProcessor<'T,'U> -> StackProcessor<'S,'U>
/// A memory-aware pipeline builder with the specified processing constraints.
val pipeline:
  availableMemory: uint64 ->
    width: uint -> height: uint -> depth: uint -> PipelineBuilder
/// Pipeline helper functions
val singleton: x: 'In -> StackProcessor<'In,'In>
val private runWith:
  input: FSharp.Control.AsyncSeq<'In> ->
    p: StackProcessor<'In,'T> -> FSharp.Control.AsyncSeq<'T>
val private run: p: StackProcessor<unit,'T> -> FSharp.Control.AsyncSeq<'T>
val fromReducer:
  name: string ->
    profile: MemoryProfile ->
    reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
    StackProcessor<'In,'Out>
val fromConsumer:
  name: string ->
    profile: MemoryProfile ->
    consume: (FSharp.Control.AsyncSeq<'T> -> Async<unit>) ->
    StackProcessor<'T,unit>
val print<'T> : StackProcessor<'T,unit>
val plot:
  plt: (float list -> float list -> unit) ->
    StackProcessor<(float * float) list,unit>
val show:
  plt: (Slice.Slice<'a> -> unit) -> StackProcessor<Slice.Slice<'a>,unit>
    when 'a: equality
val writeSlices:
  path: string -> suffix: string -> StackProcessor<Slice.Slice<'a>,unit>
    when 'a: equality
/// Join two StackProcessors<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
val join:
  f: ('A -> 'B -> 'C) ->
    p1: StackProcessor<'In,'A> ->
    p2: StackProcessor<'In,'B> -> StackProcessor<'In,'C>
/// Split a StackProcessor<'In,'T> into two branches that
///   • read the upstream only once
///   • keep at most one item in memory
///   • terminate correctly when both sides finish
type private Request<'T> =
    | Left of AsyncReplyChannel<Option<'T>>
    | Right of AsyncReplyChannel<Option<'T>>
val tee:
  p: StackProcessor<'In,'T> -> StackProcessor<'In,'T> * StackProcessor<'In,'T>
/// Fan out a StackProcessor<'In,'T> to two branches:
///   • processes input once using tee
///   • applies separate processors to each branch
///   • zips outputs into a tuple
val fanOut:
  p: StackProcessor<'In,'T> ->
    f1: StackProcessor<'T,'U> ->
    f2: StackProcessor<'T,'V> -> StackProcessor<'In,('U * 'V)>
/// Run two StackProcessors<unit, _> in parallel:
///   • executes each with its own consumer function
///   • waits for both to finish before returning
val runBoth2:
  p1: StackProcessor<unit,'T1> ->
    f1: (StackProcessor<unit,'T1> -> Async<unit>) ->
    p2: StackProcessor<unit,'T2> ->
    f2: (StackProcessor<unit,'T2> -> Async<unit>) -> unit
val sinkLst: processors: StackProcessor<unit,unit> list -> unit
val sink: p: StackProcessor<unit,unit> -> unit
val sourceLst:
  availableMemory: uint64 ->
    width: uint ->
    height: uint ->
    depth: uint ->
    processors: StackProcessor<unit,Slice.Slice<'T>> list ->
    StackProcessor<unit,Slice.Slice<'T>> list when 'T: equality
val source:
  availableMemory: uint64 ->
    width: uint ->
    height: uint ->
    depth: uint ->
    p: StackProcessor<unit,Slice.Slice<'T>> ->
    StackProcessor<unit,Slice.Slice<'T>> when 'T: equality
