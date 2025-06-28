namespace FSharp
module Core
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant
    | StreamingConstant
    | Streaming
    | SlidingConstant of uint
    | Sliding of uint
    | BufferedConstant
    | Buffered
    member EstimateUsage: width: uint -> height: uint -> depth: uint -> uint64
    member
      RequiresBuffering: availableMemory: uint64 ->
                           width: uint -> height: uint -> depth: uint -> bool
    member combineProfile: other: MemoryProfile -> MemoryProfile
/// A configurable image processing step that operates on image slices.
/// A monadic layered transformer - a monadic morphism, to hide the underlying
/// AsyncSeq<'T> Monad. This can, perhaps, be called a Kleisli category over
/// Pipe<'S,'T> morphisms
type Pipe<'S,'T> =
    {
      Name: string
      Profile: MemoryProfile
      Apply: (FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>)
    }
val internal isScalar: profile: MemoryProfile -> bool
val internal requiresFullInput: profile: MemoryProfile -> bool
val internal lift:
  name: string ->
    profile: MemoryProfile -> f: ('In -> Async<'Out>) -> Pipe<'In,'Out>
val internal reduce:
  name: string ->
    profile: MemoryProfile ->
    reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) -> Pipe<'In,'Out>
val internal consumeWith:
  name: string ->
    profile: MemoryProfile ->
    consume: (FSharp.Control.AsyncSeq<'T> -> Async<unit>) -> Pipe<'T,unit>
/// Pipeline computation expression
type PipelineBuilder =
    new: availableMemory: uint64 * width: uint * height: uint * depth: uint ->
           PipelineBuilder
    member Bind: p: Pipe<'S,'T> * f: ('T -> Pipe<'S,'U>) -> Pipe<'S,'U>
    /// Chain two <c>Pipe</c> instances, optionally inserting intermediate disk I/O
    member Bind: p: Pipe<'S,'T> * f: (Pipe<'S,'T> -> Pipe<'S,'T>) -> Pipe<'S,'T>
    /// Wraps a processor value for use in the pipeline computation expression.
    member Return: p: Pipe<'S,'T> -> Pipe<'S,'T>
    /// Allows returning a processor directly from another computation expression.
    member ReturnFrom: p: Pipe<'S,'T> -> Pipe<'S,'T>
    /// Provides a default identity processor using streaming as the memory profile.
    member Zero: unit -> Pipe<'a,'a>
/// A memory-aware pipeline builder with the specified processing constraints.
val pipeline:
  availableMemory: uint64 ->
    width: uint -> height: uint -> depth: uint -> PipelineBuilder
module Helpers =
    /// Pipeline helper functions
    val singleton: x: 'In -> Pipe<'In,'In>
module Routing
val private run: p: Core.Pipe<unit,'T> -> FSharp.Control.AsyncSeq<'T>
val sinkLst: processors: Core.Pipe<unit,unit> list -> unit
val sink: p: Core.Pipe<unit,unit> -> unit
val sourceLst:
  availableMemory: uint64 ->
    width: uint ->
    height: uint ->
    depth: uint ->
    processors: Core.Pipe<unit,'T> list -> Core.Pipe<unit,'T> list
val source:
  availableMemory: uint64 ->
    width: uint ->
    height: uint -> depth: uint -> p: Core.Pipe<unit,'T> -> Core.Pipe<unit,'T>
/// Split a Pipe<'In,'T> into two branches that
///   • read the upstream only once
///   • keep at most one item in memory
///   • terminate correctly when both sides finish
type private Request<'T> =
    | Left of AsyncReplyChannel<Option<'T>>
    | Right of AsyncReplyChannel<Option<'T>>
val tee: p: Core.Pipe<'In,'T> -> Core.Pipe<'In,'T> * Core.Pipe<'In,'T>
/// zipWith two Pipes<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
val zipWith:
  f: ('A -> 'B -> 'C) ->
    p1: Core.Pipe<'In,'A> -> p2: Core.Pipe<'In,'B> -> Core.Pipe<'In,'C>
val cacheScalar: name: string -> p: Core.Pipe<unit,'T> -> Core.Pipe<'In,'T>
/// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
val composePipe:
  p1: Core.Pipe<'S,'T> -> p2: Core.Pipe<'T,'U> -> Core.Pipe<'S,'U>
val (>=>) : p1: Core.Pipe<'a,'b> -> p2: Core.Pipe<'b,'c> -> Core.Pipe<'a,'c>
val tap: label: string -> Core.Pipe<'T,'T>
module StackPipeline
module internal InternalHelpers =
    val plotListAsync:
      plt: (float list -> float list -> unit) ->
        vectorSeq: FSharp.Control.AsyncSeq<(float * float) list> -> Async<unit>
    val showSliceAsync:
      plt: (Slice.Slice<'T> -> unit) ->
        slices: FSharp.Control.AsyncSeq<Slice.Slice<'T>> -> Async<unit>
        when 'T: equality
    val printAsync: slices: FSharp.Control.AsyncSeq<'T> -> Async<unit>
    val writeSlicesAsync:
      outputDir: string ->
        suffix: string ->
        slices: FSharp.Control.AsyncSeq<Slice.Slice<'T>> -> Async<unit>
        when 'T: equality
    val readSlicesAsync:
      inputDir: string ->
        suffix: string -> FSharp.Control.AsyncSeq<Slice.Slice<'T>>
        when 'T: equality
/// Source parts
val create:
  width: uint -> height: uint -> depth: uint -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readSlices:
  inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readSliceN:
  idx: uint ->
    inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readRandomSlices:
  count: uint ->
    inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
/// Sink parts
val print<'T> : Core.Pipe<'T,unit>
val plot:
  plt: (float list -> float list -> unit) ->
    Core.Pipe<(float * float) list,unit>
val show:
  plt: (Slice.Slice<'a> -> unit) -> Core.Pipe<Slice.Slice<'a>,unit>
    when 'a: equality
val writeSlices:
  path: string -> suffix: string -> Core.Pipe<Slice.Slice<'a>,unit>
    when 'a: equality
val ignore<'T> : Core.Pipe<'T,unit>
