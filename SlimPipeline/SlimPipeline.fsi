namespace FSharp
module SlimPipeline
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant
    | Streaming
    | Sliding of uint * uint * uint * uint
    | Full
module MemoryProfile =
    val estimateUsage:
      profile: MemoryProfile -> memPerElement: uint64 -> depth: uint -> uint64
    val requiresBuffering:
      profile: MemoryProfile ->
        availableMemory: uint64 -> memPerElement: uint64 -> depth: uint -> bool
    val combine: prof1: MemoryProfile -> prof2: MemoryProfile -> MemoryProfile
/// A configurable image processing step that operates on image slices.
/// Pipe describes *how* to do it:
/// - Encapsulates the concrete execution logic
/// - Defines memory usage behavior
/// - Takes and returns AsyncSeq streams
/// - Pipe + WindowedProcessor: How it’s computed 
type Pipe<'S,'T> =
    {
      Name: string
      Apply: (FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>)
      Profile: MemoryProfile
    }
module Pipe =
    val create:
      name: string ->
        apply: (FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>) ->
        profile: MemoryProfile -> Pipe<'S,'T>
    val runWith: input: 'S -> pipe: Pipe<'S,'T> -> Async<unit>
    val run: pipe: Pipe<unit,unit> -> unit
    val lift:
      name: string -> profile: MemoryProfile -> f: ('S -> 'T) -> Pipe<'S,'T>
    val init:
      name: string ->
        depth: uint ->
        mapper: (uint -> 'T) -> profile: MemoryProfile -> Pipe<unit,'T>
    val map: name: string -> f: ('U -> 'V) -> pipe: Pipe<'In,'U> -> Pipe<'In,'V>
    val map2:
      name: string ->
        f: ('U -> 'V -> 'W) ->
        pipe1: Pipe<'In,'U> -> pipe2: Pipe<'In,'V> -> Pipe<'In,'W>
    val reduce:
      label: string ->
        reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: MemoryProfile -> Pipe<'In,'Out>
    val consumeWith:
      name: string ->
        consume: (FSharp.Control.AsyncSeq<'T> -> Async<unit>) ->
        profile: MemoryProfile -> Pipe<'T,unit>
    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    val compose: p1: Pipe<'S,'T> -> p2: Pipe<'T,'U> -> Pipe<'S,'U>
    val memNeed:
      memPerElement: uint64 -> depth: uint -> p: Pipe<'a,'b> -> uint64
    /// Try to shrink a too‑hungry pipe to a cheaper profile.
    /// *You* control the downgrade policy here.
    val shrinkProfile:
      avail: uint64 ->
        memPerElement: uint64 -> depth: uint -> p: Pipe<'S,'T> -> Pipe<'S,'T>
    /// mapWindowed keeps a running window along the slice direction of depth images
    /// and processes them by f. The stepping size of the running window is stride.
    /// So if depth is 3 and stride is 1 then first image 0,1,2 is sent to f, then 1, 2, 3
    /// and so on. If depth is 3 and stride is 3, then it'll be image 0, 1, 2 followed by
    /// 3, 4, 5. It is also possible to use this for sampling, e.g., setting depth to 1
    /// and stride to 2 sends every second image to f.  
    val mapWindowed:
      label: string ->
        depth: uint ->
        updateId: (uint -> 'S -> 'S) ->
        pad: uint ->
        zeroMaker: ('S -> 'S) ->
        stride: uint ->
        emitStart: uint ->
        emitCount: uint -> f: ('S list -> 'T list) -> Pipe<'S,'T>
        when 'S: equality
    val ignore: unit -> Pipe<'T,unit>
    /// Split a Pipe<'In,'T> into two branches that
    ///   • read the upstream only once
    ///   • keep at most one item in memory
    ///   • terminate correctly when both sides finish
    type private Request<'T> =
        | Left of AsyncReplyChannel<Option<'T>>
        | Right of AsyncReplyChannel<Option<'T>>
    val internal tee: p: Pipe<'In,'T> -> Pipe<'In,'T> * Pipe<'In,'T>
/// MemoryTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
type MemoryTransition =
    {
      From: MemoryProfile
      To: MemoryProfile
    }
/// Stage describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Stage + MemoryTransition: what happens
type Stage<'S,'T,'Shape> =
    {
      Name: string
      Pipe: Pipe<'S,'T>
      Transition: MemoryTransition
      ShapeUpdate: ('Shape -> 'Shape)
    }
module Stage =
    val create:
      name: string ->
        pipe: Pipe<'S,'T> ->
        transition: MemoryTransition ->
        shapeUpdate: ('Shape -> 'Shape) -> Stage<'S,'T,'Shape>
    val init<'S,'T,'Shape> :
      name: string ->
        depth: uint ->
        mapper: (uint -> 'T) ->
        transition: MemoryTransition ->
        shapeUpdate: ('Shape -> 'Shape) -> Stage<unit,'T,'Shape>
    val transition:
      fromProfile: MemoryProfile -> toProfile: MemoryProfile -> MemoryTransition
    val id: unit -> Stage<'T,'T,'Shape>
    val toPipe: op: Stage<'a,'b,'c> -> Pipe<'a,'b>
    val fromPipe:
      name: string ->
        transition: MemoryTransition ->
        shapeUpdate: ('Shape -> 'Shape) ->
        pipe: Pipe<'S,'T> -> Stage<'S,'T,'Shape>
    val compose:
      op1: Stage<'S,'T,'Shape> ->
        op2: Stage<'T,'U,'Shape> -> Stage<'S,'U,'Shape>
    val (-->) : (Stage<'a,'b,'c> -> Stage<'b,'d,'c> -> Stage<'a,'d,'c>)
    val map:
      name: string ->
        f: ('U -> 'V) -> stage: Stage<'In,'U,'Shape> -> Stage<'In,'V,'Shape>
    val map2:
      name: string ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U,'Shape> * stage2: Stage<'In,'V,'Shape> ->
          Stage<'In,'W,'Shape>
    val liftUnary: name: string -> f: ('T -> 'T) -> Stage<'T,'T,'Shape>
    val tap: label: string -> Stage<'T,'T,'Shape>
    val tapIt: toString: ('T -> string) -> Stage<'T,'T,'Shape>
    val internal tee:
      op: Stage<'In,'T,'Shape> -> Stage<'In,'T,'Shape> * Stage<'In,'T,'Shape>
    val ignore: unit -> Stage<'T,unit,'Shape>
type ShapeContext<'S> =
    {
      memPerElement: ('S -> uint64)
      depth: ('S -> uint)
    }
module ShapeContext =
    val create:
      memPerElement: ('S -> uint64) -> depth: ('S -> uint) -> ShapeContext<'S>
type MemFlow<'S,'T,'Shape> =
    uint64 ->
      'Shape option ->
      ShapeContext<'Shape> -> Stage<'S,'T,'Shape> * uint64 * 'Shape option
module MemFlow =
    val returnM:
      op: Stage<'S,'T,'Shape> ->
        bytes: uint64 ->
        shape: 'Shape option ->
        shapeContext: ShapeContext<'Shape> ->
        Stage<'S,'T,'Shape> * uint64 * 'Shape option
    val bindM:
      m: MemFlow<'A,'B,'Shape> ->
        k: (Stage<'A,'B,'Shape> ->
              uint64 ->
              'Shape option ->
              ShapeContext<'Shape> ->
              Stage<'B,'C,'Shape> * uint64 * 'Shape option) ->
        bytes: uint64 ->
        shape: 'Shape option ->
        shapeContext: ShapeContext<'Shape> ->
        Stage<'A,'C,'Shape> * uint64 * 'Shape option
type Pipeline<'S,'T,'Shape> =
    {
      flow: MemFlow<'S,'T,'Shape>
      mem: uint64
      shape: 'Shape option
      context: ShapeContext<'Shape>
      debug: bool
    }
module Pipeline =
    val create:
      flow: MemFlow<'S,'T,'Shape> ->
        mem: uint64 ->
        shape: 'Shape option ->
        context: ShapeContext<'Shape> -> debug: bool -> Pipeline<'S,'T,'Shape>
        when 'T: equality
    val source:
      context: ShapeContext<'Shape> ->
        availableMemory: uint64 -> Pipeline<unit,unit,'Shape>
    val debug:
      context: ShapeContext<'Shape> ->
        availableMemory: uint64 -> Pipeline<unit,unit,'Shape>
    val compose:
      pl: Pipeline<'a,'b,'Shape> ->
        next: Stage<'b,'c,'Shape> -> Pipeline<'a,'c,'Shape> when 'c: equality
    val (>=>) :
      (Pipeline<'a,'b,'c> -> Stage<'b,'d,'c> -> Pipeline<'a,'d,'c>)
        when 'd: equality
    val sink: pl: Pipeline<unit,unit,'Shape> -> unit
    val sinkList: pipelines: Pipeline<unit,unit,'Shape> list -> unit
    val tee:
      pl: Pipeline<'In,'T,'Shape> ->
        Pipeline<'In,'T,'Shape> * Pipeline<'In,'T,'Shape> when 'T: equality
    val asStage: pl: Pipeline<'In,'Out,'Shape> -> Stage<'In,'Out,'Shape>
/// Represents a pipeline that has been shared (split into synchronized branches)
type SharedPipeline<'T,'U,'V,'Shape> =
    {
      flow: MemFlow<'T,'U,'Shape>
      branches: Stage<'T,'U,'Shape> * Stage<'T,'V,'Shape>
      mem: uint64
      shape: 'Shape option
      context: ShapeContext<'Shape>
      debug: bool
    }
module SharedPipeline =
    val create<'T,'U,'V,'Shape when 'T: equality> :
      flow: MemFlow<'T,'U,'Shape> ->
        Stage<'T,'U,'Shape> * Stage<'T,'V,'Shape> ->
          mem: uint64 ->
          shape: 'Shape option ->
          context: ShapeContext<'Shape> ->
          debug: bool -> SharedPipeline<'T,'U,'V,'Shape> when 'T: equality
module Routing =
    val runToScalar:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'T> -> Async<'R>) ->
        pl: Pipeline<'In,'T,'Shape> -> 'R
    val drainSingle: name: string -> pl: Pipeline<'S,'T,'Shape> -> 'T
    val drainList: name: string -> pl: Pipeline<'S,'T,'Shape> -> 'T list
    val drainLast: name: string -> pl: Pipeline<'S,'T,'Shape> -> 'T
    /// parallel fanout with synchronization
    /// Synchronously split the shared stream into two parallel pipelines
    val (>=>>) :
      pl: Pipeline<'In,'T,'Shape> ->
        op1: Stage<'T,'T,'Shape> * op2: Stage<'T,'V,'Shape> ->
          SharedPipeline<'In,'T,'V,'Shape> when 'In: equality
    val (>>=>) :
      shared: SharedPipeline<'In,'U,'V,'Shape> ->
        combineFn: ('U -> 'V -> 'W) -> Pipeline<'In,'W,'Shape> when 'W: equality
