namespace FSharp
module SlimPipeline
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant
    | Streaming
    | Sliding of uint * uint * uint * uint
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
module private Pipe =
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
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: MemoryProfile -> Pipe<'In,'Out>
    val fold:
      name: string ->
        folder: ('State -> 'In -> 'State) ->
        initial: 'State -> profile: MemoryProfile -> Pipe<'In,'State>
    val mapNFold:
      name: string ->
        mapFn: ('In -> 'Mapped) ->
        folder: ('State -> 'Mapped -> 'State) ->
        state: 'State -> profile: MemoryProfile -> Pipe<'In,'State>
    val consumeWith:
      name: string ->
        consume: ('T -> unit) -> profile: MemoryProfile -> Pipe<'T,unit>
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
      name: string ->
        depth: uint ->
        updateId: (uint -> 'S -> 'S) ->
        pad: uint ->
        zeroMaker: ('S -> 'S) ->
        stride: uint ->
        emitStart: uint ->
        emitCount: uint -> f: ('S list -> 'T list) -> Pipe<'S,'T>
        when 'S: equality
    val ignore: unit -> Pipe<'T,unit>
/// MemoryTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
type MemoryTransition =
    {
      From: MemoryProfile
      To: MemoryProfile
    }
module MemoryTransition =
    val create:
      fromProfile: MemoryProfile -> toProfile: MemoryProfile -> MemoryTransition
/// Stage describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Stage + MemoryTransition: what happens
type Stage<'S,'T,'ShapeS,'ShapeT> =
    {
      Name: string
      Pipe: Pipe<'S,'T>
      Transition: MemoryTransition
      ShapeUpdate: ('ShapeS option -> 'ShapeT option)
    }
module Stage =
    val create:
      name: string ->
        pipe: Pipe<'S,'T> ->
        transition: MemoryTransition ->
        shapeUpdate: ('ShapeS option -> 'ShapeT option) ->
        Stage<'S,'T,'ShapeS,'ShapeT>
    val init<'S,'T,'ShapeS,'ShapeT> :
      name: string ->
        depth: uint ->
        mapper: (uint -> 'T) ->
        transition: MemoryTransition ->
        shapeUpdate: ('ShapeS option -> 'ShapeT option) ->
        Stage<unit,'T,'ShapeS,'ShapeT>
    val toPipe: stage: Stage<'a,'b,'c,'d> -> Pipe<'a,'b>
    val fromPipe:
      name: string ->
        transition: MemoryTransition ->
        shapeUpdate: ('ShapeA option -> 'ShapeB option) ->
        pipe: Pipe<'S,'T> -> Stage<'S,'T,'ShapeA,'ShapeB>
    val compose:
      stage1: Stage<'S,'T,'ShapeA,'ShapeB> ->
        stage2: Stage<'T,'U,'ShapeB,'ShapeC> -> Stage<'S,'U,'ShapeA,'ShapeC>
    val (-->) : (Stage<'a,'b,'c,'d> -> Stage<'b,'e,'d,'f> -> Stage<'a,'e,'c,'f>)
    val map:
      name: string ->
        f: ('S -> 'T) ->
        shapeUpdate: ('ShapeS option -> 'ShapeT option) ->
        Stage<'S,'T,'ShapeS,'ShapeT>
    val map2:
      name: string ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U,'ShapeIn,'ShapeU> *
        stage2: Stage<'In,'V,'ShapeIn,'ShapeV> ->
          shapeUpdate: ('ShapeIn option -> 'ShapeW option) ->
          Stage<'In,'W,'ShapeIn,'ShapeW>
    val reduce:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: MemoryProfile ->
        shapeUpdate: ('ShapeIn option -> 'ShapeOut option) ->
        Stage<'In,'Out,'ShapeIn,'ShapeOut>
    val fold<'S,'T,'ShapeS,'ShapeT> :
      name: string ->
        folder: ('T -> 'S -> 'T) ->
        initial: 'T ->
        shapeUpdate: ('ShapeS option -> 'ShapeT option) ->
        Stage<'S,'T,'ShapeS,'ShapeT>
    val mapNFold:
      name: string ->
        mapFn: ('In -> 'Mapped) ->
        folder: ('State -> 'Mapped -> 'State) ->
        state: 'State ->
        profile: MemoryProfile ->
        shapeUpdate: ('ShapeIn option -> 'ShapeState option) ->
        Stage<'In,'State,'ShapeIn,'ShapeState>
    val liftUnary:
      name: string ->
        f: ('S -> 'T) ->
        shapeUpdate: ('ShapeS option -> 'ShapeT option) ->
        Stage<'S,'T,'ShapeS,'ShapeT>
    val liftWindowed:
      name: string ->
        updateId: (uint -> 'S -> 'S) ->
        window: uint ->
        pad: uint ->
        zeroMaker: ('S -> 'S) ->
        stride: uint ->
        emitStart: uint ->
        emitCount: uint ->
        f: ('S list -> 'T list) ->
        shapeUpdate: ('ShapeS option -> 'ShapeT option) ->
        Stage<'S,'T,'ShapeS,'ShapeT> when 'S: equality and 'T: equality
    val tap: name: string -> Stage<'T,'T,'ShapeT,'ShapeT>
    val tapIt: toString: ('T -> string) -> Stage<'T,'T,'ShapeT,'ShapeT>
    val ignore: unit -> Stage<'T,unit,'Shape,'Shape>
    val consumeWith:
      name: string -> consume: ('T -> unit) -> Stage<'T,unit,'ShapeT,'ShapeT>
    val cast:
      name: string -> f: ('S -> 'T) -> Stage<'S,'T,'ShapeS,'ShapeS>
        when 'S: equality and 'T: equality
    val promoteConstantToStreaming:
      name: string -> depth: uint -> value: 'T -> Stage<unit,'T,'Shape,'Shape>
    val promoteStreamingToSliding:
      name: string ->
        depth: uint ->
        updateId: (uint -> 'T -> 'T) ->
        pad: uint ->
        zeroMaker: ('T -> 'T) ->
        stride: uint ->
        emitStart: uint -> emitCount: uint -> Stage<'T,'T,'Shape,'Shape>
        when 'T: equality
    val promoteSlidingToSliding:
      name: string ->
        depth: uint ->
        updateId: (uint -> 'T -> 'T) ->
        pad: uint ->
        zeroMaker: ('T -> 'T) ->
        stride: uint ->
        emitStart: uint -> emitCount: uint -> Stage<'T,'T,'Shape,'Shape>
        when 'T: equality
type ShapeContext<'S> =
    {
      memPerElement: ('S -> uint64)
      depth: ('S -> uint)
    }
module ShapeContext =
    val create:
      memPerElement: ('S -> uint64) -> depth: ('S -> uint) -> ShapeContext<'S>
type MemFlow<'S,'T,'ShapeS,'ShapeT> =
    uint64 ->
      'ShapeS option ->
      ShapeContext<'ShapeS> ->
      Stage<'S,'T,'ShapeS,'ShapeT> * uint64 * 'ShapeT option
module MemFlow =
    val returnM:
      stage: Stage<'S,'T,'ShapeS,'ShapeT> ->
        bytes: uint64 ->
        shape: 'ShapeS option ->
        shapeContext: ShapeContext<'ShapeS> ->
        Stage<'S,'T,'ShapeS,'ShapeT> * uint64 * 'ShapeT option
    val bindM:
      k: (Stage<'A,'B,'ShapeA,'ShapeB> ->
            uint64 ->
            'ShapeB option ->
            ShapeContext<'ShapeB> ->
            Stage<'B,'C,'ShapeB,'ShapeC> * uint64 * 'ShapeC option) ->
        flowAB: MemFlow<'A,'B,'ShapeA,'ShapeB> ->
        bytes: uint64 ->
        shape: 'ShapeA option ->
        shapeContextA: ShapeContext<'ShapeA> ->
        Stage<'A,'C,'ShapeA,'ShapeC> * uint64 * 'ShapeC option
type Pipeline<'S,'T,'ShapeS,'ShapeT> =
    {
      flow: MemFlow<'S,'T,'ShapeS,'ShapeT>
      mem: uint64
      shape: 'ShapeS option
      context: ShapeContext<'ShapeS>
      debug: bool
    }
module Pipeline =
    val create:
      flow: MemFlow<'S,'T,'ShapeS,'ShapeT> ->
        mem: uint64 ->
        shape: 'ShapeS option ->
        context: ShapeContext<'ShapeS> ->
        debug: bool -> Pipeline<'S,'T,'ShapeS,'ShapeT> when 'T: equality
    val asStage:
      pl: Pipeline<'In,'Out,'ShapeIn,'ShapeOut> ->
        Stage<'In,'Out,'ShapeIn,'ShapeOut>
    /// Source type operators
    val source:
      context: ShapeContext<'Shape> ->
        availableMemory: uint64 -> Pipeline<unit,unit,'Shape,'Shape>
    val debug:
      context: ShapeContext<'Shape> ->
        availableMemory: uint64 -> Pipeline<unit,unit,'Shape,'Shape>
    /// Composition operators
    val compose:
      pl: Pipeline<'a,'b,'Shapea,'Shapeb> ->
        stage: Stage<'b,'c,'Shapeb,'Shapec> -> Pipeline<'a,'c,'Shapea,'Shapec>
        when 'c: equality
    val (>=>) :
      (Pipeline<'a,'b,'c,'d> -> Stage<'b,'e,'d,'f> -> Pipeline<'a,'e,'c,'f>)
        when 'e: equality
    /// parallel fanout with synchronization
    val (>=>>) :
      pl: Pipeline<'In,'S,'ShapeIn,'ShapeS> ->
        stg1: Stage<'S,'U,'ShapeS,'ShapeU> * stg2: Stage<'S,'V,'ShapeS,'ShapeV> ->
          Pipeline<'In,('U * 'V),'ShapeIn,('ShapeU * 'ShapeV)>
        when 'U: equality and 'V: equality
    val (>>=>) :
      f: ('a -> 'b -> 'c) ->
        pl: Pipeline<'d,'e,'f,'g> ->
        Stage<'e,'a,'g,'h> * Stage<'e,'b,'g,'i> ->
          shapeUpdate: ('g option -> 'j option) -> Pipeline<'d,'c,'f,'j>
        when 'c: equality
    /// sink type operators
    val sink: pl: Pipeline<unit,unit,'Shape,'Shape> -> unit
    val sinkList: pipelines: Pipeline<unit,unit,'Shape,'Shape> list -> unit
    val internal runToScalar:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'T> -> Async<'R>) ->
        pl: Pipeline<'In,'T,'ShapeIn,'ShapeT> -> 'R
    val drainSingle: name: string -> pl: Pipeline<'S,'T,'ShapeS,'ShapeT> -> 'T
    val drainList:
      name: string -> pl: Pipeline<'S,'T,'ShapeS,'ShapeT> -> 'T list
    val drainLast: name: string -> pl: Pipeline<'S,'T,'ShapeS,'ShapeT> -> 'T
