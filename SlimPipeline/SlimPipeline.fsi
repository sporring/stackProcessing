namespace FSharp
module SlimPipeline
/// The memory usage strategies during image processing.
type Profile =
    | Constant
    | Streaming
    | Sliding of uint * uint * uint * uint
module Profile =
    val estimateUsage: profile: Profile -> memPerElement: uint64 -> uint64
    val combine: prof1: Profile -> prof2: Profile -> Profile
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
      Profile: Profile
    }
module private Pipe =
    val create:
      name: string ->
        apply: (FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>) ->
        profile: Profile -> Pipe<'S,'T>
    val runWith: input: 'S -> pipe: Pipe<'S,'T> -> Async<unit>
    val run: pipe: Pipe<unit,unit> -> unit
    val lift: name: string -> profile: Profile -> f: ('S -> 'T) -> Pipe<'S,'T>
    val init:
      name: string ->
        depth: uint -> mapper: (uint -> 'T) -> profile: Profile -> Pipe<unit,'T>
    val skip: name: string -> count: uint -> Pipe<'a,'a>
    val take: name: string -> count: uint -> Pipe<'a,'a>
    val map: name: string -> f: ('U -> 'V) -> pipe: Pipe<'In,'U> -> Pipe<'In,'V>
    val map2:
      name: string ->
        f: ('U -> 'V -> 'W) ->
        pipe1: Pipe<'In,'U> -> pipe2: Pipe<'In,'V> -> Pipe<'In,'W>
    val reduce:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: Profile -> Pipe<'In,'Out>
    val fold:
      name: string ->
        folder: ('State -> 'In -> 'State) ->
        initial: 'State -> profile: Profile -> Pipe<'In,'State>
    val mapNFold:
      name: string ->
        mapFn: ('In -> 'Mapped) ->
        folder: ('State -> 'Mapped -> 'State) ->
        state: 'State -> profile: Profile -> Pipe<'In,'State>
    val consumeWith:
      name: string -> consume: ('T -> unit) -> profile: Profile -> Pipe<'T,unit>
    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    val compose: p1: Pipe<'S,'T> -> p2: Pipe<'T,'U> -> Pipe<'S,'U>
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
        winSz: uint ->
        updateId: (uint -> 'S -> 'S) ->
        pad: uint ->
        zeroMaker: ('S -> 'S) ->
        stride: uint ->
        emitStart: uint ->
        emitCount: uint -> f: ('S list -> 'T list) -> Pipe<'S,'T>
        when 'S: equality
    val ignore: unit -> Pipe<'T,unit>
/// ProfileTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
type ProfileTransition =
    {
      From: Profile
      To: Profile
    }
module ProfileTransition =
    val create: fromProfile: Profile -> toProfile: Profile -> ProfileTransition
/// Stage describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Stage + ProfileTransition: what happens
type Stage<'S,'T> =
    {
      Name: string
      Pipe: Pipe<'S,'T>
      Transition: ProfileTransition
      SizeUpdate: (uint64 -> uint64)
    }
module Stage =
    val create:
      name: string ->
        pipe: Pipe<'S,'T> ->
        transition: ProfileTransition ->
        sizeUpdate: (uint64 -> uint64) -> Stage<'S,'T>
    val init<'S,'T> :
      name: string ->
        depth: uint ->
        mapper: (uint -> 'T) ->
        transition: ProfileTransition ->
        sizeUpdate: (uint64 -> uint64) -> Stage<unit,'T>
    val toPipe: stage: Stage<'a,'b> -> Pipe<'a,'b>
    val fromPipe:
      name: string ->
        transition: ProfileTransition ->
        sizeUpdate: (uint64 -> uint64) -> pipe: Pipe<'S,'T> -> Stage<'S,'T>
    val compose: stage1: Stage<'S,'T> -> stage2: Stage<'T,'U> -> Stage<'S,'U>
    val (-->) : (Stage<'a,'b> -> Stage<'b,'c> -> Stage<'a,'c>)
    val skip: name: string -> n: uint -> Stage<'S,'S>
    val take: name: string -> n: uint -> Stage<'S,'S>
    val map:
      name: string ->
        f: ('S -> 'T) -> sizeUpdate: (uint64 -> uint64) -> Stage<'S,'T>
    val map2:
      name: string ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U> * stage2: Stage<'In,'V> ->
          sizeUpdate: (uint64 -> uint64) -> Stage<'In,'W>
    val reduce:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: Profile -> sizeUpdate: (uint64 -> uint64) -> Stage<'In,'Out>
    val fold<'S,'T> :
      name: string ->
        folder: ('T -> 'S -> 'T) ->
        initial: 'T -> sizeUpdate: (uint64 -> uint64) -> Stage<'S,'T>
    val mapNFold:
      name: string ->
        mapFn: ('In -> 'Mapped) ->
        folder: ('State -> 'Mapped -> 'State) ->
        state: 'State ->
        profile: Profile -> sizeUpdate: (uint64 -> uint64) -> Stage<'In,'State>
    val liftUnary:
      name: string ->
        f: ('S -> 'T) -> sizeUpdate: (uint64 -> uint64) -> Stage<'S,'T>
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
        sizeUpdate: (uint64 -> uint64) -> Stage<'S,'T>
        when 'S: equality and 'T: equality
    val tap: name: string -> Stage<'T,'T>
    val tapIt: toString: ('T -> string) -> Stage<'T,'T>
    val ignore: unit -> Stage<'T,unit>
    val consumeWith: name: string -> consume: ('T -> unit) -> Stage<'T,unit>
    val cast:
      name: string -> f: ('S -> 'T) -> Stage<'S,'T>
        when 'S: equality and 'T: equality
    val promoteConstantToStreaming:
      name: string -> depth: uint -> value: 'T -> Stage<unit,'T>
    val promoteStreamingToSliding:
      name: string ->
        winSz: uint ->
        updateId: (uint -> 'T -> 'T) ->
        pad: uint ->
        zeroMaker: ('T -> 'T) ->
        stride: uint -> emitStart: uint -> emitCount: uint -> Stage<'T,'T>
        when 'T: equality
    val promoteSlidingToSliding:
      name: string ->
        winSz: uint ->
        updateId: (uint -> 'T -> 'T) ->
        pad: uint ->
        zeroMaker: ('T -> 'T) ->
        stride: uint -> emitStart: uint -> emitCount: uint -> Stage<'T,'T>
        when 'T: equality
type SizeUpdate = uint64 -> uint64
type Flow<'S,'T> =
    uint64 -> uint64 -> SizeUpdate -> Stage<'S,'T> * uint64 * uint64
module Flow =
    val returnM:
      stage: Stage<'S,'T> ->
        memAvailable: uint64 ->
        sizeS: uint64 ->
        sizeUpdate: SizeUpdate -> Stage<'S,'T> * uint64 * uint64
    val bindM:
      k: (Stage<'A,'B> ->
            uint64 -> uint64 -> SizeUpdate -> Stage<'B,'C> * uint64 * uint64) ->
        flowAB: Flow<'A,'B> ->
        memAvailable: uint64 ->
        shape: uint64 ->
        shapeContextA: SizeUpdate -> Stage<'A,'C> * uint64 * uint64
type Pipeline<'S,'T> =
    {
      flow: Flow<'S,'T>
      sizeUpdate: SizeUpdate
      elmSize: uint64
      nElems: uint64
      mem: uint64
      debug: bool
    }
module Pipeline =
    val create:
      flow: Flow<'S,'T> ->
        mem: uint64 ->
        elmSize: uint64 ->
        nElems: uint64 ->
        sizeUpdate: SizeUpdate -> debug: bool -> Pipeline<'S,'T>
        when 'T: equality
    val asStage: pl: Pipeline<'In,'Out> -> Stage<'In,'Out>
    /// Source type operators
    val source:
      sizeUpdate: SizeUpdate -> availableMemory: uint64 -> Pipeline<unit,unit>
    val debug:
      sizeUpdate: SizeUpdate -> availableMemory: uint64 -> Pipeline<unit,unit>
    /// Composition operators
    val compose:
      pl: Pipeline<'a,'b> -> stage: Stage<'b,'c> -> Pipeline<'a,'c>
        when 'c: equality
    val (>=>) :
      (Pipeline<'a,'b> -> Stage<'b,'c> -> Pipeline<'a,'c>) when 'c: equality
    /// parallel fanout with synchronization
    val (>=>>) :
      pl: Pipeline<'In,'S> ->
        stg1: Stage<'S,'U> * stg2: Stage<'S,'V> -> Pipeline<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    val (>>=>) :
      f: ('a -> 'b -> 'c) ->
        pl: Pipeline<'d,'e> ->
        Stage<'e,'a> * Stage<'e,'b> ->
          sizeUpdate: (uint64 -> uint64) -> Pipeline<'d,'c> when 'c: equality
    /// sink type operators
    val sink: pl: Pipeline<unit,unit> -> unit
    val sinkList: pipelines: Pipeline<unit,unit> list -> unit
    val internal runToScalar:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'T> -> Async<'R>) ->
        pl: Pipeline<'In,'T> -> 'R
    val drainSingle: name: string -> pl: Pipeline<'S,'T> -> 'T
    val drainList: name: string -> pl: Pipeline<'S,'T> -> 'T list
    val drainLast: name: string -> pl: Pipeline<'S,'T> -> 'T
