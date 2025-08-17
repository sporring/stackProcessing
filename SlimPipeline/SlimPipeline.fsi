namespace FSharp
module SlimPipeline
type SingleOrPair =
    | Single of uint64
    | Pair of uint64 * uint64
module SingleOrPair =
    val isSingle: v: SingleOrPair -> bool
    val map: f: (uint64 -> uint64) -> v: SingleOrPair -> SingleOrPair
    val mapPair:
      f: (uint64 -> uint64) * g: (uint64 -> uint64) ->
        v: SingleOrPair -> SingleOrPair
    val fst: v: SingleOrPair -> uint64
    val snd: v: SingleOrPair -> uint64
    val sum: v: SingleOrPair -> uint64
    val add: v: SingleOrPair -> w: SingleOrPair -> SingleOrPair
/// The memory usage strategies during image processing.
type Profile =
    | Unit
    | Streaming
    | Sliding of uint * uint * uint * uint * uint
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
      Apply:
        (bool -> FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>)
      Profile: Profile
    }
module private Pipe =
    val create:
      name: string ->
        apply: (bool ->
                  FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>) ->
        profile: Profile -> Pipe<'S,'T>
    val runWith: debug: bool -> input: 'S -> pipe: Pipe<'S,'T> -> Async<unit>
    val run: debug: bool -> pipe: Pipe<unit,unit> -> unit
    val lift: name: string -> profile: Profile -> f: ('S -> 'T) -> Pipe<'S,'T>
    val empty: name: string -> Pipe<unit,unit>
    val init:
      name: string ->
        depth: uint -> mapper: (int -> 'T) -> profile: Profile -> Pipe<unit,'T>
    val liftConsume:
      name: string ->
        profile: Profile ->
        release: ('S -> unit) -> f: ('S -> 'T) -> Pipe<'S,'T>
    val skip: name: string -> count: uint -> Pipe<'a,'a>
    val take: name: string -> count: uint -> Pipe<'a,'a>
    val map:
      name: string -> mapper: ('U -> 'V) -> pipe: Pipe<'In,'U> -> Pipe<'In,'V>
    type TeeMsg<'T> =
        | Left of AsyncReplyChannel<'T option>
        | Right of AsyncReplyChannel<'T option>
    val tee: debug: bool -> p: Pipe<'In,'T> -> Pipe<'In,'T> * Pipe<'In,'T>
    val id: name: string -> Pipe<'T,'T>
    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    val compose: p1: Pipe<'S,'T> -> p2: Pipe<'T,'U> -> Pipe<'S,'U>
    val map2Sync:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        pipe1: Pipe<'In,'U> -> pipe2: Pipe<'In,'V> -> Pipe<'In,'W>
    val map2:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        pipe1: Pipe<'In,'U> -> pipe2: Pipe<'In,'V> -> Pipe<'In,'W>
    val reduce:
      name: string ->
        reducer: (bool -> FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: Profile -> Pipe<'In,'Out>
    val fold:
      name: string ->
        folder: ('State -> 'In -> 'State) ->
        initial: 'State -> profile: Profile -> Pipe<'In,'State>
    val consumeWith:
      name: string ->
        consume: (bool -> int -> 'T -> unit) ->
        profile: Profile -> Pipe<'T,unit>
    val window:
      name: string ->
        winSz: uint ->
        pad: uint ->
        zeroMaker: (int -> 'T -> 'T) -> stride: uint -> Pipe<'T,'T list>
        when 'T: equality
    val collect: name: string -> Pipe<'T list,'T>
    val ignore: clean: ('T -> unit) -> Pipe<'T,unit>
    val ignorePairs:
      cleanFst: ('S -> unit) * cleanSnd: ('T -> unit) -> Pipe<('S * 'T),unit>
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
type MemoryNeed = uint64 -> uint64
type MemoryNeedWrapped = SingleOrPair -> SingleOrPair
type NElemsTransformation = uint64 -> uint64
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
      MemoryNeed: MemoryNeedWrapped
      NElemsTransformation: NElemsTransformation
    }
module Stage =
    val create:
      name: string ->
        pipe: Pipe<'S,'T> ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<'S,'T>
    val createWrapped:
      name: string ->
        pipe: Pipe<'S,'T> ->
        transition: ProfileTransition ->
        wrapMemoryNeed: MemoryNeedWrapped ->
        nElemsTransformation: NElemsTransformation -> Stage<'S,'T>
    val empty: name: string -> Stage<unit,unit>
    val init<'S,'T> :
      name: string ->
        depth: uint ->
        mapper: (int -> 'T) ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<unit,'T>
    val idOp: name: string -> Stage<'T,'T>
    val toPipe: stage: Stage<'a,'b> -> Pipe<'a,'b>
    val fromPipe:
      name: string ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation ->
        pipe: Pipe<'S,'T> -> Stage<'S,'T>
    val compose: stage1: Stage<'S,'T> -> stage2: Stage<'T,'U> -> Stage<'S,'U>
    val (-->) : (Stage<'a,'b> -> Stage<'b,'c> -> Stage<'a,'c>)
    val skip: name: string -> n: uint -> Stage<'S,'S>
    val take: name: string -> n: uint -> Stage<'S,'S>
    val map:
      name: string ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<'S,'T>
    val map1:
      name: string ->
        f: ('U -> 'V) ->
        stage: Stage<'In,'U> ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<'In,'V>
    val map2:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U> ->
        stage2: Stage<'In,'V> ->
        memoryNeed: MemoryNeedWrapped ->
        nElemsTransformation: NElemsTransformation -> Stage<'In,'W>
    val map2Sync:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U> ->
        stage2: Stage<'In,'V> ->
        memoryNeed: MemoryNeedWrapped ->
        nElemsTransformation: NElemsTransformation -> Stage<'In,'W>
    val reduce:
      name: string ->
        reducer: (bool -> FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: Profile ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<'In,'Out>
    val fold<'S,'T> :
      name: string ->
        folder: ('T -> 'S -> 'T) ->
        initial: 'T ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<'S,'T>
    val window:
      name: string ->
        winSz: uint ->
        pad: uint ->
        zeroMaker: (int -> 'T -> 'T) -> stride: uint -> Stage<'T,'T list>
        when 'T: equality
    val collect: name: string -> Stage<'T list,'T>
    val liftUnary:
      name: string ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<'S,'T>
    val liftConsumeUnary:
      name: string ->
        release: ('S -> unit) ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        nElemsTransformation: NElemsTransformation -> Stage<'S,'T>
    val tapItOp: name: string -> toString: ('T -> string) -> Stage<'T,'T>
    val tapIt: toString: ('T -> string) -> Stage<'T,'T>
    val tap: name: string -> Stage<'T,'T>
    val ignore: clean: ('T -> unit) -> Stage<'T,unit>
    val ignorePairs:
      cleanFst: ('S -> unit) * cleanSnd: ('T -> unit) -> Stage<('S * 'T),unit>
    val consumeWith:
      name: string ->
        consume: (bool -> int -> 'T -> unit) ->
        memoryNeed: MemoryNeed -> Stage<'T,unit>
    val cast:
      name: string -> f: ('S -> 'T) -> memoryNeed: MemoryNeed -> Stage<'S,'T>
        when 'S: equality and 'T: equality
type Pipeline<'S,'T> =
    {
      stage: Stage<'S,'T> option
      nElems: SingleOrPair
      length: uint64
      memAvail: uint64
      memPeak: uint64
      debug: bool
    }
module Pipeline =
    val create:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElems: uint64 -> length: uint64 -> debug: bool -> Pipeline<'S,'T>
        when 'T: equality
    val createWrapped:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElems: SingleOrPair -> length: uint64 -> debug: bool -> Pipeline<'S,'T>
        when 'T: equality
    /// Source type operators
    val source: availableMemory: uint64 -> Pipeline<unit,unit>
    val debug: availableMemory: uint64 -> Pipeline<unit,unit>
    /// Composition operators
    val composeOp:
      name: string ->
        pl: Pipeline<'a,'b> -> stage: Stage<'b,'c> -> Pipeline<'a,'c>
        when 'c: equality
    val (>=>) :
      pl: Pipeline<'a,'b> -> stage: Stage<'b,'c> -> Pipeline<'a,'c>
        when 'c: equality
    val map:
      name: string -> f: ('U -> 'V) -> pl: Pipeline<'In,'U> -> Pipeline<'In,'V>
        when 'V: equality
    /// parallel execution of non-synchronised streams
    val internal zipOp:
      name: string ->
        pl1: Pipeline<'In,'U> ->
        pl2: Pipeline<'In,'V> -> Pipeline<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    /// parallel execution of non-synchronised streams
    val zip:
      pl1: Pipeline<'In,'U> -> pl2: Pipeline<'In,'V> -> Pipeline<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    /// parallel execution of synchronised streams
    val (>=>>) :
      pl: Pipeline<'In,'S> ->
        stg1: Stage<'S,'U> * stg2: Stage<'S,'V> -> Pipeline<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    val (>>=>) :
      pl: Pipeline<'In,('U * 'V)> -> f: ('U -> 'V -> 'W) -> Pipeline<'In,'W>
        when 'W: equality
    val (>>=>>) :
      f: ('U * 'V -> 'S * 'T) ->
        pl: Pipeline<'In,('U * 'V)> ->
        stage: Stage<('U * 'V),('S * 'T)> -> Pipeline<'In,('S * 'T)>
        when 'S: equality and 'T: equality
    /// sink type operators
    val sink: pl: Pipeline<unit,unit> -> unit
    val sinkList: pipelines: Pipeline<unit,unit> list -> unit
    val internal runToScalar:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'T> -> Async<'R>) ->
        pl: Pipeline<unit,'T> -> 'R
    val drainSingle: name: string -> pl: Pipeline<unit,'T> -> 'T
    val drainList: name: string -> pl: Pipeline<unit,'T> -> 'T list
    val drainLast: name: string -> pl: Pipeline<unit,'T> -> 'T
