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
    | Constant
    | Streaming
    | Window of uint * uint * uint * uint * uint
module Profile =
    val estimateUsage: profile: Profile -> memPerElement: uint64 -> uint64
    val combine: prof1: Profile -> prof2: Profile -> Profile
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
    val liftRelease:
      name: string ->
        profile: Profile ->
        release: ('S -> unit) -> f: ('S -> 'T) -> Pipe<'S,'T>
    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    val compose: p1: Pipe<'S,'T> -> p2: Pipe<'T,'U> -> Pipe<'S,'U>
    val skip: name: string -> count: uint -> Pipe<'a,'a>
    val take: name: string -> count: uint -> Pipe<'a,'a>
    val map:
      name: string -> mapper: ('U -> 'V) -> pipe: Pipe<'In,'U> -> Pipe<'In,'V>
    /// Prepend a sequence produced by a Pipe<unit,'S> to the input stream.
    val prepend: name: string -> pre: Pipe<unit,'S> -> Pipe<'S,'S>
    /// Append a sequence produced by a Pipe<unit,'S> to the input stream.
    val append: name: string -> post: Pipe<unit,'S> -> Pipe<'S,'S>
    type TeeMsg<'T> =
        | Left of AsyncReplyChannel<'T option>
        | Right of AsyncReplyChannel<'T option>
    val tee: debug: bool -> p: Pipe<'In,'T> -> Pipe<'In,'T> * Pipe<'In,'T>
    val id: name: string -> Pipe<'T,'T>
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
    val collect: name: string -> mapper: ('S -> 'T list) -> Pipe<'S,'T>
    val flatten: name: string -> Pipe<'T list,'T>
    val ignore: clean: ('T -> unit) -> Pipe<'T,unit>
    val ignorePairs:
      cleanFst: ('S -> unit) * cleanSnd: ('T -> unit) -> Pipe<('S * 'T),unit>
type MemoryNeed = uint64 -> uint64
type MemoryNeedWrapped = SingleOrPair -> SingleOrPair
type LengthTransformation = uint64 -> uint64
/// Stage describes *what* should be done:
type Stage<'S,'T> =
    {
      Name: string
      Build: (unit -> Pipe<'S,'T>)
      Transition: ProfileTransition
      MemoryNeed: MemoryNeedWrapped
      LengthTransformation: LengthTransformation
    }
module Stage =
    val create:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
    val createWrapped:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        wrapMemoryNeed: MemoryNeedWrapped ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
    val empty: name: string -> Stage<unit,unit>
    val init<'S,'T> :
      name: string ->
        depth: uint ->
        mapper: (int -> 'T) ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<unit,'T>
    val compose: stage1: Stage<'S,'T> -> stage2: Stage<'T,'U> -> Stage<'S,'U>
    val (-->) : (Stage<'a,'b> -> Stage<'b,'c> -> Stage<'a,'c>)
    val prepend: name: string -> pre: Stage<unit,'S> -> Stage<'S,'S>
    val append: name: string -> app: Stage<unit,'S> -> Stage<'S,'S>
    val idOp: name: string -> Stage<'T,'T>
    val toPipe: stage: Stage<'a,'b> -> (unit -> Pipe<'a,'b>)
    val fromPipe:
      name: string ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation ->
        pipe: Pipe<'S,'T> -> Stage<'S,'T>
    val skip: name: string -> n: uint -> Stage<'S,'S>
    val take: name: string -> n: uint -> Stage<'S,'S>
    val map:
      name: string ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
    val map1:
      name: string ->
        f: ('U -> 'V) ->
        stage: Stage<'In,'U> ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'In,'V>
    val map2:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U> ->
        stage2: Stage<'In,'V> ->
        memoryNeed: MemoryNeedWrapped ->
        lengthTransformation: LengthTransformation -> Stage<'In,'W>
    val map2Sync:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U> ->
        stage2: Stage<'In,'V> ->
        memoryNeed: MemoryNeedWrapped ->
        lengthTransformation: LengthTransformation -> Stage<'In,'W>
    val reduce:
      name: string ->
        reducer: (bool -> FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: Profile ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'In,'Out>
    val fold<'S,'T> :
      name: string ->
        folder: ('T -> 'S -> 'T) ->
        initial: 'T ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
    val window:
      name: string ->
        winSz: uint ->
        pad: uint ->
        zeroMaker: (int -> 'T -> 'T) -> stride: uint -> Stage<'T,'T list>
        when 'T: equality
    val flatten: name: string -> Stage<'T list,'T>
    val liftUnary:
      name: string ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
    val liftReleaseUnary:
      name: string ->
        release: ('S -> unit) ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
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
type Plan<'S,'T> =
    {
      stage: Stage<'S,'T> option
      nElems: SingleOrPair
      length: uint64
      memAvail: uint64
      memPeak: uint64
      debug: bool
    }
module Plan =
    val create:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElems: uint64 -> length: uint64 -> debug: bool -> Plan<'S,'T>
        when 'T: equality
    val createWrapped:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElems: SingleOrPair -> length: uint64 -> debug: bool -> Plan<'S,'T>
        when 'T: equality
    /// Source type operators
    val source: availableMemory: uint64 -> Plan<unit,unit>
    val debug: availableMemory: uint64 -> Plan<unit,unit>
    /// Composition operators
    val composeOp:
      name: string -> pl: Plan<'a,'b> -> stage: Stage<'b,'c> -> Plan<'a,'c>
        when 'c: equality
    val (>=>) :
      pl: Plan<'a,'b> -> stage: Stage<'b,'c> -> Plan<'a,'c> when 'c: equality
    val map:
      name: string -> f: ('U -> 'V) -> pl: Plan<'In,'U> -> Plan<'In,'V>
        when 'V: equality
    /// parallel execution of non-synchronised streams
    val internal zipOp:
      name: string ->
        pl1: Plan<'In,'U> -> pl2: Plan<'In,'V> -> Plan<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    /// parallel execution of non-synchronised streams
    val zip:
      pl1: Plan<'In,'U> -> pl2: Plan<'In,'V> -> Plan<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    /// parallel execution of synchronised streams
    val (>=>>) :
      pl: Plan<'In,'S> ->
        stg1: Stage<'S,'U> * stg2: Stage<'S,'V> -> Plan<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    val (>>=>) :
      pl: Plan<'In,('U * 'V)> -> f: ('U -> 'V -> 'W) -> Plan<'In,'W>
        when 'W: equality
    val (>>=>>) :
      f: ('U * 'V -> 'S * 'T) ->
        pl: Plan<'In,('U * 'V)> ->
        stage: Stage<('U * 'V),('S * 'T)> -> Plan<'In,('S * 'T)>
        when 'S: equality and 'T: equality
    /// sink type operators
    val sink: pl: Plan<unit,unit> -> unit
    val sinkList: plans: Plan<unit,unit> list -> unit
    val internal runToScalar:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'T> -> Async<'R>) ->
        pl: Plan<unit,'T> -> 'R
    val drainSingle: name: string -> pl: Plan<unit,'T> -> 'T
    val drainList: name: string -> pl: Plan<unit,'T> -> 'T list
    val drainLast: name: string -> pl: Plan<unit,'T> -> 'T
