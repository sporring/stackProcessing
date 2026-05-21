namespace FSharp
module SlimPipeline
type SingleOrPair =
    | Single of uint64
    | Pair of uint64 * uint64
type Window<'T> =
    {
      Items: 'T list
      EmitRange: uint * uint
      ReleaseCount: uint
    }
module Window =
    val create:
      emitStart: uint -> emitCount: uint -> items: 'a list -> Window<'a>
    val createWithRelease:
      emitStart: uint ->
        emitCount: uint -> releaseCount: uint -> items: 'a list -> Window<'a>
    val singleton: item: 'a -> Window<'a>
    val emitItems: window: Window<'a> -> 'a list
module SingleOrPair =
    val isSingle: v: SingleOrPair -> bool
    val map: f: (uint64 -> uint64) -> v: SingleOrPair -> SingleOrPair
    val mapPair:
      f: (uint64 -> uint64) * g: (uint64 -> uint64) ->
        v: SingleOrPair -> SingleOrPair
    val index1: v: SingleOrPair -> SingleOrPair
    val index2: v: SingleOrPair -> SingleOrPair
    val fst: v: SingleOrPair -> uint64
    val snd: v: SingleOrPair -> uint64
    val sum: v: SingleOrPair -> SingleOrPair
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
    type TeeSide =
        | Left
        | Right
    type TeeMsg<'T> = | Next of TeeSide * AsyncReplyChannel<'T option>
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
        zeroMaker: (int -> 'T -> 'T) -> stride: uint -> Pipe<'T,Window<'T>>
    val collect: name: string -> mapper: ('S -> 'T list) -> Pipe<'S,'T>
    val flatten: name: string -> Pipe<'T list,'T>
    val flattenWindow: name: string -> Pipe<Window<'T>,'T>
    val ignore: clean: ('T -> unit) -> Pipe<'T,unit>
    val ignorePairs:
      cleanFst: ('S -> unit) * cleanSnd: ('T -> unit) -> Pipe<('S * 'T),unit>
type MemoryNeed = uint64 -> uint64
type MemoryNeedWrapped = SingleOrPair -> SingleOrPair
type ElementTransformation = uint64 -> uint64
type SliceEnd =
    | RelativeToInputEnd of int64
    | CountFromStart of uint64
    | UnknownEnd
type SliceDomain =
    {
      StartOffset: int64
      End: SliceEnd
    }
type SliceCardinality =
    | Domain of SliceDomain
    | ReduceTo of uint64
    | UnknownCardinality
module SliceDomain =
    val preserve: SliceDomain
    val trim: before: uint -> after: uint -> SliceDomain
    val expand: before: uint -> after: uint -> SliceDomain
    val skip: count: uint -> SliceDomain
    val take: count: uint64 -> SliceDomain
    val private tryUint64: value: int64 -> uint64 option
    val private composeEnd: left: SliceDomain -> right: SliceDomain -> SliceEnd
    val compose: left: SliceDomain -> right: SliceDomain -> SliceDomain
    val private stop: inputLength: uint64 -> domain: SliceDomain -> int64 option
    val length: inputLength: uint64 -> domain: SliceDomain -> uint64 option
module SliceCardinality =
    val preserve: SliceCardinality
    val reduceTo: count: uint64 -> SliceCardinality
    val unknown: SliceCardinality
    val compose:
      left: SliceCardinality -> right: SliceCardinality -> SliceCardinality
    val length:
      inputLength: uint64 -> cardinality: SliceCardinality -> uint64 option
type StageEvaluation =
    | Source
    | Map
    | Iter
    | Windowed of windowSize: uint * stride: uint * pad: uint
    | Flatten
    | Reduce
    | Fanout of branchCount: int
    | Join
    | Sink
    | Custom of string
type StageMemoryEstimate =
    {
      InputLive: uint64
      OutputLive: uint64
      WorkLive: uint64
      RetainedLive: uint64
      Peak: uint64
    }
module StageMemoryEstimate =
    val create:
      inputLive: uint64 ->
        outputLive: uint64 ->
        workLive: uint64 -> retainedLive: uint64 -> StageMemoryEstimate
    val fromPeak: peak: uint64 -> StageMemoryEstimate
type StageMemoryModel =
    {
      Evaluation: StageEvaluation
      Estimate: (SingleOrPair -> StageMemoryEstimate)
    }
module StageMemoryModel =
    val private bytesOf: input: SingleOrPair -> uint64
    val fromPeak:
      evaluation: StageEvaluation ->
        memoryNeed: MemoryNeedWrapped -> StageMemoryModel
    val fromSinglePeak:
      evaluation: StageEvaluation -> memoryNeed: MemoryNeed -> StageMemoryModel
    val mapLike:
      outputBytes: MemoryNeed -> workBytes: MemoryNeed -> StageMemoryModel
    val iterLike: workBytes: MemoryNeed -> StageMemoryModel
    val windowLike: winSz: uint -> stride: uint -> pad: uint -> StageMemoryModel
    val reduceLike:
      accumulatorBytes: MemoryNeed -> workBytes: MemoryNeed -> StageMemoryModel
    val memoryNeed:
      model: StageMemoryModel -> input: SingleOrPair -> SingleOrPair
type StageTimeCostKind =
    | Cpu
    | Native
    | Io
    | Mixed
type StageTimeCostEstimate =
    {
      CpuCostUnits: float
      NativeCostUnits: float
      IoReadBytes: uint64
      IoWriteBytes: uint64
      IoReadOps: uint64
      IoWriteOps: uint64
      CalibrationKey: string option
      Tags: (string * string) list
    }
module StageTimeCostEstimate =
    val zero: StageTimeCostEstimate
    val create:
      cpuCostUnits: float ->
        nativeCostUnits: float ->
        ioReadBytes: uint64 ->
        ioWriteBytes: uint64 ->
        ioReadOps: uint64 ->
        ioWriteOps: uint64 ->
        calibrationKey: string option -> StageTimeCostEstimate
    val withTags:
      tags: (string * string) list ->
        estimate: StageTimeCostEstimate -> StageTimeCostEstimate
    val scale:
      factor: uint64 -> estimate: StageTimeCostEstimate -> StageTimeCostEstimate
    val private isZero: estimate: StageTimeCostEstimate -> bool
    val private hasNoIo: estimate: StageTimeCostEstimate -> bool
    val add:
      left: StageTimeCostEstimate ->
        right: StageTimeCostEstimate -> StageTimeCostEstimate
type StageTimeCostModel =
    {
      Kind: StageTimeCostKind
      Evaluation: StageEvaluation
      Estimate: (SingleOrPair -> StageTimeCostEstimate)
    }
type StageTimeCoefficients =
    {
      CpuMillisecondsPerUnit: float
      NativeMillisecondsPerUnit: float
      IoReadMillisecondsPerByte: float
      IoWriteMillisecondsPerByte: float
      IoReadMillisecondsPerOp: float
      IoWriteMillisecondsPerOp: float
    }
module StageTimeCoefficients =
    val zero: StageTimeCoefficients
    val estimateMilliseconds:
      coefficients: StageTimeCoefficients ->
        estimate: StageTimeCostEstimate -> float
module StageTimeCalibration =
    val mutable private coefficientsByKey: Map<string,StageTimeCoefficients>
    val clear: unit -> unit
    val register: key: string -> coefficients: StageTimeCoefficients -> unit
    val tryFind: key: string -> StageTimeCoefficients option
    val estimateMilliseconds: estimate: StageTimeCostEstimate -> float option
    val private propertyDouble:
      name: string -> element: System.Text.Json.JsonElement -> float
    val loadJson: path: string -> bool
module StageTimeCostModel =
    val private elementCount: input: SingleOrPair -> float
    val zero: evaluation: StageEvaluation -> StageTimeCostModel
    val cpu:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        costUnits: (SingleOrPair -> float) -> StageTimeCostModel
    val native:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        costUnits: (SingleOrPair -> float) -> StageTimeCostModel
    val ioRead:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        bytes: (SingleOrPair -> uint64) ->
        ops: (SingleOrPair -> uint64) -> StageTimeCostModel
    val ioWrite:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        bytes: (SingleOrPair -> uint64) ->
        ops: (SingleOrPair -> uint64) -> StageTimeCostModel
    val fromEvaluation:
      evaluation: StageEvaluation ->
        calibrationKey: string option -> StageTimeCostModel
type StageCostEstimate =
    {
      Memory: StageMemoryEstimate
      Time: StageTimeCostEstimate
    }
module StageCostEstimate =
    val add:
      left: StageCostEstimate -> right: StageCostEstimate -> StageCostEstimate
type StageCostModel =
    {
      Memory: StageMemoryModel
      Time: StageTimeCostModel
    }
module StageCostModel =
    val create:
      memory: StageMemoryModel -> time: StageTimeCostModel -> StageCostModel
    val fromMemory: memory: StageMemoryModel -> StageCostModel
    val estimate:
      model: StageCostModel -> input: SingleOrPair -> StageCostEstimate
    val memoryNeed: model: StageCostModel -> input: SingleOrPair -> SingleOrPair
    val combine:
      evaluation: StageEvaluation ->
        left: StageCostModel -> right: StageCostModel -> StageCostModel
type SourcePeek =
    {
      Name: string
      ElementBytes: uint64
      Length: uint64 option
      Shape: Map<string,string>
    }
module SourcePeek =
    val create:
      name: string ->
        elementBytes: uint64 ->
        length: uint64 option -> shape: Map<string,string> -> SourcePeek
module MemoryProbe =
    type Snapshot =
        {
          Baseline: uint64
          Peak: uint64
          Delta: uint64
        }
    type private Probe =
        {
          Baseline: uint64
          mutable Peak: uint64
          Cancellation: System.Threading.CancellationTokenSource
        }
    val currentRssBytes: unit -> uint64
    val private sample: probe: Probe -> unit
    val private start: sampleIntervalMs: int -> Probe
    val private stop: probe: Probe -> Snapshot
    val measure: sampleIntervalMs: int -> f: (unit -> 'a) -> 'a * Snapshot
    val formatBytes: bytes: uint64 -> string
type ResourceOps<'T> =
    {
      Retain: ('T -> unit)
      Release: ('T -> unit)
      MemoryOf: ('T -> uint64 option)
    }
module ResourceOps =
    val none<'T> : ResourceOps<'T>
    val retainAndReturn: ops: ResourceOps<'T> -> value: 'T -> 'T
    val release: ops: ResourceOps<'T> -> value: 'T -> unit
    val memoryOf: ops: ResourceOps<'T> -> value: 'T -> uint64 option
module DebugLevel =
    val mutable private level: uint32
    val set: value: uint32 -> unit
    val current: unit -> uint32
    val isEnabled: unit -> bool
    val rssEnabled: unit -> bool
type PipelineGraphNode =
    {
      Id: int
      Name: string
      Transition: ProfileTransition
    }
type PipelineGraphEdge =
    {
      From: int
      To: int
      Label: string
    }
type PipelineGraph =
    {
      Nodes: PipelineGraphNode list
      Edges: PipelineGraphEdge list
      Entries: int list
      Exits: int list
    }
module PipelineGraph =
    val private nextNodeId: (unit -> int)
    val empty: PipelineGraph
    val singleton:
      name: string -> transition: ProfileTransition -> PipelineGraph
    val merge: left: PipelineGraph -> right: PipelineGraph -> PipelineGraph
    val connect:
      label: string ->
        left: PipelineGraph -> right: PipelineGraph -> PipelineGraph
    val appendNode:
      label: string ->
        name: string ->
        transition: ProfileTransition -> graph: PipelineGraph -> PipelineGraph
    val compose: left: PipelineGraph -> right: PipelineGraph -> PipelineGraph
    val parallelJoin:
      label: string ->
        left: PipelineGraph -> right: PipelineGraph -> PipelineGraph
/// Stage describes *what* should be done: 
type Stage<'S,'T> =
    {
      Name: string
      Build: (unit -> Pipe<'S,'T>)
      Transition: ProfileTransition
      MemoryNeed: MemoryNeedWrapped
      MemoryModel: StageMemoryModel
      CostModel: StageCostModel
      ElementTransformation: ElementTransformation
      SliceCardinality: SliceCardinality
      Graph: PipelineGraph
      Cleaning: (unit -> unit) list
    }
module Stage =
    val createWithCostModelAndSlice:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        costModel: StageCostModel ->
        elementTransformation: ElementTransformation ->
        sliceCardinality: SliceCardinality ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWithCostModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        costModel: StageCostModel ->
        elementTransformation: ElementTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWithModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryModel: StageMemoryModel ->
        elementTransformation: ElementTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWithModelAndSlice:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryModel: StageMemoryModel ->
        elementTransformation: ElementTransformation ->
        sliceCardinality: SliceCardinality ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val create:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWithSlice:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation ->
        sliceCardinality: SliceCardinality ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrapped:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        wrapMemoryNeed: MemoryNeedWrapped ->
        elementTransformation: ElementTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrappedWithSlice:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        wrapMemoryNeed: MemoryNeedWrapped ->
        elementTransformation: ElementTransformation ->
        sliceCardinality: SliceCardinality ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrappedWithModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryModel: StageMemoryModel ->
        elementTransformation: ElementTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrappedWithCostModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        costModel: StageCostModel ->
        elementTransformation: ElementTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrappedWithCostModelAndSlice:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        costModel: StageCostModel ->
        elementTransformation: ElementTransformation ->
        sliceCardinality: SliceCardinality ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val private withGraph:
      graph: PipelineGraph -> stage: Stage<'a,'b> -> Stage<'a,'b>
    val withSliceCardinality:
      sliceCardinality: SliceCardinality -> stage: Stage<'a,'b> -> Stage<'a,'b>
    val empty: name: string -> Stage<unit,unit>
    val init<'S,'T> :
      name: string ->
        depth: uint ->
        mapper: (int -> 'T) ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<unit,'T>
    val compose: stage1: Stage<'S,'T> -> stage2: Stage<'T,'U> -> Stage<'S,'U>
    val (-->) : (Stage<'a,'b> -> Stage<'b,'c> -> Stage<'a,'c>)
    val prepend: name: string -> pre: Stage<unit,'S> -> Stage<'S,'S>
    val append: name: string -> app: Stage<unit,'S> -> Stage<'S,'S>
    val toPipe: stage: Stage<'a,'b> -> (unit -> Pipe<'a,'b>)
    val fromPipe:
      name: string ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation ->
        pipe: Pipe<'S,'T> -> Stage<'S,'T>
    val skip: name: string -> n: uint -> Stage<'S,'S>
    val take: name: string -> n: uint -> Stage<'S,'S>
    val map:
      name: string ->
        f: (bool -> 'S -> 'T) ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'S,'T>
    val mapi:
      name: string ->
        f: (bool -> int64 -> 'S -> 'T) ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'S,'T>
    val map1:
      name: string ->
        f: ('U -> 'V) ->
        stage: Stage<'In,'U> ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'In,'V>
    val map2:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U> ->
        stage2: Stage<'In,'V> ->
        memoryNeed: MemoryNeedWrapped ->
        elementTransformation: ElementTransformation -> Stage<'In,'W>
    val map2Sync:
      name: string ->
        debug: bool ->
        f: ('U -> 'V -> 'W) ->
        stage1: Stage<'In,'U> ->
        stage2: Stage<'In,'V> ->
        memoryNeed: MemoryNeedWrapped ->
        elementTransformation: ElementTransformation -> Stage<'In,'W>
    val mapPairSync:
      name: string ->
        debug: bool ->
        stage1: Stage<'U,'S> ->
        stage2: Stage<'V,'T> ->
        memoryNeed: MemoryNeedWrapped ->
        elementTransformation: ElementTransformation ->
        Stage<('U * 'V),('S * 'T)>
    val teeFst: stage: Stage<'A,'A> -> Stage<('A * 'B),('A * 'B)>
    val teeSnd: stage: Stage<'B,'B> -> Stage<('A * 'B),('A * 'B)>
    val reduce:
      name: string ->
        reducer: (bool -> FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
        profile: Profile ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'In,'Out>
    val fold<'S,'T> :
      name: string ->
        folder: ('T -> 'S -> 'T) ->
        initial: 'T ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'S,'T>
    val window:
      name: string ->
        winSz: uint ->
        pad: uint ->
        zeroMaker: (int -> 'T -> 'T) -> stride: uint -> Stage<'T,Window<'T>>
    val flatten: name: string -> Stage<'T list,'T>
    val flattenWindow: name: string -> Stage<Window<'T>,'T>
    val liftUnary:
      name: string ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'S,'T>
    val liftReleaseUnary:
      name: string ->
        release: ('S -> unit) ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'S,'T>
    val retainWith: name: string -> ops: ResourceOps<'T> -> Stage<'T,'T>
    val releaseWith: name: string -> ops: ResourceOps<'T> -> Stage<'T,unit>
    val liftResourceUnary:
      name: string ->
        ops: ResourceOps<'S> ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        elementTransformation: ElementTransformation -> Stage<'S,'T>
    val tapItStage: name: string -> toString: ('T -> string) -> Stage<'T,'T>
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
      graph: PipelineGraph
      sourcePeek: SourcePeek option
      costPeak: StageCostEstimate option
      costObservations: StageCostEstimate list
      costTerms: PipelineCostTerm list
      nElemsPerSlice: SingleOrPair
      length: uint64
      memAvail: uint64
      memPeak: uint64
      debug: bool
      debugLevel: uint
      optimize: bool
      costDiscrepancy: bool
    }
and PipelineCostTerm =
    {
      StageName: string
      InputLength: uint64
      OutputLength: uint64
      Multiplicity: uint64
      Memory: StageMemoryEstimate
      Time: StageTimeCostEstimate
    }
module Plan =
    val private graphOfStage: stage: Stage<'a,'b> option -> PipelineGraph
    val private levelOf: debug: bool -> uint32
    val createWithOptimizer:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElemsPerSlice: uint64 ->
        length: uint64 -> debug: bool -> optimize: bool -> Plan<'S,'T>
        when 'T: equality
    val create:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElemsPerSlice: uint64 -> length: uint64 -> debug: bool -> Plan<'S,'T>
        when 'T: equality
    val createWrappedWithOptimizer:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElemsPerSlice: SingleOrPair ->
        length: uint64 -> debug: bool -> optimize: bool -> Plan<'S,'T>
        when 'T: equality
    val createWrapped:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElemsPerSlice: SingleOrPair ->
        length: uint64 -> debug: bool -> Plan<'S,'T> when 'T: equality
    val withSourcePeek: sourcePeek: SourcePeek -> pl: Plan<'S,'T> -> Plan<'S,'T>
    val withCostDiscrepancyReporting:
      enabled: bool -> pl: Plan<'S,'T> -> Plan<'S,'T>
    val withRuntimeOptionsFrom:
      source: Plan<'A,'B> -> target: Plan<'S,'T> -> Plan<'S,'T>
    val private mergeCostPeak:
      current: StageCostEstimate option ->
        candidate: StageCostEstimate -> StageCostEstimate option
    val private costScore: estimate: StageTimeCostEstimate -> float
    val private hasNoIoCost: estimate: StageTimeCostEstimate -> bool
    val private tryParseTagFloat:
      name: string -> tags: (string * string) list -> float option
    val private ceilDiv: value: uint64 -> divisor: uint64 -> uint64
    val private termMultiplicity:
      inputLength: uint64 ->
        outputLength: uint64 -> time: StageTimeCostEstimate -> uint64
    val private makeCostTerm:
      stageName: string ->
        inputLength: uint64 ->
        outputLength: uint64 -> stageCost: StageCostEstimate -> PipelineCostTerm
    val private trySumEstimatedTimeMillisecondsFromTerms:
      terms: PipelineCostTerm list -> float option
    val private totalCostScoreFromTerms: terms: PipelineCostTerm list -> float
    val private printOptimizationSummary: label: 'a -> pl: Plan<'S,'T> -> unit
    val private formatMilliseconds: milliseconds: float -> string
    val private estimatedRunTimeText: pl: Plan<'S,'T> -> string
    val private ratioAway: expected: float -> actual: float -> float
    val private csvEscape: value: string -> string
    val private invariantFloat: value: float -> string
    val private invariantUInt64: value: uint64 -> string
    val private tryFindRepositoryRoot: unit -> string option
    val private resolveCostFlagPath: path: string -> string
    val private defaultCostFlagPath: unit -> string option
    val private appendCostFlag:
      label: string ->
        kind: string ->
        expected: float option ->
        actual: float ->
        ratio: float option ->
        pl: Plan<'S,'T> ->
        actualTime: float -> actualMemoryDelta: uint64 -> unit
    val private printCostDiscrepancies:
      label: string ->
        pl: Plan<'S,'T> ->
        estimatedTime: float option ->
        actualTime: float -> actualMemoryDelta: uint64 -> unit
    val private runMeasured:
      label: string -> pl: Plan<'S,'T> -> run: (unit -> 'a) -> 'a
    val graph: pl: Plan<'S,'T> -> PipelineGraph
    /// Source type operators
    val source: availableMemory: uint64 -> Plan<unit,unit>
    val sourceWithOptimizer:
      optimize: bool -> availableMemory: uint64 -> Plan<unit,unit>
    val debug:
      level: uint ->
        optimize: bool -> availableMemory: uint64 -> Plan<unit,unit>
    /// Composition operators
    val composePlan:
      name: string -> pl: Plan<'a,'b> -> stage: Stage<'b,'c> -> Plan<'a,'c>
        when 'c: equality
    val (>=>) :
      pl: Plan<'a,'b> -> stage: Stage<'b,'c> -> Plan<'a,'c> when 'c: equality
    val map:
      name: string -> f: ('U -> 'V) -> pl: Plan<'In,'U> -> Plan<'In,'V>
        when 'V: equality
    /// parallel execution of non-synchronised streams
    val internal zipPlan:
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
        stage1: Stage<'S,'U> * stage2: Stage<'S,'V> -> Plan<'In,('U * 'V)>
        when 'U: equality and 'V: equality
    val (>>=>) :
      pl: Plan<'In,('U * 'V)> -> f: ('U -> 'V -> 'W) -> Plan<'In,'W>
        when 'W: equality
    val (>>=>>) :
      pl: Plan<'In,('U * 'V)> ->
        stage1: Stage<'U,'S> * stage2: Stage<'V,'T> -> Plan<'In,('S * 'T)>
        when 'S: equality and 'T: equality
    /// sink type operators
    val doCleaning: pl: Plan<'a,'b> -> unit option
    val sink: pl: Plan<unit,unit> -> unit
    val sinkList: plans: Plan<unit,unit> list -> unit
    val internal runToScalar:
      name: string ->
        reducer: (FSharp.Control.AsyncSeq<'T> -> Async<'R>) ->
        pl: Plan<unit,'T> -> 'R
    val drainSingle: name: string -> pl: Plan<unit,'T> -> 'T
    val drainList: name: string -> pl: Plan<unit,'T> -> 'T list
    val drainLast: name: string -> pl: Plan<unit,'T> -> 'T
