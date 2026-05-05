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
type StageMemoryPressure =
    {
      InputLive: uint64
      OutputLive: uint64
      WorkLive: uint64
      RetainedLive: uint64
      Peak: uint64
    }
module StageMemoryPressure =
    val create:
      inputLive: uint64 ->
        outputLive: uint64 ->
        workLive: uint64 -> retainedLive: uint64 -> StageMemoryPressure
    val fromPeak: peak: uint64 -> StageMemoryPressure
type StageMemoryModel =
    {
      Evaluation: StageEvaluation
      Estimate: (SingleOrPair -> StageMemoryPressure)
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
type StageCostKind =
    | Cpu
    | Native
    | Io
    | Mixed
type StageWorkPressure =
    {
      CpuWorkUnits: float
      NativeWorkUnits: float
      IoReadBytes: uint64
      IoWriteBytes: uint64
      IoReadOps: uint64
      IoWriteOps: uint64
      CalibrationKey: string option
    }
module StageWorkPressure =
    val zero: StageWorkPressure
    val create:
      cpuWorkUnits: float ->
        nativeWorkUnits: float ->
        ioReadBytes: uint64 ->
        ioWriteBytes: uint64 ->
        ioReadOps: uint64 ->
        ioWriteOps: uint64 -> calibrationKey: string option -> StageWorkPressure
    val add:
      left: StageWorkPressure -> right: StageWorkPressure -> StageWorkPressure
type StageWorkModel =
    {
      Kind: StageCostKind
      Evaluation: StageEvaluation
      Estimate: (SingleOrPair -> StageWorkPressure)
    }
type StageCostCoefficients =
    {
      CpuMillisecondsPerUnit: float
      NativeMillisecondsPerUnit: float
      IoReadMillisecondsPerByte: float
      IoWriteMillisecondsPerByte: float
      IoReadMillisecondsPerOp: float
      IoWriteMillisecondsPerOp: float
    }
module StageCostCoefficients =
    val zero: StageCostCoefficients
    val estimateMilliseconds:
      coefficients: StageCostCoefficients ->
        pressure: StageWorkPressure -> float
module StageCostCalibration =
    val mutable private coefficientsByKey: Map<string,StageCostCoefficients>
    val clear: unit -> unit
    val register: key: string -> coefficients: StageCostCoefficients -> unit
    val tryFind: key: string -> StageCostCoefficients option
    val estimateMilliseconds: pressure: StageWorkPressure -> float option
    val private propertyDouble:
      name: string -> element: System.Text.Json.JsonElement -> float
    val loadJson: path: string -> bool
module StageWorkModel =
    val private elementCount: input: SingleOrPair -> float
    val zero: evaluation: StageEvaluation -> StageWorkModel
    val cpu:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        workUnits: (SingleOrPair -> float) -> StageWorkModel
    val native:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        workUnits: (SingleOrPair -> float) -> StageWorkModel
    val ioRead:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        bytes: (SingleOrPair -> uint64) ->
        ops: (SingleOrPair -> uint64) -> StageWorkModel
    val ioWrite:
      evaluation: StageEvaluation ->
        calibrationKey: string option ->
        bytes: (SingleOrPair -> uint64) ->
        ops: (SingleOrPair -> uint64) -> StageWorkModel
    val fromEvaluation:
      evaluation: StageEvaluation ->
        calibrationKey: string option -> StageWorkModel
type StageCostPressure =
    {
      Memory: StageMemoryPressure
      Work: StageWorkPressure
    }
module StageCostPressure =
    val add:
      left: StageCostPressure -> right: StageCostPressure -> StageCostPressure
type StageCostModel =
    {
      Memory: StageMemoryModel
      Work: StageWorkModel
    }
module StageCostModel =
    val create:
      memory: StageMemoryModel -> work: StageWorkModel -> StageCostModel
    val fromMemory: memory: StageMemoryModel -> StageCostModel
    val estimate:
      model: StageCostModel -> input: SingleOrPair -> StageCostPressure
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
      LengthTransformation: LengthTransformation
      Graph: PipelineGraph
      Cleaning: (unit -> unit) list
    }
module Stage =
    val createWithCostModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        costModel: StageCostModel ->
        lengthTransformation: LengthTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWithModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryModel: StageMemoryModel ->
        lengthTransformation: LengthTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val create:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrapped:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        wrapMemoryNeed: MemoryNeedWrapped ->
        lengthTransformation: LengthTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrappedWithModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        memoryModel: StageMemoryModel ->
        lengthTransformation: LengthTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val createWrappedWithCostModel:
      name: string ->
        build: (unit -> Pipe<'S,'T>) ->
        transition: ProfileTransition ->
        costModel: StageCostModel ->
        lengthTransformation: LengthTransformation ->
        cleaning: (unit -> unit) list -> Stage<'S,'T>
    val private withGraph:
      graph: PipelineGraph -> stage: Stage<'a,'b> -> Stage<'a,'b>
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
    val idStage: name: string -> Stage<'T,'T>
    val clean: name: string -> fct: (unit -> unit) -> Stage<'T,'T>
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
        f: (bool -> 'S -> 'T) ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
    val mapi:
      name: string ->
        f: (bool -> int64 -> 'S -> 'T) ->
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
    val teeFst: stage: Stage<'A,'A> -> Stage<('A * 'B),('A * 'B)>
    val teeSnd: stage: Stage<'B,'B> -> Stage<('A * 'B),('A * 'B)>
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
    val retainWith: name: string -> ops: ResourceOps<'T> -> Stage<'T,'T>
    val releaseWith: name: string -> ops: ResourceOps<'T> -> Stage<'T,unit>
    val liftResourceUnary:
      name: string ->
        ops: ResourceOps<'S> ->
        f: ('S -> 'T) ->
        memoryNeed: MemoryNeed ->
        lengthTransformation: LengthTransformation -> Stage<'S,'T>
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
type OptimizationCandidate<'S,'T> =
    {
      Name: string
      Stage: Stage<'S,'T>
      Explanation: string
    }
type OptimizationDecision =
    {
      CandidateName: string
      Accepted: bool
      EstimatedMemoryBytes: uint64
      EstimatedMilliseconds: float option
      EstimatedWorkScore: float
      Reason: string
    }
type OptimizationResult<'S,'T> =
    {
      Selected: OptimizationCandidate<'S,'T> option
      Decisions: OptimizationDecision list
    }
module Optimizer =
    val chooseStage:
      availableMemory: uint64 ->
        inputShape: SingleOrPair ->
        candidates: OptimizationCandidate<'S,'T> list ->
        OptimizationResult<'S,'T>
    val chooseStageOrThrow:
      availableMemory: uint64 ->
        inputShape: SingleOrPair ->
        candidates: OptimizationCandidate<'a,'b> list ->
        Stage<'a,'b> * OptimizationResult<'a,'b>
type Plan<'S,'T> =
    {
      stage: Stage<'S,'T> option
      graph: PipelineGraph
      sourcePeek: SourcePeek option
      costPeak: StageCostPressure option
      costObservations: StageCostPressure list
      nElemsPerSlice: SingleOrPair
      length: uint64
      memAvail: uint64
      memPeak: uint64
      debug: bool
      debugLevel: uint
    }
module Plan =
    val private graphOfStage: stage: Stage<'a,'b> option -> PipelineGraph
    val private levelOf: debug: bool -> uint32
    val create:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElemsPerSlice: uint64 -> length: uint64 -> debug: bool -> Plan<'S,'T>
        when 'T: equality
    val createWrapped:
      stage: Stage<'S,'T> option ->
        memAvail: uint64 ->
        memPeak: uint64 ->
        nElemsPerSlice: SingleOrPair ->
        length: uint64 -> debug: bool -> Plan<'S,'T> when 'T: equality
    val withSourcePeek: sourcePeek: SourcePeek -> pl: Plan<'S,'T> -> Plan<'S,'T>
    val private mergeCostPeak:
      current: StageCostPressure option ->
        candidate: StageCostPressure -> StageCostPressure option
    val graph: pl: Plan<'S,'T> -> PipelineGraph
    /// Source type operators
    val source: availableMemory: uint64 -> Plan<unit,unit>
    val debug: level: uint -> availableMemory: uint64 -> Plan<unit,unit>
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
      f: ('U * 'V -> 'S * 'T) ->
        pl: Plan<'In,('U * 'V)> ->
        stage: Stage<('U * 'V),('S * 'T)> -> Plan<'In,('S * 'T)>
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
