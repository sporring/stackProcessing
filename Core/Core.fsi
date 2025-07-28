namespace FSharp
module Core
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant
    | Streaming
    | Sliding of uint * uint * uint * uint
    | Full
    member EstimateUsage: width: uint -> height: uint -> depth: uint -> uint64
    member
      RequiresBuffering: availableMemory: uint64 ->
                           width: uint -> height: uint -> depth: uint -> bool
    member combineProfile: other: MemoryProfile -> MemoryProfile
/// A configurable image processing step that operates on image slices.
/// Pipe describes *how* to do it:
/// - Encapsulates the concrete execution logic
/// - Defines memory usage behavior
/// - Takes and returns AsyncSeq streams
/// - Pipe + WindowedProcessor: How it’s computed 
type Pipe<'S,'T> =
    {
      Name: string
      Profile: MemoryProfile
      Apply: (FSharp.Control.AsyncSeq<'S> -> FSharp.Control.AsyncSeq<'T>)
    }
val internal run: p: Pipe<'S,'T> -> FSharp.Control.AsyncSeq<'T>
val internal lift:
  name: string ->
    profile: MemoryProfile -> f: ('In -> Async<'Out>) -> Pipe<'In,'Out>
val map: label: string -> profile: MemoryProfile -> f: ('S -> 'T) -> Pipe<'S,'T>
val internal reduce:
  name: string ->
    profile: MemoryProfile ->
    reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) -> Pipe<'In,'Out>
val fold:
  label: string ->
    profile: MemoryProfile ->
    folder: ('State -> 'In -> 'State) -> state0: 'State -> Pipe<'In,'State>
val internal consumeWith:
  name: string ->
    profile: MemoryProfile ->
    consume: (FSharp.Control.AsyncSeq<'T> -> Async<unit>) -> Pipe<'T,unit>
/// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
val composePipe: p1: Pipe<'S,'T> -> p2: Pipe<'T,'U> -> Pipe<'S,'U>
/// SliceShape describes the dimensions of a stacked slice.
/// Conventionally: [width; height; depth]
/// Used for validating if transitions are feasible (e.g., sliding window depth fits).
type SliceShape = uint list
/// MemoryTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
/// - Check: a predicate on slice shape to validate if the transition is allowed
type MemoryTransition =
    {
      From: MemoryProfile
      To: MemoryProfile
      Check: (SliceShape -> bool)
    }
/// Operation describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Operation + MemoryTransition: what happens
type Operation<'S,'T> =
    {
      Name: string
      Transition: MemoryTransition
      Pipe: Pipe<'S,'T>
    }
val defaultCheck: 'a -> bool
val transition:
  fromProfile: MemoryProfile -> toProfile: MemoryProfile -> MemoryTransition
val idOp: unit -> Operation<'T,'T>
val inline asPipe: op: Operation<'a,'b> -> Pipe<'a,'b>
val pipe2Operation:
  name: string ->
    transition: MemoryTransition -> pipe: Pipe<'S,'T> -> Operation<'S,'T>
type MemFlow<'S,'T> =
    uint64 -> SliceShape option -> Operation<'S,'T> * uint64 * SliceShape option
val private memNeed: shape: uint list -> p: Pipe<'a,'b> -> uint64
/// Try to shrink a too‑hungry pipe to a cheaper profile.
/// *You* control the downgrade policy here.
val private shrinkProfile:
  avail: uint64 -> shape: uint list -> p: Pipe<'S,'T> -> Pipe<'S,'T>
val returnM:
  op: Operation<'S,'T> ->
    bytes: uint64 ->
    shapeOpt: SliceShape option -> Operation<'S,'T> * uint64 * SliceShape option
val composeOp:
  op1: Operation<'S,'T> -> op2: Operation<'T,'U> -> Operation<'S,'U>
val bindM:
  m: ('a -> 'b -> Operation<'c,'d> * 'e * 'f) ->
    k: (Operation<'c,'d> -> 'e -> 'f -> Operation<'d,'g> * 'h * 'i) ->
    bytes: 'a -> shapeOpt: 'b -> Operation<'c,'g> * 'h * 'i
val (-->) : op1: Operation<'A,'B> -> op2: Operation<'B,'C> -> Operation<'A,'C>
type Pipeline<'S,'T> =
    {
      flow: MemFlow<'S,'T>
      mem: uint64
      shape: SliceShape option
    }
val operation2Pipeline:
  op: Operation<'S,'T> ->
    mem: uint64 -> shape: SliceShape option -> Pipeline<'S,'T>
val sourceOp: availableMemory: uint64 -> Pipeline<unit,unit>
val attachFirst:
  Operation<unit,'T> * (unit -> SliceShape) ->
    pl: Pipeline<unit,'T> -> Pipeline<unit,'T>
val (>=>) : pl: Pipeline<'a,'b> -> next: Operation<'b,'c> -> Pipeline<'a,'c>
val sinkOp: pl: Pipeline<unit,unit> -> unit
val sinkListOp: pipelines: Pipeline<unit,unit> list -> unit
val asOperation: pl: Pipeline<'In,'Out> -> Operation<'In,'Out>
module Routing
/// Split a Pipe<'In,'T> into two branches that
///   • read the upstream only once
///   • keep at most one item in memory
///   • terminate correctly when both sides finish
type private Request<'T> =
    | Left of AsyncReplyChannel<Option<'T>>
    | Right of AsyncReplyChannel<Option<'T>>
val internal tee: p: Core.Pipe<'In,'T> -> Core.Pipe<'In,'T> * Core.Pipe<'In,'T>
val internal teeOp:
  op: Core.Operation<'In,'T> -> Core.Operation<'In,'T> * Core.Operation<'In,'T>
val teePipeline:
  pl: Core.Pipeline<'In,'T> -> Core.Pipeline<'In,'T> * Core.Pipeline<'In,'T>
/// zipWith two Pipes<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
val internal zipWithOp:
  f: ('A -> 'B -> 'C) ->
    op1: Core.Operation<'In,'A> ->
    op2: Core.Operation<'In,'B> -> Core.Operation<'In,'C>
val zipWith:
  f: ('A -> 'B -> 'C) ->
    p1: Core.Pipeline<'In,'A> ->
    p2: Core.Pipeline<'In,'B> -> Core.Pipeline<'In,'C>
val runToScalar:
  name: 'a ->
    reducer: (FSharp.Control.AsyncSeq<'T> -> Async<'R>) ->
    pl: Core.Pipeline<'In,'T> -> 'R
val drainSingle: name: 'a -> pl: Core.Pipeline<'b,'c> -> 'c
val drainList: name: 'a -> pl: Core.Pipeline<'b,'c> -> 'c list
val drainLast: name: 'a -> pl: Core.Pipeline<'b,'c> -> 'c
val tap: label: string -> Core.Pipe<'T,'T>
/// quick constructor for Streaming→Streaming unary ops
val liftUnaryOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val tapOp: label: string -> Core.Operation<'T,'T>
/// Represents a pipeline that has been shared (split into synchronized branches)
type SharedPipeline<'T,'U,'V> =
    {
      flow:
        (uint64 ->
           Core.SliceShape option ->
           (Core.Operation<'T,'U> * Core.Operation<'T,'V>) * uint64 *
           Core.SliceShape option)
      branches: Core.Operation<'T,'U> * Core.Operation<'T,'V>
      mem: uint64
      shape: Core.SliceShape option
    }
/// parallel fanout with synchronization
/// Synchronously split the shared stream into two parallel pipelines
val (>=>>) :
  pl: Core.Pipeline<'In,'T> ->
    op1: Core.Operation<'T,'U> * op2: Core.Operation<'T,'V> ->
      SharedPipeline<'In,'U,'V>
val (>>=>) :
  shared: SharedPipeline<'In,'U,'V> ->
    combine: (Core.Operation<'In,'U> * Core.Operation<'In,'V> ->
                Core.Operation<'In,'W>) -> Core.Pipeline<'In,'W>
val unitPipeline: unit -> Core.Pipeline<'T,unit>
val combineIgnore:
  op1: Core.Operation<'In,'U> * op2: Core.Operation<'In,'V> ->
    Core.Operation<'In,unit>
module SourceSink
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
    val readSlices:
      inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
        when 'T: equality
    val readRandomSlices:
      count: uint ->
        inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
        when 'T: equality
/// Source parts
val createPipe:
  width: uint -> height: uint -> depth: uint -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val createOp:
  width: uint ->
    height: uint ->
    depth: uint ->
    pl: Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<'T>>
    when 'T: equality
val readOp:
  inputDir: string ->
    suffix: string ->
    pl: Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<'T>>
    when 'T: equality
val readRandomOp:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    pl: Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<'T>>
    when 'T: equality
val writeOp:
  path: string -> suffix: string -> Core.Operation<Slice.Slice<'a>,unit>
    when 'a: equality
val showOp:
  plt: (Slice.Slice<'T> -> unit) -> Core.Operation<Slice.Slice<'T>,unit>
    when 'T: equality
val plotOp:
  plt: (float list -> float list -> unit) ->
    Core.Operation<(float * float) list,unit>
val printOp: unit -> Core.Operation<'T,unit>
val liftImageSource:
  name: string -> img: Slice.Slice<'T> -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val axisSourceOp:
  axis: int ->
    size: int list ->
    pl: Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<uint>>
val finiteDiffFilter3DOp:
  direction: uint ->
    order: uint ->
    pl: Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<float>>
module Processing
val skipFirstLast: n: int -> lst: 'a list -> 'a list
/// mapWindowed keeps a running window along the slice direction of depth images
/// and processes them by f. The stepping size of the running window is stride.
/// So if depth is 3 and stride is 1 then first image 0,1,2 is sent to f, then 1, 2, 3
/// and so on. If depth is 3 and stride is 3, then it'll be image 0, 1, 2 followed by
/// 3, 4, 5. It is also possible to use this for sampling, e.g., setting depth to 1
/// and stride to 2 sends every second image to f.  
val internal mapWindowed:
  label: string ->
    depth: uint ->
    stride: uint ->
    emitStart: uint ->
    emitCount: uint -> f: ('S list -> 'T list) -> Core.Pipe<'S,'T>
val internal liftWindowedOp:
  name: string ->
    window: uint ->
    stride: uint ->
    emitStart: uint ->
    emitCount: uint ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'S>,Slice.Slice<'T>>
    when 'S: equality and 'T: equality
val internal liftWindowedTrimOp:
  name: string ->
    window: uint ->
    stride: uint ->
    emitStart: uint ->
    emitCount: uint ->
    trim: uint ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'S>,Slice.Slice<'T>>
    when 'S: equality and 'T: equality
/// quick constructor for Streaming→Streaming unary ops
val internal liftUnaryOpInt:
  name: string ->
    f: (Slice.Slice<int> -> Slice.Slice<int>) ->
    Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val internal liftUnaryOpFloat32:
  name: string ->
    f: (Slice.Slice<float32> -> Slice.Slice<float32>) ->
    Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val internal liftUnaryOpFloat:
  name: string ->
    f: (Slice.Slice<float> -> Slice.Slice<float>) ->
    Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val internal liftBinaryOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<(Slice.Slice<'T> * Slice.Slice<'T>),Slice.Slice<'T>>
    when 'T: equality
val internal liftBinaryOpFloat:
  name: string ->
    f: (Slice.Slice<float> -> Slice.Slice<float> -> Slice.Slice<float>) ->
    Core.Operation<(Slice.Slice<float> * Slice.Slice<float>),Slice.Slice<float>>
val internal liftFullOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val internal liftFullParamOp:
  name: string ->
    f: ('P -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    param: 'P -> Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val internal liftFullParam2Op:
  name: string ->
    f: ('P -> 'Q -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    param1: 'P -> param2: 'Q -> Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val internal liftMapOp:
  name: string ->
    f: (Slice.Slice<'T> -> 'U) -> Core.Operation<Slice.Slice<'T>,'U>
    when 'T: comparison
val inline castOp:
  name: string ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'S>,Slice.Slice<'T>>
    when 'S: equality and 'T: equality
val castUInt8ToInt8Op:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<int8>>
val castUInt8ToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint16>>
val castUInt8ToInt16Op:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<int16>>
val castUInt8ToUIntOp:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint>>
val castUInt8ToIntOp:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<int>>
val castUInt8ToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint64>>
val castUInt8ToInt64Op:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<int64>>
val castUInt8ToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<float32>>
val castUInt8ToFloatOp:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<float>>
val castInt8ToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<uint8>>
val castInt8ToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<uint16>>
val castInt8ToInt16Op:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<int16>>
val castInt8ToUIntOp:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<uint>>
val castInt8ToIntOp:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<int>>
val castInt8ToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<uint64>>
val castInt8ToInt64Op:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<int64>>
val castInt8ToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<float32>>
val castInt8ToFloatOp:
  name: string -> Core.Operation<Slice.Slice<int8>,Slice.Slice<float>>
val castUInt16ToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<uint8>>
val castUInt16ToInt8Op:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<int8>>
val castUInt16ToInt16Op:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<int16>>
val castUInt16ToUIntOp:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<uint>>
val castUInt16ToIntOp:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<int>>
val castUInt16ToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<uint64>>
val castUInt16ToInt64Op:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<int64>>
val castUInt16ToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<float32>>
val castUInt16ToFloatOp:
  name: string -> Core.Operation<Slice.Slice<uint16>,Slice.Slice<float>>
val castInt16ToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<uint8>>
val castInt16ToInt8Op:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<int8>>
val castInt16ToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<uint16>>
val castInt16ToUIntOp:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<uint>>
val castInt16ToIntOp:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<int>>
val castInt16ToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<uint64>>
val castInt16ToInt64Op:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<int64>>
val castInt16ToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<float32>>
val castInt16ToFloatOp:
  name: string -> Core.Operation<Slice.Slice<int16>,Slice.Slice<float>>
val castUIntToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<uint8>>
val castUIntToInt8Op:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<int8>>
val castUIntToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<uint16>>
val castUIntToInt16Op:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<int16>>
val castUIntToIntOp:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<int>>
val castUIntToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<uint64>>
val castUIntToInt64Op:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<int64>>
val castUIntToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<float32>>
val castUIntToFloatOp:
  name: string -> Core.Operation<Slice.Slice<uint>,Slice.Slice<float>>
val castIntToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<uint8>>
val castIntToInt8Op:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int8>>
val castIntToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<uint16>>
val castIntToInt16Op:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int16>>
val castIntToUIntOp:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<uint>>
val castIntToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<uint64>>
val castIntToInt64Op:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int64>>
val castIntToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<float32>>
val castIntToFloatOp:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<float>>
val castUInt64ToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint8>>
val castUInt64ToInt8Op:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<int8>>
val castUInt64ToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint16>>
val castUInt64ToInt16Op:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<int16>>
val castUInt64ToUIntOp:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint>>
val castUInt64ToIntOp:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<int>>
val castUInt64ToInt64Op:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<int64>>
val castUInt64ToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<float32>>
val castUInt64ToFloatOp:
  name: string -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<float>>
val castInt64ToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<uint8>>
val castInt64ToInt8Op:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<int8>>
val castInt64ToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<uint16>>
val castInt64ToInt16Op:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<int16>>
val castInt64ToUIntOp:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<uint>>
val castInt64ToIntOp:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<int>>
val castInt64ToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<uint64>>
val castInt64ToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<float32>>
val castInt64ToFloatOp:
  name: string -> Core.Operation<Slice.Slice<int64>,Slice.Slice<float>>
val castFloat32ToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<uint8>>
val castFloat32ToInt8Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<int8>>
val castFloat32ToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<uint16>>
val castFloat32ToInt16Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<int16>>
val castFloat32ToUIntOp:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<uint>>
val castFloat32ToIntOp:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<int>>
val castFloat32ToUInt64Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<uint64>>
val castFloat32ToInt64Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<int64>>
val castFloat32ToFloatOp:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float>>
val castFloatToUInt8Op:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<uint8>>
val castFloatToInt8Op:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<int8>>
val castFloatToUInt16Op:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<uint16>>
val castFloatToInt16Op:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<int16>>
val castFloatToUIntOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<uint>>
val castFloatToIntOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<int>>
val castFloatToUIn64Op:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<uint64>>
val castFloatToInt64Op:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<int64>>
val castFloatToFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float32>>
/// Basic arithmetic
val addOp:
  name: string ->
    slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarAddSliceOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceAddScalarOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val subOp:
  name: string ->
    slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarSubSliceOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceSubScalarOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val mulOp:
  name: string ->
    slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarMulSliceOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceMulScalarOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val divOp:
  name: string ->
    slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarDivSliceOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceDivScalarOp:
  name: string -> i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
/// Simple functions
val absFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val absFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val absIntOp: name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val acosFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val acosFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val asinFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val asinFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val atanFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val atanFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val cosFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val cosFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val expFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val expFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val log10Float32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val log10FloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val logFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val logFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val roundFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val roundFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sinFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val sinFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sqrtFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val sqrtFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sqrtIntOp: name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val squareFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val squareFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val squareIntOp:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val tanFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val tanFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
/// Histogram related functions
val histogramOp:
  name: string -> Core.Operation<Slice.Slice<'T>,Map<'T,uint64>>
    when 'T: comparison
val map2pairsOp:
  name: string -> Core.Operation<Map<'T,'S>,('T * 'S) list> when 'T: comparison
val inline pairs2floatsOp:
  name: string -> Core.Operation<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2intsOp:
  name: string -> Core.Operation<(^T * ^S) list,(int * int) list>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
type ImageStats = Slice.ImageStats
val computeStatsOp:
  name: string -> Core.Operation<Slice.Slice<'T>,ImageStats> when 'T: equality
/// Convolution like operators
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
val discreteGaussianOp:
  name: string ->
    sigma: float ->
    bc: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val convolveOp:
  name: string ->
    kernel: Slice.Slice<'T> ->
    bc: Slice.BoundaryCondition option ->
    winSz: uint option -> Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val private makeMorphOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    core: (uint -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val binaryErodeOp:
  name: string ->
    radius: uint ->
    winSz: uint option -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryDilateOp:
  name: string ->
    radius: uint ->
    winSz: uint option -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryOpeningOp:
  name: string ->
    radius: uint ->
    winSz: uint option -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryClosingOp:
  name: string ->
    radius: uint ->
    winSz: uint option -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryFillHolesOp:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val connectedComponentsOp:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint64>>
val piecewiseConnectedComponentsOp:
  name: string ->
    windowSize: uint option ->
    Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint64>>
val otsuThresholdOp:
  name: string -> Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val otsuMultiThresholdOp:
  name: string -> n: byte -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val momentsThresholdOp:
  name: string -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val signedDistanceMapOp:
  name: string -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<float>>
val watershedOp:
  name: string -> a: float -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val thresholdOp:
  name: string ->
    a: float -> b: float -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val addNormalNoiseOp:
  name: string ->
    a: float -> b: float -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val relabelComponentsOp:
  name: string ->
    a: uint -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint64>>
val constantPad2DOp<'T when 'T: equality> :
  name: string ->
    padLower: uint list ->
    padUpper: uint list ->
    c: double -> Core.Operation<Slice.Slice<obj>,Slice.Slice<obj>>
    when 'T: equality
type FileInfo = Slice.FileInfo
val getStackDepth: (string -> string -> uint)
val getStackInfo: (string -> string -> Slice.FileInfo)
val getStackSize: (string -> string -> uint * uint * uint)
val getStackWidth: (string -> string -> uint64)
val getStackHeight: (string -> string -> uint64)
