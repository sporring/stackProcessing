namespace FSharp
module Core
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant
    | StreamingConstant
    | Streaming
    | SlidingConstant of uint
    | Sliding of uint
    | FullConstant
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
/// Creates a MemoryTransition record:
/// - Describes expected memory layout before and after an operation
/// - Default check always passes; can be replaced with shape-aware checks
val defaultCheck: 'a -> bool
val transition:
  fromProfile: MemoryProfile -> toProfile: MemoryProfile -> MemoryTransition
/// Represents a 3D image processing operation:
/// - Operates on a stacked 3D Slice built from a sliding window of 2D slices
/// - Independent of streaming logic — only processes one 3D slice at a time
/// - Typically wrapped via `fromWindowed` to integrate into a streaming pipeline
type WindowedProcessor<'S,'T when 'S: equality and 'T: equality> =
    {
      Name: string
      Window: uint
      Stride: uint
      Process: (Slice.Slice<'S> -> Slice.Slice<'T>)
    }
val validate: op1: Operation<'A,'B> -> op2: Operation<'B,'C> -> unit
val describeOp: op: Operation<'a,'b> -> string
val plan: ops: Operation<'a,'b> list -> string
val (>=>!) : op1: Operation<'A,'B> -> op2: Operation<'B,'C> -> Operation<'A,'C>
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
    new: availableMemory: uint64 -> PipelineBuilder
    /// Wraps a processor value for use in the pipeline computation expression.
    member Return: p: Pipe<'S,'T> -> Pipe<'S,'T>
    member availableMemory: uint64
/// A memory-aware pipeline builder with the specified processing constraints.
val pipeline: availableMemory: uint64 -> PipelineBuilder
/// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
val composePipe: p1: Pipe<'S,'T> -> p2: Pipe<'T,'U> -> Pipe<'S,'U>
val (>=>) : p1: Pipe<'a,'b> -> p2: Pipe<'b,'c> -> Pipe<'a,'c>
module Helpers =
    /// Pipeline helper functions
    val singletonPipe:
      name: string -> seq: FSharp.Control.AsyncSeq<'T> -> Pipe<'S,'T>
    val bindPipe: p: Pipe<'S,Pipe<'S,'T>> -> Pipe<'S,'T>
    /// Operator and such
    /// pull the runnable pipe out of an operation
    val inline asPipe: op: Operation<'a,'b> -> Pipe<'a,'b>
    /// quick constructor for Streaming→Streaming unary ops
    val liftUnaryOp:
      name: string ->
        f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
        Operation<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
    val validate: op1: Operation<'a,'b> -> op2: Operation<'c,'d> -> bool
module Routing
module internal Builder =
    type MemFlow<'S,'T> =
        uint64 ->
          Core.SliceShape option ->
          Core.Operation<'S,'T> * uint64 * Core.SliceShape option
    val private memNeed: shape: uint list -> p: Core.Pipe<'a,'b> -> uint64
    /// Try to shrink a too‑hungry pipe to a cheaper profile.
    /// *You* control the downgrade policy here.
    val private shrinkProfile:
      avail: uint64 ->
        shape: uint list -> p: Core.Pipe<'S,'T> -> Core.Pipe<'S,'T>
    val returnM:
      op: Core.Operation<'S,'T> ->
        bytes: uint64 ->
        shapeOpt: Core.SliceShape option ->
        Core.Operation<'S,'T> * uint64 * Core.SliceShape option
    val composeOp:
      op1: Core.Operation<'S,'T> ->
        op2: Core.Operation<'T,'U> -> Core.Operation<'S,'U>
    val bindM:
      m: ('a -> Core.SliceShape option -> Core.Operation<'b,'c> * 'd * 'e) ->
        k: (Core.Operation<'b,'c> -> 'd -> 'e -> Core.Operation<'c,'f> * 'g * 'h) ->
        bytes: 'a ->
        shapeOpt: Core.SliceShape option -> Core.Operation<'b,'f> * 'g * 'h
    type Pipeline<'S,'T> =
        {
          flow: MemFlow<'S,'T>
          mem: uint64
          shape: Core.SliceShape option
        }
    val source: memBudget: uint64 -> Pipeline<'a,'b>
    val attachFirst:
      Core.Operation<unit,'T> * (unit -> Core.SliceShape) ->
        pl: Pipeline<unit,'T> -> Pipeline<unit,'T>
    val (>>=>) :
      pl: Pipeline<'a,'b> -> next: Core.Operation<'b,'b> -> Pipeline<'a,'b>
    val sink: pl: Pipeline<'S,'T> -> Core.Pipe<'S,'T>
val run: p: Core.Pipe<unit,'T> -> FSharp.Control.AsyncSeq<'T>
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
val zipWithPipe:
  f: ('A -> 'B -> Core.Pipe<'In,'C>) ->
    pa: Core.Pipe<'In,'A> -> pb: Core.Pipe<'In,'B> -> Core.Pipe<'In,'C>
val cacheScalar: name: string -> p: Core.Pipe<unit,'T> -> Core.Pipe<'In,'T>
val tap: label: string -> Core.Pipe<'T,'T>
val sequentialJoin:
  p1: Core.Pipe<'S,'T> -> p2: Core.Pipe<'S,'T> -> Core.Pipe<'S,'T>
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
val sourceLst:
  availableMemory: uint64 ->
    processors: Core.Pipe<unit,'T> list -> Core.Pipe<unit,'T> list
val source:
  availableMemory: uint64 -> p: Core.Pipe<unit,'T> -> Core.Pipe<unit,'T>
val sinkLst: processors: Core.Pipe<unit,unit> list -> unit
val sink: p: Core.Pipe<unit,unit> -> unit
val readSlices:
  inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val read:
  inputDir: string ->
    suffix: string ->
    transform: (Core.Pipe<unit,Slice.Slice<'T>> ->
                  Core.Pipe<unit,Slice.Slice<'T>>) ->
    Core.Pipe<unit,Slice.Slice<'T>> when 'T: equality
val readSliceN:
  idx: uint ->
    inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readRandomSlices:
  count: uint ->
    inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readRandom:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    transform: (Core.Pipe<unit,Slice.Slice<'T>> ->
                  Core.Pipe<unit,Slice.Slice<'T>>) ->
    Core.Pipe<unit,Slice.Slice<'T>> when 'T: equality
/// Source parts
val createPipe:
  width: uint -> height: uint -> depth: uint -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val create:
  width: uint ->
    height: uint ->
    depth: uint ->
    transform: (Core.Pipe<unit,Slice.Slice<'T>> ->
                  Core.Pipe<unit,Slice.Slice<'T>>) ->
    Core.Pipe<unit,Slice.Slice<'T>> when 'T: equality
val liftImageSource:
  name: string -> img: Slice.Slice<'T> -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val gaussSource:
  sigma: float -> kernelSize: uint option -> Core.Pipe<unit,Slice.Slice<float>>
val axisSource: axis: int -> size: int list -> Core.Pipe<unit,Slice.Slice<uint>>
val gauss:
  sigma: float -> kernelSize: uint option -> Core.Pipe<unit,Slice.Slice<float>>
val finiteDiffFilter2D:
  direction: uint -> order: uint -> Core.Pipe<unit,Slice.Slice<float>>
val finiteDiffFilter3D:
  direction: uint -> order: uint -> Core.Pipe<unit,Slice.Slice<float>>
/// Sink parts
val print<'T> : Core.Pipe<'T,unit>
val plot:
  plt: (float list -> float list -> unit) ->
    Core.Pipe<(float * float) list,unit>
val show:
  plt: (Slice.Slice<'a> -> unit) -> Core.Pipe<Slice.Slice<'a>,unit>
    when 'a: equality
val write:
  path: string -> suffix: string -> Core.Pipe<Slice.Slice<'a>,unit>
    when 'a: equality
val ignore<'T> : Core.Pipe<'T,unit>
module Processing
val private explodeSlice:
  slices: Slice.Slice<'T> -> FSharp.Control.AsyncSeq<Slice.Slice<'T>>
    when 'T: equality
val private reduce:
  label: string ->
    profile: Core.MemoryProfile ->
    reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
    Core.Pipe<'In,'Out>
val fold:
  label: string ->
    profile: Core.MemoryProfile ->
    folder: ('State -> 'In -> 'State) -> state0: 'State -> Core.Pipe<'In,'State>
val map:
  label: string ->
    profile: Core.MemoryProfile -> f: ('S -> 'T) -> Core.Pipe<'S,'T>
/// mapWindowed keeps a running window along the slice direction of depth images
/// and processes them by f. The stepping size of the running window is stride.
/// So if depth is 3 and stride is 1 then first image 0,1,2 is sent to f, then 1, 2, 3
/// and so on. If depth is 3 and stride is 3, then it'll be image 0, 1, 2 followed by
/// 3, 4, 5. It is also possible to use this for sampling, e.g., setting depth to 1
/// and stride to 2 sends every second image to f.  
val mapWindowed:
  label: string ->
    depth: uint -> stride: uint -> f: ('S list -> 'T list) -> Core.Pipe<'S,'T>
val inline cast:
  label: string ->
    fct: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    Core.Pipe<Slice.Slice<'S>,Slice.Slice<'T>>
    when 'S: equality and 'T: equality
val castUInt8ToInt8: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<int8>>
val castUInt8ToUInt16: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint16>>
val castUInt8ToInt16: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<int16>>
val castUInt8ToUInt: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint>>
val castUInt8ToInt: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<int>>
val castUInt8ToUInt64: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint64>>
val castUInt8ToInt64: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<int64>>
val castUInt8ToFloat32: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<float32>>
val castUInt8ToFloat: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<float>>
val castInt8ToUInt8: Core.Pipe<Slice.Slice<int8>,Slice.Slice<uint8>>
val castInt8ToUInt16: Core.Pipe<Slice.Slice<int8>,Slice.Slice<uint16>>
val castInt8ToInt16: Core.Pipe<Slice.Slice<int8>,Slice.Slice<int16>>
val castInt8ToUInt: Core.Pipe<Slice.Slice<int8>,Slice.Slice<uint>>
val castInt8ToInt: Core.Pipe<Slice.Slice<int8>,Slice.Slice<int>>
val castInt8ToUInt64: Core.Pipe<Slice.Slice<int8>,Slice.Slice<uint64>>
val castInt8ToInt64: Core.Pipe<Slice.Slice<int8>,Slice.Slice<int64>>
val castInt8ToFloat32: Core.Pipe<Slice.Slice<int8>,Slice.Slice<float32>>
val castInt8ToFloat: Core.Pipe<Slice.Slice<int8>,Slice.Slice<float>>
val castUInt16ToUInt8: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<uint8>>
val castUInt16ToInt8: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<int8>>
val castUInt16ToInt16: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<int16>>
val castUInt16ToUInt: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<uint>>
val castUInt16ToInt: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<int>>
val castUInt16ToUInt64: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<uint64>>
val castUInt16ToInt64: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<int64>>
val castUInt16ToFloat32: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<float32>>
val castUInt16ToFloat: Core.Pipe<Slice.Slice<uint16>,Slice.Slice<float>>
val castInt16ToUInt8: Core.Pipe<Slice.Slice<int16>,Slice.Slice<uint8>>
val castInt16ToInt8: Core.Pipe<Slice.Slice<int16>,Slice.Slice<int8>>
val castInt16ToUInt16: Core.Pipe<Slice.Slice<int16>,Slice.Slice<uint16>>
val castInt16ToUInt: Core.Pipe<Slice.Slice<int16>,Slice.Slice<uint>>
val castInt16ToInt: Core.Pipe<Slice.Slice<int16>,Slice.Slice<int>>
val castInt16ToUInt64: Core.Pipe<Slice.Slice<int16>,Slice.Slice<uint64>>
val castInt16ToInt64: Core.Pipe<Slice.Slice<int16>,Slice.Slice<int64>>
val castInt16ToFloat32: Core.Pipe<Slice.Slice<int16>,Slice.Slice<float32>>
val castInt16ToFloat: Core.Pipe<Slice.Slice<int16>,Slice.Slice<float>>
val castUIntToUInt8: Core.Pipe<Slice.Slice<uint>,Slice.Slice<uint8>>
val castUIntToInt8: Core.Pipe<Slice.Slice<uint>,Slice.Slice<int8>>
val castUIntToUInt16: Core.Pipe<Slice.Slice<uint>,Slice.Slice<uint16>>
val castUIntToInt16: Core.Pipe<Slice.Slice<uint>,Slice.Slice<int16>>
val castUIntToInt: Core.Pipe<Slice.Slice<uint>,Slice.Slice<int>>
val castUIntToUInt64: Core.Pipe<Slice.Slice<uint>,Slice.Slice<uint64>>
val castUIntToInt64: Core.Pipe<Slice.Slice<uint>,Slice.Slice<int64>>
val castUIntToFloat32: Core.Pipe<Slice.Slice<uint>,Slice.Slice<float32>>
val castUIntToFloat: Core.Pipe<Slice.Slice<uint>,Slice.Slice<float>>
val castIntToUInt8: Core.Pipe<Slice.Slice<int>,Slice.Slice<uint8>>
val castIntToInt8: Core.Pipe<Slice.Slice<int>,Slice.Slice<int8>>
val castIntToUInt16: Core.Pipe<Slice.Slice<int>,Slice.Slice<uint16>>
val castIntToInt16: Core.Pipe<Slice.Slice<int>,Slice.Slice<int16>>
val castIntToUInt: Core.Pipe<Slice.Slice<int>,Slice.Slice<uint>>
val castIntToUInt64: Core.Pipe<Slice.Slice<int>,Slice.Slice<uint64>>
val castIntToInt64: Core.Pipe<Slice.Slice<int>,Slice.Slice<int64>>
val castIntToFloat32: Core.Pipe<Slice.Slice<int>,Slice.Slice<float32>>
val castIntToFloat: Core.Pipe<Slice.Slice<int>,Slice.Slice<float>>
val castUInt64ToUInt8: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<uint8>>
val castUInt64ToInt8: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<int8>>
val castUInt64ToUInt16: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<uint16>>
val castUInt64ToInt16: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<int16>>
val castUInt64ToUInt: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<uint>>
val castUInt64ToInt: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<int>>
val castUInt64ToInt64: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<int64>>
val castUInt64ToFloat32: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<float32>>
val castUInt64ToFloat: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<float>>
val castInt64ToUInt8: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint8>>
val castInt64ToInt8: Core.Pipe<Slice.Slice<int64>,Slice.Slice<int8>>
val castInt64ToUInt16: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint16>>
val castInt64ToInt16: Core.Pipe<Slice.Slice<int64>,Slice.Slice<int16>>
val castInt64ToUInt: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint>>
val castInt64ToInt: Core.Pipe<Slice.Slice<int64>,Slice.Slice<int>>
val castInt64ToUInt64: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint64>>
val castInt64ToFloat32: Core.Pipe<Slice.Slice<int64>,Slice.Slice<float32>>
val castInt64ToFloat: Core.Pipe<Slice.Slice<int64>,Slice.Slice<float>>
val castFloat32ToUInt8: Core.Pipe<Slice.Slice<float32>,Slice.Slice<uint8>>
val castFloat32ToInt8: Core.Pipe<Slice.Slice<float32>,Slice.Slice<int8>>
val castFloat32ToUInt16: Core.Pipe<Slice.Slice<float32>,Slice.Slice<uint16>>
val castFloat32ToInt16: Core.Pipe<Slice.Slice<float32>,Slice.Slice<int16>>
val castFloat32ToUInt: Core.Pipe<Slice.Slice<float32>,Slice.Slice<uint>>
val castFloat32ToInt: Core.Pipe<Slice.Slice<float32>,Slice.Slice<int>>
val castFloat32ToUInt64: Core.Pipe<Slice.Slice<float32>,Slice.Slice<uint64>>
val castFloat32ToInt64: Core.Pipe<Slice.Slice<float32>,Slice.Slice<int64>>
val castFloat32ToFloat: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float>>
val castFloatToUInt8: Core.Pipe<Slice.Slice<float>,Slice.Slice<uint8>>
val castFloatToInt8: Core.Pipe<Slice.Slice<float>,Slice.Slice<int8>>
val castFloatToUInt16: Core.Pipe<Slice.Slice<float>,Slice.Slice<uint16>>
val castFloatToInt16: Core.Pipe<Slice.Slice<float>,Slice.Slice<int16>>
val castFloatToUInt: Core.Pipe<Slice.Slice<float>,Slice.Slice<uint>>
val castFloatToInt: Core.Pipe<Slice.Slice<float>,Slice.Slice<int>>
val castFloatToUIn64: Core.Pipe<Slice.Slice<float>,Slice.Slice<uint64>>
val castFloatToInt64: Core.Pipe<Slice.Slice<float>,Slice.Slice<int64>>
val castFloatToFloat32: Core.Pipe<Slice.Slice<float>,Slice.Slice<float32>>
val add:
  slice: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val inline scalarAddSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceAddScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val sub:
  slice: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val inline scalarSubSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceSubScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val mul:
  slice: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val inline scalarMulSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceMulScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val div:
  slice: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val inline scalarDivSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceDivScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val absProcess<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val sqrtProcess<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val logProcess<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val expProcess<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val histogram<'T when 'T: comparison> :
  Core.Pipe<Slice.Slice<'T>,Map<'T,uint64>> when 'T: comparison
val map2pairs<'T,'S when 'T: comparison> :
  Core.Pipe<Map<'T,'S>,('T * 'S) list> when 'T: comparison
val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  Core.Pipe<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2int<^T,^S
                       when ^T: (static member op_Explicit: ^T -> int) and
                            ^S: (static member op_Explicit: ^S -> int)> :
  Core.Pipe<(^T * ^S) list,(int * int) list>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
val addNormalNoise:
  mean: float -> stddev: float -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val threshold:
  lower: float -> upper: float -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
val convolve:
  kern: Slice.Slice<'T> ->
    boundaryCondition: Slice.BoundaryCondition option ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val conv:
  kern: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val convolveStreams:
  kernelSrc: Core.Pipe<'S,Slice.Slice<'T>> ->
    imageSrc: Core.Pipe<'S,Slice.Slice<'T>> -> Core.Pipe<'S,Slice.Slice<'T>>
    when 'T: equality
val discreteGaussian:
  sigma: float ->
    kernelSize: uint option ->
    boundaryCondition: Slice.BoundaryCondition option ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val convGauss:
  sigma: float ->
    boundaryCondition: Slice.BoundaryCondition option ->
    Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val skipFirstLast: n: int -> lst: 'a list -> 'a list
val private binaryMathMorph:
  name: string ->
    f: (uint -> Slice.Slice<uint8> -> Slice.Slice<uint8>) ->
    radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryErode:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryDilate:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryOpening:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryClosing:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val piecewiseConnectedComponents:
  windowSize: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint64>>
type ImageStats = Slice.ImageStats
val computeStats<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<'T>,ImageStats> when 'T: equality
val liftUnaryOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val liftWindowedOp:
  name: string ->
    window: uint ->
    stride: uint ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'S>,Slice.Slice<'T>>
    when 'S: equality and 'T: equality
val liftWindowedTrimOp:
  name: string ->
    window: uint ->
    stride: uint ->
    trim: uint ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'S>,Slice.Slice<'T>>
    when 'S: equality and 'T: equality
val liftUnaryOpInt:
  name: string ->
    f: (Slice.Slice<int> -> Slice.Slice<int>) ->
    Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val liftUnaryOpFloat32:
  name: string ->
    f: (Slice.Slice<float32> -> Slice.Slice<float32>) ->
    Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val liftUnaryOpFloat:
  name: string ->
    f: (Slice.Slice<float> -> Slice.Slice<float>) ->
    Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val liftBinaryOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<(Slice.Slice<'T> * Slice.Slice<'T>),Slice.Slice<'T>>
    when 'T: equality
val liftBinaryOpFloat:
  name: string ->
    f: (Slice.Slice<float> -> Slice.Slice<float> -> Slice.Slice<float>) ->
    Core.Operation<(Slice.Slice<float> * Slice.Slice<float>),Slice.Slice<float>>
val liftBinaryZipOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    p1: Core.Pipe<'In,Slice.Slice<'T>> ->
    p2: Core.Pipe<'In,Slice.Slice<'T>> -> Core.Pipe<'In,Slice.Slice<'T>>
    when 'T: equality
val liftFullOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val liftFullParamOp:
  name: string ->
    f: ('P -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    param: 'P -> Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val liftFullParam2Op:
  name: string ->
    f: ('P -> 'Q -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    param1: 'P -> param2: 'Q -> Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val liftMapOp:
  name: string ->
    f: (Slice.Slice<'T> -> 'U) -> Core.Operation<Slice.Slice<'T>,'U>
    when 'T: comparison
val absIntOp: name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val absFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val absFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val logFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val logFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val log10Float32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val log10FloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val expFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val expFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sqrtIntOp: name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val sqrtFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val sqrtFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val squareIntOp:
  name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val squareFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val squareFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sinFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val sinFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val cosFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val cosFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val tanFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val tanFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val asinFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val asinFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val acosFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val acosFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val atanFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val atanFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val roundFloat32Op:
  name: string -> Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val roundFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val roundFloatToUint: v: float -> uint
val discreteGaussianOp:
  name: string ->
    sigma: float ->
    bc: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val windowFromSlices:
  a: Slice.Slice<'T> -> b: Slice.Slice<'T> -> uint when 'T: equality
val windowFromKernel: k: Slice.Slice<'T> -> uint when 'T: equality
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
val addOpFloat:
  Core.Operation<(Slice.Slice<float> * Slice.Slice<float>),Slice.Slice<float>>
val sNotOp: name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
type FileInfo = Slice.FileInfo
val getStackDepth: (string -> string -> uint)
val getStackInfo: (string -> string -> Slice.FileInfo)
val getStackSize: (string -> string -> uint * uint * uint)
val getStackWidth: (string -> string -> uint64)
val getStackHeight: (string -> string -> uint64)
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
val histogramOp:
  name: string -> Core.Operation<Slice.Slice<'T>,Map<'T,uint64>>
    when 'T: comparison
val computeStatsOp:
  name: string -> Core.Operation<Slice.Slice<'T>,ImageStats> when 'T: comparison
val constantPad2DOp<'T when 'T: equality> :
  name: string ->
    padLower: uint list ->
    padUpper: uint list ->
    c: double -> Core.Operation<Slice.Slice<obj>,Slice.Slice<obj>>
    when 'T: equality
module Ops
val sqrtFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val absFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val logFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val log10Float: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val expFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val squareFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val sinFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val cosFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val tanFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val asinFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val acosFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val atanFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val roundFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val sqrtFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val absFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val logFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val log10Float32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val expFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val squareFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val sinFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val cosFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val tanFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val asinFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val acosFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val atanFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val roundFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val sqrtInt: Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val absInt: Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val squareInt: Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val discreteGaussian:
  sigma: float ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val convGauss: sigma: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val convolve:
  kernel: Slice.Slice<'a> ->
    bc: Slice.BoundaryCondition option ->
    winSz: uint option -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val conv:
  kernel: Slice.Slice<'a> -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
/// these only works on uint8
val binaryErode:
  r: uint ->
    winSz: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val erode: r: uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryDilate:
  r: uint ->
    winSz: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val dilate: r: uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryOpening:
  r: uint ->
    winSz: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val opening: r: uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryClosing:
  r: uint ->
    winSz: uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val closing: r: uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val binaryFillHoles: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val connectedComponents: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint64>>
val add:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val sub:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val mul:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val div:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val isGreaterEqual:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val isGreater:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val isEqual:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val isNotEqual:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val isLessThanEqual:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val isLessThan:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val sAnd:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val sOr:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val sXor:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val sNot: unit -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val pow:
  im1: Slice.Slice<'T> -> im2: Slice.Slice<'T> -> Slice.Slice<'T>
    when 'T: equality
val otsuThreshold<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<obj>,Slice.Slice<obj>> when 'T: equality
val otsuMultiThreshold:
  n: byte -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val momentsThreshold<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<obj>,Slice.Slice<obj>> when 'T: equality
val signedDistanceMap: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<float>>
val watershed:
  a: float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val threshold:
  a: float -> b: float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val addNormalNoise:
  a: float -> b: float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val relabelComponents:
  a: uint -> Core.Pipe<Slice.Slice<uint64>,Slice.Slice<uint64>>
val histPipe: Core.Pipe<Slice.Slice<float>,Map<float,uint64>>
type ImageStats = ImageFunctions.ImageStats
val computeStats<'T when 'T: comparison> :
  Core.Pipe<Slice.Slice<'T>,Processing.ImageStats> when 'T: comparison
val constantPad2D<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list ->
    c: double -> Core.Pipe<Slice.Slice<obj>,Slice.Slice<obj>> when 'T: equality
