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
/// Source parts
val create:
  width: uint -> height: uint -> depth: uint -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
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
/// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
val composePipe:
  p1: Core.Pipe<'S,'T> -> p2: Core.Pipe<'T,'U> -> Core.Pipe<'S,'U>
val (>=>) : p1: Core.Pipe<'a,'b> -> p2: Core.Pipe<'b,'c> -> Core.Pipe<'a,'c>
val tap: label: string -> Core.Pipe<'T,'T>
val validate: op1: Core.Operation<'a,'b> -> op2: Core.Operation<'c,'d> -> bool
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
val gauss:
  sigma: float -> kernelSize: uint option -> Core.Pipe<unit,Slice.Slice<float>>
val finiteDiffFilter1D: order: uint -> Core.Pipe<unit,Slice.Slice<float>>
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
val writeSlices:
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
val castUInt8ToFloat: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<float>>
val castFloatToUInt8: Core.Pipe<Slice.Slice<float>,Slice.Slice<uint8>>
val addFloat: value: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val addInt: value: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val addUInt8: value: uint8 -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val add:
  image: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val subFloat: value: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val subInt: value: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val sub:
  image: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val mulFloat: value: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val mulInt: value: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val mulUInt8: value: uint8 -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val mul:
  image: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val divFloat: value: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val divInt: value: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val div:
  image: Slice.Slice<'T> -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
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
    f: (uint -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val binaryErode:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val binaryDilate:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val binaryOpening:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val binaryClosing:
  radius: uint ->
    windowSize: uint option ->
    stride: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val piecewiseConnectedComponents:
  windowSize: uint option -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
type FileInfo = Slice.FileInfo
val getStackDepth: inputDir: string -> suffix: string -> uint
val getStackInfo: inputDir: string -> suffix: string -> FileInfo
val getStackSize: inputDir: string -> suffix: string -> uint64 list
val getStackWidth: inputDir: string -> suffix: string -> uint64
val getStackHeigth: inputDir: string -> suffix: string -> uint64
type ImageStats = Slice.ImageStats
val computeStats<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<'T>,ImageStats> when 'T: equality
val liftUnaryOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val liftImageScalarOpInt:
  name: string ->
    scalar: int ->
    core: (Slice.Slice<int> -> int -> Slice.Slice<int>) ->
    Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val liftImageScalarOpUInt8:
  name: string ->
    scalar: uint8 ->
    core: (Slice.Slice<uint8> -> uint8 -> Slice.Slice<uint8>) ->
    Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val liftImageScalarOpFloat32:
  name: string ->
    scalar: float32 ->
    core: (Slice.Slice<float32> -> float32 -> Slice.Slice<float32>) ->
    Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val liftImageScalarOpFloat:
  name: string ->
    scalar: float ->
    core: (Slice.Slice<float> -> float -> Slice.Slice<float>) ->
    Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
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
    winSz: uint option -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val binaryDilateOp:
  name: string ->
    radius: uint ->
    winSz: uint option -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val binaryOpeningOp:
  name: string ->
    radius: uint ->
    winSz: uint option -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val binaryClosingOp:
  name: string ->
    radius: uint ->
    winSz: uint option -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val addIntOp:
  name: string ->
    scalar: int -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val addUInt8Op:
  name: string ->
    scalar: uint8 -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val addFloatOp:
  name: string ->
    scalar: float -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val subIntOp:
  name: string ->
    scalar: int -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val subFloatOp:
  name: string ->
    scalar: float -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val mulIntOp:
  name: string ->
    scalar: int -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val mulUInt8Op:
  name: string ->
    scalar: uint8 -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val mulFloatOp:
  name: string ->
    scalar: float -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val divIntOp:
  name: string ->
    scalar: int -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val divFloatOp:
  name: string ->
    scalar: float -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val addOpFloat:
  Core.Operation<(Slice.Slice<float> * Slice.Slice<float>),Slice.Slice<float>>
val sNotOp: name: string -> Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
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
val binaryErode:
  r: uint -> winSz: uint option -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val erode:
  r: uint -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val binaryDilate:
  r: uint -> winSz: uint option -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val dilate:
  r: uint -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val binaryOpening:
  r: uint -> winSz: uint option -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val opening:
  r: uint -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val binaryClosing:
  r: uint -> winSz: uint option -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val closing:
  r: uint -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val addInt: scalar: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val addUInt8: scalar: uint8 -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val addFloat: scalar: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val subInt: scalar: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val subFloat: scalar: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val mulInt: scalar: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val mulUInt8: scalar: uint8 -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val mulFloat: scalar: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val divInt: scalar: int -> Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val divFloat: scalar: float -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
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
