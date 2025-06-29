namespace FSharp
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
val liftWindowedOp:
  name: string ->
    window: uint ->
    stride: uint ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    Core.Operation<Slice.Slice<'S>,Slice.Slice<'T>>
    when 'S: equality and 'T: equality
val roundFloatToUint: v: float -> uint
val discreteGaussianOp:
  name: string ->
    sigma: float ->
    bc: ImageFunctions.BoundaryCondition option ->
    Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sqrtFloatOp:
  name: string -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
module Ops =
    val sqrtFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
    val discreteGaussian:
      sigma: float ->
        boundaryCondition: ImageFunctions.BoundaryCondition option ->
        Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
