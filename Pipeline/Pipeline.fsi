namespace FSharp
module Pipeline
type Pipe<'S,'T> = Core.Pipe<'S,'T>
type Operation<'S,'T> = Core.Operation<'S,'T>
type MemoryProfile = Core.MemoryProfile
type MemoryTransition = Core.MemoryTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>
val sourceOp:
  availableMemory: uint64 -> Core.Builder.Pipeline<unit,Slice<'T>>
    when 'T: equality
val source<'T> : (uint64 -> Core.Pipe<unit,'T> -> Core.Pipe<unit,'T>)
val sourceLst<'T> :
  (uint64 -> Core.Pipe<unit,'T> list -> Core.Pipe<unit,'T> list)
val sinkOp: pl: Core.Builder.Pipeline<unit,unit> -> unit
val sink: (Core.Pipe<unit,unit> -> unit)
val sinkLst: (Core.Pipe<unit,unit> list -> unit)
val (>=>) : (Core.Pipe<'a,'b> -> Core.Pipe<'b,'c> -> Core.Pipe<'a,'c>)
val (>>=>) :
  (Core.Builder.Pipeline<'a,'b> ->
     Core.Operation<'b,'c> -> Core.Builder.Pipeline<'a,'c>)
val tee: (Core.Pipe<'a,'b> -> Core.Pipe<'a,'b> * Core.Pipe<'a,'b>)
val zipWith:
  (('a -> 'b -> 'c) -> Core.Pipe<'d,'a> -> Core.Pipe<'d,'b> -> Core.Pipe<'d,'c>)
val cacheScalar: (string -> Core.Pipe<unit,'a> -> Core.Pipe<'b,'a>)
val tap: (string -> Core.Pipe<'a,'a>)
val create:
  (uint ->
     uint ->
     uint ->
     (Core.Pipe<unit,Slice.Slice<'a>> -> Core.Pipe<unit,Slice.Slice<'a>>) ->
     Core.Pipe<unit,Slice.Slice<'a>>) when 'a: equality
val readOp:
  inputDir: string ->
    suffix: string ->
    pl: Core.Builder.Pipeline<unit,Slice<'T>> ->
    Core.Builder.Pipeline<unit,Slice.Slice<'T>> when 'T: equality
val readRandom:
  (uint ->
     string ->
     string ->
     (Core.Pipe<unit,Slice.Slice<'T>> -> Core.Pipe<unit,Slice.Slice<'T>>) ->
     Core.Pipe<unit,Slice.Slice<'T>>) when 'T: equality
val writeOp:
  (string -> string -> Core.Operation<Slice.Slice<'a>,unit>) when 'a: equality
val write:
  (string -> string -> Core.Pipe<Slice.Slice<'a>,unit>) when 'a: equality
val print<'T> : Pipe<'T,unit>
val plot:
  ((float list -> float list -> unit) -> Core.Pipe<(float * float) list,unit>)
val show:
  ((Slice.Slice<'a> -> unit) -> Core.Pipe<Slice.Slice<'a>,unit>)
    when 'a: equality
val finiteDiffFilter2D:
  (uint ->
     uint ->
     (Core.Pipe<unit,Slice.Slice<float>> -> Core.Pipe<unit,Slice.Slice<float>>) ->
     Core.Pipe<unit,Slice.Slice<float>>)
val finiteDiffFilter3D:
  (uint ->
     uint ->
     (Core.Pipe<unit,Slice.Slice<float>> -> Core.Pipe<unit,Slice.Slice<float>>) ->
     Core.Pipe<unit,Slice.Slice<float>>)
val gaussSource:
  (float -> uint option -> (Core.Pipe<unit,Slice.Slice<float>> -> 'a) -> 'a)
val axisSource:
  (int -> int list -> (Core.Pipe<unit,Slice.Slice<uint>> -> 'a) -> 'a)
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
val castUInt64ToFloat32: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<float32>>
val castUInt64ToInt64: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<int64>>
val castUInt64ToFloat: Core.Pipe<Slice.Slice<uint64>,Slice.Slice<float>>
val castInt64ToUInt8: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint8>>
val castInt64ToInt8: Core.Pipe<Slice.Slice<int64>,Slice.Slice<int8>>
val castInt64ToUInt16: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint16>>
val castInt64ToInt16: Core.Pipe<Slice.Slice<int64>,Slice.Slice<int16>>
val castInt64ToUInt: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint>>
val castInt64ToInt: Core.Pipe<Slice.Slice<int64>,Slice.Slice<int>>
val castInt64ToUInt64: Core.Pipe<Slice.Slice<int64>,Slice.Slice<uint64>>
val castInt64ToFloat32: Core.Pipe<Slice.Slice<float>,Slice.Slice<float32>>
val castInt64ToIntFloat: Core.Pipe<Slice.Slice<int64>,Slice.Slice<float>>
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
val castFloatToUInt8Op: Core.Operation<Slice.Slice<float>,Slice.Slice<uint8>>
/// Basic arithmetic
val add:
  (Slice.Slice<'a> -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val inline scalarAddSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceAddScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val sub:
  (Slice.Slice<'a> -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val inline scalarSubSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceSubScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val mul:
  (Slice.Slice<'a> -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val inline scalarMulSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceMulScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val div:
  (Slice.Slice<'a> -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val inline scalarDivSlice:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceDivScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
/// Simple functions
val absFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val absFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val absInt: Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val acosFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val acosFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val asinFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val asinFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val atanFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val atanFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val cosFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val cosFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val expFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val expFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val log10Float: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val log10Float32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val logFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val logFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val roundFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val roundFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val sinFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val sinFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val sqrtFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val sqrtFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val sqrtInt: Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val squareFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val squareFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val squareInt: Core.Pipe<Slice.Slice<int>,Slice.Slice<int>>
val tanFloat: Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>
val tanFloat32: Core.Pipe<Slice.Slice<float32>,Slice.Slice<float32>>
val histogram<'T when 'T: comparison> :
  Core.Pipe<Slice.Slice<'T>,Map<'T,uint64>> when 'T: comparison
val inline map2pairs<^T,^S
                       when ^T: comparison and
                            ^T: (static member op_Explicit: ^T -> float) and
                            ^S: (static member op_Explicit: ^S -> float)> :
  Core.Pipe<Map<^T,^S>,(^T * ^S) list>
    when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  Core.Pipe<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2ints<^T,^S
                        when ^T: (static member op_Explicit: ^T -> int) and
                             ^S: (static member op_Explicit: ^S -> int)> :
  Core.Pipe<(^T * ^S) list,(int * int) list>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
type ImageStats = Ops.ImageStats
val computeStats<'T when 'T: comparison> :
  Core.Pipe<Slice.Slice<System.IComparable>,Processing.ImageStats>
    when 'T: comparison
/// Convolution like operators
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
val discreteGaussian:
  (float ->
     ImageFunctions.BoundaryCondition option ->
     uint option -> Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>)
val convGauss:
  (float ->
     ImageFunctions.BoundaryCondition option ->
     Core.Pipe<Slice.Slice<float>,Slice.Slice<float>>)
val convolve:
  (Slice.Slice<'a> ->
     Slice.BoundaryCondition option ->
     uint option -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val conv:
  (Slice.Slice<'a> -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val convGaussOp:
  (float ->
     ImageFunctions.BoundaryCondition option ->
     Core.Operation<Slice.Slice<float>,Slice.Slice<float>>)
val erode: (uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
val dilate: (uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
val opening: (uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
val closing: (uint -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
/// Full stack operators
val binaryFillHoles: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>
val connectedComponents: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint64>>
val piecewiseConnectedComponents:
  (uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint64>>)
val otsuThreshold<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<obj>,Slice.Slice<obj>> when 'T: equality
val otsuMultiThreshold:
  (byte -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>) when 'a: equality
val momentsThreshold<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<obj>,Slice.Slice<obj>> when 'T: equality
val signedDistanceMap: Core.Pipe<Slice.Slice<uint8>,Slice.Slice<float>>
val watershed:
  (float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>) when 'a: equality
val threshold:
  (float -> float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val addNormalNoise:
  (float -> float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val relabelComponents:
  (uint -> Core.Pipe<Slice.Slice<uint64>,Slice.Slice<uint64>>)
val constantPad2D:
  (uint list ->
     uint list -> double -> Core.Pipe<Slice.Slice<obj>,Slice.Slice<obj>>)
type FileInfo = Slice.FileInfo
val getStackDepth: (string -> string -> uint)
val getStackHeight: (string -> string -> uint64)
val getStackInfo: (string -> string -> Slice.FileInfo)
val getStackSize: (string -> string -> uint * uint * uint)
val getStackWidth: (string -> string -> uint64)
