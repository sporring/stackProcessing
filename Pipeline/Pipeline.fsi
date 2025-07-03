namespace FSharp
module Pipeline
type Pipe<'S,'T> = Core.Pipe<'S,'T>
type Operation<'S,'T> = Core.Operation<'S,'T>
type MemoryProfile = Core.MemoryProfile
type MemoryTransition = Core.MemoryTransition
val source<'T> :
  (uint64 -> uint -> uint -> uint -> Core.Pipe<unit,'T> -> Core.Pipe<unit,'T>)
val sink: (Core.Pipe<unit,unit> -> unit)
val sinkLst: (Core.Pipe<unit,unit> list -> unit)
val (>=>) : (Core.Pipe<'a,'b> -> Core.Pipe<'b,'c> -> Core.Pipe<'a,'c>)
val tee: (Core.Pipe<'a,'b> -> Core.Pipe<'a,'b> * Core.Pipe<'a,'b>)
val zipWith:
  (('a -> 'b -> 'c) -> Core.Pipe<'d,'a> -> Core.Pipe<'d,'b> -> Core.Pipe<'d,'c>)
val cacheScalar: (string -> Core.Pipe<unit,'a> -> Core.Pipe<'b,'a>)
val create:
  (uint ->
     uint ->
     uint ->
     (Core.Pipe<unit,Slice.Slice<'a>> -> Core.Pipe<unit,Slice.Slice<'a>>) ->
     Core.Pipe<unit,Slice.Slice<'a>>) when 'a: equality
val read:
  (string ->
     string ->
     (Core.Pipe<unit,Slice.Slice<'T>> -> Core.Pipe<unit,Slice.Slice<'T>>) ->
     Core.Pipe<unit,Slice.Slice<'T>>) when 'T: equality
val readRandom:
  (uint ->
     string ->
     string ->
     (Core.Pipe<unit,Slice.Slice<'T>> -> Core.Pipe<unit,Slice.Slice<'T>>) ->
     Core.Pipe<unit,Slice.Slice<'T>>) when 'T: equality
val write:
  (string -> string -> Core.Pipe<Slice.Slice<'a>,unit>) when 'a: equality
val print<'T> : Pipe<'T,unit>
val plot:
  ((float list -> float list -> unit) -> Core.Pipe<(float * float) list,unit>)
val show:
  ((Slice.Slice<'a> -> unit) -> Core.Pipe<Slice.Slice<'a>,unit>)
    when 'a: equality
val getStackDepth: (string -> string -> uint)
val getStackHeight: (string -> string -> uint64)
val getStackInfo: (string -> string -> Slice.FileInfo)
val getStackSize: (string -> string -> uint * uint * uint)
val getStackWidth: (string -> string -> uint64)
val finiteDiffFilter2D: (uint -> uint -> Core.Pipe<unit,Slice.Slice<float>>)
val finiteDiffFilter3D: (uint -> uint -> Core.Pipe<unit,Slice.Slice<float>>)
val gauss: (float -> uint option -> Core.Pipe<unit,Slice.Slice<float>>)
val axisSource: (int -> int list -> Core.Pipe<unit,Slice.Slice<uint>>)
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
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
val addNormalNoise:
  (float -> float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val discreteGaussian:
  (float ->
     uint option ->
     Slice.BoundaryCondition option ->
     uint option -> uint option -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val convGauss:
  (float ->
     Slice.BoundaryCondition option ->
     Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>) when 'a: equality
val inline addScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline subScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline mulScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline divScalar:
  i: ^T -> Core.Pipe<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val add:
  (Slice.Slice<'a> -> Slice.Slice<'a> -> Slice.Slice<'a>) when 'a: equality
val sub:
  (Slice.Slice<'a> -> Slice.Slice<'a> -> Slice.Slice<'a>) when 'a: equality
val mul:
  (Slice.Slice<'a> -> Slice.Slice<'a> -> Slice.Slice<'a>) when 'a: equality
val div:
  (Slice.Slice<'a> -> Slice.Slice<'a> -> Slice.Slice<'a>) when 'a: equality
val computeStatistics<'T when 'T: comparison> :
  Core.Pipe<Slice.Slice<'T>,Processing.ImageStats> when 'T: comparison
val threshold:
  (float -> float -> Core.Pipe<Slice.Slice<'a>,Slice.Slice<'a>>)
    when 'a: equality
val binaryErode:
  (uint ->
     uint option ->
     uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
val binaryDilate:
  (uint ->
     uint option ->
     uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
val binaryOpening:
  (uint ->
     uint option ->
     uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
val binaryClosing:
  (uint ->
     uint option ->
     uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint8>>)
val piecewiseConnectedComponents:
  (uint option -> Core.Pipe<Slice.Slice<uint8>,Slice.Slice<uint64>>)
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
