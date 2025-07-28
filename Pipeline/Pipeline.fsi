namespace FSharp
module Pipeline
type Pipe<'S,'T> = Core.Pipe<'S,'T>
type Operation<'S,'T> = Core.Operation<'S,'T>
type MemoryProfile = Core.MemoryProfile
type MemoryTransition = Core.MemoryTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>
val idOp: (unit -> Core.Operation<'a,'a>)
val (-->) :
  (Core.Operation<'a,'b> -> Core.Operation<'b,'c> -> Core.Operation<'a,'c>)
val source: (uint64 -> Core.Pipeline<unit,unit>)
val sink: pl: Core.Pipeline<unit,unit> -> unit
val sinkList: plLst: Core.Pipeline<unit,unit> list -> unit
val (>=>) :
  (Core.Pipeline<'a,'b> -> Core.Operation<'b,'c> -> Core.Pipeline<'a,'c>)
val (>=>>) :
  (Core.Pipeline<'a,'b> ->
     Core.Operation<'b,'c> * Core.Operation<'b,'d> ->
       Routing.SharedPipeline<'a,'c,'d>)
val (>>=>>) :
  (Routing.SharedPipeline<'a,'b,'c> ->
     (Core.Pipeline<'a,'b> * Core.Pipeline<'a,'c> ->
        Core.Pipeline<'a,'b> * Core.Pipeline<'a,'c>) ->
     Routing.SharedPipeline<'a,'b,'c>)
val (>>=>) :
  (Routing.SharedPipeline<'a,'b,'c> ->
     (Core.Pipeline<'a,'b> * Core.Pipeline<'a,'c> -> Core.Pipeline<'a,'d>) ->
     Core.Pipeline<'a,'d>)
val unitPipeline: (unit -> Core.Pipeline<'a,unit>)
val combineIgnore:
  (Core.Pipeline<'a,'b> * Core.Pipeline<'a,'c> -> Core.Pipeline<'a,unit>)
val drainSingle: pl: Core.Pipeline<'a,'b> -> 'b
val drainList: pl: Core.Pipeline<'a,'b> -> 'b list
val drainLast: pl: Core.Pipeline<'a,'b> -> 'b
val tap: (string -> Core.Operation<'a,'a>)
val liftUnary:
  f: (Slice<'T> -> Slice<'T>) -> Core.Operation<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val create<'T when 'T: equality> :
  (uint ->
     uint ->
     uint -> Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<'T>>)
    when 'T: equality
val readAs<'T when 'T: equality> :
  (string ->
     string -> Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<'T>>)
    when 'T: equality
val readRandomAs<'T when 'T: equality> :
  (uint ->
     string ->
     string -> Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<'T>>)
    when 'T: equality
val write:
  (string -> string -> Core.Operation<Slice.Slice<'a>,unit>) when 'a: equality
val print: (unit -> Core.Operation<'a,unit>)
val plot:
  ((float list -> float list -> unit) ->
     Core.Operation<(float * float) list,unit>)
val show:
  ((Slice.Slice<'a> -> unit) -> Core.Operation<Slice.Slice<'a>,unit>)
    when 'a: equality
val finiteDiffFilter3D:
  (uint ->
     uint -> Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<float>>)
val axisSource:
  (int ->
     int list ->
     Core.Pipeline<unit,unit> -> Core.Pipeline<unit,Slice.Slice<uint>>)
val castUInt8ToInt8: Core.Operation<Slice.Slice<uint8>,Slice.Slice<int8>>
val castUInt8ToUInt16: Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint16>>
val castUInt8ToInt16: Core.Operation<Slice.Slice<uint8>,Slice.Slice<int16>>
val castUInt8ToUInt: Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint>>
val castUInt8ToInt: Core.Operation<Slice.Slice<uint8>,Slice.Slice<int>>
val castUInt8ToUInt64: Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint64>>
val castUInt8ToInt64: Core.Operation<Slice.Slice<uint8>,Slice.Slice<int64>>
val castUInt8ToFloat32: Core.Operation<Slice.Slice<uint8>,Slice.Slice<float32>>
val castUInt8ToFloat: Core.Operation<Slice.Slice<uint8>,Slice.Slice<float>>
val castInt8ToUInt8: Core.Operation<Slice.Slice<int8>,Slice.Slice<uint8>>
val castInt8ToUInt16: Core.Operation<Slice.Slice<int8>,Slice.Slice<uint16>>
val castInt8ToInt16: Core.Operation<Slice.Slice<int8>,Slice.Slice<int16>>
val castInt8ToUInt: Core.Operation<Slice.Slice<int8>,Slice.Slice<uint>>
val castInt8ToInt: Core.Operation<Slice.Slice<int8>,Slice.Slice<int>>
val castInt8ToUInt64: Core.Operation<Slice.Slice<int8>,Slice.Slice<uint64>>
val castInt8ToInt64: Core.Operation<Slice.Slice<int8>,Slice.Slice<int64>>
val castInt8ToFloat32: Core.Operation<Slice.Slice<int8>,Slice.Slice<float32>>
val castInt8ToFloat: Core.Operation<Slice.Slice<int8>,Slice.Slice<float>>
val castUInt16ToUInt8: Core.Operation<Slice.Slice<uint16>,Slice.Slice<uint8>>
val castUInt16ToInt8: Core.Operation<Slice.Slice<uint16>,Slice.Slice<int8>>
val castUInt16ToInt16: Core.Operation<Slice.Slice<uint16>,Slice.Slice<int16>>
val castUInt16ToUInt: Core.Operation<Slice.Slice<uint16>,Slice.Slice<uint>>
val castUInt16ToInt: Core.Operation<Slice.Slice<uint16>,Slice.Slice<int>>
val castUInt16ToUInt64: Core.Operation<Slice.Slice<uint16>,Slice.Slice<uint64>>
val castUInt16ToInt64: Core.Operation<Slice.Slice<uint16>,Slice.Slice<int64>>
val castUInt16ToFloat32:
  Core.Operation<Slice.Slice<uint16>,Slice.Slice<float32>>
val castUInt16ToFloat: Core.Operation<Slice.Slice<uint16>,Slice.Slice<float>>
val castInt16ToUInt8: Core.Operation<Slice.Slice<int16>,Slice.Slice<uint8>>
val castInt16ToInt8: Core.Operation<Slice.Slice<int16>,Slice.Slice<int8>>
val castInt16ToUInt16: Core.Operation<Slice.Slice<int16>,Slice.Slice<uint16>>
val castInt16ToUInt: Core.Operation<Slice.Slice<int16>,Slice.Slice<uint>>
val castInt16ToInt: Core.Operation<Slice.Slice<int16>,Slice.Slice<int>>
val castInt16ToUInt64: Core.Operation<Slice.Slice<int16>,Slice.Slice<uint64>>
val castInt16ToInt64: Core.Operation<Slice.Slice<int16>,Slice.Slice<int64>>
val castInt16ToFloat32: Core.Operation<Slice.Slice<int16>,Slice.Slice<float32>>
val castInt16ToFloat: Core.Operation<Slice.Slice<int16>,Slice.Slice<float>>
val castUIntToUInt8: Core.Operation<Slice.Slice<uint>,Slice.Slice<uint8>>
val castUIntToInt8: Core.Operation<Slice.Slice<uint>,Slice.Slice<int8>>
val castUIntToUInt16: Core.Operation<Slice.Slice<uint>,Slice.Slice<uint16>>
val castUIntToInt16: Core.Operation<Slice.Slice<uint>,Slice.Slice<int16>>
val castUIntToInt: Core.Operation<Slice.Slice<uint>,Slice.Slice<int>>
val castUIntToUInt64: Core.Operation<Slice.Slice<uint>,Slice.Slice<uint64>>
val castUIntToInt64: Core.Operation<Slice.Slice<uint>,Slice.Slice<int64>>
val castUIntToFloat32: Core.Operation<Slice.Slice<uint>,Slice.Slice<float32>>
val castUIntToFloat: Core.Operation<Slice.Slice<uint>,Slice.Slice<float>>
val castIntToUInt8: Core.Operation<Slice.Slice<int>,Slice.Slice<uint8>>
val castIntToInt8: Core.Operation<Slice.Slice<int>,Slice.Slice<int8>>
val castIntToUInt16: Core.Operation<Slice.Slice<int>,Slice.Slice<uint16>>
val castIntToInt16: Core.Operation<Slice.Slice<int>,Slice.Slice<int16>>
val castIntToUInt: Core.Operation<Slice.Slice<int>,Slice.Slice<uint>>
val castIntToUInt64: Core.Operation<Slice.Slice<int>,Slice.Slice<uint64>>
val castIntToInt64: Core.Operation<Slice.Slice<int>,Slice.Slice<int64>>
val castIntToFloat32: Core.Operation<Slice.Slice<int>,Slice.Slice<float32>>
val castIntToFloat: Core.Operation<Slice.Slice<int>,Slice.Slice<float>>
val castUInt64ToUInt8: Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint8>>
val castUInt64ToInt8: Core.Operation<Slice.Slice<uint64>,Slice.Slice<int8>>
val castUInt64ToUInt16: Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint16>>
val castUInt64ToInt16: Core.Operation<Slice.Slice<uint64>,Slice.Slice<int16>>
val castUInt64ToUInt: Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint>>
val castUInt64ToInt: Core.Operation<Slice.Slice<uint64>,Slice.Slice<int>>
val castUInt64ToInt64: Core.Operation<Slice.Slice<uint64>,Slice.Slice<int64>>
val castUInt64ToFloat32:
  Core.Operation<Slice.Slice<uint64>,Slice.Slice<float32>>
val castUInt64ToFloat: Core.Operation<Slice.Slice<uint64>,Slice.Slice<float>>
val castInt64ToUInt8: Core.Operation<Slice.Slice<int64>,Slice.Slice<uint8>>
val castInt64ToInt8: Core.Operation<Slice.Slice<int64>,Slice.Slice<int8>>
val castInt64ToUInt16: Core.Operation<Slice.Slice<int64>,Slice.Slice<uint16>>
val castInt64ToInt16: Core.Operation<Slice.Slice<int64>,Slice.Slice<int16>>
val castInt64ToUInt: Core.Operation<Slice.Slice<int64>,Slice.Slice<uint>>
val castInt64ToInt: Core.Operation<Slice.Slice<int64>,Slice.Slice<int>>
val castInt64ToUInt64: Core.Operation<Slice.Slice<int64>,Slice.Slice<uint64>>
val castInt64ToFloat32: Core.Operation<Slice.Slice<int64>,Slice.Slice<float32>>
val castInt64ToFloat: Core.Operation<Slice.Slice<int64>,Slice.Slice<float>>
val castFloat32ToUInt8: Core.Operation<Slice.Slice<float32>,Slice.Slice<uint8>>
val castFloat32ToInt8: Core.Operation<Slice.Slice<float32>,Slice.Slice<int8>>
val castFloat32ToUInt16:
  Core.Operation<Slice.Slice<float32>,Slice.Slice<uint16>>
val castFloat32ToInt16: Core.Operation<Slice.Slice<float32>,Slice.Slice<int16>>
val castFloat32ToUInt: Core.Operation<Slice.Slice<float32>,Slice.Slice<uint>>
val castFloat32ToInt: Core.Operation<Slice.Slice<float32>,Slice.Slice<int>>
val castFloat32ToUInt64:
  Core.Operation<Slice.Slice<float32>,Slice.Slice<uint64>>
val castFloat32ToInt64: Core.Operation<Slice.Slice<float32>,Slice.Slice<int64>>
val castFloat32ToFloat: Core.Operation<Slice.Slice<float32>,Slice.Slice<float>>
val castFloatToUInt8: Core.Operation<Slice.Slice<float>,Slice.Slice<uint8>>
val castFloatToInt8: Core.Operation<Slice.Slice<float>,Slice.Slice<int8>>
val castFloatToUInt16: Core.Operation<Slice.Slice<float>,Slice.Slice<uint16>>
val castFloatToInt16: Core.Operation<Slice.Slice<float>,Slice.Slice<int16>>
val castFloatToUInt: Core.Operation<Slice.Slice<float>,Slice.Slice<uint>>
val castFloatToInt: Core.Operation<Slice.Slice<float>,Slice.Slice<int>>
val castFloatToUIn64: Core.Operation<Slice.Slice<float>,Slice.Slice<uint64>>
val castFloatToInt64: Core.Operation<Slice.Slice<float>,Slice.Slice<int64>>
val castFloatToFloat32: Core.Operation<Slice.Slice<float>,Slice.Slice<float32>>
/// Basic arithmetic
val add:
  slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarAddSlice:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceAddScalar:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val sub:
  slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarSubSlice:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceSubScalar:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val mul:
  slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarMulSlice:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceMulScalar:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val div:
  slice: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val inline scalarDivSlice:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceDivScalar:
  i: ^T -> Core.Operation<Slice.Slice<^T>,Slice.Slice<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
/// Simple functions
val absFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val absFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val absInt: Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val acosFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val acosFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val asinFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val asinFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val atanFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val atanFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val cosFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val cosFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val expFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val expFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val log10Float: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val log10Float32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val logFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val logFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val roundFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val roundFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val sinFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sinFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val sqrtFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val sqrtFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val sqrtInt: Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val squareFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val squareFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val squareInt: Core.Operation<Slice.Slice<int>,Slice.Slice<int>>
val tanFloat: Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val tanFloat32: Core.Operation<Slice.Slice<float32>,Slice.Slice<float32>>
val histogram<'T when 'T: comparison> :
  Core.Operation<Slice.Slice<'T>,Map<'T,uint64>> when 'T: comparison
val inline map2pairs<^T,^S
                       when ^T: comparison and
                            ^T: (static member op_Explicit: ^T -> float) and
                            ^S: (static member op_Explicit: ^S -> float)> :
  Core.Operation<Map<^T,^S>,(^T * ^S) list>
    when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  Core.Operation<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2ints<^T,^S
                        when ^T: (static member op_Explicit: ^T -> int) and
                             ^S: (static member op_Explicit: ^S -> int)> :
  Core.Operation<(^T * ^S) list,(int * int) list>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
type ImageStats = ImageFunctions.ImageStats
val computeStats<'T when 'T: comparison> :
  Core.Operation<Slice.Slice<'T>,Processing.ImageStats> when 'T: comparison
/// Convolution like operators
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
val discreteGaussian:
  sigma: float ->
    bc: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val convGauss:
  sigma: float ->
    bc: ImageFunctions.BoundaryCondition option ->
    Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val convolve:
  kernel: Slice.Slice<'a> ->
    bc: Slice.BoundaryCondition option ->
    winSz: uint option -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val conv:
  kernel: Slice.Slice<'a> -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val convGaussOp:
  sigma: float ->
    bc: ImageFunctions.BoundaryCondition option ->
    Core.Operation<Slice.Slice<float>,Slice.Slice<float>>
val erode: r: uint -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val dilate: r: uint -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val opening: r: uint -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val closing: r: uint -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
/// Full stack operators
val binaryFillHoles: Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint8>>
val connectedComponents: Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint64>>
val piecewiseConnectedComponents:
  wz: uint option -> Core.Operation<Slice.Slice<uint8>,Slice.Slice<uint64>>
val otsuThreshold<'T when 'T: equality> :
  Core.Operation<Slice.Slice<obj>,Slice.Slice<obj>> when 'T: equality
val otsuMultiThreshold:
  n: byte -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val momentsThreshold<'T when 'T: equality> :
  Core.Operation<Slice.Slice<obj>,Slice.Slice<obj>> when 'T: equality
val signedDistanceMap: Core.Operation<Slice.Slice<uint8>,Slice.Slice<float>>
val watershed:
  a: float -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>> when 'a: equality
val threshold:
  a: float -> b: float -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val addNormalNoise:
  a: float -> b: float -> Core.Operation<Slice.Slice<'a>,Slice.Slice<'a>>
    when 'a: equality
val relabelComponents:
  a: uint -> Core.Operation<Slice.Slice<uint64>,Slice.Slice<uint64>>
val constantPad2D<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list ->
    c: double -> Core.Operation<Slice.Slice<obj>,Slice.Slice<obj>>
    when 'T: equality
type FileInfo = Slice.FileInfo
val getStackDepth: (string -> string -> uint)
val getStackHeight: (string -> string -> uint64)
val getStackInfo: (string -> string -> Slice.FileInfo)
val getStackSize: (string -> string -> uint * uint * uint)
val getStackWidth: (string -> string -> uint64)
