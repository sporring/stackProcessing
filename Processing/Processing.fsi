namespace FSharp
module Processing
val private unstack:
  slices: Slice.Slice<'T> -> FSharp.Control.AsyncSeq<Slice.Slice<'T>>
    when 'T: equality
val private fromReducer:
  name: string ->
    profile: StackPipeline.MemoryProfile ->
    reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
    StackPipeline.StackProcessor<'In,'Out>
val mapSlices:
  label: string ->
    profile: StackPipeline.MemoryProfile ->
    f: ('S -> 'T) -> StackPipeline.StackProcessor<'S,'T>
val mapSlicesWindowed:
  label: string ->
    depth: uint ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val mapSlicesChunked:
  label: string ->
    chunkSize: uint ->
    baseIndex: uint ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val addFloat:
  value: float ->
    StackPipeline.StackProcessor<Slice.Slice<float>,Slice.Slice<float>>
val addInt:
  value: int -> StackPipeline.StackProcessor<Slice.Slice<int>,Slice.Slice<int>>
val addUInt8:
  value: uint8 ->
    StackPipeline.StackProcessor<Slice.Slice<uint8>,Slice.Slice<uint8>>
val add:
  image: Slice.Slice<'T> ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val subFloat:
  value: float ->
    StackPipeline.StackProcessor<Slice.Slice<float>,Slice.Slice<float>>
val subInt:
  value: int -> StackPipeline.StackProcessor<Slice.Slice<int>,Slice.Slice<int>>
val sub:
  image: Slice.Slice<'T> ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val mulFloat:
  value: float ->
    StackPipeline.StackProcessor<Slice.Slice<float>,Slice.Slice<float>>
val mulInt:
  value: int -> StackPipeline.StackProcessor<Slice.Slice<int>,Slice.Slice<int>>
val mulUInt8:
  value: uint8 ->
    StackPipeline.StackProcessor<Slice.Slice<uint8>,Slice.Slice<uint8>>
val mul:
  image: Slice.Slice<'T> ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val divFloat:
  value: float ->
    StackPipeline.StackProcessor<Slice.Slice<float>,Slice.Slice<float>>
val divInt:
  value: int -> StackPipeline.StackProcessor<Slice.Slice<int>,Slice.Slice<int>>
val div:
  image: Slice.Slice<'T> ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val absProcess<'T when 'T: equality> :
  StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val sqrtProcess<'T when 'T: equality> :
  StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val logProcess<'T when 'T: equality> :
  StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val expProcess<'T when 'T: equality> :
  StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val histogram<'T when 'T: comparison> :
  StackPipeline.StackProcessor<Slice.Slice<'T>,Map<'T,uint64>>
    when 'T: comparison
val map2pairs<'T,'S when 'T: comparison> :
  StackPipeline.StackProcessor<Map<'T,'S>,('T * 'S) list> when 'T: comparison
val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  StackPipeline.StackProcessor<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2int<^T,^S
                       when ^T: (static member op_Explicit: ^T -> int) and
                            ^S: (static member op_Explicit: ^S -> int)> :
  StackPipeline.StackProcessor<(^T * ^S) list,(int * int) list>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
val create:
  width: uint ->
    height: uint ->
    depth: uint -> StackPipeline.StackProcessor<unit,Slice.Slice<'T>>
    when 'T: equality
val readSlices:
  inputDir: string ->
    suffix: string -> StackPipeline.StackProcessor<unit,Slice.Slice<'T>>
    when 'T: equality
val readSliceN:
  idx: uint ->
    inputDir: string ->
    suffix: string -> StackPipeline.StackProcessor<unit,Slice.Slice<'T>>
    when 'T: equality
val readRandomSlices:
  count: uint ->
    inputDir: string ->
    suffix: string -> StackPipeline.StackProcessor<unit,Slice.Slice<'T>>
    when 'T: equality
val addNormalNoise:
  mean: float ->
    stddev: float ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val threshold:
  lower: float ->
    upper: float ->
    StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val discreteGaussian:
  sigma: float -> StackPipeline.StackProcessor<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
type FileInfo = Slice.FileInfo
val getFileInfo: fname: string -> FileInfo
val getVolumeSize: inputDir: string -> suffix: string -> uint * uint * uint
type ImageStats = Slice.ImageStats
val computeStats<'T when 'T: equality> :
  StackPipeline.StackProcessor<Slice.Slice<'T>,ImageStats> when 'T: equality
