namespace FSharp
module Processing
val private reduce:
  name: string ->
    profile: Core.MemoryProfile ->
    reducer: (FSharp.Control.AsyncSeq<'In> -> Async<'Out>) ->
    Core.Pipe<'In,'Out>
val private explodeSlice:
  slices: Slice.Slice<'T> -> FSharp.Control.AsyncSeq<Slice.Slice<'T>>
    when 'T: equality
val map:
  label: string ->
    profile: Core.MemoryProfile -> f: ('S -> 'T) -> Core.Pipe<'S,'T>
val mapWindowed:
  label: string ->
    depth: uint ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
val mapChunked:
  label: string ->
    chunkSize: uint ->
    baseIndex: uint ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
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
val create:
  width: uint -> height: uint -> depth: uint -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readSlices:
  inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readSliceN:
  idx: uint ->
    inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val readRandomSlices:
  count: uint ->
    inputDir: string -> suffix: string -> Core.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val addNormalNoise:
  mean: float -> stddev: float -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val threshold:
  lower: float -> upper: float -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>>
    when 'T: equality
val discreteGaussian:
  sigma: float -> Core.Pipe<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality
type FileInfo = Slice.FileInfo
val getFileInfo: fname: string -> FileInfo
val getVolumeSize: inputDir: string -> suffix: string -> uint * uint * uint
type ImageStats = Slice.ImageStats
val computeStats<'T when 'T: equality> :
  Core.Pipe<Slice.Slice<'T>,ImageStats> when 'T: equality
