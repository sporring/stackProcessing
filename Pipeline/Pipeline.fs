module Pipeline

// Core processing model
open Core

// Combinators and routing logic
open Routing

// Sources and sinks (file IO, streaming)
open SourceSink

// Common image operators
open Ops

// Image and slice types
open Slice
//open Image

// AsyncSeq helpers
//open AsyncSeqExtensions

type Pipe<'S,'T> = Core.Pipe<'S,'T>
type Operation<'S,'T> = Core.Operation<'S,'T>
type MemoryProfile = Core.MemoryProfile
type MemoryTransition = Core.MemoryTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>

let source<'T> = SourceSink.source<'T>
let sourceLst<'T> = SourceSink.sourceLst<'T>
let sink = SourceSink.sink
let sinkLst = SourceSink.sinkLst
let (>=>) = Core.composePipe
let tee = Routing.tee
let zipWith = Routing.zipWith
let cacheScalar = Routing.cacheScalar


let create = SourceSink.create
let read = SourceSink.read<'T>
let readRandom = SourceSink.readRandom<'T>
let write = SourceSink.write
let print<'T> : Pipe<'T, unit> = SourceSink.print
let plot = SourceSink.plot
let show = SourceSink.show

let getStackDepth = Slice.getStackDepth
let getStackHeight = Slice.getStackHeight
let getStackInfo = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth

let finiteDiffFilter2D = SourceSink.finiteDiffFilter2D
let finiteDiffFilter3D = SourceSink.finiteDiffFilter3D
let gauss = SourceSink.gauss
let axisSource = SourceSink.axisSource

let zeroPad = Processing.zeroPad
let periodicPad = Processing.periodicPad
let zeroFluxNeumannPad = Processing.zeroFluxNeumannPad
let valid = Processing.valid
let same = Processing.same

let castUInt8ToInt8 = Processing.castUInt8ToInt8
let castUInt8ToUInt16 = Processing.castUInt8ToUInt16
let castUInt8ToInt16 = Processing.castUInt8ToInt16
let castUInt8ToUInt = Processing.castUInt8ToUInt
let castUInt8ToInt = Processing.castUInt8ToInt
let castUInt8ToUInt64 = Processing.castUInt8ToUInt64
let castUInt8ToInt64 = Processing.castUInt8ToInt64
let castUInt8ToFloat32 = Processing.castUInt8ToFloat32
let castUInt8ToFloat = Processing.castUInt8ToFloat
let castInt8ToUInt8 = Processing.castInt8ToUInt8
let castInt8ToUInt16 = Processing.castInt8ToUInt16
let castInt8ToInt16 = Processing.castInt8ToInt16
let castInt8ToUInt = Processing.castInt8ToUInt
let castInt8ToInt = Processing.castInt8ToInt
let castInt8ToUInt64 = Processing.castInt8ToUInt64
let castInt8ToInt64 = Processing.castInt8ToInt64
let castInt8ToFloat32 = Processing.castInt8ToFloat32
let castInt8ToFloat = Processing.castInt8ToFloat

let castUInt16ToUInt8 = Processing.castUInt16ToUInt8
let castUInt16ToInt8 = Processing.castUInt16ToInt8
let castUInt16ToInt16 = Processing.castUInt16ToInt16
let castUInt16ToUInt = Processing.castUInt16ToUInt
let castUInt16ToInt = Processing.castUInt16ToInt
let castUInt16ToUInt64 = Processing.castUInt16ToUInt64
let castUInt16ToInt64 = Processing.castUInt16ToInt64
let castUInt16ToFloat32 = Processing.castUInt16ToFloat32
let castUInt16ToFloat = Processing.castUInt16ToFloat
let castInt16ToUInt8 = Processing.castInt16ToUInt8
let castInt16ToInt8 = Processing.castInt16ToInt8
let castInt16ToUInt16 = Processing.castInt16ToUInt16
let castInt16ToUInt = Processing.castInt16ToUInt
let castInt16ToInt = Processing.castInt16ToInt
let castInt16ToUInt64 = Processing.castInt16ToUInt64
let castInt16ToInt64 = Processing.castInt16ToInt64
let castInt16ToFloat32 = Processing.castInt16ToFloat32
let castInt16ToFloat = Processing.castInt16ToFloat

let castUIntToUInt8 = Processing.castUIntToUInt8
let castUIntToInt8 = Processing.castUIntToInt8
let castUIntToUInt16 = Processing.castUIntToUInt16
let castUIntToInt16 = Processing.castUIntToInt16
let castUIntToInt = Processing.castUIntToInt
let castUIntToUInt64 = Processing.castUIntToUInt64
let castUIntToInt64 = Processing.castUIntToInt64
let castUIntToFloat32 = Processing.castUIntToFloat32
let castUIntToFloat = Processing.castUIntToFloat
let castIntToUInt8 = Processing.castIntToUInt8
let castIntToInt8 = Processing.castIntToInt8
let castIntToUInt16 = Processing.castIntToUInt16
let castIntToInt16 = Processing.castIntToInt16
let castIntToUInt = Processing.castIntToUInt
let castIntToUInt64 = Processing.castIntToUInt64
let castIntToInt64 = Processing.castIntToInt64
let castIntToFloat32 = Processing.castIntToFloat32
let castIntToFloat = Processing.castIntToFloat

let castUInt64ToUInt8 = Processing.castUInt64ToUInt8
let castUInt64ToInt8 = Processing.castUInt64ToInt8
let castUInt64ToUInt16 = Processing.castUInt64ToUInt16
let castUInt64ToInt16 = Processing.castUInt64ToInt16
let castUInt64ToUInt = Processing.castUInt64ToUInt
let castUInt64ToInt = Processing.castUInt64ToInt
let castUInt64ToFloat32 = Processing.castUInt64ToFloat32
let castUInt64ToInt64 = Processing.castUInt64ToInt64
let castUInt64ToFloat = Processing.castUInt64ToFloat
let castInt64ToUInt8 = Processing.castInt64ToUInt8
let castInt64ToInt8 = Processing.castInt64ToInt8
let castInt64ToUInt16 = Processing.castInt64ToUInt16
let castInt64ToInt16 = Processing.castInt64ToInt16
let castInt64ToUInt = Processing.castInt64ToUInt
let castInt64ToInt = Processing.castInt64ToInt
let castInt64ToUInt64 = Processing.castInt64ToUInt64
let castInt64ToFloat32 = Processing.castFloatToFloat32
let castInt64ToIntFloat = Processing.castInt64ToFloat

let castFloat32ToUInt8 = Processing.castFloat32ToUInt8
let castFloat32ToInt8 = Processing.castFloat32ToInt8
let castFloat32ToUInt16 = Processing.castFloat32ToUInt16
let castFloat32ToInt16 = Processing.castFloat32ToInt16
let castFloat32ToUInt = Processing.castFloat32ToUInt
let castFloat32ToInt = Processing.castFloat32ToInt
let castFloat32ToUInt64 = Processing.castFloat32ToUInt64
let castFloat32ToInt64 = Processing.castFloat32ToInt64
let castFloat32ToFloat = Processing.castFloat32ToFloat
let castFloatToUInt8 = Processing.castFloatToUInt8
let castFloatToInt8 = Processing.castFloatToInt8
let castFloatToUInt16 = Processing.castFloatToUInt16
let castFloatToInt16 = Processing.castFloatToInt16
let castFloatToUInt = Processing.castFloatToUInt
let castFloatToInt = Processing.castFloatToInt
let castFloatToUIn64 = Processing.castFloatToUIn64
let castFloatToInt64 = Processing.castFloatToInt64
let castFloatToFloat32 = Processing.castFloatToFloat32

let addNormalNoise = Processing.addNormalNoise
let discreteGaussian = Processing.discreteGaussian
let convGauss = Processing.convGauss
let inline addScalar
        (i : ^T)                             // ← just the scalar
        : Core.Pipe<Slice.Slice< ^T >,
                    Slice.Slice< ^T >>
    when ^T : equality
    and  ^T : (static member op_Explicit : ^T -> float) =
    Processing.sliceAddScalar i
let inline subScalar
        (i : ^T)                             // ← just the scalar
        : Core.Pipe<Slice.Slice< ^T >,
                    Slice.Slice< ^T >>
    when ^T : equality
    and  ^T : (static member op_Explicit : ^T -> float) =
    Processing.sliceAddScalar i
let inline mulScalar
        (i : ^T)                             // ← just the scalar
        : Core.Pipe<Slice.Slice< ^T >,
                    Slice.Slice< ^T >>
    when ^T : equality
    and  ^T : (static member op_Explicit : ^T -> float) =
    Processing.sliceMulScalar i
let inline divScalar
        (i : ^T)                             // ← just the scalar
        : Core.Pipe<Slice.Slice< ^T >,
                    Slice.Slice< ^T >>
    when ^T : equality
    and  ^T : (static member op_Explicit : ^T -> float) =
    Processing.sliceDivScalar i

let add = Slice.add
let sub = Slice.sub
let mul = Slice.mul
let div = Slice.div
let computeStatistics<'T when 'T: comparison> :
  Core.Pipe<Slice.Slice<'T>,Processing.ImageStats> when 'T: comparison = 
  Processing.computeStats


let threshold = Processing.threshold
let binaryErode = Processing.binaryErode
let binaryDilate = Processing.binaryDilate
let binaryOpening = Processing.binaryOpening
let binaryClosing = Processing.binaryClosing
let piecewiseConnectedComponents = Processing.piecewiseConnectedComponents

let histogram<'T when 'T: comparison> : Core.Pipe<Slice.Slice<'T>,Map<'T,uint64>> when 'T: comparison = 
    Processing.histogram
let map2pairs<'T,'S when 'T: comparison> :
  Core.Pipe<Map<'T,'S>,('T * 'S) list> when 'T: comparison =
  Processing.map2pairs
let inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  Core.Pipe<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float) = 
        Processing.pairs2floats

module Ops = Ops
