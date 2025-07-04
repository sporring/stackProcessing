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
let tap = Routing.tap

let create = SourceSink.create
let read = SourceSink.read<'T>
let readRandom = SourceSink.readRandom<'T>
let write = SourceSink.write
let print<'T> : Pipe<'T, unit> = SourceSink.print
let plot = SourceSink.plot
let show = SourceSink.show

let finiteDiffFilter2D = SourceSink.finiteDiffFilter2D
let finiteDiffFilter3D = SourceSink.finiteDiffFilter3D
let gaussSource = SourceSink.gaussSource
let axisSource = SourceSink.axisSource

let castUInt8ToInt8 = Ops.castUInt8ToInt8
let castUInt8ToUInt16 = Ops.castUInt8ToUInt16
let castUInt8ToInt16 = Ops.castUInt8ToInt16
let castUInt8ToUInt = Ops.castUInt8ToUInt
let castUInt8ToInt = Ops.castUInt8ToInt
let castUInt8ToUInt64 = Ops.castUInt8ToUInt64
let castUInt8ToInt64 = Ops.castUInt8ToInt64
let castUInt8ToFloat32 = Ops.castUInt8ToFloat32
let castUInt8ToFloat = Ops.castUInt8ToFloat
let castInt8ToUInt8 = Ops.castInt8ToUInt8
let castInt8ToUInt16 = Ops.castInt8ToUInt16
let castInt8ToInt16 = Ops.castInt8ToInt16
let castInt8ToUInt = Ops.castInt8ToUInt
let castInt8ToInt = Ops.castInt8ToInt
let castInt8ToUInt64 = Ops.castInt8ToUInt64
let castInt8ToInt64 = Ops.castInt8ToInt64
let castInt8ToFloat32 = Ops.castInt8ToFloat32
let castInt8ToFloat = Ops.castInt8ToFloat

let castUInt16ToUInt8 = Ops.castUInt16ToUInt8
let castUInt16ToInt8 = Ops.castUInt16ToInt8
let castUInt16ToInt16 = Ops.castUInt16ToInt16
let castUInt16ToUInt = Ops.castUInt16ToUInt
let castUInt16ToInt = Ops.castUInt16ToInt
let castUInt16ToUInt64 = Ops.castUInt16ToUInt64
let castUInt16ToInt64 = Ops.castUInt16ToInt64
let castUInt16ToFloat32 = Ops.castUInt16ToFloat32
let castUInt16ToFloat = Ops.castUInt16ToFloat
let castInt16ToUInt8 = Ops.castInt16ToUInt8
let castInt16ToInt8 = Ops.castInt16ToInt8
let castInt16ToUInt16 = Ops.castInt16ToUInt16
let castInt16ToUInt = Ops.castInt16ToUInt
let castInt16ToInt = Ops.castInt16ToInt
let castInt16ToUInt64 = Ops.castInt16ToUInt64
let castInt16ToInt64 = Ops.castInt16ToInt64
let castInt16ToFloat32 = Ops.castInt16ToFloat32
let castInt16ToFloat = Ops.castInt16ToFloat

let castUIntToUInt8 = Ops.castUIntToUInt8
let castUIntToInt8 = Ops.castUIntToInt8
let castUIntToUInt16 = Ops.castUIntToUInt16
let castUIntToInt16 = Ops.castUIntToInt16
let castUIntToInt = Ops.castUIntToInt
let castUIntToUInt64 = Ops.castUIntToUInt64
let castUIntToInt64 = Ops.castUIntToInt64
let castUIntToFloat32 = Ops.castUIntToFloat32
let castUIntToFloat = Ops.castUIntToFloat
let castIntToUInt8 = Ops.castIntToUInt8
let castIntToInt8 = Ops.castIntToInt8
let castIntToUInt16 = Ops.castIntToUInt16
let castIntToInt16 = Ops.castIntToInt16
let castIntToUInt = Ops.castIntToUInt
let castIntToUInt64 = Ops.castIntToUInt64
let castIntToInt64 = Ops.castIntToInt64
let castIntToFloat32 = Ops.castIntToFloat32
let castIntToFloat = Ops.castIntToFloat

let castUInt64ToUInt8 = Ops.castUInt64ToUInt8
let castUInt64ToInt8 = Ops.castUInt64ToInt8
let castUInt64ToUInt16 = Ops.castUInt64ToUInt16
let castUInt64ToInt16 = Ops.castUInt64ToInt16
let castUInt64ToUInt = Ops.castUInt64ToUInt
let castUInt64ToInt = Ops.castUInt64ToInt
let castUInt64ToFloat32 = Ops.castUInt64ToFloat32
let castUInt64ToInt64 = Ops.castUInt64ToInt64
let castUInt64ToFloat = Ops.castUInt64ToFloat
let castInt64ToUInt8 = Ops.castInt64ToUInt8
let castInt64ToInt8 = Ops.castInt64ToInt8
let castInt64ToUInt16 = Ops.castInt64ToUInt16
let castInt64ToInt16 = Ops.castInt64ToInt16
let castInt64ToUInt = Ops.castInt64ToUInt
let castInt64ToInt = Ops.castInt64ToInt
let castInt64ToUInt64 = Ops.castInt64ToUInt64
let castInt64ToFloat32 = Ops.castFloatToFloat32
let castInt64ToIntFloat = Ops.castInt64ToFloat

let castFloat32ToUInt8 = Ops.castFloat32ToUInt8
let castFloat32ToInt8 = Ops.castFloat32ToInt8
let castFloat32ToUInt16 = Ops.castFloat32ToUInt16
let castFloat32ToInt16 = Ops.castFloat32ToInt16
let castFloat32ToUInt = Ops.castFloat32ToUInt
let castFloat32ToInt = Ops.castFloat32ToInt
let castFloat32ToUInt64 = Ops.castFloat32ToUInt64
let castFloat32ToInt64 = Ops.castFloat32ToInt64
let castFloat32ToFloat = Ops.castFloat32ToFloat
let castFloatToUInt8 = Ops.castFloatToUInt8
let castFloatToInt8 = Ops.castFloatToInt8
let castFloatToUInt16 = Ops.castFloatToUInt16
let castFloatToInt16 = Ops.castFloatToInt16
let castFloatToUInt = Ops.castFloatToUInt
let castFloatToInt = Ops.castFloatToInt
let castFloatToUIn64 = Ops.castFloatToUIn64
let castFloatToInt64 = Ops.castFloatToInt64
let castFloatToFloat32 = Ops.castFloatToFloat32

/// Basic arithmetic
let add = Ops.add
let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.scalarAddSlice i
let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.sliceAddScalar i

let sub = Ops.sub
let inline scalarSubSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.scalarSubSlice i
let inline sliceSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.sliceSubScalar i

let mul = Ops.mul
let inline scalarMulSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.scalarMulSlice i
let inline sliceMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.sliceMulScalar i

let div = Ops.div
let inline scalarDivSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.scalarDivSlice i
let inline sliceDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = Ops.sliceDivScalar i

/// Simple functions
let absFloat = Ops.absFloat
let absFloat32 = Ops.absFloat32
let absInt = Ops.absInt
let acosFloat = Ops.acosFloat
let acosFloat32 = Ops.acosFloat32
let asinFloat = Ops.asinFloat
let asinFloat32 = Ops.asinFloat32
let atanFloat = Ops.atanFloat
let atanFloat32 = Ops.atanFloat32
let cosFloat = Ops.cosFloat
let cosFloat32 = Ops.cosFloat32
let expFloat = Ops.expFloat
let expFloat32 = Ops.expFloat32
let log10Float = Ops.log10Float
let log10Float32 = Ops.log10Float32
let logFloat = Ops.logFloat
let logFloat32 = Ops.logFloat32
let roundFloat = Ops.roundFloat
let roundFloat32 = Ops.roundFloat32
let sinFloat = Ops.sinFloat
let sinFloat32 = Ops.sinFloat32
let sqrtFloat = Ops.sqrtFloat
let sqrtFloat32 = Ops.sqrtFloat32
let sqrtInt = Ops.sqrtInt
let squareFloat = Ops.squareFloat
let squareFloat32 = Ops.squareFloat32
let squareInt = Ops.squareInt
let tanFloat = Ops.tanFloat
let tanFloat32 = Ops.tanFloat32

let histogram<'T when 'T: comparison> = Ops.histogram<'T>
let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) >  = Ops.map2pairs<'T,'S>
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = Ops.pairs2floats<'T,'S>
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = Ops.pairs2ints<'T,'S>

type ImageStats = Ops.ImageStats
let computeStats<'T when 'T: equality and 'T: comparison> = Ops.computeStats

/// Convolution like operators
// Chained type definitions do expose the originals
let zeroPad = Ops.zeroPad
let periodicPad = Ops.periodicPad
let zeroFluxNeumannPad = Ops.zeroFluxNeumannPad
let valid = Ops.valid
let same = Ops.same

let discreteGaussian = Ops.discreteGaussian
let convGauss = Ops.convGauss
let convolve =  Ops.convolve
let conv = Ops.conv

let erode = Ops.erode
let dilate = Ops.dilate
let opening = Ops.opening
let closing = Ops.closing

/// Full stack operators
let binaryFillHoles = Ops.binaryFillHoles
let connectedComponents = Ops.connectedComponents
let piecewiseConnectedComponents = Ops.piecewiseConnectedComponents

// Annoying F# value restriction requires explicit types here, sigh
let otsuThreshold<'T when 'T: equality> = Ops.otsuThreshold
let otsuMultiThreshold = Ops.otsuMultiThreshold
let momentsThreshold<'T when 'T: equality> = Ops.momentsThreshold
let signedDistanceMap = Ops.signedDistanceMap
let watershed = Ops.watershed
let threshold = Ops.threshold
let addNormalNoise = Ops.addNormalNoise
let relabelComponents = Ops.relabelComponents

let constantPad2D = Ops.constantPad2D

// Not Pipes nor Operators
type FileInfo = Slice.FileInfo
let getStackDepth = Slice.getStackDepth
let getStackHeight = Slice.getStackHeight
let getStackInfo = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth

