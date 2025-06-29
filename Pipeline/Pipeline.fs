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

let getStackSize = Slice.getStackSize
let readSlices = SourceSink.readSlices
let writeSlices = SourceSink.writeSlices
let source = Routing.source
let sink = Routing.sink
let (>=>) = Routing.composePipe

let zeroPad = Processing.zeroPad
let periodicPad = Processing.periodicPad
let zeroFluxNeumannPad = Processing.zeroFluxNeumannPad
let valid = Processing.valid
let same = Processing.same
let castFloatToUInt8 = Processing.castFloatToUInt8

// Re-export useful types and functions
(*
module Ops = 
    let sqrtFloat = Processing.Ops.sqrtFloat
    let discreteGaussian = Processing.Ops.discreteGaussian
*)

module Ops = Ops