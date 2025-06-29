namespace FSharp
module Pipeline
type Pipe<'S,'T> = Core.Pipe<'S,'T>
type Operation<'S,'T> = Core.Operation<'S,'T>
type MemoryProfile = Core.MemoryProfile
type MemoryTransition = Core.MemoryTransition
val getStackSize: (string -> string -> uint64 list)
val readSlices:
  (string -> string -> Core.Pipe<unit,Slice.Slice<'a>>) when 'a: equality
val writeSlices:
  (string -> string -> Core.Pipe<Slice.Slice<'a>,unit>) when 'a: equality
val source:
  (uint64 -> uint -> uint -> uint -> Core.Pipe<unit,'a> -> Core.Pipe<unit,'a>)
val sink: (Core.Pipe<unit,unit> -> unit)
val (>=>) : (Core.Pipe<'a,'b> -> Core.Pipe<'b,'c> -> Core.Pipe<'a,'c>)
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
val castFloatToUInt8: Core.Pipe<Slice.Slice<float>,Slice.Slice<uint8>>
