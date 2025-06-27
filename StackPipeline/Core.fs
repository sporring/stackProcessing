module Core

open FSharp.Control
open AsyncSeqExtensions

/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant // Slice by slice independently
    | Streaming // Slice by slice independently
    | Sliding of uint // Sliding window of slices of depth
    | Buffered // All slices of depth

    member this.EstimateUsage (width: uint) (height: uint) (depth: uint) : uint64 =
        let pixelSize = 1UL // Assume 1 byte per pixel for UInt8
        let sliceBytes = (uint64 width) * (uint64 height) * pixelSize
        match this with
            | Constant -> 0uL
            | Streaming -> sliceBytes
            | Sliding windowSize -> sliceBytes * uint64 windowSize
            | Buffered -> sliceBytes * uint64 depth

    member this.RequiresBuffering (availableMemory: uint64) (width: uint) (height: uint) (depth: uint) : bool =
        let usage = this.EstimateUsage width height depth
        usage > availableMemory
    member this.combineProfile (other: MemoryProfile): MemoryProfile  = 
        match this, other with
        | Buffered, _ 
        | _, Buffered -> Buffered // conservative fallback
        | Sliding sz1, Sliding sz2 -> Sliding (max sz1 sz2)
        | Sliding sz, _ 
        | _, Sliding sz -> Sliding sz
        | Streaming, _
        | _, Streaming -> Streaming
        | Constant, Constant -> Constant

/// A configurable image processing step that operates on image slices.
type Pipe<'S,'T> = {
    Name: string // Name of the process
    Profile: MemoryProfile
    Apply: AsyncSeq<'S> -> AsyncSeq<'T>
}

let internal lift
    (name: string)
    (profile: MemoryProfile)
    (f: 'In -> Async<'Out>)
    : Pipe<'In, 'Out> =
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            input |> AsyncSeq.mapAsync f
    }

let internal reduce (name: string) (profile: MemoryProfile) (reducer: AsyncSeq<'In> -> Async<'Out>) : Pipe<'In, 'Out> =
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            reducer input |> ofAsync
    }

let internal consumeWith
        (name    : string)
        (profile : MemoryProfile)
        (consume : AsyncSeq<'T> -> Async<unit>)
        : Pipe<'T, unit> =

    let reducer (s : AsyncSeq<'T>) = consume s          // Async<unit>
    reduce name profile reducer                    // gives AsyncSeq<unit>

/// Pipeline computation expression
type PipelineBuilder(availableMemory: uint64, width: uint, height: uint, depth: uint) =
    /// Chain two <c>Pipe</c> instances, optionally inserting intermediate disk I/O
    member _.Bind(p: Pipe<'S,'T>, f: Pipe<'S,'T> -> Pipe<'S,'T>) : Pipe<'S,'T> =
        let composed = f p
        (*
        let combinedProfile = composed.Profile
        if combinedProfile.RequiresBuffering availableMemory  width  height depth then
            printfn "[Memory] Exceeded memory limits. Splitting pipeline."
            let tempDir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName())
            Directory.CreateDirectory(tempDir) |> ignore
            let intermediate = fun input ->
                // Step 1: Write intermediate slices to disk
                writeSlicesAsync tempDir ".tif" input
                |> Async.RunSynchronously
                // Step 2: Read them back for next stage
                readSlices tempDir ".tif"

            { Name = $"{composed.Name} {p.Name}"; Profile = Streaming; Apply = composed.Apply << intermediate } // The profile needs to be reset here. How to do that?
        else *)
        composed

    member _.Bind(p: Pipe<'S, 'T>, f: 'T -> Pipe<'S, 'U>) : Pipe<'S, 'U> =
        {
            Name = $"bindResult({p.Name})"
            Profile = p.Profile // You could also recompute based on `f` if needed
            Apply = fun input ->
                asyncSeq {
                    let! result = p.Apply input |> AsyncSeq.toListAsync
                    match result with
                    | [single] ->
                        let nextProcessor = f single
                        yield! nextProcessor.Apply input
                    | _ ->
                        failwithf "Expected reducer output to produce a single result, but got %d values." result.Length
                }
        }

    /// Wraps a processor value for use in the pipeline computation expression.
    member _.Return(p: Pipe<'S,'T>) = p

    /// Allows returning a processor directly from another computation expression.
    member _.ReturnFrom(p: Pipe<'S,'T>) = p

    /// Provides a default identity processor using streaming as the memory profile.
    member _.Zero() = { Name=""; Profile = Streaming; Apply = id }

/// A memory-aware pipeline builder with the specified processing constraints.
let pipeline availableMemory width height depth = PipelineBuilder(availableMemory, width, height, depth)

module Helpers =
    /// Pipeline helper functions
    let singleton (x: 'In) : Pipe<'In, 'In> =
        {
            Name = "[singleton]"
            Profile = Constant
            Apply = fun _ -> AsyncSeq.singleton x
        }
