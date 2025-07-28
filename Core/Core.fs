module Core

open FSharp.Control
open AsyncSeqExtensions
open Slice

////////////////////////////////////////////////////////////
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant // Slice by slice independently
    | Streaming // Slice by slice independently
    | Sliding of uint // Sliding window of slices of depth
    | Full // All slices of depth

    member this.EstimateUsage (width: uint) (height: uint) (depth: uint) : uint64 =
        let pixelSize = 1UL // Assume 1 byte per pixel for UInt8
        let sliceBytes = (uint64 width) * (uint64 height) * pixelSize
        match this with
            | Constant -> 0uL
            | Streaming -> sliceBytes
            | Sliding windowSize -> sliceBytes * uint64 windowSize
            | Full -> sliceBytes * uint64 depth

    member this.RequiresBuffering (availableMemory: uint64) (width: uint) (height: uint) (depth: uint) : bool = // Not used yet...
        let usage = this.EstimateUsage width height depth
        usage > availableMemory

    member this.combineProfile (other: MemoryProfile): MemoryProfile  = 
        match this, other with
        | Full, _ 
        | _, Full -> Full // conservative fallback
        | Sliding sz1, Sliding sz2 -> Sliding (max sz1 sz2)
        | Sliding sz, _ 
        | _, Sliding sz -> Sliding sz
        | Streaming, _
        | _, Streaming -> Streaming
        | Constant, Constant -> Constant

////////////////////////////////////////////////////////////
/// A configurable image processing step that operates on image slices.
/// Pipe describes *how* to do it:
/// - Encapsulates the concrete execution logic
/// - Defines memory usage behavior
/// - Takes and returns AsyncSeq streams
/// - Pipe + WindowedProcessor: How it’s computed 
type Pipe<'S,'T> = {
    Name: string // Name of the process
    Profile: MemoryProfile
    Apply: AsyncSeq<'S> -> AsyncSeq<'T>
}

let internal run (p: Pipe<'S,'T>) : AsyncSeq<'T> =
    printfn "[run]"
    AsyncSeq.singleton Unchecked.defaultof<'S> |> p.Apply

let internal lift (name: string) (profile: MemoryProfile) (f: 'In -> Async<'Out>) : Pipe<'In, 'Out> =
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            input |> AsyncSeq.mapAsync f
    }

let map (label: string) (profile: MemoryProfile) (f: 'S -> 'T) : Pipe<'S,'T> =
    {
        Name = label; 
        Profile = profile
        Apply = fun input -> input |> AsyncSeq.map f
    }

let internal reduce (name: string) (profile: MemoryProfile) (reducer: AsyncSeq<'In> -> Async<'Out>) : Pipe<'In, 'Out> =
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            reducer input |> ofAsync
    }

let fold (label: string) (profile: MemoryProfile)  (folder: 'State -> 'In -> 'State) (state0: 'State) 
    : Pipe<'In, 'State> =
    reduce label profile (fun stream ->
        async {
            let! result = stream |> AsyncSeq.fold folder state0
            return result
        })

let internal consumeWith (name    : string) (profile : MemoryProfile) (consume : AsyncSeq<'T> -> Async<unit>) : Pipe<'T, unit> =

    let reducer (s : AsyncSeq<'T>) = consume s          // Async<unit>
    reduce name profile reducer                    // gives AsyncSeq<unit>

/// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
let composePipe (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
    printfn "[composePipe]"
    {
        Name = $"{p2.Name} {p1.Name}"; 
        Profile = p1.Profile.combineProfile p2.Profile
        Apply = fun input -> input |> p1.Apply |> p2.Apply
    }


////////////////////////////////////////////////////////////
// Operation between pipes
/// SliceShape describes the dimensions of a stacked slice.
/// Conventionally: [width; height; depth]
/// Used for validating if transitions are feasible (e.g., sliding window depth fits).
type SliceShape = uint list

/// MemoryTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
/// - Check: a predicate on slice shape to validate if the transition is allowed
type MemoryTransition =
    { From  : MemoryProfile
      To    : MemoryProfile
      Check : SliceShape -> bool }

/// Operation describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Operation + MemoryTransition: what happens
type Operation<'S,'T> =
    { Name       : string
      Transition : MemoryTransition
      Pipe       : Pipe<'S,'T> } 

let defaultCheck _ = true
let transition (fromProfile: MemoryProfile) (toProfile: MemoryProfile) : MemoryTransition =
    {
        From = fromProfile
        To   = toProfile
        Check = defaultCheck
    }

let idOp<'T> () : Operation<'T, 'T> =
    {
        Name = "id"
        Transition = transition Streaming Streaming
        Pipe = lift "id" Streaming (fun x -> async.Return x)
    }

let inline asPipe (op : Operation<_,_>) = op.Pipe

let pipe2Operation name transition pipe : Operation<'S, 'T> =
    { Name = name; Transition = transition; Pipe = pipe }

////////////////////////////////////////////////////////////
// MemFlow state monad
type MemFlow<'S,'T> =
        uint64 -> SliceShape option -> Operation<'S,'T> * uint64 * SliceShape option

// How many bytes does this pipe claim for one full slice?
let private memNeed (shape: uint list) (p : Pipe<_,_>) =
    p.Profile.EstimateUsage shape[0] shape[1] shape[2]

/// Try to shrink a too‑hungry pipe to a cheaper profile.
/// *You* control the downgrade policy here.
let private shrinkProfile avail shape (p : Pipe<'S,'T>) =
    let needed = memNeed shape p
    if needed <= avail then p                              // fits as is
    else
        match p.Profile with
        | Full         -> { p with Profile = Streaming }
        | Sliding w when w > 1u ->
            { p with Profile = Sliding (w/2u) }
        | _ ->
            failwith
                $"Stage “{p.Name}” needs {needed}s B but only {avail}s B are left."

let returnM (op : Operation<'S,'T>) : MemFlow<'S,'T> =
    fun bytes shapeOpt ->
        match shapeOpt with
        | None      -> op, bytes, shapeOpt       // can’t check memory yet
        | Some sh   ->
            let p'     = shrinkProfile bytes sh op.Pipe
            let need   = memNeed sh p'
            { op with Pipe = p' }, bytes - need, shapeOpt

let composeOp (op1 : Operation<'S,'T>) (op2 : Operation<'T,'U>) : Operation<'S,'U> =
    {
        Name       = $"{op2.Name} ∘ {op1.Name}"
        Transition = transition op1.Transition.From op2.Transition.To
        Pipe       = composePipe (asPipe op1) (asPipe op2)
    }

let bindM m k =
    fun bytes shapeOpt ->
        let op1, bytes', shapeOpt' = m bytes shapeOpt
        let m2 = k op1
        let op2, bytes'', shapeOpt'' = m2 bytes' shapeOpt'

        (* 2025/07/25 Is this step necessary? The processing is taking care of the interfacing...
        // validate memory transition, if shape is known
        shapeOpt
        |> Option.iter (fun shape ->
            if op1.Transition.To <> op2.Transition.From || not (op2.Transition.Check shape) then 
                failwith $"Invalid memory transition: {op1} → {op2}")
        *)
        composeOp op1 op2, bytes'', shapeOpt''

let (-->) (op1: Operation<'A, 'B>) (op2: Operation<'B, 'C>) : Operation<'A, 'C> =
    composeOp op1 op2

////////////////////////////////////////////////////////////
// Pipeline flow controler
type Pipeline<'S,'T> = { 
    flow  : MemFlow<'S,'T>
    mem   : uint64
    shape : SliceShape option }

let operation2Pipeline op mem shape : Pipeline<'S, 'T> =
    { flow = returnM op; mem = mem; shape = shape }

// source: only memory budget, shape = None
let sourceOp (availableMemory: uint64) : Pipeline<unit,unit> =
    { 
        flow  = fun _ _ -> failwith "pipeline not started yet"
        mem   = availableMemory
        shape = None 
    }

// attach the first stage that *discovers* the shape
let attachFirst ((op, probeShape) : Operation<unit,'T> * (unit -> SliceShape))
                (pl : Pipeline<unit,'T>) =
    let shape = probeShape ()
    { 
        flow  = returnM op
        mem   = pl.mem
        shape = Some shape 
    }

// later compositions Pipeline composition
let (>=>) (pl: Pipeline<'a,'b>) (next: Operation<'b,'c>) : Pipeline<'a,'c> =
    {
        mem = pl.mem
        shape = pl.shape
        flow = bindM pl.flow (fun _op -> returnM next)
    }

// sink
let sinkOp (pl: Pipeline<unit, unit>) : unit =
    match pl.shape with
    | None -> failwith "No stage provided slice dimensions."

    | Some sh ->
        let op, rest, _ = pl.flow pl.mem pl.shape
        printfn $"Pipeline built - {rest} B still free."

        let pipe = asPipe op
        // Dummy source: feed a single unit input (can change later)
        ()
        |> AsyncSeq.singleton
        |> pipe.Apply
        |> AsyncSeq.iterAsync (fun () -> async.Return())
        |> Async.RunSynchronously

(*
let sinkListOp (plList: Pipeline<unit, unit> list) =
    if plList.Length > 1 then
        printfn "[Compile time analysis: sinkList parallel]"

    plList
    |> List.map sinkOp
    |> ignore    
*)
let sinkListOp (pipelines: Pipeline<unit, unit> list) : unit =
    if pipelines.Length > 1 then
        printfn "Compile time analysis: sinkList parallel"

    let runs =
        pipelines
        |> List.map (fun pl ->
            async {
                let op, _, _ = pl.flow pl.mem pl.shape
                let pipe = op.Pipe
                do!
                    AsyncSeq.singleton ()
                    |> pipe.Apply
                    |> AsyncSeq.iterAsync (fun () -> async.Return())
            })

    runs
    |> Async.Parallel
    |> Async.Ignore
    |> Async.RunSynchronously

