module SlimPipeline

open FSharp.Control
open AsyncSeqExtensions

////////////////////////////////////////////////////////////
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant // Slice by slice independently
    | Streaming // Slice by slice independently
    | Sliding of uint * uint * uint * uint  // (window, stride, emitStart, emitCount)
    | Full // All slices of depth

module MemoryProfile =
    let estimateUsage (profile: MemoryProfile) (memPerElement: uint64) (depth: uint) : uint64 =
        match profile with
            | Constant -> 0uL
            | Streaming -> memPerElement
            | Sliding (windowSize, _, _, _) -> memPerElement * uint64 windowSize
            | Full -> memPerElement * uint64 depth

    let requiresBuffering (profile: MemoryProfile) (availableMemory: uint64) (memPerElement: uint64) (depth: uint) : bool = // Not used yet...
        estimateUsage profile memPerElement depth > availableMemory

    let combine (prof1: MemoryProfile) (prof2: MemoryProfile): MemoryProfile  = 
        match prof1, prof2 with
        | Full, _ 
        | _, Full -> Full // conservative fallback
        | Sliding (sz1,str1,emitS1,emitN1), Sliding (sz2,str2,emitS2,emitN2) -> Sliding ((max sz1 sz2), min str1 str2, min emitS1 emitS2, max emitN1 emitN2) // don't really know what stride rule should be
        | Sliding (sz,str,emitS,emitN), _ 
        | _, Sliding (sz,str,emitS,emitN) -> Sliding (sz,str,emitS,emitN)
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
    Apply: AsyncSeq<'S> -> AsyncSeq<'T>
    Profile: MemoryProfile
}

module Pipe =
    let create<'T> (name: string) (depth: uint) (mapper: uint -> 'T) (profile: MemoryProfile) : Pipe<unit,'T> =
        { Name = name; Apply = (fun _ -> AsyncSeq.init (int64 depth) (fun (i:int64) -> mapper (uint i))); Profile = profile }

    let runWith (input: 'S) (pipe: Pipe<'S, 'T>) : Async<unit> =
        AsyncSeq.singleton input
        |> pipe.Apply
        |> AsyncSeq.iterAsync (fun _ -> async.Return())

    let run (pipe: Pipe<unit, unit>) : unit =
        runWith () pipe |> Async.RunSynchronously

    let lift (label: string) (profile: MemoryProfile) (f: 'In -> Async<'Out>) : Pipe<'In, 'Out> =
        {
            Name = label
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

    let reduce (label: string) (profile: MemoryProfile) (reducer: AsyncSeq<'In> -> Async<'Out>) : Pipe<'In, 'Out> =
        {
            Name = label
            Profile = profile
            Apply = fun input ->
                reducer input |> ofAsync
        }

    let fold (label: string) (profile: MemoryProfile)  (folder: 'State -> 'In -> 'State) (state0: 'State) : Pipe<'In, 'State> =
        reduce label profile (fun stream ->
            async {
                let! result = stream |> AsyncSeq.fold folder state0
                return result
            })

    let concatenate (label: string) (seqs: AsyncSeq<'T> list) : Pipe<'T, 'T> =
        {
            Name = label
            Profile = Streaming
            Apply = fun _input ->
                seqs |> Seq.fold AsyncSeq.append AsyncSeq.empty
        }

    let consumeWith (name: string) (profile: MemoryProfile) (consume: AsyncSeq<'T> -> Async<unit>) : Pipe<'T, unit> =

        let reducer (s : AsyncSeq<'T>) = consume s          // Async<unit>
        reduce name profile reducer                    // gives AsyncSeq<unit>

    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    let compose (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
        printfn "[composePipe]"
        {
            Name = $"{p2.Name} {p1.Name}"; 
            Profile = MemoryProfile.combine p1.Profile p2.Profile
            Apply = fun input -> input |> p1.Apply |> p2.Apply
        }

    // How many bytes does this pipe claim for one full slice?
    let memNeed (memPerElement: uint64) (depth: uint) (p : Pipe<_,_>) : uint64 =
        MemoryProfile.estimateUsage p.Profile memPerElement depth

    /// Try to shrink a too‑hungry pipe to a cheaper profile.
    /// *You* control the downgrade policy here.
    let shrinkProfile (avail: uint64) (memPerElement: uint64) (depth: uint) (p : Pipe<'S,'T>) =
        let needed = memNeed memPerElement depth p
        if needed <= avail then p                              // fits as is
        else
            match p.Profile with
            | Full         -> { p with Profile = Streaming }
            | Sliding (w,s,es,ec) when w > 1u ->
                { p with Profile = Sliding ((w/2u),s,es,ec) }
            | _ ->
                failwith
                    $"Stage “{p.Name}” needs {needed}s B but only {avail}s B are left."

    /// mapWindowed keeps a running window along the slice direction of depth images
    /// and processes them by f. The stepping size of the running window is stride.
    /// So if depth is 3 and stride is 1 then first image 0,1,2 is sent to f, then 1, 2, 3
    /// and so on. If depth is 3 and stride is 3, then it'll be image 0, 1, 2 followed by
    /// 3, 4, 5. It is also possible to use this for sampling, e.g., setting depth to 1
    /// and stride to 2 sends every second image to f.  
    let mapWindowed (label: string) (depth: uint) (updateId: uint->'S->'S) (pad: uint) (zeroMaker: 'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) : Pipe<'S,'T> =
        {
            Name = label; 
            Profile = Sliding (depth,stride,emitStart,emitCount)
            Apply = fun input ->
    //            AsyncSeqExtensions.windowed depth stride input
                AsyncSeqExtensions.windowedWithPad depth updateId stride pad pad zeroMaker input
                    |> AsyncSeq.collect (f >> AsyncSeq.ofSeq)
        }

    let tap label : Pipe<'T, 'T> = // I don't think this is used
        printfn "[tap]"
        lift $"tap: {label}" Streaming (fun x ->
            printfn "[%s] %A" label x
            async.Return x)

    /// Split a Pipe<'In,'T> into two branches that
    ///   • read the upstream only once
    ///   • keep at most one item in memory
    ///   • terminate correctly when both sides finish
    type private Request<'T> =
        | Left of AsyncReplyChannel<Option<'T>>
        | Right of AsyncReplyChannel<Option<'T>>

    let internal tee (p : Pipe<'In,'T>): Pipe<'In, 'T> * Pipe<'In, 'T> =
        printfn "[tee]"

        // ---------- 1. Create the broadcaster exactly *once* per pipeline run ----------
        // We store it in a mutable cell that is initialised on first Apply().
        let mutable shared : Lazy<AsyncSeq<'T> * AsyncSeq<'T>> option = None
        let syncRoot = obj()

        let makeShared (input : AsyncSeq<'In>) =
            let src = p.Apply input                        // drives upstream only ONCE
            let agent = MailboxProcessor.Start(fun inbox ->
                async {
                    let enum = (AsyncSeq.toAsyncEnum src).GetAsyncEnumerator()
                    let mutable current   : 'T option = None
                    let mutable consumed  = (true, true)
                    let mutable finished  = false

                    let rec loop () = async {
                        // Pull next slice if needed
                        if current.IsNone && not finished then
                            let! hasNext = enum.MoveNextAsync().AsTask() |> Async.AwaitTask
                            if hasNext then
                                current  <- Some enum.Current
                                consumed <- (false, false)
                            else
                                finished <- true

                        // Serve whichever side is asking
                        let! msg = inbox.Receive()
                        match msg, current with
                        | Left ch, Some v when not (fst consumed) ->
                            ch.Reply(Some v)
                            consumed <- (true, snd consumed)
                        | Right ch, Some v when not (snd consumed) ->
                            ch.Reply(Some v)
                            consumed <- (fst consumed, true)
                        | Left ch, None when finished ->
                            ch.Reply(None)
                        | Right ch, None when finished ->
                            ch.Reply(None)
                        | _ -> ()

                        // Release the slice when both sides have seen it
                        if consumed = (true, true) then current <- None
                        return! loop ()
                    }
                    do! loop ()
                })

            // Helper to build one of the two consumer streams
            let makeStream tag =
                asyncSeq {
                    let mutable done_ = false
                    while not done_ do
                        let! vOpt = agent.PostAndAsyncReply(tag)
                        match vOpt with
                        | Some v -> yield v
                        | None   -> done_ <- true
                }

            makeStream Left, makeStream Right

        // Thread‑safe initialisation-on-first-use
        let getShared input =
            match shared with
            | Some lazyStreams -> lazyStreams.Value
            | None ->
                lock syncRoot (fun () ->
                    match shared with
                    | Some lazyStreams -> lazyStreams.Value
                    | None ->
                        let lazyStreams = lazy (makeShared input)
                        shared <- Some lazyStreams
                        lazyStreams.Value)

        // ---------- 2. Build the two outward‑facing Pipes ----------
        let mkPipe name pick =
            { 
            Name  = $"{p.Name} - {name}"
            Profile = p.Profile
            Apply   = 
                fun input ->
                    let left, right = getShared input
                    pick (left, right) 
            }

        mkPipe "left" fst, mkPipe "right" snd

////////////////////////////////////////////////////////////
// Stage between pipes

/// MemoryTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
type MemoryTransition =
    { From  : MemoryProfile
      To    : MemoryProfile }

/// Stage describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Stage + MemoryTransition: what happens
type Stage<'S,'T,'Shape> =
    { Name        : string
      Pipe        : Pipe<'S,'T> 
      Transition  : MemoryTransition
      ShapeUpdate : 'Shape -> 'Shape (*A shape indication such as uint list -> uint list*) } 

module Stage =
    let create<'S,'T,'Shape> (name: string) (depth: uint) (mapper: uint -> 'T) (transition: MemoryTransition) (shapeUpdate: 'Shape->'Shape) =
        let pipe = Pipe.create<'T> name depth mapper transition.From
        { Name = name; Pipe = pipe; Transition = transition; ShapeUpdate = shapeUpdate}

    let transition (fromProfile: MemoryProfile) (toProfile: MemoryProfile) : MemoryTransition =
        {
            From = fromProfile
            To   = toProfile
        }

    let id<'T,'Shape> () : Stage<'T, 'T, 'Shape> = // I don't think this is used
        {
            Name = "id"
            Transition = transition Streaming Streaming
            Pipe = Pipe.lift "id" Streaming (fun x -> async.Return x)
            ShapeUpdate = id
        }

    let toPipe (op : Stage<_,_,_>) = op.Pipe

    let fromPipe (name: string) (transition: MemoryTransition) (shapeUpdate: 'Shape -> 'Shape) (pipe: Pipe<'S, 'T>) : Stage<'S, 'T, 'Shape> =
        {
            Name = name
            Transition = transition
            Pipe = pipe
            ShapeUpdate = shapeUpdate
        }

    let compose (op1 : Stage<'S,'T,'Shape>) (op2 : Stage<'T,'U,'Shape>) : Stage<'S,'U,'Shape> =
        {
            Name       = $"{op2.Name} ∘ {op1.Name}"
            Transition = transition op1.Transition.From op2.Transition.To
            Pipe       = Pipe.compose op1.Pipe op2.Pipe
            ShapeUpdate = fun shape -> shape |>  op1.ShapeUpdate |> op2.ShapeUpdate
        }

    let (-->) = compose

    // this assumes too much: Streaming and identity ShapeUpdate!!!
    let liftUnary<'T,'Shape> (name: string) (f: 'T -> 'T) : Stage<'T, 'T, 'Shape> =
        {
            Name = name
            Transition = transition Streaming Streaming
            Pipe = Pipe.map name Streaming f
            ShapeUpdate = fun s -> s // shape unchanged
        }

    let tap (label: string) : Stage<'T, 'T, 'Shape> =
        liftUnary $"tap: {label}" (fun x -> printfn "[%s] %A" label x; x)

    let internal tee (op: Stage<'In, 'T, 'Shape>) : Stage<'In, 'T, 'Shape> * Stage<'In, 'T, 'Shape> =
        let leftPipe, rightPipe = Pipe.tee op.Pipe
        let mk name pipe =
            {
                Name = $"{op.Name} - {name}"
                Transition = op.Transition
                Pipe = pipe
                ShapeUpdate = op.ShapeUpdate
            }
        mk "left" leftPipe, mk "right" rightPipe

////////////////////////////////////////////////////////////
// MemFlow state monad
type ShapeContext<'S> = { // Do these need to be functions or just a pair of numbers? Shape update, updates shape but could perhaps also update these?
    memPerElement : 'S -> uint64
    depth         : 'S -> uint
}

type MemFlow<'S,'T,'Shape> = // memory before, shape before, shapeContext before, Stage, memory after, shape after. 
        uint64 -> 'Shape option -> ShapeContext<'Shape> -> Stage<'S,'T,'Shape> * uint64 * ('Shape option)

module MemFlow =
    let create (memPerElement : 'S -> uint64) (depth : 'S -> uint) : ShapeContext<'S> =
        {memPerElement = memPerElement; depth = depth}

    let returnM (op : Stage<'S,'T,'Shape>) : MemFlow<'S,'T,'Shape> =
        fun bytes shape shapeContext ->
            match shape with
                None ->
                    op, bytes, shape
                | Some sh ->
                    let memPerElement = shapeContext.memPerElement sh
                    let depth = shapeContext.depth sh // Is this the right way, see ShapeContext!!!
                    let p'     = Pipe.shrinkProfile bytes memPerElement depth op.Pipe
                    let need   = Pipe.memNeed memPerElement depth p' // sh -> memPerElement depth
                    { op with Pipe = p' }, bytes - need, shape

    let bindM (m: MemFlow<'A,'B,'Shape>) (k: Stage<'A,'B,'Shape> -> MemFlow<'B,'C,'Shape>) : MemFlow<'A,'C,'Shape> =
        fun bytes shape shapeContext ->
            let op1, bytes', shape' = m bytes shape shapeContext
            let m2 = k op1
            let op2, bytes'', shape'' = m2 bytes' shape' shapeContext
            (* 2025/07/25 Is this step necessary? The processing is taking care of the interfacing...
            // validate memory transition, if shape is known
            shape
            |> Option.iter (fun shape ->
                if op1.Transition.To <> op2.Transition.From || not (op2.Transition.Check shape) then 
                    failwith $"Invalid memory transition: {op1} → {op2}")
            *)
            Stage.compose op1 op2, bytes'', shape''

////////////////////////////////////////////////////////////
// Pipeline flow controler
type Pipeline<'S,'T,'Shape> = { 
    flow    : MemFlow<'S,'T,'Shape>
    mem     : uint64
    shape   : 'Shape option
    context : ShapeContext<'Shape>}

module Pipeline =
    let create<'T,'Shape when 'T: equality> (flow: MemFlow<unit,'T,'Shape>) (mem : uint64) (shape: 'Shape option) (context : ShapeContext<'Shape>): Pipeline<unit, 'T,'Shape> =
        { flow = flow; mem = mem; shape = shape; context = context}

    let stage2Pipeline (stage: Stage<'S, 'T, 'Shape>) (mem: uint64) (shape: 'Shape) (context: ShapeContext<'Shape>) : Pipeline<'S, 'T, 'Shape> =
        {
            flow = MemFlow.returnM stage
            mem = mem
            shape = Some shape
            context = context
        }

    // source: only memory budget, shape = None
    let source<'Shape> (context: ShapeContext<'Shape>) (availableMemory: uint64) : Pipeline<unit, unit, 'Shape> =
        {
            flow = fun _ _ _ -> failwith "Pipeline not started yet"
            mem = availableMemory
            shape = None
            context = context
        }

    // later compositions Pipeline composition
    let compose (pl: Pipeline<'a, 'b, 'Shape>) (next: Stage<'b, 'c, 'Shape>) : Pipeline<'a, 'c, 'Shape> =
        {
            mem = pl.mem
            shape = pl.shape
            context = pl.context
            flow = MemFlow.bindM pl.flow (fun _ -> MemFlow.returnM next)
        }

    let (>=>) = compose

    // sink
    let sink (pl: Pipeline<unit, unit, 'Shape>) : unit =
        match pl.shape with
        | None -> failwith "No stage provided shape information."
        | Some sh ->
            let stage, rest, _ = pl.flow pl.mem pl.shape pl.context
            printfn $"Pipeline built - {rest} B still free."
            stage |> Stage.toPipe |> Pipe.run

    let sinkList (pipelines: Pipeline<unit, unit, 'Shape> list) : unit =
        if pipelines.Length > 1 then
            printfn "Compile time analysis: sinkList parallel"

        pipelines |> List.iter sink

    let tee (pl: Pipeline<'In, 'T, 'Shape>) : Pipeline<'In, 'T, 'Shape> * Pipeline<'In, 'T, 'Shape> =
        let stage, mem, shape = pl.flow pl.mem pl.shape pl.context
        let left, right = Stage.tee stage
        let basePipeline pipe =
            {
                flow = MemFlow.returnM pipe
                mem = mem
                shape = shape
                context = pl.context
            }
        basePipeline left, basePipeline right

    let asStage (pl: Pipeline<'In, 'Out, 'Shape>) : Stage<'In, 'Out, 'Shape> =
        let stage, _, _ = pl.flow pl.mem pl.shape pl.context
        stage

/// Represents a pipeline that has been shared (split into synchronized branches)
type SharedPipeline<'T, 'U , 'V, 'Shape> = {
    flow: MemFlow<'T,'U,'Shape> 
    branches: Stage<'T,'U,'Shape> * Stage<'T,'V,'Shape>
    mem: uint64
    shape: 'Shape option
    context: ShapeContext<'Shape>
}

module SharedPipeline =
    let create<'T,'U,'V,'Shape when 'T: equality> (flow: MemFlow<'T,'U,'Shape>) (branches: Stage<'T,'U,'Shape> * Stage<'T,'V,'Shape>) (mem: uint64) (shape: 'Shape option) (context: ShapeContext<'Shape>) : SharedPipeline<'T, 'U , 'V, 'Shape> =
        { flow = flow; branches = branches; mem = mem; shape = shape; context = context }

module Routing =
    (*
    /// zipWith two Pipes<'In, _> into one by zipping their outputs:
    ///   • applies both processors to the same input stream
    ///   • pairs each output and combines using the given function
    ///   • assumes both sides produce values in lockstep
    let internal zipWithOp (f: 'A -> 'B -> 'C)
                (op1: Stage<'In, 'A, 'Shape>)
                (op2: Stage<'In, 'B, 'Shape>) : Stage<'In, 'C, 'Shape> =
        let name = $"zipWith({op1.Name}, {op2.Name})"
        let profile = MemoryProfile.combine op1.Pipe.Profile op2.Pipe.Profile
        let pipe =
            {
                Name = name
                Profile = profile
                Apply = fun input ->
                    let a = op1.Pipe.Apply input
                    let b = op2.Pipe.Apply input
                    match op1.Pipe.Profile, op2.Pipe.Profile with
                    | Full, Streaming | Streaming, Full ->
                        failwithf "[zipWith] Mixing Full and Streaming not supported: %s, %s"
                                (op1.Pipe.Profile.ToString()) (op2.Pipe.Profile.ToString())
                    | Constant, _ ->
                        printfn "[Runtime analysis: zipWith sequential]"
                        asyncSeq {
                            let! constant = 
                                a 
                                |> AsyncSeq.tryLast 
                                |> Async.map (Option.defaultWith (fun () -> failwith $"No constant result from {op1.Name}"))
                            yield! b |> AsyncSeq.map (fun b -> f constant b)
                        }
                    | _, Constant ->
                        printfn "[Runtime analysis: zipWith sequential]"
                        asyncSeq {
                            let! constant = 
                                b 
                                |> AsyncSeq.tryLast 
                                |> Async.map (Option.defaultWith (fun () -> failwith $"No constant result from {op2.Name}"))
                            yield! a |> AsyncSeq.map (fun a -> f a constant)
                        }
                    | _ ->
                        printfn "[Runtime analysis: zipWith parallel]"
                        AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> f x y)
            }

        {
            Name = name
            Transition = Stage.transition op1.Transition.From op2.Transition.To
            Pipe = pipe
        }

    let zipWith (f: 'A -> 'B -> 'C) (p1: Pipeline<'In, 'A>) (p2: Pipeline<'In, 'B>) : Pipeline<'In, 'C> =
        let flow (mem: uint64) (shape: SliceShape option) =
            let op1, mem1, shape1 = p1.flow mem shape
            let op2, mem2, shape2 = p2.flow mem1 shape1
            let zipped = zipWithOp f op1 op2
            zipped, mem2, shape2

        {
            flow = flow
            mem = min p1.mem p2.mem
            shape = p1.shape |> Option.orElse p2.shape
        }
    *)

    let runToScalar name (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Pipeline<'In, 'T,'Shape>) : 'R =
        let op, _, _ = pl.flow pl.mem pl.shape pl.context
        let pipe = op.Pipe
        let input = AsyncSeq.singleton Unchecked.defaultof<'In>
        pipe.Apply input |> reducer |> Async.RunSynchronously

    let drainSingle name pl =
        runToScalar name AsyncSeq.toListAsync pl
        |> function
            | [x] -> x
            | []  -> failwith $"[drainSingle] No result from {name}"
            | _   -> failwith $"[drainSingle] Multiple results from {name}, expected one."

    let drainList name pl =
        runToScalar name AsyncSeq.toListAsync pl

    let drainLast name pl =
        runToScalar name AsyncSeq.tryLast pl
        |> function
            | Some x -> x
            | None -> failwith $"[drainLast] No result from {name}"

    /// parallel fanout with synchronization
    /// Synchronously split the shared stream into two parallel pipelines
    let (>=>>) 
        (pl: Pipeline<'In, 'T, 'Shape>) 
        (op1: Stage<'T, 'U, 'Shape>, op2: Stage<'T, 'V, 'Shape>) 
        : SharedPipeline<'In, 'U, 'V, 'Shape> =

        match pl.flow pl.mem pl.shape pl.context with
        | baseOp, mem', shape' when baseOp.Transition.To = Streaming ->

            let opBase, mem', shape' = pl.flow pl.mem pl.shape pl.context
            let pipe1, pipe2 = Pipe.tee opBase.Pipe
            let op1' = Stage.compose { opBase with Pipe = pipe1 } op1
            let op2' = Stage.compose { opBase with Pipe = pipe2 } op2
            SharedPipeline.create<'T,'U,'V,'Shape> pl.flow (op1', op2') pl.mem pl.shape pl.context

        | baseOp, mem', shape' when baseOp.Transition.To = Constant ->

            let cached = lazy (
                let result =
                    AsyncSeq.singleton Unchecked.defaultof<'In>
                    |> baseOp.Pipe.Apply
                    |> AsyncSeq.tryLast
                    |> Async.RunSynchronously
                result |> Option.defaultWith (fun () -> failwithf "No constant result from %s" baseOp.Name)
            )

            let applyOp (op: Stage<'T, 'X, 'Shape>) (value: 'T) : 'X =
                AsyncSeq.singleton value
                |> op.Pipe.Apply
                |> AsyncSeq.tryLast
                |> Async.RunSynchronously
                |> Option.defaultWith (fun () -> failwithf "applyOp: No output from %s" op.Name)

            let makeConstOp (op: Stage<'T, 'X, 'Shape>) label : Stage<'In, 'X, 'Shape> =
                {
                    Name = $"shared-const:{label}"
                    Transition = Stage.transition Constant Constant
                    Pipe = Pipe.lift label Constant (fun _ -> async { return applyOp op (cached.Value) })
                    ShapeUpdate = fun s -> s // I don't know what this should be!!!!
                }

            let op1' = makeConstOp op1 "left"
            let op2' = makeConstOp op2 "right"
            SharedPipeline.create pl.flow (op1', op2') pl.mem pl.shape pl.context

        | _ -> failwith "Unsupported transition kind in >=>>"

    let (>>=>)
        (shared: SharedPipeline<'In, 'U, 'V, 'Shape>)
        (combine: Stage<'In, 'U, 'Shape> * Stage<'In, 'V, 'Shape> -> Stage<'In, 'W, 'Shape>)
        : Pipeline<'In, 'W, 'Shape> =

        let stage = combine shared.branches
        let flow = MemFlow.returnM stage
        let mem = shared.mem
        let shape = shared.shape
        let context = shared.context
        Pipeline.create flow mem shape context

    let combineIgnore : Stage<'In, 'U, 'Shape> * Stage<'In, 'V, 'Shape> -> Stage<'In, unit, 'Shape> =
        fun (op1, op2) ->
            {
                Name = $"combineIgnore({op1.Name}, {op2.Name})"
                Transition = Stage.transition op1.Transition.From op2.Transition.To
                ShapeUpdate = id
                Pipe =
                    {
                        Name = "combineIgnore"
                        Profile = MemoryProfile.combine op1.Pipe.Profile op2.Pipe.Profile
                        Apply = fun input ->
                            let out1 = op1.Pipe.Apply input |> AsyncSeq.iterAsync (fun _ -> async.Return())
                            let out2 = op2.Pipe.Apply input |> AsyncSeq.iterAsync (fun _ -> async.Return())
                            asyncSeq {
                                do! Async.Parallel [ out1; out2 ] |> Async.Ignore
                                yield ()
                            }
                    }
            }

