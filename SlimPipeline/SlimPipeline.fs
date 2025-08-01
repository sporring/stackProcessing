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
    let create (name: string) (apply: AsyncSeq<'S> -> AsyncSeq<'T>) (profile: MemoryProfile) : Pipe<'S,'T> =
        { Name = name; Apply = apply; Profile = profile }

    let runWith (input: 'S) (pipe: Pipe<'S, 'T>) : Async<unit> =
        AsyncSeq.singleton input
        |> pipe.Apply
        |> AsyncSeq.iterAsync (fun _ -> async.Return())

    let run (pipe: Pipe<unit, unit>) : unit =
        runWith () pipe |> Async.RunSynchronously

    let lift (name: string) (profile: MemoryProfile) (f: 'S -> 'T) : Pipe<'S,'T> =
        let apply input = input |> AsyncSeq.map f
        create name apply profile

    let init<'T> (name: string) (depth: uint) (mapper: uint -> 'T) (profile: MemoryProfile) : Pipe<unit,'T> =
        let apply _ = AsyncSeq.init (int64 depth) (fun (i:int64) -> mapper (uint i))
        create name apply profile

    let map (name: string) (f: 'U -> 'V) (pipe: Pipe<'In, 'U>) : Pipe<'In, 'V> =
        let apply input = input |> pipe.Apply |> AsyncSeq.map f
        create name apply pipe.Profile

    let map2 (name: string) (f: 'U -> 'V -> 'W) (pipe1: Pipe<'In, 'U>) (pipe2: Pipe<'In, 'V>) : Pipe<'In, 'W> =
        let apply input =
                let stream1 = pipe1.Apply input
                let stream2 = pipe2.Apply input
                AsyncSeq.zip stream1 stream2 |> AsyncSeq.map (fun (u, v) -> f u v)
        let profile = max pipe1.Profile pipe2.Profile
        create name apply profile

    let reduce (name: string) (reducer: AsyncSeq<'In> -> Async<'Out>) (profile: MemoryProfile) : Pipe<'In, 'Out> =
        let apply input = input |> reducer |> ofAsync
        create name apply profile

    let fold (name: string) (folder: 'State -> 'In -> 'State) (initial: 'State) (profile: MemoryProfile) : Pipe<'In, 'State> =
        let reducer (s: AsyncSeq<'In>) : Async<'State> =
            AsyncSeqExtensions.fold folder initial s
        reduce name reducer profile

    let mapNFold (name: string) (mapFn: 'In -> 'Mapped) (folder: 'State -> 'Mapped -> 'State) (state: 'State) (profile: MemoryProfile)
        : Pipe<'In, 'State> =
        
        let reducer (s: AsyncSeq<'In>) : Async<'State> =
            s
            |> AsyncSeq.map mapFn
            |> AsyncSeqExtensions.fold folder state

        reduce name reducer profile

//    let consumeWith (name: string) (consume: AsyncSeq<'T> -> Async<unit>) (profile: MemoryProfile) : Pipe<'T, unit> =
//        let reducer (s : AsyncSeq<'T>) = consume s          // Async<unit>
//        reduce name reducer  profile              // gives AsyncSeq<unit>

    let consumeWith (name: string) (consume: 'T -> unit) (profile: MemoryProfile) : Pipe<'T, unit> =
        let apply input = 
            asyncSeq {
                do! AsyncSeq.iterAsync (fun x -> async { consume x }) input
                yield ()
            }
        create name apply profile

    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    let compose (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
        let profile = MemoryProfile.combine p1.Profile p2.Profile
        let apply input = input |> p1.Apply |> p2.Apply
        create $"{p2.Name} {p1.Name}" apply profile

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
    let mapWindowed (name: string) (depth: uint) (updateId: uint->'S->'S) (pad: uint) (zeroMaker: 'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) : Pipe<'S,'T> =
        let apply input =
            // AsyncSeqExtensions.windowed depth stride input
            AsyncSeqExtensions.windowedWithPad depth updateId stride pad pad zeroMaker input
            |> AsyncSeq.collect (f >> AsyncSeq.ofSeq)
        let profile = Sliding (depth,stride,emitStart,emitCount)
        create name apply profile

    let ignore () : Pipe<'T, unit> =
        let apply input =
            asyncSeq {
                // Consume the stream without doing anything
                do! AsyncSeq.iterAsync (fun _ -> async.Return()) input
                // Then emit one unit
                yield ()
            }
        create "ignore" apply Constant

    /// Split a Pipe<'In,'T> into two branches that
    ///   • read the upstream only once
    ///   • keep at most one item in memory
    ///   • terminate correctly when both sides finish
    type private Request<'T> =
        | Left of AsyncReplyChannel<Option<'T>>
        | Right of AsyncReplyChannel<Option<'T>>

    let internal tee (p : Pipe<'In,'T>): Pipe<'In, 'T> * Pipe<'In, 'T> =
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
            let apply input =
                let left, right = getShared input
                pick (left, right) 
            create $"{p.Name} - {name}" apply p.Profile

        mkPipe "left" fst, mkPipe "right" snd

////////////////////////////////////////////////////////////
// Stage between pipes

/// MemoryTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
type MemoryTransition =
    { From  : MemoryProfile
      To    : MemoryProfile }

module MemoryTransition =
    let create (fromProfile: MemoryProfile) (toProfile: MemoryProfile) : MemoryTransition =
        {
            From = fromProfile
            To   = toProfile
        }

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
    let create<'S,'T,'Shape> (name: string) (pipe: Pipe<'S,'T>) (transition: MemoryTransition) (shapeUpdate: 'Shape->'Shape) =
        { Name = name; Pipe = pipe; Transition = transition; ShapeUpdate = shapeUpdate}

    let init<'S,'T,'Shape> (name: string) (depth: uint) (mapper: uint -> 'T) (transition: MemoryTransition) (shapeUpdate: 'Shape->'Shape) =
        let pipe = Pipe.init name depth mapper transition.From
        create name pipe transition shapeUpdate

    let id<'T,'Shape> () : Stage<'T, 'T, 'Shape> = // I don't think this is used
        let transition = MemoryTransition.create Streaming Streaming
        let pipe = Pipe.lift "id" Streaming (fun x -> x)
        create "id" pipe transition id

    let toPipe (op : Stage<_,_,_>) = op.Pipe

    let fromPipe (name: string) (transition: MemoryTransition) (shapeUpdate: 'Shape -> 'Shape) (pipe: Pipe<'S, 'T>) : Stage<'S, 'T, 'Shape> =
        create name pipe transition shapeUpdate

    let compose (op1 : Stage<'S,'T,'Shape>) (op2 : Stage<'T,'U,'Shape>) : Stage<'S,'U,'Shape> =
        let transition  = MemoryTransition.create op1.Transition.From op2.Transition.To
        let pipe        = Pipe.compose op1.Pipe op2.Pipe
        let shapeUpdate = fun shape -> shape |>  op1.ShapeUpdate |> op2.ShapeUpdate
        create $"{op2.Name} ∘ {op1.Name}" pipe transition shapeUpdate

    let (-->) = compose

    let map<'S,'T,'Shape> (name: string) (f: 'S -> 'T) : Stage<'S, 'T, 'Shape> =
        let transition = MemoryTransition.create Streaming Streaming
        let apply input = input |> AsyncSeq.map f
        let pipe : Pipe<'S,'T> = Pipe.create name apply Streaming
        create name pipe transition (fun s -> s)

(*
    let map (name: string) (f: 'U -> 'V) (stage: Stage<'In, 'U, 'Shape>) : Stage<'In, 'V, 'Shape> =
        let pipe = Pipe.map name f stage.Pipe
        let transition = MemoryTransition.create Streaming Streaming
        create name pipe transition (fun s -> s)
*)

    let map2 (name: string) (f: 'U -> 'V -> 'W) (stage1: Stage<'In, 'U, 'Shape>, stage2: Stage<'In, 'V, 'Shape>) : Stage<'In, 'W, 'Shape> =
        let pipe = Pipe.map2 name f stage1.Pipe stage2.Pipe
        let transition = MemoryTransition.create Streaming Streaming
        create name pipe transition (fun s -> s) 

    let reduce (name: string) (reducer: AsyncSeq<'In> -> Async<'Out>) (profile: MemoryProfile) : Stage<'In, 'Out, 'Shape> =
        let transition = MemoryTransition.create Streaming Constant
        let pipe = Pipe.reduce name reducer profile
        create name pipe transition (fun s -> s)

    let fold<'S,'T,'Shape> (name: string) (folder: 'T -> 'S -> 'T) (initial: 'T) : Stage<'S, 'T, 'Shape> =
        let transition = MemoryTransition.create Streaming Constant
        let pipe : Pipe<'S,'T> = Pipe.fold name folder initial Streaming
        create name pipe transition (fun s -> s)

    let mapNFold (name: string) (mapFn: 'In -> 'Mapped) (folder: 'State -> 'Mapped -> 'State) (state: 'State) (profile: MemoryProfile) : Stage<'In, 'State, 'Shape> =
        let transition = MemoryTransition.create Streaming Constant
        let pipe = Pipe.mapNFold name mapFn folder state profile
        create name pipe transition (fun s -> s)

    // this assumes too much: Streaming and identity ShapeUpdate!!!
    let liftUnary<'S,'T,'Shape> (name: string) (f: 'S -> 'T) : Stage<'S, 'T, 'Shape> =
        let transition = MemoryTransition.create Streaming Streaming
        let pipe = Pipe.lift name Streaming f
        create name pipe transition (fun s -> s) 

    let liftWindowed<'S,'T,'Shape when 'S: equality and 'T: equality> (name: string) (updateId: uint->'S->'S) (window: uint) (pad: uint) (zeroMaker: 'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) : Stage<'S, 'T,'Shape> =
        let transition = MemoryTransition.create (Sliding (window,stride,emitStart,emitCount)) Streaming
        let pipe = Pipe.mapWindowed name window updateId pad zeroMaker stride emitStart emitCount f
        create name pipe transition (fun s -> s)

    let tap (name: string) : Stage<'T, 'T, 'Shape> =
        liftUnary $"tap: {name}" (fun x -> printfn "[%s] %A" name x; x)

    let tapIt (toString: 'T -> string) : Stage<'T, 'T, 'Shape> =
        liftUnary "tapIt" (fun x -> printfn "%s" (toString x); x)

    let internal tee (op: Stage<'In, 'T, 'Shape>) : Stage<'In, 'T, 'Shape> * Stage<'In, 'T, 'Shape> =
        let leftPipe, rightPipe = Pipe.tee op.Pipe
        let mk name pipe = 
            create $"{op.Name} - {name}" pipe op.Transition op.ShapeUpdate
        mk "left" leftPipe, mk "right" rightPipe

    let ignore<'T,'Shape> () : Stage<'T, unit, 'Shape> =
        let pipe = Pipe.ignore ()
        let transition = MemoryTransition.create Streaming Constant
        create "ignore" pipe transition (fun s -> s)

    let consumeWith (name: string) (consume: 'T -> unit) : Stage<'T, unit, 'Shape> =
        let pipe = Pipe.consumeWith name consume Streaming
        let transition = MemoryTransition.create Streaming Constant
        create name pipe transition (fun s -> s)

    let cast<'S,'T,'Shape when 'S: equality and 'T: equality> name f : Stage<'S,'T, 'Shape> =
        let apply input = input |> AsyncSeq.map f 
        let pipe = Pipe.create name apply Streaming
        let transition = MemoryTransition.create Streaming Streaming
        create name pipe transition (fun s -> s)

////////////////////////////////////////////////////////////
// MemFlow state monad
type ShapeContext<'S> = { // Do these need to be functions or just a pair of numbers? Shape update, updates shape but could perhaps also update these?
    memPerElement : 'S -> uint64
    depth         : 'S -> uint
}

module ShapeContext =
    let create (memPerElement : 'S -> uint64) (depth : 'S -> uint) : ShapeContext<'S> =
        {memPerElement = memPerElement; depth = depth}

type MemFlow<'S,'T,'Shape> = // memory before, shape before, shapeContext before, Stage, memory after, shape after. 
        uint64 -> 'Shape option -> ShapeContext<'Shape> -> Stage<'S,'T,'Shape> * uint64 * ('Shape option)

module MemFlow =
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
    context : ShapeContext<'Shape>
    debug   : bool }

module Pipeline =
    let create<'S,'T,'Shape when 'T: equality> (flow: MemFlow<'S,'T,'Shape>) (mem : uint64) (shape: 'Shape option) (context : ShapeContext<'Shape>) (debug: bool): Pipeline<'S, 'T,'Shape> =
        { flow = flow; mem = mem; shape = shape; context = context; debug = debug }

    // source: only memory budget, shape = None
    let source<'Shape> (context: ShapeContext<'Shape>) (availableMemory: uint64) : Pipeline<unit, unit, 'Shape> =
        let flow = fun _ _ _ -> failwith "Pipeline not started yet"
        create flow availableMemory None context false

    let debug<'Shape> (context: ShapeContext<'Shape>) (availableMemory: uint64) : Pipeline<unit, unit, 'Shape> =
        printfn $"Preparing pipeline - {availableMemory} B available"
        let flow = fun _ _ _ -> failwith "Pipeline not started yet"
        create flow availableMemory None context true

    // later compositions Pipeline composition
    let compose (pl: Pipeline<'a, 'b, 'Shape>) (next: Stage<'b, 'c, 'Shape>) : Pipeline<'a, 'c, 'Shape> =
        if pl.debug then printfn $"[>=>] {next.Name}"
        let flow = MemFlow.bindM pl.flow (fun _ -> MemFlow.returnM next)
        create flow pl.mem pl.shape pl.context pl.debug

    let (>=>) = compose

    // sink
    let sink (pl: Pipeline<unit, unit, 'Shape>) : unit =
        if pl.debug then printfn "sink"
        match pl.shape with
        | None -> failwith "No stage provided shape information."
        | Some sh ->
            let stage, rest, _ = pl.flow pl.mem pl.shape pl.context
            if pl.debug then printfn $"Pipeline built - {rest} B still free\nRunning"
            stage |> Stage.toPipe |> Pipe.run
            if pl.debug then printfn "Done"

    let sinkList (pipelines: Pipeline<unit, unit, 'Shape> list) : unit =
        pipelines |> List.iter sink

    let tee (pl: Pipeline<'In, 'T, 'Shape>) : Pipeline<'In, 'T, 'Shape> * Pipeline<'In, 'T, 'Shape> =
        if pl.debug then printfn "tee"
        let stage, mem, shape = pl.flow pl.mem pl.shape pl.context
        let left, right = Stage.tee stage
        let basePipeline pipe =
            create (MemFlow.returnM pipe) mem shape pl.context pl.debug
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
    debug: bool
}

module SharedPipeline =
    let create<'T,'U,'V,'Shape when 'T: equality> (flow: MemFlow<'T,'U,'Shape>) (branches: Stage<'T,'U,'Shape> * Stage<'T,'V,'Shape>) (mem: uint64) (shape: 'Shape option) (context: ShapeContext<'Shape>) (debug: bool): SharedPipeline<'T, 'U , 'V, 'Shape> =
        { flow = flow; branches = branches; mem = mem; shape = shape; context = context; debug = debug}

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

    let runToScalar (name:string) (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Pipeline<'In, 'T,'Shape>) : 'R =
        if pl.debug then printfn "[runToScalar]"
        let op, _, _ = pl.flow pl.mem pl.shape pl.context
        let pipe = op.Pipe
        let input = AsyncSeq.singleton Unchecked.defaultof<'In>
        pipe.Apply input |> reducer |> Async.RunSynchronously

    let drainSingle (name:string) (pl: Pipeline<'S, 'T,'Shape>) =
        if pl.debug then printfn "[drainSingle]"
        runToScalar name AsyncSeq.toListAsync pl
        |> function
            | [x] -> x
            | []  -> failwith $"[drainSingle] No result from {name}"
            | _   -> failwith $"[drainSingle] Multiple results from {name}, expected one."

    let drainList (name:string) (pl: Pipeline<'S, 'T,'Shape>) =
        runToScalar name AsyncSeq.toListAsync pl

    let drainLast (name:string) (pl: Pipeline<'S, 'T,'Shape>) =
        if pl.debug then printfn "[drainLast]"
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
        if pl.debug then printfn $"[>=>>] ({op1.Name}, {op2.Name})"
        match pl.flow pl.mem pl.shape pl.context with
        | baseOp, mem', shape' ->

            match (op1.Transition.From, op2.Transition.From) with

            // === Both Streaming: no windowing needed ===
            | Streaming, Streaming ->
                let pipe1, pipe2 = Pipe.tee baseOp.Pipe
                let op1': Stage<'In, 'U, 'Shape>  = Stage.compose { baseOp with Pipe = pipe1 } op1
                let op2': Stage<'In, 'V, 'Shape>  = Stage.compose { baseOp with Pipe = pipe2 } op2
                SharedPipeline.create pl.flow (op1', op2') mem' shape' pl.context pl.debug
            | _ -> failwith "Unsupported transition pattern"

            (*
            // === One or both Sliding: compute merged window ===
            | Sliding (size1,stride1,start1,count1), Sliding (size2,stride2,start2,count2) ->
                let winSize = max size1 size2
                let stride = max stride1 stride2
                let prePad = max start1 start2
                let postPad = max (size1 - (start1 + count1)) (size2 - (start2 + count2))

                let windowedPipe =
                    Pipe.lift "windowed" Streaming (fun source ->
                        windowedWithPad
                            (uint winSize)
                            (fun _ x -> x)         // id updater
                            (uint stride)
                            (uint prePad)
                            (uint postPad)
                            id                      // zero-maker
                            (baseOp.Pipe.Apply source)
                    )

                let opBase' =
                    {
                        baseOp with
                            Pipe = windowedPipe
                            Transition = Stage.transition Streaming Streaming
                    }

                // Compose tee and continue
                let pipe1, pipe2 = Pipe.tee opBase'.Pipe
                let op1' = Stage.compose { opBase' with Pipe = pipe1 } op1
                let op2' = Stage.compose { opBase' with Pipe = pipe2 } op2

                SharedPipeline.create pl.flow (op1', op2') mem' shape' pl.context

            // === Mixed: one Streaming, one Sliding ===
            | Streaming, Sliding w2
            | Sliding w2, Streaming ->

                let winSize = w2.WindowSize
                let stride = w2.Stride
                let prePad = w2.EmitStart
                let postPad = w2.WindowSize - (w2.EmitStart + w2.EmitCount)

                let windowedPipe =
                    Pipe.lift "windowed" Streaming (fun source ->
                        windowedWithPad
                            (uint winSize)
                            (fun _ x -> x)
                            (uint stride)
                            (uint prePad)
                            (uint postPad)
                            id
                            (baseOp.Pipe.Apply source)
                    )

                let opBase' =
                    {
                        baseOp with
                            Pipe = windowedPipe
                            Transition = Stage.transition Streaming Streaming
                    }

                let pipe1, pipe2 = Pipe.tee opBase'.Pipe

                // Handle which side is sliding
                let op1' =
                    match op1.Transition.From with
                    | Sliding _ -> Stage.compose { opBase' with Pipe = pipe1 } op1
                    | _         -> Stage.compose { baseOp with Pipe = pipe1 } op1

                let op2' =
                    match op2.Transition.From with
                    | Sliding _ -> Stage.compose { opBase' with Pipe = pipe2 } op2
                    | _         -> Stage.compose { baseOp with Pipe = pipe2 } op2

                SharedPipeline.create pl.flow (op1', op2') mem' shape' pl.context

            // Constant handling unchanged
            | Constant, Constant ->
                // ... your existing constant logic here ...
                failwith "Handle constants as before"
*)

    let (>>=>)
        (shared: SharedPipeline<'In, 'U, 'V, 'Shape>)
        (combineFn: 'U -> 'V -> 'W)
        : Pipeline<'In, 'W, 'Shape> =
        if shared.debug then 
            let b1,b2 = shared.branches
            printfn "[>>=>]"
        let stage = Stage.map2 ">>=>" combineFn shared.branches
        let flow = MemFlow.returnM stage
        Pipeline.create flow shared.mem shared.shape shared.context shared.debug

(*
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
*)