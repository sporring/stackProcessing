module SlimPipeline

open FSharp.Control
open AsyncSeqExtensions

////////////////////////////////////////////////////////////
/// The memory usage strategies during image processing.
type MemoryProfile =
    | Constant // Slice by slice independently
    | Streaming // Slice by slice independently
    | Sliding of uint * uint * uint * uint  // (window, stride, emitStart, emitCount)
//    | Full // All slices of depth // Full = Sliding depth 1 0 depth

module MemoryProfile =
    let estimateUsage (profile: MemoryProfile) (memPerElement: uint64) (depth: uint) : uint64 =
        match profile with
            | Constant -> 0uL
            | Streaming -> memPerElement
            | Sliding (windowSize, _, _, _) -> memPerElement * uint64 windowSize
//            | Full -> memPerElement * uint64 depth

    let requiresBuffering (profile: MemoryProfile) (availableMemory: uint64) (memPerElement: uint64) (depth: uint) : bool = // Not used yet...
        estimateUsage profile memPerElement depth > availableMemory

    let combine (prof1: MemoryProfile) (prof2: MemoryProfile): MemoryProfile  = 
        match prof1, prof2 with
//        | Full, _ 
//        | _, Full -> Full // conservative fallback
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

module private Pipe =
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
//            | Full         -> { p with Profile = Streaming }
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
(*
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
*)

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
type Stage<'S,'T,'ShapeS,'ShapeT> =
    { Name        : string
      Pipe        : Pipe<'S,'T> 
      Transition  : MemoryTransition
      ShapeUpdate : 'ShapeS option -> 'ShapeT option (*A shape indication such as uint list -> uint list*) } 

module Stage =
    let create<'S,'T,'ShapeS,'ShapeT> (name: string) (pipe: Pipe<'S,'T>) (transition: MemoryTransition) (shapeUpdate: 'ShapeS option -> 'ShapeT option) =
        { Name = name; Pipe = pipe; Transition = transition; ShapeUpdate = shapeUpdate}

    let init<'S,'T,'ShapeS,'ShapeT> (name: string) (depth: uint) (mapper: uint -> 'T) (transition: MemoryTransition) (shapeUpdate: 'ShapeS option ->'ShapeT option) =
        let pipe = Pipe.init name depth mapper transition.From
        create name pipe transition shapeUpdate

(*
    let id<'T,'ShapeA,'ShapeB> () : Stage<'T, 'T, 'ShapeA,'ShapeB> = // I don't think this is used
        let transition = MemoryTransition.create Streaming Streaming
        let pipe = Pipe.lift "id" Streaming (fun x -> x)
        create "id" pipe transition (fun s -> s)
*)

    let toPipe (stage : Stage<_,_,_,_>) = stage.Pipe

    let fromPipe (name: string) (transition: MemoryTransition) (shapeUpdate: 'ShapeA option -> 'ShapeB option) (pipe: Pipe<'S, 'T>) : Stage<'S, 'T, 'ShapeA, 'ShapeB> =
        create name pipe transition shapeUpdate

    let compose (stage1 : Stage<'S,'T,'ShapeA,'ShapeB>) (stage2 : Stage<'T,'U,'ShapeB,'ShapeC>) : Stage<'S,'U,'ShapeA,'ShapeC> =
        let transition = MemoryTransition.create stage1.Transition.From stage2.Transition.To
        let pipe = Pipe.compose stage1.Pipe stage2.Pipe
        let shapeUpdate = stage1.ShapeUpdate >> stage2.ShapeUpdate
        create $"{stage2.Name} ∘ {stage1.Name}" pipe transition shapeUpdate

    let (-->) = compose

    let map<'S,'T,'ShapeS,'ShapeT> (name: string) (f: 'S -> 'T) (shapeUpdate: 'ShapeS option ->'ShapeT option) : Stage<'S, 'T, 'ShapeS,'ShapeT> =
        let transition = MemoryTransition.create Streaming Streaming
        let apply input = input |> AsyncSeq.map f
        let pipe : Pipe<'S,'T> = Pipe.create name apply Streaming
        create name pipe transition shapeUpdate

(*
    let map (name: string) (f: 'U -> 'V) (stage: Stage<'In, 'U, 'Shape>) : Stage<'In, 'V, 'Shape> =
        let pipe = Pipe.map name f stage.Pipe
        let transition = MemoryTransition.create Streaming Streaming
        create name pipe transition (fun s -> s)
*)

    let map2 (name: string) (f: 'U -> 'V -> 'W) (stage1: Stage<'In, 'U, 'ShapeIn,'ShapeU>, stage2: Stage<'In, 'V, 'ShapeIn,'ShapeV>) (shapeUpdate: 'ShapeIn option -> 'ShapeW option): Stage<'In, 'W, 'ShapeIn,'ShapeW> =
        let pipe = Pipe.map2 name f stage1.Pipe stage2.Pipe
        let transition = MemoryTransition.create Streaming Streaming
        create name pipe transition shapeUpdate

    let reduce (name: string) (reducer: AsyncSeq<'In> -> Async<'Out>) (profile: MemoryProfile) (shapeUpdate:'ShapeIn option -> 'ShapeOut option): Stage<'In, 'Out, 'ShapeIn, 'ShapeOut> =
        let transition = MemoryTransition.create Streaming Constant
        let pipe = Pipe.reduce name reducer profile
        create name pipe transition shapeUpdate

    let fold<'S,'T,'ShapeS,'ShapeT> (name: string) (folder: 'T -> 'S -> 'T) (initial: 'T) (shapeUpdate:'ShapeS option->'ShapeT option): Stage<'S, 'T, 'ShapeS, 'ShapeT> =
        let transition = MemoryTransition.create Streaming Constant
        let pipe : Pipe<'S,'T> = Pipe.fold name folder initial Streaming
        create name pipe transition shapeUpdate

    let mapNFold (name: string) (mapFn: 'In -> 'Mapped) (folder: 'State -> 'Mapped -> 'State) (state: 'State) (profile: MemoryProfile) (shapeUpdate:'ShapeIn option->'ShapeState option) : Stage<'In, 'State, 'ShapeIn,'ShapeState> =
        let transition = MemoryTransition.create Streaming Constant
        let pipe = Pipe.mapNFold name mapFn folder state profile
        create name pipe transition shapeUpdate

    // this assumes too much: Streaming and identity ShapeUpdate!!!
    let liftUnary<'S,'T,'ShapeS,'ShapeT> (name: string) (f: 'S -> 'T) (shapeUpdate: 'ShapeS option -> 'ShapeT option): Stage<'S, 'T, 'ShapeS,'ShapeT> =
        let transition = MemoryTransition.create Streaming Streaming
        let pipe = Pipe.lift name Streaming f
        create name pipe transition shapeUpdate

    let liftWindowed<'S,'T,'ShapeS,'ShapeT when 'S: equality and 'T: equality> (name: string) (updateId: uint->'S->'S) (window: uint) (pad: uint) (zeroMaker: 'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) (shapeUpdate: 'ShapeS option -> 'ShapeT option): Stage<'S, 'T,'ShapeS,'ShapeT> =
        let transition = MemoryTransition.create (Sliding (window,stride,emitStart,emitCount)) Streaming
        let pipe = Pipe.mapWindowed name window updateId pad zeroMaker stride emitStart emitCount f
        create name pipe transition shapeUpdate

    let tap (name: string) : Stage<'T, 'T, 'ShapeT,'ShapeT> =
        liftUnary $"tap: {name}" (fun x -> printfn "[%s] %A" name x; x) (fun s->s)

    let tapIt (toString: 'T -> string) : Stage<'T, 'T, 'ShapeT, 'ShapeT> =
        liftUnary "tapIt" (fun x -> printfn "%s" (toString x); x) (fun s->s)

(*
    let internal tee (op: Stage<'In, 'T, 'Shape>) : Stage<'In, 'T, 'Shape> * Stage<'In, 'T, 'Shape> =
        let leftPipe, rightPipe = Pipe.tee op.Pipe
        let mk name pipe = 
            create $"{op.Name} - {name}" pipe op.Transition op.ShapeUpdate
        mk "left" leftPipe, mk "right" rightPipe
*)
    let ignore<'T,'Shape> () : Stage<'T, unit, 'Shape, 'Shape> =
        let pipe = Pipe.ignore ()
        let transition = MemoryTransition.create Streaming Constant
        create "ignore" pipe transition (fun s -> s)

    let consumeWith (name: string) (consume: 'T -> unit) : Stage<'T, unit, 'ShapeT, 'ShapeT> =  // Should this cause Shape to be option type, so we can pare unit with None?
        let pipe = Pipe.consumeWith name consume Streaming
        let transition = MemoryTransition.create Streaming Constant
        create name pipe transition (fun s -> s)

    let cast<'S,'T,'ShapeS when 'S: equality and 'T: equality> name f : Stage<'S,'T, 'ShapeS,'ShapeS> = // cast cannot change
        let apply input = input |> AsyncSeq.map f 
        let pipe = Pipe.create name apply Streaming
        let transition = MemoryTransition.create Streaming Streaming
        create name pipe transition (fun s -> s)

    let promoteConstantToStreaming (name: string) (depth: uint) (value: 'T) : Stage<unit, 'T, 'Shape, 'Shape> = // Does not change shape, and again, do we need a Shape for unit?
        let transition = MemoryTransition.create Constant Streaming
        let pipe = Pipe.init $"stream-of:{name}" depth (fun _ -> value) Streaming
        create $"promote:{name}" pipe transition (fun s->s)

    let promoteStreamingToSliding 
        (name: string)
        (depth: uint)
        (updateId: uint -> 'T -> 'T)
        (pad: uint)
        (zeroMaker: 'T -> 'T)
        (stride: uint)
        (emitStart: uint)
        (emitCount: uint)
        : Stage<'T, 'T, 'Shape,'Shape> = // Does not change shape

        let f (window: 'T list) =
            window
            |> List.skip (int emitStart)
            |> List.take (int emitCount)

        let pipe = Pipe.mapWindowed name depth updateId pad zeroMaker stride emitStart emitCount f
        let transition = MemoryTransition.create Streaming Streaming

        create $"promote:{name}" pipe transition (fun s -> s)

    let promoteSlidingToSliding
        (name: string)
        (depth: uint)
        (updateId: uint -> 'T -> 'T)
        (pad: uint)
        (zeroMaker: 'T -> 'T)
        (stride: uint)
        (emitStart: uint)
        (emitCount: uint)
        : Stage<'T, 'T, 'Shape,'Shape> = // Does not change shape

        let f (window: 'T list) =
            window
            |> List.skip (int emitStart)
            |> List.take (int emitCount)

        let pipe = Pipe.mapWindowed name depth updateId pad zeroMaker stride emitStart emitCount f
        let transition = MemoryTransition.create Streaming Streaming

        create $"promote:{name}" pipe transition (fun s -> s)


////////////////////////////////////////////////////////////
// MemFlow state monad
type ShapeContext<'S> = { // Do these need to be functions or just a pair of numbers? Shape update, updates shape but could perhaps also update these?
    memPerElement : 'S -> uint64
    depth         : 'S -> uint
}

module ShapeContext =
    let create (memPerElement : 'S -> uint64) (depth : 'S -> uint) : ShapeContext<'S> =
        {memPerElement = memPerElement; depth = depth}

type MemFlow<'S,'T,'ShapeS,'ShapeT> = // memory before, shape before, shapeContext before, Stage, memory after, shape after. 
        uint64 -> 'ShapeS option -> ShapeContext<'ShapeS> -> Stage<'S,'T,'ShapeS,'ShapeT> * uint64 * ('ShapeT option)

module MemFlow =
    let returnM (stage : Stage<'S,'T,'ShapeS,'ShapeT>) : MemFlow<'S,'T,'ShapeS,'ShapeT> =
        fun bytes shape shapeContext ->
            match shape with
                None ->
                    stage, bytes, None
                | Some sh ->
                    let memPerElement = shapeContext.memPerElement sh
                    let depth = shapeContext.depth sh // Is this the right way, see ShapeContext!!!
                    //let p' = Pipe.shrinkProfile bytes memPerElement depth stage.Pipe
                    let need = Pipe.memNeed memPerElement depth stage.Pipe // sh -> memPerElement depth
                    let newShape = stage.ShapeUpdate shape
                    stage, bytes - need, newShape

    let bindM (k: Stage<'A,'B,'ShapeA,'ShapeB> -> MemFlow<'B,'C,'ShapeB,'ShapeC>) (flowAB: MemFlow<'A,'B,'ShapeA,'ShapeB>) : MemFlow<'A,'C,'ShapeA,'ShapeC> =
        fun bytes shape shapeContextA ->
            let stage1, bytes1, shape1 = flowAB bytes shape shapeContextA
            let flowBC = k stage1
            let shapeContextB = // this is where shapeContext is updated. Return here to later!!!!
                match shape with
                    | Some sh ->
                        {
                            memPerElement = fun (s2:'ShapeB)->shapeContextA.memPerElement sh
                            depth = fun(s2:'ShapeB) -> shapeContextA.depth sh
                        }
                    | _ ->
                        {
                            memPerElement = fun (s2:'ShapeB)-> 0UL
                            depth = fun(s2:'ShapeB) -> 0u
                        }
            let stage2, bytes2, shape2 = flowBC bytes1 shape1 shapeContextB
            (* 2025/07/25 Is this step necessary? The processing is taking care of the interfacing...
            // validate memory transition, if shape is known
            shape
            |> Option.iter (fun shape ->
                if stage1.Transition.To <> stage2.Transition.From || not (stage2.Transition.Check shape) then 
                    failwith $"Invalid memory transition: {stage1} → {stage2}")
            *)
            Stage.compose stage1 stage2, bytes2, shape2

////////////////////////////////////////////////////////////
// Pipeline flow controler
type Pipeline<'S,'T,'ShapeS,'ShapeT> = { 
    flow    : MemFlow<'S,'T,'ShapeS,'ShapeT>
    mem     : uint64 // memory available before
    shape   : 'ShapeS option // shape before transformation
    context : ShapeContext<'ShapeS> // shape element's shape descriptor before transformation
    debug   : bool }

module Pipeline =
    let create<'S,'T,'ShapeS,'ShapeT when 'T: equality> (flow: MemFlow<'S,'T,'ShapeS,'ShapeT>) (mem : uint64) (shape: 'ShapeS option) (context : ShapeContext<'ShapeS>) (debug: bool): Pipeline<'S, 'T,'ShapeS,'ShapeT> =
        { flow = flow; mem = mem; shape = shape; context = context; debug = debug }

    let asStage (pl: Pipeline<'In, 'Out, 'ShapeIn,'ShapeOut>) : Stage<'In, 'Out, 'ShapeIn,'ShapeOut> =
        let stage, _, _ = pl.flow pl.mem pl.shape pl.context
        stage

    //////////////////////////////////////////////////
    /// Source type operators
    let source<'Shape> (context: ShapeContext<'Shape>) (availableMemory: uint64) : Pipeline<unit, unit, 'Shape,'Shape> =
        let flow = fun _ _ _ -> failwith "Pipeline not started yet"
        create flow availableMemory None context false

    let debug<'Shape> (context: ShapeContext<'Shape>) (availableMemory: uint64) : Pipeline<unit, unit, 'Shape,'Shape> =
        printfn $"Preparing pipeline - {availableMemory} B available"
        let flow = fun _ _ _ -> failwith "Pipeline not started yet"
        create flow availableMemory None context true

    //////////////////////////////////////////////////////////////
    /// Composition operators
    let compose (pl: Pipeline<'a, 'b, 'Shapea, 'Shapeb>) (stage: Stage<'b, 'c, 'Shapeb, 'Shapec>) : Pipeline<'a, 'c, 'Shapea,'Shapec> =
        if pl.debug then printfn $"[>=>] {stage.Name}"
        let flow = MemFlow.bindM (fun _ -> MemFlow.returnM stage) pl.flow  // This seems wrong. Why is the function ignoring input? 
        create flow pl.mem pl.shape pl.context pl.debug

    let (>=>) = compose

    //let shape' = pl.shape |> Option.map stage.ShapeUpdate


(*
    let tee (pl: Pipeline<'In, 'T, 'Shape>) : Pipeline<'In, 'T, 'Shape> * Pipeline<'In, 'T, 'Shape> =
        if pl.debug then printfn "tee"
        let stage, mem, shape = pl.flow pl.mem pl.shape pl.context
        let left, right = Stage.tee stage
        let basePipeline pipe =
            create (MemFlow.returnM pipe) mem shape pl.context pl.debug
        basePipeline left, basePipeline right
*)

    /// parallel fanout with synchronization
    let (>=>>) 
        (pl: Pipeline<'In, 'S, 'ShapeIn, 'ShapeS>) 
        (stg1: Stage<'S, 'U, 'ShapeS, 'ShapeU>, stg2: Stage<'S, 'V, 'ShapeS, 'ShapeV>) 
        : Pipeline<'In, ('U * 'V), 'ShapeIn, ('ShapeU * 'ShapeV)> =

        if pl.debug then printfn $"[>=>>] ({stg1.Name}, {stg2.Name})"

        match pl.flow pl.mem pl.shape pl.context with
        | baseStg, mem', shape' ->

            // Compose both stages with the base
            //let stage1 = Stage.compose baseStg stg1
            //let stage2 = Stage.compose baseStg stg2

            // Combine them
            let shapeUpdate1 sh = sh |> stg1.ShapeUpdate
            let shapeUpdate2 sh = sh |> stg2.ShapeUpdate
            let shapeUpdate sh =
                match shapeUpdate1 sh, shapeUpdate2 sh with
                | Some sh1, Some sh2 -> Some (sh1,sh2)
                | _ -> None
            let stage = Stage.map2 ">=>>" (fun u v -> (u,v)) (stg1, stg2) shapeUpdate

            let flow = MemFlow.bindM (fun _ -> MemFlow.returnM stage) pl.flow // This seems wrong, why is input ignored?
            //let shape'' = shape' |> Option.map stage.ShapeUpdate

            create flow pl.mem pl.shape pl.context pl.debug

    let (>>=>) f pl stage shapeUpdate = 
        (>=>) pl (Stage.map2 "zip2" f stage shapeUpdate)

    ///////////////////////////////////////////
    /// sink type operators
    let sink (pl: Pipeline<unit, unit, 'Shape,'Shape>) : unit = // Shape of unit?
        if pl.debug then printfn "sink"
        match pl.shape with
        | None -> failwith "No stage provided shape information."
        | Some sh ->
            let stage, rest, _ = pl.flow pl.mem pl.shape pl.context
            if pl.debug then printfn $"Pipeline built - {rest} B still free\nRunning"
            stage |> Stage.toPipe |> Pipe.run
            if pl.debug then printfn "Done"

    let sinkList (pipelines: Pipeline<unit, unit, 'Shape, 'Shape> list) : unit = // shape of unit?
        pipelines |> List.iter sink

    let internal runToScalar (name:string) (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Pipeline<'In, 'T,'ShapeIn,'ShapeT>) : 'R =
        if pl.debug then printfn "[runToScalar]"
        let stage, _, _ = pl.flow pl.mem pl.shape pl.context
        let pipe = stage.Pipe
        let input = AsyncSeq.singleton Unchecked.defaultof<'In>
        pipe.Apply input |> reducer |> Async.RunSynchronously

    let drainSingle (name:string) (pl: Pipeline<'S, 'T,'ShapeS,'ShapeT>) =
        if pl.debug then printfn "[drainSingle]"
        runToScalar name AsyncSeq.toListAsync pl
        |> function
            | [x] -> x
            | []  -> failwith $"[drainSingle] No result from {name}"
            | _   -> failwith $"[drainSingle] Multiple results from {name}, expected one."

    let drainList (name:string) (pl: Pipeline<'S, 'T,'ShapeS,'ShapeT>) =
        runToScalar name AsyncSeq.toListAsync pl

    let drainLast (name:string) (pl: Pipeline<'S, 'T,'ShapeS,'ShapeT>) =
        if pl.debug then printfn "[drainLast]"
        runToScalar name AsyncSeq.tryLast pl
        |> function
            | Some x -> x
            | None -> failwith $"[drainLast] No result from {name}"
