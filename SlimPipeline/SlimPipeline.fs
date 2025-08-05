module SlimPipeline

open FSharp.Control
open AsyncSeqExtensions

////////////////////////////////////////////////////////////
/// The memory usage strategies during image processing.
type Profile =
    | Constant // Slice by slice independently
    | Streaming // Slice by slice independently
    | Sliding of uint * uint * uint * uint  // (window, stride, emitStart, emitCount)
    // | Full // Full = Sliding depth 1 0 depth

module Profile =
    let estimateUsage (profile: Profile) (memPerElement: uint64) : uint64 =
        match profile with
            | Constant -> 0uL
            | Streaming -> memPerElement
            | Sliding (windowSize, _, _, _) -> memPerElement * uint64 windowSize

    let combine (prof1: Profile) (prof2: Profile): Profile  = 
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
    Profile: Profile
}

module private Pipe =
    let create (name: string) (apply: AsyncSeq<'S> -> AsyncSeq<'T>) (profile: Profile) : Pipe<'S,'T> =
        { Name = name; Apply = apply; Profile = profile }

    let runWith (input: 'S) (pipe: Pipe<'S, 'T>) : Async<unit> =
        AsyncSeq.singleton input
        |> pipe.Apply
        |> AsyncSeq.iterAsync (fun _ -> async.Return())

    let run (pipe: Pipe<unit, unit>) : unit =
        runWith () pipe |> Async.RunSynchronously

    let lift (name: string) (profile: Profile) (f: 'S -> 'T) : Pipe<'S,'T> =
        let apply input = input |> AsyncSeq.map f
        create name apply profile

    let init<'T> (name: string) (depth: uint) (mapper: uint -> 'T) (profile: Profile) : Pipe<unit,'T> =
        let apply _ = AsyncSeq.init (int64 depth) (fun (i:int64) -> mapper (uint i))
        create name apply profile

    let skip (name: string) (count: uint) =
        let apply input = input |> AsyncSeq.skip (int count)
        create "skip" apply (Sliding (2u, 2u, 0u, System.UInt32.MaxValue))

    let take (name: string) (count: uint) =
        let apply input = input |> AsyncSeq.take (int count)
        create name apply Streaming

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

    let reduce (name: string) (reducer: AsyncSeq<'In> -> Async<'Out>) (profile: Profile) : Pipe<'In, 'Out> =
        let apply input = input |> reducer |> ofAsync
        create name apply profile

    let fold (name: string) (folder: 'State -> 'In -> 'State) (initial: 'State) (profile: Profile) : Pipe<'In, 'State> =
        let reducer (s: AsyncSeq<'In>) : Async<'State> =
            AsyncSeqExtensions.fold folder initial s
        reduce name reducer profile

    let mapNFold (name: string) (mapFn: 'In -> 'Mapped) (folder: 'State -> 'Mapped -> 'State) (state: 'State) (profile: Profile)
        : Pipe<'In, 'State> =
        
        let reducer (s: AsyncSeq<'In>) : Async<'State> =
            s
            |> AsyncSeq.map mapFn
            |> AsyncSeqExtensions.fold folder state

        reduce name reducer profile

//    let consumeWith (name: string) (consume: AsyncSeq<'T> -> Async<unit>) (profile: Profile) : Pipe<'T, unit> =
//        let reducer (s : AsyncSeq<'T>) = consume s          // Async<unit>
//        reduce name reducer  profile              // gives AsyncSeq<unit>

    let consumeWith (name: string) (consume: 'T -> unit) (profile: Profile) : Pipe<'T, unit> =
        let apply input = 
            asyncSeq {
                do! AsyncSeq.iterAsync (fun x -> async { consume x }) input
                yield ()
            }
        create name apply profile

    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    let compose (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
        let profile = Profile.combine p1.Profile p2.Profile
        let apply input = input |> p1.Apply |> p2.Apply
        create $"{p2.Name} {p1.Name}" apply profile

    /// Try to shrink a too‑hungry pipe to a cheaper profile.
    /// *You* control the downgrade policy here.
    let shrinkProfile (avail: uint64) (memPerElement: uint64) (depth: uint) (p : Pipe<'S,'T>) =
        let needed = Profile.estimateUsage p.Profile memPerElement
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
    let mapWindowed (name: string) (winSz: uint) (updateId: uint->'S->'S) (pad: uint) (zeroMaker: 'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) : Pipe<'S,'T> =
        let apply input =
            // AsyncSeqExtensions.windowed depth stride input
            AsyncSeqExtensions.windowedWithPad winSz updateId stride pad pad zeroMaker input
            |> AsyncSeq.collect (f >> AsyncSeq.ofSeq)
        let profile = Sliding (winSz,stride,emitStart,emitCount)
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

/// ProfileTransition describes *how* memory layout is expected to change:
/// - From: the input memory profile
/// - To: the expected output memory profile
type ProfileTransition =
    { From  : Profile
      To    : Profile }

module ProfileTransition =
    let create (fromProfile: Profile) (toProfile: Profile) : ProfileTransition =
        {
            From = fromProfile
            To   = toProfile
        }

/// Stage describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Stage + ProfileTransition: what happens
type Stage<'S,'T> =
    { Name        : string
      Pipe        : Pipe<'S,'T> 
      Transition  : ProfileTransition
      SizeUpdate  : uint64 -> uint64 (*A shape indication such as uint list -> uint list*) } 

module Stage =
    let create<'S,'T> (name: string) (pipe: Pipe<'S,'T>) (transition: ProfileTransition) (sizeUpdate: uint64 -> uint64) =
        { Name = name; Pipe = pipe; Transition = transition; SizeUpdate = sizeUpdate}

    let init<'S,'T> (name: string) (depth: uint) (mapper: uint -> 'T) (transition: ProfileTransition) (sizeUpdate: uint64 -> uint64) =
        let pipe = Pipe.init name depth mapper transition.From
        create name pipe transition sizeUpdate

(*
    let id<'T> () : Stage<'T, 'T> = // I don't think this is used
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.lift "id" Streaming (fun x -> x)
        create "id" pipe transition (fun s -> s)
*)

    let toPipe (stage : Stage<_,_>) = stage.Pipe

    let fromPipe (name: string) (transition: ProfileTransition) (sizeUpdate: uint64 -> uint64) (pipe: Pipe<'S, 'T>) : Stage<'S, 'T> =
        create name pipe transition sizeUpdate

    let compose (stage1 : Stage<'S,'T>) (stage2 : Stage<'T,'U>) : Stage<'S,'U> =
        let transition = ProfileTransition.create stage1.Transition.From stage2.Transition.To
        let pipe = Pipe.compose stage1.Pipe stage2.Pipe
        let sizeUpdate = stage1.SizeUpdate >> stage2.SizeUpdate
        create $"{stage2.Name} ∘ {stage1.Name}" pipe transition sizeUpdate

    let (-->) = compose

    let skip (name: string) (n:uint)  : Stage<'S, 'S> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.skip name n 
        create name pipe transition id

    let take (name: string) (n:uint)  : Stage<'S, 'S> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.take name n 
        create name pipe transition id

    let map<'S,'T> (name: string) (f: 'S -> 'T) (sizeUpdate: uint64 -> uint64) : Stage<'S, 'T> =
        let transition = ProfileTransition.create Streaming Streaming
        let apply input = input |> AsyncSeq.map f
        let pipe : Pipe<'S,'T> = Pipe.create name apply Streaming
        create name pipe transition sizeUpdate

(*
    let map (name: string) (f: 'U -> 'V) (stage: Stage<'In, 'U>) : Stage<'In, 'V> =
        let pipe = Pipe.map name f stage.Pipe
        let transition = ProfileTransition.create Streaming Streaming
        create name pipe transition (fun s -> s)
*)

    let map2 (name: string) (f: 'U -> 'V -> 'W) (stage1: Stage<'In, 'U>, stage2: Stage<'In, 'V>) (sizeUpdate: uint64 -> uint64): Stage<'In, 'W> =
        let pipe = Pipe.map2 name f stage1.Pipe stage2.Pipe
        let transition = ProfileTransition.create Streaming Streaming
        create name pipe transition sizeUpdate

    let reduce (name: string) (reducer: AsyncSeq<'In> -> Async<'Out>) (profile: Profile) (sizeUpdate: uint64 -> uint64): Stage<'In, 'Out> =
        let transition = ProfileTransition.create Streaming Constant
        let pipe = Pipe.reduce name reducer profile
        create name pipe transition sizeUpdate

    let fold<'S,'T> (name: string) (folder: 'T -> 'S -> 'T) (initial: 'T) (sizeUpdate: uint64 -> uint64): Stage<'S, 'T> =
        let transition = ProfileTransition.create Streaming Constant
        let pipe : Pipe<'S,'T> = Pipe.fold name folder initial Streaming
        create name pipe transition sizeUpdate

    let mapNFold (name: string) (mapFn: 'In -> 'Mapped) (folder: 'State -> 'Mapped -> 'State) (state: 'State) (profile: Profile) (sizeUpdate: uint64 -> uint64) : Stage<'In, 'State> =
        let transition = ProfileTransition.create Streaming Constant
        let pipe = Pipe.mapNFold name mapFn folder state profile
        create name pipe transition sizeUpdate

    // this assumes too much: Streaming and identity sizeUpdate!!!
    let liftUnary<'S,'T> (name: string) (f: 'S -> 'T) (sizeUpdate: uint64 -> uint64): Stage<'S, 'T> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.lift name Streaming f
        create name pipe transition sizeUpdate

    let liftWindowed<'S,'T when 'S: equality and 'T: equality> (name: string) (updateId: uint->'S->'S) (window: uint) (pad: uint) (zeroMaker: 'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) (sizeUpdate: uint64 -> uint64): Stage<'S, 'T> =
        let transition = ProfileTransition.create (Sliding (window,stride,emitStart,emitCount)) Streaming
        let pipe = Pipe.mapWindowed name window updateId pad zeroMaker stride emitStart emitCount f
        create name pipe transition sizeUpdate

    let tap (name: string) : Stage<'T, 'T> =
        liftUnary $"tap: {name}" (fun x -> printfn "[%s] %A" name x; x) (fun s->s)

    let tapIt (toString: 'T -> string) : Stage<'T, 'T> =
        liftUnary "tapIt" (fun x -> printfn "%s" (toString x); x) (fun s->s)

(*
    let internal tee (op: Stage<'In, 'T>) : Stage<'In, 'T> * Stage<'In, 'T> =
        let leftPipe, rightPipe = Pipe.tee op.Pipe
        let mk name pipe = 
            create $"{op.Name} - {name}" pipe op.Transition op.SizeUpdate
        mk "left" leftPipe, mk "right" rightPipe
*)
    let ignore<'T> () : Stage<'T, unit> =
        let pipe = Pipe.ignore ()
        let transition = ProfileTransition.create Streaming Constant
        create "ignore" pipe transition (fun s -> s)

    let consumeWith (name: string) (consume: 'T -> unit) : Stage<'T, unit> = 
        let pipe = Pipe.consumeWith name consume Streaming
        let transition = ProfileTransition.create Streaming Constant
        create name pipe transition (fun s -> s)

    let cast<'S,'T when 'S: equality and 'T: equality> name f : Stage<'S,'T> = // cast cannot change
        let apply input = input |> AsyncSeq.map f 
        let pipe = Pipe.create name apply Streaming
        let transition = ProfileTransition.create Streaming Streaming
        create name pipe transition (fun s -> s)

    let promoteConstantToStreaming (name: string) (depth: uint) (value: 'T) : Stage<unit, 'T> =
        let transition = ProfileTransition.create Constant Streaming
        let pipe = Pipe.init $"stream-of:{name}" depth (fun _ -> value) Streaming
        create $"promote:{name}" pipe transition (fun s->s)

    let promoteStreamingToSliding 
        (name: string)
        (winSz: uint)
        (updateId: uint -> 'T -> 'T)
        (pad: uint)
        (zeroMaker: 'T -> 'T)
        (stride: uint)
        (emitStart: uint)
        (emitCount: uint)
        : Stage<'T, 'T> = // Does not change shape

        let f (window: 'T list) =
            window
            |> List.skip (int emitStart)
            |> List.take (int emitCount)

        let pipe = Pipe.mapWindowed name winSz updateId pad zeroMaker stride emitStart emitCount f
        let transition = ProfileTransition.create Streaming Streaming

        create $"promote:{name}" pipe transition (fun s -> s)

    let promoteSlidingToSliding
        (name: string)
        (winSz: uint)
        (updateId: uint -> 'T -> 'T)
        (pad: uint)
        (zeroMaker: 'T -> 'T)
        (stride: uint)
        (emitStart: uint)
        (emitCount: uint)
        : Stage<'T, 'T> = // Does not change shape

        let f (window: 'T list) =
            window
            |> List.skip (int emitStart)
            |> List.take (int emitCount)

        let pipe = Pipe.mapWindowed name winSz updateId pad zeroMaker stride emitStart emitCount f
        let transition = ProfileTransition.create Streaming Streaming

        create $"promote:{name}" pipe transition (fun s -> s)

////////////////////////////////////////////////////////////
// Flow state monad
type SizeUpdate = uint64 -> uint64
type Flow<'S,'T> = // memory before, shape before, shapeContext before, Stage, memory after, shape after. 
        uint64 -> uint64 -> SizeUpdate -> Stage<'S,'T> * uint64 * uint64

module Flow =
    let returnM (stage : Stage<'S,'T>) : Flow<'S,'T> =
        fun memAvailable sizeS sizeUpdate ->
            if sizeS = 0UL then 
                stage, memAvailable, 0UL
            else
                let sizeT = sizeUpdate sizeS // Check again !!!!****!!!!*****!!!
                //let p' = Pipe.shrinkProfile memAvailable memPerElement depth stage.Pipe
                let memNeeded = Profile.estimateUsage stage.Pipe.Profile sizeT // sh -> memPerElement depth
                let shapeT = stage.SizeUpdate sizeS
                stage, memAvailable - memNeeded, shapeT

    let bindM (k: Stage<'A,'B> -> Flow<'B,'C>) (flowAB: Flow<'A,'B>) : Flow<'A,'C> =
        fun memAvailable shape shapeContextA ->
            let stage1, memAvailable1, shape1 = flowAB memAvailable shape shapeContextA
            let flowBC = k stage1
            let shapeContextB = fun sz -> if sz = 0UL then 0UL else shapeContextA sz
            let stage2, memAvailable2, shape2 = flowBC memAvailable1 shape1 shapeContextB
            (* 2025/07/25 Is this step necessary? The processing is taking care of the interfacing...
            // validate memory transition, if shape is known
            shape
            |> Option.iter (fun shape ->
                if stage1.Transition.To <> stage2.Transition.From || not (stage2.Transition.Check shape) then 
                    failwith $"Invalid memory transition: {stage1} → {stage2}")
            *)
            Stage.compose stage1 stage2, memAvailable2, shape2

////////////////////////////////////////////////////////////
// Pipeline flow controler
type Pipeline<'S,'T> = { 
    flow    : Flow<'S,'T>
    sizeUpdate : SizeUpdate // shape element's shape descriptor before transformation
    elmSize : uint64 // elment size before transformation
    nElems  : uint64  // number of elements before transformation
    mem     : uint64 // memory available before
    debug   : bool }

module Pipeline =
    let create<'S,'T when 'T: equality> (flow: Flow<'S,'T>) (mem : uint64) (elmSize: uint64) (nElems: uint64) (sizeUpdate : SizeUpdate) (debug: bool): Pipeline<'S, 'T> =
        { flow = flow; mem = mem; elmSize = elmSize; nElems = nElems; sizeUpdate = sizeUpdate; debug = debug }

    let asStage (pl: Pipeline<'In, 'Out>) : Stage<'In, 'Out> =
        let stage, _, _ = pl.flow pl.mem pl.elmSize pl.sizeUpdate
        stage

    //////////////////////////////////////////////////
    /// Source type operators
    let source (sizeUpdate: SizeUpdate) (availableMemory: uint64) : Pipeline<unit, unit> =
        let flow = fun _ _ _ -> failwith "Pipeline not started yet"
        create flow availableMemory 0UL 0UL sizeUpdate false

    let debug (sizeUpdate: SizeUpdate) (availableMemory: uint64) : Pipeline<unit, unit> =
        printfn $"Preparing pipeline - {availableMemory} B available"
        let flow = fun _ _ _ -> failwith "Pipeline not started yet"
        create flow availableMemory 0UL 0UL sizeUpdate true

    //////////////////////////////////////////////////////////////
    /// Composition operators
    let compose (pl: Pipeline<'a, 'b>) (stage: Stage<'b, 'c>) : Pipeline<'a, 'c> =
        if pl.debug then printfn $"[>=>] {stage.Name}"
        let flow = Flow.bindM (fun _ -> Flow.returnM stage) pl.flow 
        let elmSize,nElems, sizeUpdate = pl.elmSize, pl.nElems, pl.sizeUpdate  // stage may change the elmSize and nElems, which is to job of sizeUpdate
        create flow pl.mem elmSize nElems sizeUpdate pl.debug

    let (>=>) = compose

    //let shape' = pl.elmSize |> Option.map stage.SizeUpdate

(*
    let tee (pl: Pipeline<'In, 'T>) : Pipeline<'In, 'T> * Pipeline<'In, 'T> =
        if pl.debug then printfn "tee"
        let stage, mem, shape = pl.flow pl.mem pl.elmSize pl.sizeUpdate
        let left, right = Stage.tee stage
        let basePipeline pipe =
            create (Flow.returnM pipe) mem shape pl.sizeUpdate pl.debug
        basePipeline left, basePipeline right
*)

    /// parallel fanout with synchronization
    let (>=>>) 
        (pl: Pipeline<'In, 'S>) 
        (stg1: Stage<'S, 'U>, stg2: Stage<'S, 'V>) 
        : Pipeline<'In, ('U * 'V)> =

        if pl.debug then printfn $"[>=>>] ({stg1.Name}, {stg2.Name})"

        match pl.flow pl.mem pl.elmSize pl.sizeUpdate with
        | baseStg, mem', shape' ->

            // Compose both stages with the base
            //let stage1 = Stage.compose baseStg stg1
            //let stage2 = Stage.compose baseStg stg2

            // Combine them
            let sizeUpdate sh = (sh |> stg1.SizeUpdate) + (sh |> stg2.SizeUpdate)
            let stage = Stage.map2 ">=>>" (fun u v -> (u,v)) (stg1, stg2) sizeUpdate

            let flow = Flow.bindM (fun _ -> Flow.returnM stage) pl.flow // This seems wrong, why is input ignored?
            //let shape'' = shape' |> Option.map stage.SizeUpdate

            create flow pl.mem pl.elmSize pl.nElems pl.sizeUpdate pl.debug

    let (>>=>) f pl stage sizeUpdate = 
        (>=>) pl (Stage.map2 "zip2" f stage sizeUpdate)

    ///////////////////////////////////////////
    /// sink type operators
    let sink (pl: Pipeline<unit, unit>) : unit =
        if pl.debug then printfn "sink"
        if pl.elmSize = 0UL then 
            failwith "No stage provided shape information."
        else
            let stage, rest, _ = pl.flow pl.mem pl.elmSize pl.sizeUpdate
            if pl.debug then printfn $"Pipeline built - {rest} B still free\nRunning"
            stage |> Stage.toPipe |> Pipe.run
            if pl.debug then printfn "Done"

    let sinkList (pipelines: Pipeline<unit, unit> list) : unit = // shape of unit?
        pipelines |> List.iter sink

    let internal runToScalar (name:string) (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Pipeline<'In, 'T>) : 'R =
        if pl.debug then printfn "[runToScalar]"
        let stage, _, _ = pl.flow pl.mem pl.elmSize pl.sizeUpdate
        let pipe = stage.Pipe
        let input = AsyncSeq.singleton Unchecked.defaultof<'In>
        pipe.Apply input |> reducer |> Async.RunSynchronously

    let drainSingle (name:string) (pl: Pipeline<'S, 'T>) =
        if pl.debug then printfn "[drainSingle]"
        runToScalar name AsyncSeq.toListAsync pl
        |> function
            | [x] -> x
            | []  -> failwith $"[drainSingle] No result from {name}"
            | _   -> failwith $"[drainSingle] Multiple results from {name}, expected one."

    let drainList (name:string) (pl: Pipeline<'S, 'T>) =
        runToScalar name AsyncSeq.toListAsync pl

    let drainLast (name:string) (pl: Pipeline<'S, 'T>) =
        if pl.debug then printfn "[drainLast]"
        runToScalar name AsyncSeq.tryLast pl
        |> function
            | Some x -> x
            | None -> failwith $"[drainLast] No result from {name}"
