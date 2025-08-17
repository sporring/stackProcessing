module SlimPipeline

open FSharp.Control
open AsyncSeqExtensions

type SingleOrPair = Single of uint64 | Pair of uint64*uint64

module SingleOrPair =
    let isSingle v = 
        match v with Single _ -> true | _ -> false

    let map f v =
        match v with
            | Single elm -> Single (f elm)
            | Pair (left,right) -> Pair (f left, f right)

    let mapPair (f,g) v =
        match v with
            | Single elm -> Pair (f elm, g elm)
            | Pair (left,right) -> Pair (f left, g right)

    let fst v = 
        match v with
            | Single elm -> elm
            | Pair (left,_) -> left

    let snd v = 
        match v with
            | Single elm -> printfn "[SingleOrPair] internal warning: snd of Single"; elm
            | Pair (_,right) -> right

    let sum v =
        match v with
            | Single elm -> elm
            | Pair (left,right) -> left+right

    let add v w =
        match v,w with
            | Single elm1, Single elm2 -> Single (elm1+elm2)
            | Single elm, Pair (left,right) 
            | Pair (left,right), Single elm -> Pair (elm+left,elm+right)
            | Pair (left1,right1), Pair (left2,right2) -> Pair (left1+left2,right1+right2)

////////////////////////////////////////////////////////////
/// The memory usage strategies during image processing.
type Profile =
    | Unit // Slice by slice independently
    | Streaming // Slice by slice independently
    | Sliding of uint * uint * uint * uint * uint  // (window, stride, pad, emitStart, emitCount)
    // | Full // Full = Sliding depth 1 0 depth

module Profile =
    let estimateUsage (profile: Profile) (memPerElement: uint64) : uint64 =
        match profile with
            | Unit -> 0uL
            | Streaming -> memPerElement
            | Sliding (windowSize, _, _, _, _) -> memPerElement * uint64 windowSize

    let combine (prof1: Profile) (prof2: Profile): Profile  = 
        let result =
            match prof1, prof2 with
    //        | Full, _ 
    //        | _, Full -> Full // conservative fallback
            | Sliding (sz1,str1,pad1, emitS1,emitN1), Sliding (sz2,str2,pad2, emitS2,emitN2) -> Sliding ((max sz1 sz2), min str1 str2, max pad1 pad2, min emitS1 emitS2, max emitN1 emitN2) // don't really know what stride rule should be
            | Sliding (sz,str,pad, emitS,emitN), _ 
            | _, Sliding (sz,str,pad,emitS,emitN) -> Sliding (sz,str,pad,emitS,emitN)
            | Streaming, _
            | _, Streaming -> Streaming
            | Unit, Unit -> Unit

        result

////////////////////////////////////////////////////////////
/// A configurable image processing step that operates on image slices.
/// Pipe describes *how* to do it:
/// - Encapsulates the concrete execution logic
/// - Defines memory usage behavior
/// - Takes and returns AsyncSeq streams
/// - Pipe + WindowedProcessor: How it’s computed 
type Pipe<'S,'T> = {
    Name: string // Name of the process
    Apply: bool -> AsyncSeq<'S> -> AsyncSeq<'T>
    Profile: Profile
}

module private Pipe =
    let create (name: string) (apply: bool -> AsyncSeq<'S> -> AsyncSeq<'T>) (profile: Profile) : Pipe<'S,'T> =
        { Name = name; Apply = apply; Profile = profile }

    let runWith (debug: bool) (input: 'S) (pipe: Pipe<'S, 'T>) : Async<unit> =
        AsyncSeq.singleton input
        |> pipe.Apply debug
        |> AsyncSeq.iterAsync (fun _ -> async.Return())

    let run (debug: bool) (pipe: Pipe<unit, unit>) : unit =
        runWith debug () pipe |> Async.RunSynchronously

    let lift (name: string) (profile: Profile) (f: 'S -> 'T) : Pipe<'S,'T> =
        let apply debug  input = input |> AsyncSeq.map f
        create name apply profile

    let empty (name: string) : Pipe<unit,unit> =
        let apply debug input = AsyncSeq.empty
        create name apply Streaming

    let init<'T> (name: string) (depth: uint) (mapper: int -> 'T) (profile: Profile) : Pipe<unit,'T> =
        let apply debug input = AsyncSeq.init (int64 depth) (fun (i:int64) -> mapper (int i))
        create name apply profile

    let liftConsume (name: string) (profile: Profile) (release: 'S->unit) (f: 'S -> 'T) : Pipe<'S,'T> =
        let folder input = 
            let output = f input
            release input
            output
        let apply debug  input = input |> AsyncSeq.map folder
        create name apply profile

    let skip (name: string) (count: uint) =
        let apply debug input = input |> AsyncSeq.skip (int count)
        create "skip" apply (Sliding (2u, 2u, 0u, 0u, System.UInt32.MaxValue))

    let take (name: string) (count: uint) =
        let apply debug input = input |> AsyncSeq.take (int count)
        create name apply Streaming

    let map (name: string) (mapper: 'U -> 'V) (pipe: Pipe<'In, 'U>) : Pipe<'In, 'V> =
        let apply debug input = input |> pipe.Apply debug |> AsyncSeq.map mapper
        create name apply pipe.Profile
(*
    let wrap (name: string) (wrapper: ('In * 'U) -> 'V) (pipe: Pipe<'In, 'U>) : Pipe<'In, 'V> =
        let apply debug input = 
            let output = input |> pipe.Apply debug
            let zipped = AsyncSeq.zip input output
            let mapped = AsyncSeq.map wrapper zipped
            mapped
        create name apply pipe.Profile
*)
    type TeeMsg<'T> =
        | Left of AsyncReplyChannel<'T option>
        | Right of AsyncReplyChannel<'T option>

    let tee (debug: bool) (p: Pipe<'In, 'T>) : Pipe<'In, 'T> * Pipe<'In, 'T> =
        // Shared lazy value to ensure Apply is only triggered once
        let mutable shared: Lazy<AsyncSeq<'T> * AsyncSeq<'T>> option = None
        let syncRoot = obj()

        let makeShared (input: AsyncSeq<'In>) =
            let src = p.Apply debug input  // Only called once

            let agent = MailboxProcessor.Start(fun inbox ->
                async {
                    let enum = (AsyncSeq.toAsyncEnum src).GetAsyncEnumerator()
                    let mutable current: 'T option = None
                    let mutable consumed = (true, true)
                    let mutable finished = false

                    let rec loop () = async {
                        if current.IsNone && not finished then
                            let! hasNext = enum.MoveNextAsync().AsTask() |> Async.AwaitTask
                            if hasNext then
                                current <- Some enum.Current
                                consumed <- (false, false)
                            else
                                finished <- true

                        let! msg = inbox.Receive()
                        match msg, current with
                        | Left reply, Some v when not (fst consumed) ->
                            reply.Reply(Some v)
                            consumed <- (true, snd consumed)
                        | Right reply, Some v when not (snd consumed) ->
                            reply.Reply(Some v)
                            consumed <- (fst consumed, true)
                        | Left reply, None when finished ->
                            reply.Reply(None)
                        | Right reply, None when finished ->
                            reply.Reply(None)
                        | _ -> ()  // Ignore duplicate requests

                        if consumed = (true, true) then
                            current <- None

                        return! loop ()
                    }

                    do! loop ()
                })

            let makeStream tag =
                asyncSeq {
                    let mutable done_ = false
                    while not done_ do
                        let! vOpt = agent.PostAndAsyncReply(tag)
                        match vOpt with
                        | Some v -> yield v
                        | None -> done_ <- true
                }

            makeStream Left, makeStream Right

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
                        lazyStreams.Value
                )

        let makeTeePipe name pick =
            let apply (debug: bool) input =
                let left, right = getShared input
                pick (left, right)
            create $"{p.Name}-tee-{name}" apply p.Profile

        makeTeePipe "left" fst, makeTeePipe "right" snd

    let id (name: string) : Pipe<'T, 'T> =
        let apply (debug: bool) input = input
        create name apply Streaming  // or use profile passed as arg if you prefer

    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    let compose (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
        let profile = Profile.combine p1.Profile p2.Profile
        let apply debug input = input |> p1.Apply debug |> p2.Apply debug 
        create $"{p2.Name} {p1.Name}" apply profile

    let map2Sync (name: string) (debug: bool) (f: 'U -> 'V -> 'W) (pipe1: Pipe<'In, 'U>) (pipe2: Pipe<'In, 'V>) : Pipe<'In, 'W> =
        let teeBasePipe = id "tee" // Or take the upstream pipe shared by both
        let shared1, shared2 = tee debug teeBasePipe

        let composedPipe1 = compose shared1 pipe1
        let composedPipe2 = compose shared2 pipe2

        let apply debug input =
            let stream1 = composedPipe1.Apply debug input
            let stream2 = composedPipe2.Apply debug input
            AsyncSeqExtensions.zipConcurrent stream1 stream2 
            |> AsyncSeq.map (fun (u, v) -> 
                let result = f u v; 
                result)
            
        let profile = max pipe1.Profile pipe2.Profile
        create name apply profile

    let map2 (name: string) (debug: bool) (f: 'U -> 'V -> 'W) (pipe1: Pipe<'In, 'U>) (pipe2: Pipe<'In, 'V>) : Pipe<'In, 'W> =
        let apply debug input =
                let stream1 = input |> pipe1.Apply debug
                let stream2 = input |> pipe2.Apply debug
                AsyncSeq.zip stream1 stream2 |> AsyncSeq.map (fun (u, v) -> f u v)
        let profile = max pipe1.Profile pipe2.Profile
        create name apply profile

    let reduce (name: string) (reducer: bool -> AsyncSeq<'In> -> Async<'Out>) (profile: Profile) : Pipe<'In, 'Out> =
        let apply debug input = input |> reducer debug |> ofAsync
        create name apply profile

    let fold (name: string) (folder: 'State -> 'In -> 'State) (initial: 'State) (profile: Profile) : Pipe<'In, 'State> =
        let reducer debug (s: AsyncSeq<'In>) : Async<'State> =
            AsyncSeq.fold folder initial s
        reduce name reducer profile

    let consumeWith (name: string) (consume: bool -> int -> 'T -> unit) (profile: Profile) : Pipe<'T, unit> =
        let apply debug input = 
            asyncSeq {
                do! AsyncSeq.iteriAsync (fun i x -> async { consume debug i x; }) input
                yield ()
            }
        create name apply profile

(*
    /// mapWindowed keeps a running window along the slice direction of depth images
    /// and processes them by f. The stepping size of the running window is stride.
    /// So if depth is 3 and stride is 1 then first image 0,1,2 is sent to f, then 1, 2, 3
    /// and so on. If depth is 3 and stride is 3, then it'll be image 0, 1, 2 followed by
    /// 3, 4, 5. It is also possible to use this for sampling, e.g., setting depth to 1
    /// and stride to 2 sends every second image to f.  
    let mapWindowed (name: string) (winSz: uint) (pad: uint) (zeroMaker: int->'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) : Pipe<'S,'T> =
        //printfn "mapWindowed winSz=%A stride=%A pad=%A" winSz stride pad
        let apply debug input =
            // AsyncSeqExtensions.windowed depth stride input
            AsyncSeqExtensions.windowedWithPad winSz stride pad pad zeroMaker input
            |> AsyncSeq.collect (fun input -> 
            input |> f |> AsyncSeq.ofSeq)
        let profile = Sliding (winSz,stride,pad,emitStart,emitCount)
        create name apply profile
*)

    // window : wraps AsyncSeqExtensions.windowedWithPad
    let window
        (name       : string)
        (winSz      : uint)
        (pad        : uint)
        (zeroMaker  : int -> 'T -> 'T)
        (stride     : uint)
        : Pipe<'T, 'T list> =

        let apply _debug (input: AsyncSeq<'T>) : AsyncSeq<'T list> =
            // Produces an AsyncSeq of windows (each window is a 'T list)
            AsyncSeqExtensions.windowedWithPad winSz stride pad pad zeroMaker input
        let profile = Sliding (winSz, stride, pad, 0u, winSz)
        create name apply profile

    // collect : flattens an AsyncSeq<'T list> to AsyncSeq<'T>
    let collect<'T> (name: string) : Pipe<'T list, 'T> =
        let apply _debug (input: AsyncSeq<'T list>) : AsyncSeq<'T> =
            input |> AsyncSeq.collect (fun (xs: 'T list) -> AsyncSeq.ofSeq xs)
        let profile = Streaming
        create name apply profile

    let ignore clean : Pipe<'T, unit> =
        let apply debug input =
            asyncSeq {
                // Consume the stream without doing anything
                do! AsyncSeq.iterAsync (fun elm -> clean elm; async.Return()) input
                // Then emit one unit
                yield ()
            }
        create "ignore" apply Unit

    let ignorePairs (cleanFst, cleanSnd) : Pipe<'S*'T, unit> =
        let apply debug input =
            asyncSeq {
                // Consume the stream without doing anything
                do! AsyncSeq.iterAsync (fun (a,b) -> cleanFst a; cleanSnd b; async.Return()) input
                // Then emit one unit
                yield ()
            }
        create "ignorePairs" apply Unit

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

type MemoryNeed = uint64->uint64
type MemoryNeedWrapped = SingleOrPair -> SingleOrPair
type NElemsTransformation = uint64 -> uint64

/// Stage describes *what* should be done:
/// - Contains high-level metadata
/// - Encodes memory transition intent
/// - Suitable for planning, validation, and analysis
/// - Stage + ProfileTransition: what happens
type Stage<'S,'T> =
    { Name       : string
      Pipe       : Pipe<'S,'T> 
      Transition : ProfileTransition
      MemoryNeed : MemoryNeedWrapped
      NElemsTransformation : NElemsTransformation
      } 

module Stage =
    let create<'S,'T> (name: string) (pipe: Pipe<'S,'T>) (transition: ProfileTransition) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation) =
        let wrapMemoryNeed = SingleOrPair.sum >> memoryNeed >> Single
        { Name = name; Pipe = pipe; Transition = transition; MemoryNeed = wrapMemoryNeed; NElemsTransformation = nElemsTransformation}

    let createWrapped<'S,'T> (name: string) (pipe: Pipe<'S,'T>) (transition: ProfileTransition) (wrapMemoryNeed: MemoryNeedWrapped) (nElemsTransformation: NElemsTransformation) =
        { Name = name; Pipe = pipe; Transition = transition; MemoryNeed = wrapMemoryNeed; NElemsTransformation = nElemsTransformation}

    let empty (name: string) =
        let pipe = Pipe.empty name
        let transition = ProfileTransition.create Streaming Streaming
        let memoryNeed _ = 0UL
        let NElemsTransformation = id
        create name pipe transition memoryNeed  NElemsTransformation

    let init<'S,'T> (name: string) (depth: uint) (mapper: int -> 'T) (transition: ProfileTransition) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation) =
        let pipe = Pipe.init name depth mapper transition.From
        create name pipe transition memoryNeed nElemsTransformation

    let idOp<'T> (name: string) : Stage<'T, 'T> = // I don't think this is used
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.id name
        create "id" pipe transition id id

    let toPipe (stage : Stage<_,_>) = stage.Pipe

    let fromPipe (name: string) (transition: ProfileTransition) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation) (pipe: Pipe<'S, 'T>) : Stage<'S, 'T> =
        create name pipe transition memoryNeed nElemsTransformation

    let compose (stage1 : Stage<'S,'T>) (stage2 : Stage<'T,'U>) : Stage<'S,'U> =
        let transition = ProfileTransition.create stage1.Transition.From stage2.Transition.To
        let pipe = Pipe.compose stage1.Pipe stage2.Pipe
        let memoryNeed nElems = SingleOrPair.add (stage1.MemoryNeed nElems) (stage2.MemoryNeed nElems)
        let nElemsTransformation = stage1.NElemsTransformation >> stage2.NElemsTransformation
        createWrapped $"{stage2.Name} o {stage1.Name}" pipe transition memoryNeed nElemsTransformation

    let (-->) = compose

    let skip (name: string) (n:uint) : Stage<'S, 'S> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.skip name n 
        create name pipe transition id id

    let take (name: string) (n:uint) : Stage<'S, 'S> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.take name n 
        create name pipe transition id id

    let map<'S,'T> (name: string) (f: 'S -> 'T) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation) : Stage<'S, 'T> =
        let transition = ProfileTransition.create Streaming Streaming
        let apply debug input = input |> AsyncSeq.map f
        let pipe : Pipe<'S,'T> = Pipe.create name apply Streaming
        create name pipe transition memoryNeed nElemsTransformation // Not right!!!

(*
    let wrap (name: string) (f: ('In * 'U) -> 'V) (stage: Stage<'In, 'U>) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation) : Stage<'In, 'V> =
        let pipe = Pipe.wrap name f stage.Pipe
        let transition = ProfileTransition.create Streaming Streaming
        create name pipe transition memoryNeed nElemsTransformation
*)
    let map1 (name: string) (f: 'U -> 'V) (stage: Stage<'In, 'U>) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation) : Stage<'In, 'V> =
        let pipe = Pipe.map name f stage.Pipe
        let transition = ProfileTransition.create Streaming Streaming
        create name pipe transition memoryNeed nElemsTransformation

    let map2 (name: string) (debug: bool) (f: 'U -> 'V -> 'W) (stage1: Stage<'In, 'U>) (stage2: Stage<'In, 'V>) (memoryNeed: MemoryNeedWrapped) (nElemsTransformation: NElemsTransformation): Stage<'In, 'W> =
        let pipe = Pipe.map2 name debug f stage1.Pipe stage2.Pipe
        let transition = ProfileTransition.create Streaming Streaming
        createWrapped name pipe transition memoryNeed nElemsTransformation

    let map2Sync (name: string) (debug: bool) (f: 'U -> 'V -> 'W) (stage1: Stage<'In, 'U>) (stage2: Stage<'In, 'V>) (memoryNeed: MemoryNeedWrapped) (nElemsTransformation: NElemsTransformation): Stage<'In, 'W> =
        let pipe = Pipe.map2Sync name debug f stage1.Pipe stage2.Pipe
        let transition = ProfileTransition.create Streaming Streaming
        createWrapped name pipe transition memoryNeed nElemsTransformation

    let reduce (name: string) (reducer: bool -> AsyncSeq<'In> -> Async<'Out>) (profile: Profile) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation): Stage<'In, 'Out> =
        let transition = ProfileTransition.create Streaming Unit
        let pipe = Pipe.reduce name reducer profile
        create name pipe transition memoryNeed nElemsTransformation // Check !!!

    let fold<'S,'T> (name: string) (folder: 'T -> 'S -> 'T) (initial: 'T) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation): Stage<'S, 'T> =
        let transition = ProfileTransition.create Streaming Unit
        let pipe : Pipe<'S,'T> = Pipe.fold name folder initial Streaming
        create name pipe transition memoryNeed nElemsTransformation // Check !!!

    let window
        (name       : string)
        (winSz      : uint)
        (pad        : uint)
        (zeroMaker  : int -> 'T -> 'T)
        (stride     : uint)
        : Stage<'T, 'T list> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.window name winSz pad zeroMaker stride
        fromPipe name transition id id pipe

    let collect (name: string) : Stage<'T list, 'T> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.collect name 
        fromPipe name transition id id pipe

    let liftUnary<'S,'T> (name: string) (f: 'S -> 'T) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation): Stage<'S, 'T> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.lift name Streaming f
        create name pipe transition memoryNeed nElemsTransformation

    let liftConsumeUnary<'S,'T> (name: string) (release: 'S->unit) (f: 'S -> 'T) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation): Stage<'S, 'T> =
        let transition = ProfileTransition.create Streaming Streaming
        let pipe = Pipe.liftConsume name Streaming release f
        create name pipe transition memoryNeed nElemsTransformation
(*
    let liftWindowed<'S,'T when 'S: equality and 'T: equality> (name: string) (window: uint) (pad: uint) (zeroMaker: int->'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) (memoryNeed: MemoryNeed) (nElemsTransformation: NElemsTransformation): Stage<'S, 'T> =
        let transition = ProfileTransition.create (Sliding (window,stride,pad,emitStart,emitCount)) Streaming
        let pipe = Pipe.mapWindowed name window pad zeroMaker stride emitStart emitCount f
        create name pipe transition memoryNeed nElemsTransformation
*)
    let tapItOp (name: string) (toString: 'T -> string) : Stage<'T, 'T> =
        liftUnary name (fun x -> printfn "%s" (toString x); x) id id

    let tapIt (toString: 'T -> string) : Stage<'T, 'T> =
        tapItOp "tapIt" toString

    let tap (name: string) : Stage<'T, 'T> =
        tapItOp $"tap: {name}" (fun x -> sprintf "[%s] %A" name x)

(*
    let internal tee (op: Stage<'In, 'T>) : Stage<'In, 'T> * Stage<'In, 'T> =
        let leftPipe, rightPipe = Pipe.tee op.Pipe
        let mk name pipe = 
            create $"{op.Name} - {name}" pipe op.Transition op.SizeUpdate
        mk "left" leftPipe, mk "right" rightPipe
*)
    let ignore<'T> clean : Stage<'T, unit> =
        let pipe = Pipe.ignore clean
        let transition = ProfileTransition.create Streaming Unit
        create "ignore" pipe transition id id // check !!!!

    let ignorePairs<'S,'T> (cleanFst, cleanSnd) : Stage<'S*'T, unit> =
        let pipe = Pipe.ignorePairs (cleanFst, cleanSnd)
        let transition = ProfileTransition.create Streaming Unit
        create "ignorePairs" pipe transition id id // check !!!!

    let consumeWith (name: string) (consume: bool -> int -> 'T -> unit) (memoryNeed: MemoryNeed) : Stage<'T, unit> = 
        let pipe = Pipe.consumeWith name consume Streaming
        let transition = ProfileTransition.create Streaming Unit
        create name pipe transition memoryNeed (fun _ -> 0UL) // Check !!!!

    let cast<'S,'T when 'S: equality and 'T: equality> name f (memoryNeed: MemoryNeed): Stage<'S,'T> = // cast cannot change
        let apply debug input = input |> AsyncSeq.map f 
        let pipe = Pipe.create name apply Streaming
        let transition = ProfileTransition.create Streaming Streaming
        create name pipe transition memoryNeed id // Type calculation needs to be updated!!!!

////////////////////////////////////////////////////////////
// Pipeline flow controler
type Pipeline<'S,'T> = { 
    stage      : Stage<'S,'T> option // the function to be applied, when the pipeline is run
    nElems     : SingleOrPair // number of elments before transformation - this could be single or pair
    length     : uint64 // length of the sequence, the pipeline is applied to
    memAvail   : uint64 // memory available for the pipeline
    memPeak    : uint64 // the pipeline's estimated peak memory consumption
    debug      : bool }

module Pipeline =
    let create<'S,'T when 'T: equality> (stage: Stage<'S,'T> option) (memAvail : uint64) (memPeak: uint64) (nElems: uint64) (length: uint64) (debug: bool): Pipeline<'S, 'T> =
        { stage = stage; memAvail = memAvail; memPeak = memPeak; nElems = Single nElems; length = length; debug = debug }

    let createWrapped<'S,'T when 'T: equality> (stage: Stage<'S,'T> option) (memAvail : uint64) (memPeak: uint64) (nElems: SingleOrPair) (length: uint64) (debug: bool): Pipeline<'S, 'T> =
        { stage = stage; memAvail = memAvail; memPeak = memPeak; nElems = nElems; length = length; debug = debug }
    //////////////////////////////////////////////////
    /// Source type operators
    let source (availableMemory: uint64) : Pipeline<unit, unit> =
        create None availableMemory 0UL 0UL 0UL false

    let debug (availableMemory: uint64) : Pipeline<unit, unit> =
        printfn $"[debug] Preparing pipeline - {availableMemory} B available"
        let result = create None availableMemory 0UL 0UL 0UL true
        printfn $"[debug] Done"
        result

    //////////////////////////////////////////////////////////////
    /// Composition operators
    let composeOp (name: string) (pl: Pipeline<'a, 'b>) (stage: Stage<'b, 'c>) : Pipeline<'a, 'c> =
        if pl.debug then printfn $"[{name}] {stage.Name}"

        let memNeeded = pl.nElems |> stage.MemoryNeed  |> SingleOrPair.sum // Stage.MemoryNeed must be updated as well
        let memPeak = max pl.memPeak memNeeded
        if (not pl.debug) && memPeak > pl.memAvail then
            failwith $"Out of available memory: {stage.Name} requested {memNeeded} B but have only {pl.memAvail} B"

        let stage' = Option.map (fun stg -> Stage.compose stg stage) pl.stage
        let nElems' = SingleOrPair.map stage.NElemsTransformation pl.nElems // Stage NElemsTransformation must be updated as well
        createWrapped stage' pl.memAvail memPeak nElems' pl.length pl.debug

    let (>=>) (pl: Pipeline<'a, 'b>) (stage: Stage<'b, 'c>) : Pipeline<'a, 'c> =
        composeOp $">=>" pl stage

    let map (name: string) (f: 'U->'V) (pl: Pipeline<'In,'U>) : Pipeline<'In,'V> =
        if pl.debug then printfn $"[{name}]"
        let memoryNeed m = 2UL*m // assuming simple transformation
        let nElemsTransformation = id
        let stage =
            match pl.stage with
            | Some stg -> 
                Stage.map1 name f stg memoryNeed nElemsTransformation // nElemsTransformation is unchanged by map per definition 
            | None -> failwith "Pipeline.map cannot map to empty stage"
        let nElems' = SingleOrPair.map stage.NElemsTransformation pl.nElems
        let memNeeded = pl.nElems |> stage.MemoryNeed  |> SingleOrPair.sum
        let memPeak = max pl.memPeak memNeeded
        if (not pl.debug) && memPeak > pl.memAvail then
            failwith $"Out of available memory: {stage.Name} requested {memoryNeed} B but have only {pl.memAvail} B"
        createWrapped (Some stage) pl.memAvail memPeak nElems' pl.length pl.debug
        
    /// parallel execution of non-synchronised streams
    let internal zipOp (name:string) (pl1: Pipeline<'In, 'U>) (pl2: Pipeline<'In, 'V>) : Pipeline<'In, ('U * 'V)> =
        match pl1.stage,pl2.stage with
            Some stage1, Some stage2 ->
                let debug = (pl1.debug || pl2.debug)
                if debug then printfn $"[{name}] ({stage1.Name}, {stage2.Name})"

                if pl1.length <> pl2.length then
                    failwith $"[{name}] pipelines to be ziped must be over sequences of equal lengths {pl1.length} <> {pl2.length}"

                if pl1.nElems <> pl2.nElems then
                    failwith $"[{name}] pipelines to be ziped must be of equal sized input {pl1.nElems} <> {pl2.nElems}"

                let nElems = pl1.nElems
                let nElemsTransformed1 = SingleOrPair.map stage1.NElemsTransformation  nElems
                let nElemsTransformed2 = SingleOrPair.map stage2.NElemsTransformation  nElems

                if nElemsTransformed1 <> nElemsTransformed2 then
                    failwith $"[{name}] pipelines to be ziped must be of equal sized output {nElemsTransformed1|>SingleOrPair.sum} <> {nElemsTransformed2|>SingleOrPair.sum}"
                if not (SingleOrPair.isSingle nElemsTransformed1) || not (SingleOrPair.isSingle nElemsTransformed2) then
                    failwith $"[{name}] Can't (yet) zip pipeline-pairs (isSingle({stage1.Name})={SingleOrPair.isSingle nElemsTransformed1} and isSingle({stage2.Name})={SingleOrPair.isSingle nElemsTransformed2})"

                // Create a non-synced stage
                let memoryNeed nElems = 
                    SingleOrPair.add 
                        (nElems |> SingleOrPair.fst |> Single |> stage1.MemoryNeed) 
                        (nElems |> SingleOrPair.snd |> Single |> stage2.MemoryNeed)
                let nElemsTransformation = stage1.NElemsTransformation // Transformation of result equal to any of the input
                let stage =
                    Stage.map2 $"({stage1.Name},{stage2.Name})" debug (fun U V -> (U,V)) stage1 stage2 memoryNeed nElemsTransformation

                // Check memory constraints for each individual stream
                let memNeeded = nElems |> memoryNeed
                let maxMemPeak = max pl1.memPeak pl2.memPeak
                let memPeak = max maxMemPeak (SingleOrPair.sum memNeeded)
                let maxMemAvail = max pl1.memAvail pl2.memAvail
                if (not debug) && (memPeak > maxMemAvail) then
                    failwith $"Out of available memory: {stage.Name} requested {memNeeded|>SingleOrPair.fst}+{memNeeded|>SingleOrPair.snd}={memPeak} B but have only {maxMemAvail} B"
 
                createWrapped (Some stage) maxMemAvail memPeak nElems pl1.length debug
            | _,_ -> failwith $"[{name}] Cannot zip with an empty pipeline"

    /// parallel execution of non-synchronised streams
    let zip (pl1: Pipeline<'In, 'U>) (pl2: Pipeline<'In, 'V>) : Pipeline<'In, ('U * 'V)> = zipOp "zip" pl1 pl2

    /// parallel execution of synchronised streams
    let (>=>>)
        (pl: Pipeline<'In, 'S>) 
        (stg1: Stage<'S, 'U>, stg2: Stage<'S, 'V>) 
        : Pipeline<'In, 'U * 'V> =
        if stg1.Pipe.Profile <> stg2.Pipe.Profile then 
            failwith $"[>=>>] can only apply stages with the same streaming profile ({stg1.Pipe.Profile} <> {stg2.Pipe.Profile})"

        let memoryNeed nElems = 
            SingleOrPair.add 
                (nElems |> SingleOrPair.fst |> Single |> stg1.MemoryNeed) 
                (nElems |> SingleOrPair.snd |> Single |> stg2.MemoryNeed)

        let nElemsTransformation = stg1.NElemsTransformation 
        let nElemsTransformation2 = stg2.NElemsTransformation 
        let nElems = SingleOrPair.map nElemsTransformation pl.nElems
        let nElems2 = SingleOrPair.map nElemsTransformation2 pl.nElems
        if nElems <> nElems2 then
            failwith $"[>=>>] Cannot zip pipelines with different number of elements {nElems} vs {nElems2}"

        // Combine both stages in a zip-like stage
        let stage = Stage.map2Sync $"({stg1.Name},{stg2.Name})" pl.debug (fun u v -> (u, v)) stg1 stg2 memoryNeed nElemsTransformation

        composeOp ">=>>" pl stage

    let (>>=>) (pl: Pipeline<'In,'U*'V>) ((f: 'U -> 'V -> 'W)) : Pipeline<'In,'W>  = 
        map ">>=>" (fun (u,v) -> f u v) pl
(*
    let (>>=>) (pl: Pipeline<'In,'U*'V>) (stage: Stage<'U*'V,'W>) : Pipeline<'In,'W>  = 
        composeOp ">>=>" pl stage
*)

    let (>>=>>) (f: ('U*'V) -> ('S*'T)) (pl: Pipeline<'In,'U*'V>) (stage: Stage<'U*'V,'S*'T>): Pipeline<'In,'S*'T>  = 
        map ">>=>>" f pl 

    ///////////////////////////////////////////
    /// sink type operators
    let sink (pl: Pipeline<unit, unit>) : unit =
        if not pl.debug && (pl.memPeak > pl.memAvail) then
            failwith $"Not enough memory for the pipeline {pl.memPeak} > {pl.memAvail}"
        if pl.debug then printfn $"[sink] Running pipeline with an estimated {pl.memPeak/1024UL} / {pl.memAvail/1024UL} KB of memory use"
        Option.map (fun stage -> stage |> Stage.toPipe |> Pipe.run pl.debug) pl.stage |> ignore
        if pl.debug then printfn "[sink] Done"

    let sinkList (pipelines: Pipeline<unit, unit> list) : unit = // shape of unit?
        pipelines |> List.iter sink

    let internal runToScalar (name:string) (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Pipeline<unit, 'T>) : 'R =
        if pl.memPeak > pl.memAvail then
            failwith $"Not enough memory for the pipeline {pl.memPeak} > {pl.memAvail}"
        if pl.debug then printfn $"[{name}] Running pipeline with an estimated {pl.memPeak/1024UL} / {pl.memAvail/1024UL} KB memory use" 

        let stage = pl.stage
        match stage with
            Some stg -> 
                let input = AsyncSeq.singleton ()
                input |> stg.Pipe.Apply pl.debug |> reducer |> Async.RunSynchronously
            | _ -> failwith $"[{name}] Pipeline is empty"

    let drainSingle (name:string) (pl: Pipeline<unit, 'T>) =
        runToScalar name AsyncSeq.toListAsync pl
        |> function
            | [x] -> x
            | []  -> failwith $"[drainSingle] No result from {name}"
            | _   -> failwith $"[drainSingle] Multiple results from {name}, expected one."

    let drainList (name:string) (pl: Pipeline<unit, 'T>) =
        runToScalar name AsyncSeq.toListAsync pl

    let drainLast (name:string) (pl: Pipeline<unit, 'T>) =
        runToScalar name AsyncSeq.tryLast pl
        |> function
            | Some x -> x
            | None -> failwith $"[drainLast] No result from {name}"
