module SlimPipeline

open System.Collections.Generic
open FSharp.Control
open AsyncSeqExtensions

type SingleOrPair = Single of uint64 | Pair of uint64*uint64

type Window<'T> =
    { Items: 'T list
      EmitRange: uint * uint }

module Window =
    let create emitStart emitCount items =
        { Items = items
          EmitRange = emitStart, emitCount }

    let singleton item =
        create 0u 1u [ item ]

    let emitItems window =
        let start, count = window.EmitRange
        window.Items
        |> List.skip (int start)
        |> List.take (min (int count) (max 0 (window.Items.Length - int start)))

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

    let index1 v = 
        match v with
            | Single elm -> Single elm
            | Pair (left,_) -> Single left

    let index2 v = 
        match v with
            | Single elm -> Single elm
            | Pair (_,right) -> Single right

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
            | Pair (left,right) -> left+right |> Single
            | _ -> v

    let add v w =
        match v,w with
            | Single elm1, Single elm2 -> Single (elm1+elm2)
            | Single elm, Pair (left,right) 
            | Pair (left,right), Single elm -> Pair (elm+left,elm+right)
            | Pair (left1,right1), Pair (left2,right2) -> Pair (left1+left2,right1+right2)

////////////////////////////////////////////////////////////
/// The memory usage strategies during image processing.
type Profile =
    | Unit // empty
    | Constant // independent on slices
    | Streaming // Slice by slice independently
    | Window of uint * uint * uint * uint * uint  // Sliding window of (size, stride, pad, emitStart, emitCount)
    // | Full // Full = Window depth 1 0 depth

module Profile =
    let estimateUsage (profile: Profile) (memPerElement: uint64) : uint64 =
        match profile with
            | Unit -> 0uL
            | Constant -> memPerElement
            | Streaming -> memPerElement
            | Window (windowSize, _, _, _, _) -> memPerElement * uint64 windowSize

    let combine (prof1: Profile) (prof2: Profile) : Profile  = 
        let result =
            match prof1, prof2 with
    //        | Full, _ 
    //        | _, Full -> Full // conservative fallback
            | Window (sz1,str1,pad1, emitS1,emitN1), Window (sz2,str2,pad2, emitS2,emitN2) -> Window ((max sz1 sz2), min str1 str2, max pad1 pad2, min emitS1 emitS2, max emitN1 emitN2) // don't really know what stride rule should be
            | Window (sz,str,pad, emitS,emitN), _ 
            | _, Window (sz,str,pad,emitS,emitN) -> Window (sz,str,pad,emitS,emitN)
            | Streaming, _
            | _, Streaming -> Streaming
            | Constant, _
            | _, Constant -> Constant
            | Unit, _
            | _, Unit -> Unit

        result

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
    type TeeSide =
        | Left
        | Right

    type TeeMsg<'T> =
        | Next of TeeSide * AsyncReplyChannel<'T option>

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
        let apply debug input = AsyncSeq.init (int64 depth) (fun (i: int64) -> mapper (int i))
        create name apply profile

    let liftRelease (name: string) (profile: Profile) (release: 'S->unit) (f: 'S -> 'T) : Pipe<'S,'T> =
        let mapper input = 
            let output = f input
            release input
            output
        let apply debug  input = input |> AsyncSeq.map mapper
        create name apply profile

    /// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
    let compose (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
        let profile = Profile.combine p1.Profile p2.Profile
        let apply debug input = input |> p1.Apply debug |> p2.Apply debug 
        create $"{p2.Name} {p1.Name}" apply profile

    let skip (name: string) (count: uint) =
        let apply debug input = input |> AsyncSeq.skip (int count)
        create "skip" apply (Window (2u, 2u, 0u, 0u, System.UInt32.MaxValue))

    let take (name: string) (count: uint) =
        let apply debug input = input |> AsyncSeq.take (int count)
        create name apply Streaming

    let map (name: string) (mapper: 'U -> 'V) (pipe: Pipe<'In, 'U>) : Pipe<'In, 'V> =
        let apply debug input = input |> pipe.Apply debug |> AsyncSeq.map mapper
        create name apply pipe.Profile

    /// Prepend a sequence produced by a Pipe<unit,'S> to the input stream.
    let prepend (name: string) (pre: Pipe<unit,'S>) : Pipe<'S,'S> =
        let apply debug (input: AsyncSeq<'S>) =
            let preSeq = pre.Apply debug (AsyncSeq.singleton ())
            AsyncSeq.append preSeq input
        create name apply Streaming

    /// Append a sequence produced by a Pipe<unit,'S> to the input stream.
    let append (name: string) (post: Pipe<unit,'S>) : Pipe<'S,'S> =
        let apply debug (input: AsyncSeq<'S>) =
            let postSeq = post.Apply debug (AsyncSeq.singleton ())
            AsyncSeq.append input postSeq
        create name apply Streaming

(*
    let wrap (name: string) (wrapper: ('In * 'U) -> 'V) (pipe: Pipe<'In, 'U>) : Pipe<'In, 'V> =
        let apply debug input = 
            let output = input |> pipe.Apply debug
            let zipped = AsyncSeq.zip input output
            let mapped = AsyncSeq.map wrapper zipped
            mapped
        create name apply pipe.Profile
*)

    let tee (debug: bool) (p: Pipe<'In, 'T>) : Pipe<'In, 'T> * Pipe<'In, 'T> =
        // Shared lazy value to ensure Apply is only triggered once
        let mutable shared: Lazy<AsyncSeq<'T> * AsyncSeq<'T>> option = None
        let syncRoot = obj()

        let makeShared (input: AsyncSeq<'In>) =
            let src = p.Apply debug input  // Only called once

            let agent = MailboxProcessor.Start(fun inbox ->
                async {
                    let enum = src.GetAsyncEnumerator()
                    let buffer = ResizeArray<'T>()
                    let mutable bufferStart = 0L
                    let mutable leftIndex = 0L
                    let mutable rightIndex = 0L
                    let mutable finished = false

                    let rec ensureAvailable index =
                        async {
                            while not finished && bufferStart + int64 buffer.Count <= index do
                                let! hasNext = enum.MoveNextAsync().AsTask() |> Async.AwaitTask
                                if hasNext then
                                    buffer.Add(enum.Current)
                                else
                                    finished <- true
                        }

                    let trimConsumedPrefix () =
                        let minIndex = min leftIndex rightIndex
                        let removable = minIndex - bufferStart
                        if removable > 0L then
                            buffer.RemoveRange(0, int removable)
                            bufferStart <- minIndex

                    let rec loop () = async {
                        let! msg = inbox.Receive()

                        match msg with
                        | Next(side, reply) ->
                            let index =
                                match side with
                                | Left -> leftIndex
                                | Right -> rightIndex

                            do! ensureAvailable index

                            if index < bufferStart + int64 buffer.Count then
                                let value = buffer[int (index - bufferStart)]
                                match side with
                                | Left -> leftIndex <- leftIndex + 1L
                                | Right -> rightIndex <- rightIndex + 1L
                                trimConsumedPrefix ()
                                reply.Reply(Some value)
                            else
                                reply.Reply(None)

                        return! loop ()
                    }

                    do! loop ()
                })

            let makeStream tag =
                asyncSeq {
                    let mutable done_ = false
                    while not done_ do
                        let! vOpt = agent.PostAndAsyncReply(fun reply -> Next(tag, reply))
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
        let profile = Window (winSz,stride,pad,emitStart,emitCount)
        create name apply profile
*)

    // window : wraps AsyncSeqExtensions.windowedWithPad
    let window (name: string) (winSz: uint) (pad: uint) (zeroMaker: int -> 'T -> 'T) (stride: uint) : Pipe<'T, Window<'T>> =

        let apply _debug (input: AsyncSeq<'T>) : AsyncSeq<Window<'T>> =
            // Produces an AsyncSeq of windows with the default emitted range.
            AsyncSeqExtensions.windowedWithPad winSz stride pad pad zeroMaker input
            |> AsyncSeq.map (Window.create pad stride)
        let profile = Window (winSz, stride, pad, 0u, winSz)
        create name apply profile

    // collect : flattens an AsyncSeq<'T list> to AsyncSeq<'T>
    let collect (name: string) (mapper: 'S -> ('T list)) : Pipe<'S, 'T> =
        let apply _debug (input: AsyncSeq<'S>) : AsyncSeq<'T> =
            input |> AsyncSeq.collect (mapper >> AsyncSeq.ofSeq)
        let profile = Streaming
        create name apply profile

(*
    let flattenAlt (name: string) : Pipe<'T list, 'T> =
        let apply _debug (input: AsyncSeq<'T list>) : AsyncSeq<'T> =
            input |> AsyncSeq.collect (fun (xs: 'T list) -> AsyncSeq.ofSeq xs)
        let profile = Streaming
        create name apply profile
*)

    let flatten (name: string) : Pipe<'T list, 'T> =
        collect name (fun (lst: 'T list)-> lst)

    let flattenWindow (name: string) : Pipe<Window<'T>, 'T> =
        collect name Window.emitItems

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

type MemoryNeed = uint64->uint64
type MemoryNeedWrapped = SingleOrPair -> SingleOrPair
type ElementTransformation = uint64 -> uint64

type SliceEnd =
    | RelativeToInputEnd of int64
    | CountFromStart of uint64
    | UnknownEnd

type SliceDomain =
    { StartOffset: int64
      End: SliceEnd }

type SliceCardinality =
    | Domain of SliceDomain
    | ReduceTo of uint64
    | UnknownCardinality

module SliceDomain =
    let preserve =
        { StartOffset = 0L
          End = RelativeToInputEnd 0L }

    let trim (before: uint) (after: uint) =
        { StartOffset = int64 before
          End = RelativeToInputEnd -(int64 after) }

    let expand (before: uint) (after: uint) =
        { StartOffset = -(int64 before)
          End = RelativeToInputEnd (int64 after) }

    let skip (count: uint) =
        { StartOffset = int64 count
          End = RelativeToInputEnd 0L }

    let take (count: uint64) =
        { StartOffset = 0L
          End = CountFromStart count }

    let private tryUint64 value =
        if value < 0L then None else Some(uint64 value)

    let private composeEnd left right =
        match right.End with
        | UnknownEnd -> UnknownEnd
        | CountFromStart count -> CountFromStart count
        | RelativeToInputEnd rightOffset ->
            match left.End with
            | RelativeToInputEnd leftOffset -> RelativeToInputEnd(leftOffset + rightOffset)
            | CountFromStart leftCount ->
                let composedCount = int64 leftCount + rightOffset - right.StartOffset
                match tryUint64 composedCount with
                | Some count -> CountFromStart count
                | None -> CountFromStart 0UL
            | UnknownEnd -> UnknownEnd

    let compose left right =
        { StartOffset = left.StartOffset + right.StartOffset
          End = composeEnd left right }

    let private stop inputLength domain =
        match domain.End with
        | RelativeToInputEnd offset -> Some(int64 inputLength + offset)
        | CountFromStart count -> Some(domain.StartOffset + int64 count)
        | UnknownEnd -> None

    let length inputLength domain =
        stop inputLength domain
        |> Option.map (fun stop -> max 0L (stop - domain.StartOffset) |> uint64)

module SliceCardinality =
    let preserve = Domain SliceDomain.preserve
    let reduceTo count = ReduceTo count
    let unknown = UnknownCardinality

    let compose left right =
        match left, right with
        | UnknownCardinality, _
        | _, UnknownCardinality -> UnknownCardinality
        | _, ReduceTo count -> ReduceTo count
        | ReduceTo _, Domain domain -> Domain domain
        | Domain leftDomain, Domain rightDomain -> Domain(SliceDomain.compose leftDomain rightDomain)

    let length inputLength cardinality =
        match cardinality with
        | Domain domain -> SliceDomain.length inputLength domain
        | ReduceTo count -> Some count
        | UnknownCardinality -> None

type StageEvaluation =
    | Source
    | Map
    | Iter
    | Windowed of windowSize: uint * stride: uint * pad: uint
    | Flatten
    | Reduce
    | Fanout of branchCount: int
    | Join
    | Sink
    | Custom of string

type StageMemoryPressure =
    { InputLive: uint64
      OutputLive: uint64
      WorkLive: uint64
      RetainedLive: uint64
      Peak: uint64 }

module StageMemoryPressure =
    let create inputLive outputLive workLive retainedLive =
        let peak = inputLive + outputLive + workLive + retainedLive
        { InputLive = inputLive
          OutputLive = outputLive
          WorkLive = workLive
          RetainedLive = retainedLive
          Peak = peak }

    let fromPeak peak =
        { InputLive = 0UL
          OutputLive = 0UL
          WorkLive = peak
          RetainedLive = 0UL
          Peak = peak }

type StageMemoryModel =
    { Evaluation: StageEvaluation
      Estimate: SingleOrPair -> StageMemoryPressure }

module StageMemoryModel =
    let private bytesOf input =
        input |> SingleOrPair.sum |> SingleOrPair.fst

    let fromPeak evaluation (memoryNeed: MemoryNeedWrapped) =
        { Evaluation = evaluation
          Estimate = memoryNeed >> SingleOrPair.sum >> SingleOrPair.fst >> StageMemoryPressure.fromPeak }

    let fromSinglePeak evaluation (memoryNeed: MemoryNeed) =
        let wrapped = SingleOrPair.sum >> SingleOrPair.fst >> memoryNeed >> Single
        fromPeak evaluation wrapped

    let mapLike (outputBytes: MemoryNeed) (workBytes: MemoryNeed) =
        { Evaluation = Map
          Estimate =
            fun input ->
                let inputBytes = bytesOf input
                StageMemoryPressure.create inputBytes (outputBytes inputBytes) (workBytes inputBytes) 0UL }

    let iterLike (workBytes: MemoryNeed) =
        { Evaluation = Iter
          Estimate =
            fun input ->
                let inputBytes = bytesOf input
                StageMemoryPressure.create inputBytes inputBytes (workBytes inputBytes) 0UL }

    let windowLike winSz stride pad =
        { Evaluation = Windowed(winSz, stride, pad)
          Estimate =
            fun input ->
                let inputBytes = bytesOf input
                let windowBytes = inputBytes * uint64 winSz
                StageMemoryPressure.create windowBytes windowBytes 0UL 0UL }

    let reduceLike (accumulatorBytes: MemoryNeed) (workBytes: MemoryNeed) =
        { Evaluation = Reduce
          Estimate =
            fun input ->
                let inputBytes = bytesOf input
                StageMemoryPressure.create inputBytes (accumulatorBytes inputBytes) (workBytes inputBytes) 0UL }

    let memoryNeed model input =
        model.Estimate input |> fun pressure -> Single pressure.Peak

type StageCostKind =
    | Cpu
    | Native
    | Io
    | Mixed

type StageWorkPressure =
    { CpuWorkUnits: float
      NativeWorkUnits: float
      IoReadBytes: uint64
      IoWriteBytes: uint64
      IoReadOps: uint64
      IoWriteOps: uint64
      CalibrationKey: string option }

module StageWorkPressure =
    let zero =
        { CpuWorkUnits = 0.0
          NativeWorkUnits = 0.0
          IoReadBytes = 0UL
          IoWriteBytes = 0UL
          IoReadOps = 0UL
          IoWriteOps = 0UL
          CalibrationKey = None }

    let create cpuWorkUnits nativeWorkUnits ioReadBytes ioWriteBytes ioReadOps ioWriteOps calibrationKey =
        { CpuWorkUnits = cpuWorkUnits
          NativeWorkUnits = nativeWorkUnits
          IoReadBytes = ioReadBytes
          IoWriteBytes = ioWriteBytes
          IoReadOps = ioReadOps
          IoWriteOps = ioWriteOps
          CalibrationKey = calibrationKey }

    let add left right =
        { CpuWorkUnits = left.CpuWorkUnits + right.CpuWorkUnits
          NativeWorkUnits = left.NativeWorkUnits + right.NativeWorkUnits
          IoReadBytes = left.IoReadBytes + right.IoReadBytes
          IoWriteBytes = left.IoWriteBytes + right.IoWriteBytes
          IoReadOps = left.IoReadOps + right.IoReadOps
          IoWriteOps = left.IoWriteOps + right.IoWriteOps
          CalibrationKey = None }

type StageWorkModel =
    { Kind: StageCostKind
      Evaluation: StageEvaluation
      Estimate: SingleOrPair -> StageWorkPressure }

type StageCostCoefficients =
    { CpuMillisecondsPerUnit: float
      NativeMillisecondsPerUnit: float
      IoReadMillisecondsPerByte: float
      IoWriteMillisecondsPerByte: float
      IoReadMillisecondsPerOp: float
      IoWriteMillisecondsPerOp: float }

module StageCostCoefficients =
    let zero =
        { CpuMillisecondsPerUnit = 0.0
          NativeMillisecondsPerUnit = 0.0
          IoReadMillisecondsPerByte = 0.0
          IoWriteMillisecondsPerByte = 0.0
          IoReadMillisecondsPerOp = 0.0
          IoWriteMillisecondsPerOp = 0.0 }

    let estimateMilliseconds coefficients pressure =
        pressure.CpuWorkUnits * coefficients.CpuMillisecondsPerUnit
        + pressure.NativeWorkUnits * coefficients.NativeMillisecondsPerUnit
        + float pressure.IoReadBytes * coefficients.IoReadMillisecondsPerByte
        + float pressure.IoWriteBytes * coefficients.IoWriteMillisecondsPerByte
        + float pressure.IoReadOps * coefficients.IoReadMillisecondsPerOp
        + float pressure.IoWriteOps * coefficients.IoWriteMillisecondsPerOp

module StageCostCalibration =
    let mutable private coefficientsByKey: Map<string, StageCostCoefficients> = Map.empty

    let clear () =
        coefficientsByKey <- Map.empty

    let register key coefficients =
        coefficientsByKey <- coefficientsByKey |> Map.add key coefficients

    let tryFind key =
        coefficientsByKey |> Map.tryFind key

    let estimateMilliseconds pressure =
        match pressure.CalibrationKey |> Option.bind tryFind with
        | Some coefficients -> Some (StageCostCoefficients.estimateMilliseconds coefficients pressure)
        | None -> None

    let private propertyDouble (name: string) (element: System.Text.Json.JsonElement) =
        match element.TryGetProperty(name) with
        | true, property when property.ValueKind = System.Text.Json.JsonValueKind.Number ->
            property.GetDouble()
        | _ ->
            let pascalName =
                if System.String.IsNullOrEmpty(name) then
                    name
                else
                    name.Substring(0, 1).ToUpperInvariant() + name.Substring(1)
            match element.TryGetProperty(pascalName) with
            | true, property when property.ValueKind = System.Text.Json.JsonValueKind.Number ->
                property.GetDouble()
            | _ -> 0.0

    let loadJson path =
        if System.IO.File.Exists(path) then
            use document = System.Text.Json.JsonDocument.Parse(System.IO.File.ReadAllText(path))
            match document.RootElement.TryGetProperty("calibrations") with
            | true, calibrations when calibrations.ValueKind = System.Text.Json.JsonValueKind.Object ->
                for property in calibrations.EnumerateObject() do
                    let value = property.Value
                    register
                        property.Name
                        { CpuMillisecondsPerUnit = propertyDouble "cpuMillisecondsPerUnit" value
                          NativeMillisecondsPerUnit = propertyDouble "nativeMillisecondsPerUnit" value
                          IoReadMillisecondsPerByte = propertyDouble "ioReadMillisecondsPerByte" value
                          IoWriteMillisecondsPerByte = propertyDouble "ioWriteMillisecondsPerByte" value
                          IoReadMillisecondsPerOp = propertyDouble "ioReadMillisecondsPerOp" value
                          IoWriteMillisecondsPerOp = propertyDouble "ioWriteMillisecondsPerOp" value }
                true
            | _ -> false
        else
            false

module StageWorkModel =
    let private elementCount input =
        input |> SingleOrPair.sum |> SingleOrPair.fst |> float

    let zero evaluation =
        { Kind = Cpu
          Evaluation = evaluation
          Estimate = fun _ -> StageWorkPressure.zero }

    let cpu evaluation calibrationKey workUnits =
        { Kind = Cpu
          Evaluation = evaluation
          Estimate = fun input -> StageWorkPressure.create (workUnits input) 0.0 0UL 0UL 0UL 0UL calibrationKey }

    let native evaluation calibrationKey workUnits =
        { Kind = Native
          Evaluation = evaluation
          Estimate = fun input -> StageWorkPressure.create 0.0 (workUnits input) 0UL 0UL 0UL 0UL calibrationKey }

    let ioRead evaluation calibrationKey bytes ops =
        { Kind = Io
          Evaluation = evaluation
          Estimate = fun input -> StageWorkPressure.create 0.0 0.0 (bytes input) 0UL (ops input) 0UL calibrationKey }

    let ioWrite evaluation calibrationKey bytes ops =
        { Kind = Io
          Evaluation = evaluation
          Estimate = fun input -> StageWorkPressure.create 0.0 0.0 0UL (bytes input) 0UL (ops input) calibrationKey }

    let fromEvaluation evaluation calibrationKey =
        match evaluation with
        | Source
        | Sink
        | Iter -> zero evaluation
        | Map
        | Flatten
        | Reduce
        | Fanout _
        | Join
        | Custom _ -> cpu evaluation calibrationKey elementCount
        | Windowed(winSz, _, _) ->
            native evaluation calibrationKey (fun input -> elementCount input * float winSz)

type StageCostPressure =
    { Memory: StageMemoryPressure
      Work: StageWorkPressure }

module StageCostPressure =
    let add (left: StageCostPressure) (right: StageCostPressure) =
        { Memory = StageMemoryPressure.fromPeak (left.Memory.Peak + right.Memory.Peak)
          Work = StageWorkPressure.add left.Work right.Work }

type StageCostModel =
    { Memory: StageMemoryModel
      Work: StageWorkModel }

module StageCostModel =
    let create memory work =
        { Memory = memory
          Work = work }

    let fromMemory memory =
        create memory (StageWorkModel.fromEvaluation memory.Evaluation None)

    let estimate (model: StageCostModel) input : StageCostPressure =
        { Memory = model.Memory.Estimate input
          Work = model.Work.Estimate input }

    let memoryNeed model input =
        StageMemoryModel.memoryNeed model.Memory input

    let combine evaluation left right =
        let memory =
            { Evaluation = evaluation
              Estimate =
                fun input ->
                    let leftPressure = left.Memory.Estimate input
                    let rightPressure = right.Memory.Estimate input
                    StageMemoryPressure.fromPeak (leftPressure.Peak + rightPressure.Peak) }
        let work =
            { Kind = Mixed
              Evaluation = evaluation
              Estimate =
                fun input ->
                    StageWorkPressure.add (left.Work.Estimate input) (right.Work.Estimate input) }
        create memory work

type SourcePeek =
    { Name: string
      ElementBytes: uint64
      Length: uint64 option
      Shape: Map<string, string> }

module SourcePeek =
    let create name elementBytes length shape =
        { Name = name
          ElementBytes = elementBytes
          Length = length
          Shape = shape }

module MemoryProbe =
    type Snapshot =
        { Baseline: uint64
          Peak: uint64
          Delta: uint64 }

    type private Probe =
        { Baseline: uint64
          mutable Peak: uint64
          Cancellation: System.Threading.CancellationTokenSource }

    let currentRssBytes () =
        let p = System.Diagnostics.Process.GetCurrentProcess()
        p.Refresh()
        uint64 p.WorkingSet64

    let private sample (probe: Probe) =
        let current = currentRssBytes()
        if current > probe.Peak then
            probe.Peak <- current

    let private start (sampleIntervalMs: int) =
        let baseline = currentRssBytes()
        let cancellation = new System.Threading.CancellationTokenSource()
        let probe =
            { Baseline = baseline
              Peak = baseline
              Cancellation = cancellation }

        let rec loop () =
            async {
                if not cancellation.IsCancellationRequested then
                    sample probe
                    do! Async.Sleep sampleIntervalMs
                    return! loop ()
            }

        Async.Start(loop (), cancellation.Token)
        probe

    let private stop (probe: Probe) =
        sample probe
        probe.Cancellation.Cancel()
        probe.Cancellation.Dispose()
        { Baseline = probe.Baseline
          Peak = probe.Peak
          Delta = probe.Peak - probe.Baseline }

    let measure sampleIntervalMs f =
        let probe = start sampleIntervalMs
        try
            let result = f ()
            result, stop probe
        with ex ->
            let _ = stop probe
            reraise ()

    let formatBytes (bytes: uint64) =
        $"{bytes / 1024UL} KB"

type ResourceOps<'T> =
    { Retain : 'T -> unit
      Release : 'T -> unit
      MemoryOf : 'T -> uint64 option }

module ResourceOps =

    let none<'T> : ResourceOps<'T> =
        { Retain = fun (_: 'T) -> ()
          Release = fun (_: 'T) -> ()
          MemoryOf = fun (_: 'T) -> None }

    let retainAndReturn (ops: ResourceOps<'T>) (value: 'T) =
        ops.Retain value
        value

    let release (ops: ResourceOps<'T>) (value: 'T) =
        ops.Release value

    let memoryOf (ops: ResourceOps<'T>) (value: 'T) =
        ops.MemoryOf value

module DebugLevel =
    let mutable private level = 0u
    let set value =
        level <- value
    let current () =
        level
    let isEnabled () =
        level > 0u
    let rssEnabled () =
        level >= 2u
    let optimizationDisabled () =
        level >= 3u

type PipelineGraphNode =
    { Id: int
      Name: string
      Transition: ProfileTransition }

type PipelineGraphEdge =
    { From: int
      To: int
      Label: string }

type PipelineGraph =
    { Nodes: PipelineGraphNode list
      Edges: PipelineGraphEdge list
      Entries: int list
      Exits: int list }

module PipelineGraph =
    let private nextNodeId =
        let mutable current = 0
        fun () ->
            current <- current + 1
            current

    let empty =
        { Nodes = []
          Edges = []
          Entries = []
          Exits = [] }

    let singleton name transition =
        let id = nextNodeId()
        { Nodes =
            [ { Id = id
                Name = name
                Transition = transition } ]
          Edges = []
          Entries = [ id ]
          Exits = [ id ] }

    let merge left right =
        { Nodes = left.Nodes @ right.Nodes
          Edges = left.Edges @ right.Edges
          Entries =
            match left.Entries, right.Entries with
            | [], entries -> entries
            | entries, [] -> entries
            | entries, _ -> entries
          Exits =
            match right.Exits, left.Exits with
            | [], exits -> exits
            | exits, [] -> exits
            | exits, _ -> exits }

    let connect label left right =
        let connectorEdges =
            [ for fromNode in left.Exits do
                for toNode in right.Entries do
                    yield { From = fromNode; To = toNode; Label = label } ]

        { Nodes = left.Nodes @ right.Nodes
          Edges = left.Edges @ connectorEdges @ right.Edges
          Entries =
            if left.Entries.IsEmpty then right.Entries else left.Entries
          Exits =
            if right.Exits.IsEmpty then left.Exits else right.Exits }

    let appendNode label name transition graph =
        connect label graph (singleton name transition)

    let compose left right =
        connect ">=>" left right

    let parallelJoin label left right =
        let merged = merge left right
        let join = singleton label (ProfileTransition.create Streaming Streaming)
        let connectorEdges =
            [ for fromNode in merged.Exits do
                for toNode in join.Entries do
                    yield { From = fromNode; To = toNode; Label = label } ]

        { Nodes = merged.Nodes @ join.Nodes
          Edges = merged.Edges @ connectorEdges @ join.Edges
          Entries = merged.Entries
          Exits = join.Exits }

/// Stage describes *what* should be done: 
type Stage<'S,'T> =
    { Name       : string
      Build      : unit -> Pipe<'S,'T> // the pipe creator function
      Transition : ProfileTransition // The transformation of the transition profile
      MemoryNeed : MemoryNeedWrapped // calculate the memory needed to process a specified number of element
      MemoryModel : StageMemoryModel
      CostModel : StageCostModel
      ElementTransformation : ElementTransformation // the transformation of the memory/cost shape of each stream element
      SliceCardinality : SliceCardinality // the logical z-domain transformation of the stream
      Graph: PipelineGraph
      Cleaning: (unit->unit) list // Functions to be called at sink/drain/flush
      } 

module Stage =

    let createWithCostModelAndSlice<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (costModel: StageCostModel) (elementTransformation: ElementTransformation) (sliceCardinality: SliceCardinality) (cleaning: (unit->unit) list) =
        let wrapMemoryNeed = StageCostModel.memoryNeed costModel
        { Name = name
          Build = build
          Transition = transition
          MemoryNeed = wrapMemoryNeed
          MemoryModel = costModel.Memory
          CostModel = costModel
          ElementTransformation = elementTransformation
          SliceCardinality = sliceCardinality
          Graph = PipelineGraph.singleton name transition
          Cleaning = cleaning}

    let createWithCostModel<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (costModel: StageCostModel) (elementTransformation: ElementTransformation) (cleaning: (unit->unit) list) =
        createWithCostModelAndSlice name build transition costModel elementTransformation SliceCardinality.preserve cleaning

    let createWithModel<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (memoryModel: StageMemoryModel) (elementTransformation: ElementTransformation) (cleaning: (unit->unit) list) =
        createWithCostModel name build transition (StageCostModel.fromMemory memoryModel) elementTransformation cleaning

    let createWithModelAndSlice<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (memoryModel: StageMemoryModel) (elementTransformation: ElementTransformation) (sliceCardinality: SliceCardinality) (cleaning: (unit->unit) list) =
        createWithCostModelAndSlice name build transition (StageCostModel.fromMemory memoryModel) elementTransformation sliceCardinality cleaning

    let create<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) (cleaning: (unit->unit) list) =
        let wrapMemoryNeed = SingleOrPair.sum >> SingleOrPair.fst >> memoryNeed >> Single
        let memoryModel = StageMemoryModel.fromPeak (Custom name) wrapMemoryNeed
        createWithModel name build transition memoryModel elementTransformation cleaning

    let createWithSlice<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) (sliceCardinality: SliceCardinality) (cleaning: (unit->unit) list) =
        let wrapMemoryNeed = SingleOrPair.sum >> SingleOrPair.fst >> memoryNeed >> Single
        let memoryModel = StageMemoryModel.fromPeak (Custom name) wrapMemoryNeed
        createWithModelAndSlice name build transition memoryModel elementTransformation sliceCardinality cleaning

    let createWrapped<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (wrapMemoryNeed: MemoryNeedWrapped) (elementTransformation: ElementTransformation) (cleaning: (unit->unit) list) =
        let memoryModel = StageMemoryModel.fromPeak (Custom name) wrapMemoryNeed
        createWithModel name build transition memoryModel elementTransformation cleaning

    let createWrappedWithSlice<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (wrapMemoryNeed: MemoryNeedWrapped) (elementTransformation: ElementTransformation) (sliceCardinality: SliceCardinality) (cleaning: (unit->unit) list) =
        let memoryModel = StageMemoryModel.fromPeak (Custom name) wrapMemoryNeed
        createWithModelAndSlice name build transition memoryModel elementTransformation sliceCardinality cleaning

    let createWrappedWithModel<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (memoryModel: StageMemoryModel) (elementTransformation: ElementTransformation) (cleaning: (unit->unit) list) =
        createWithModel name build transition memoryModel elementTransformation cleaning

    let createWrappedWithCostModel<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (costModel: StageCostModel) (elementTransformation: ElementTransformation) (cleaning: (unit->unit) list) =
        createWithCostModel name build transition costModel elementTransformation cleaning

    let createWrappedWithCostModelAndSlice<'S,'T> (name: string) (build: unit->Pipe<'S,'T>) (transition: ProfileTransition) (costModel: StageCostModel) (elementTransformation: ElementTransformation) (sliceCardinality: SliceCardinality) (cleaning: (unit->unit) list) =
        createWithCostModelAndSlice name build transition costModel elementTransformation sliceCardinality cleaning

    let private withGraph graph stage =
        { stage with Graph = graph }

    let withSliceCardinality sliceCardinality stage =
        { stage with SliceCardinality = sliceCardinality }

    let empty (name: string) =
        let build () = Pipe.empty name
        let transition = ProfileTransition.create Streaming Streaming
        let memoryNeed _ = 0UL
        let elementTransformation = id
        create name build transition memoryNeed  elementTransformation []

    let init<'S,'T> (name: string) (depth: uint) (mapper: int -> 'T) (transition: ProfileTransition) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) =
        let build () = Pipe.init name depth mapper transition.From
        create name build transition memoryNeed elementTransformation []

    let compose (stage1 : Stage<'S,'T>) (stage2 : Stage<'T,'U>) : Stage<'S,'U> =
        let build () = Pipe.compose (stage1.Build ()) (stage2.Build ())
        let transition = ProfileTransition.create stage1.Transition.From stage2.Transition.To
        let costModel = StageCostModel.combine (Custom $"{stage2.Name} o {stage1.Name}") stage1.CostModel stage2.CostModel
        let elementTransformation = stage1.ElementTransformation >> stage2.ElementTransformation
        let sliceCardinality = SliceCardinality.compose stage1.SliceCardinality stage2.SliceCardinality
        createWrappedWithCostModelAndSlice $"{stage2.Name} o {stage1.Name}" build transition costModel elementTransformation sliceCardinality (stage2.Cleaning@stage1.Cleaning)
        |> withGraph (PipelineGraph.compose stage1.Graph stage2.Graph)

    let (-->) = compose

    let prepend (name: string) (pre: Stage<unit,'S>) : Stage<'S,'S> =
        let build () = Pipe.prepend name (pre.Build ())
        let transition = ProfileTransition.create Streaming Streaming
        createWrappedWithSlice name build transition pre.MemoryNeed pre.ElementTransformation pre.SliceCardinality pre.Cleaning
        |> withGraph (PipelineGraph.appendNode "prepend" name transition pre.Graph)

    let append (name: string) (app: Stage<unit,'S>) : Stage<'S,'S> =
        let build () = Pipe.append name (app.Build ())
        let transition = ProfileTransition.create Streaming Streaming
        createWrappedWithSlice name build transition app.MemoryNeed app.ElementTransformation app.SliceCardinality app.Cleaning
        |> withGraph (PipelineGraph.appendNode "append" name transition app.Graph)

    let toPipe (stage : Stage<_,_>) = stage.Build

    let fromPipe (name: string) (transition: ProfileTransition) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) (pipe: Pipe<'S, 'T>) : Stage<'S, 'T> =
        create name (fun () -> pipe) transition memoryNeed elementTransformation []

    let skip (name: string) (n: uint) : Stage<'S, 'S> =
        let build () = Pipe.skip name n 
        let transition = ProfileTransition.create Streaming Streaming
        createWithSlice name build transition id id (Domain(SliceDomain.skip n)) []

    let take (name: string) (n: uint) : Stage<'S, 'S> =
        let build () = Pipe.take name n 
        let transition = ProfileTransition.create Streaming Streaming
        createWithSlice name build transition id id (Domain(SliceDomain.take (uint64 n))) []

    let map<'S,'T> (name: string) (f: bool -> 'S -> 'T) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'S, 'T> =
        let apply debug input = input |> AsyncSeq.map (f debug)
        let build () : Pipe<'S,'T> = Pipe.create name apply Streaming
        let transition = ProfileTransition.create Streaming Streaming
        createWithModel name build transition (StageMemoryModel.fromSinglePeak Map memoryNeed) elementTransformation []

    let mapi<'S,'T> (name: string) (f: bool -> int64 -> 'S -> 'T) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'S, 'T> =
        let apply debug input = input |> AsyncSeq.mapi (f debug)
        let build () : Pipe<'S,'T> = Pipe.create name apply Streaming
        let transition = ProfileTransition.create Streaming Streaming
        createWithModel name build transition (StageMemoryModel.fromSinglePeak Map memoryNeed) elementTransformation []

    let map1 (name: string) (f: 'U -> 'V) (stage: Stage<'In, 'U>) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'In, 'V> =
        let build () = Pipe.map name f (stage.Build())
        let transition = ProfileTransition.create Streaming Streaming
        create name build transition memoryNeed elementTransformation []
        |> withGraph (PipelineGraph.appendNode "map" name transition stage.Graph)

    let map2 (name: string) (debug: bool) (f: 'U -> 'V -> 'W) (stage1: Stage<'In, 'U>) (stage2: Stage<'In, 'V>) (memoryNeed: MemoryNeedWrapped) (elementTransformation: ElementTransformation) : Stage<'In, 'W> =
        let build () = Pipe.map2 name debug f (stage1.Build ()) (stage2.Build ())
        let transition = ProfileTransition.create Streaming Streaming
        let memoryModel = StageMemoryModel.fromPeak Join memoryNeed
        let workModel =
            { Kind = Mixed
              Evaluation = Join
              Estimate =
                fun input ->
                    StageWorkPressure.add (stage1.CostModel.Work.Estimate input) (stage2.CostModel.Work.Estimate input) }
        let sliceCardinality =
            if stage1.SliceCardinality = stage2.SliceCardinality then stage1.SliceCardinality else UnknownCardinality
        createWrappedWithCostModelAndSlice name build transition (StageCostModel.create memoryModel workModel) elementTransformation sliceCardinality (stage1.Cleaning@stage2.Cleaning)
        |> withGraph (PipelineGraph.parallelJoin name stage1.Graph stage2.Graph)

    let map2Sync (name: string) (debug: bool) (f: 'U -> 'V -> 'W) (stage1: Stage<'In, 'U>) (stage2: Stage<'In, 'V>) (memoryNeed: MemoryNeedWrapped) (elementTransformation: ElementTransformation) : Stage<'In, 'W> =
        let build () = Pipe.map2Sync name debug f (stage1.Build ()) (stage2.Build ())
        let transition = ProfileTransition.create Streaming Streaming
        let memoryModel = StageMemoryModel.fromPeak Join memoryNeed
        let workModel =
            { Kind = Mixed
              Evaluation = Join
              Estimate =
                fun input ->
                    StageWorkPressure.add (stage1.CostModel.Work.Estimate input) (stage2.CostModel.Work.Estimate input) }
        let sliceCardinality =
            if stage1.SliceCardinality = stage2.SliceCardinality then stage1.SliceCardinality else UnknownCardinality
        createWrappedWithCostModelAndSlice name build transition (StageCostModel.create memoryModel workModel) elementTransformation sliceCardinality (stage1.Cleaning@stage2.Cleaning)
        |> withGraph (PipelineGraph.parallelJoin name stage1.Graph stage2.Graph)

    let mapPairSync (name: string) (debug: bool) (stage1: Stage<'U, 'S>) (stage2: Stage<'V, 'T>) (memoryNeed: MemoryNeedWrapped) (elementTransformation: ElementTransformation) : Stage<'U * 'V, 'S * 'T> =
        let build () =
            let stage1Pipe = stage1.Build ()
            let stage2Pipe = stage2.Build ()

            let apply debug input =
                let leftPipe, rightPipe = Pipe.tee debug (Pipe.id "pair-input")
                let left =
                    input
                    |> leftPipe.Apply debug
                    |> AsyncSeq.map fst
                    |> stage1Pipe.Apply debug
                let right =
                    input
                    |> rightPipe.Apply debug
                    |> AsyncSeq.map snd
                    |> stage2Pipe.Apply debug

                AsyncSeqExtensions.zipConcurrent left right

            Pipe.create name apply Streaming

        let transition = ProfileTransition.create Streaming Streaming
        let memoryModel = StageMemoryModel.fromPeak Join memoryNeed
        let workModel =
            { Kind = Mixed
              Evaluation = Join
              Estimate =
                fun input ->
                    let leftInput = SingleOrPair.index1 input
                    let rightInput = SingleOrPair.index2 input
                    StageWorkPressure.add (stage1.CostModel.Work.Estimate leftInput) (stage2.CostModel.Work.Estimate rightInput) }

        let sliceCardinality =
            if stage1.SliceCardinality = stage2.SliceCardinality then stage1.SliceCardinality else UnknownCardinality
        createWrappedWithCostModelAndSlice name build transition (StageCostModel.create memoryModel workModel) elementTransformation sliceCardinality (stage1.Cleaning@stage2.Cleaning)
        |> withGraph (PipelineGraph.parallelJoin name stage1.Graph stage2.Graph)

    let teeFst (stage: Stage<'A, 'A>) : Stage<'A * 'B, 'A * 'B> =
        if stage.Transition.From <> Streaming || stage.Transition.To <> Streaming then
            invalidArg (nameof stage) $"teeFst expects a streaming identity stage, got {stage.Transition.From} -> {stage.Transition.To}"

        let build () =
            let stagePipe = stage.Build ()

            let apply debug input =
                let leftPipe, rightPipe = Pipe.tee debug (Pipe.id "teeFst-input")
                let left =
                    input
                    |> leftPipe.Apply debug
                    |> AsyncSeq.map fst
                    |> stagePipe.Apply debug
                let right =
                    input
                    |> rightPipe.Apply debug
                    |> AsyncSeq.map snd

                AsyncSeq.zip left right

            Pipe.create $"teeFst ({stage.Name})" apply Streaming

        let memoryNeed nElems =
            let leftNeed = stage.MemoryNeed (SingleOrPair.index1 nElems) |> SingleOrPair.fst
            Pair(leftNeed, 0UL)

        createWrapped $"teeFst ({stage.Name})" build (ProfileTransition.create Streaming Streaming) memoryNeed id stage.Cleaning
        |> withGraph (PipelineGraph.appendNode "teeFst" $"teeFst ({stage.Name})" (ProfileTransition.create Streaming Streaming) stage.Graph)

    let teeSnd (stage: Stage<'B, 'B>) : Stage<'A * 'B, 'A * 'B> =
        if stage.Transition.From <> Streaming || stage.Transition.To <> Streaming then
            invalidArg (nameof stage) $"teeSnd expects a streaming identity stage, got {stage.Transition.From} -> {stage.Transition.To}"

        let build () =
            let stagePipe = stage.Build ()

            let apply debug input =
                let leftPipe, rightPipe = Pipe.tee debug (Pipe.id "teeSnd-input")
                let left =
                    input
                    |> leftPipe.Apply debug
                    |> AsyncSeq.map fst
                let right =
                    input
                    |> rightPipe.Apply debug
                    |> AsyncSeq.map snd
                    |> stagePipe.Apply debug

                AsyncSeq.zip left right

            Pipe.create $"teeSnd ({stage.Name})" apply Streaming

        let memoryNeed nElems =
            let rightNeed = stage.MemoryNeed (SingleOrPair.index2 nElems) |> SingleOrPair.fst
            Pair(0UL, rightNeed)

        createWrapped $"teeSnd ({stage.Name})" build (ProfileTransition.create Streaming Streaming) memoryNeed id stage.Cleaning
        |> withGraph (PipelineGraph.appendNode "teeSnd" $"teeSnd ({stage.Name})" (ProfileTransition.create Streaming Streaming) stage.Graph)

    let reduce (name: string) (reducer: bool -> AsyncSeq<'In> -> Async<'Out>) (profile: Profile) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'In, 'Out> =
        let build () = Pipe.reduce name reducer profile
        let transition = ProfileTransition.create Streaming Constant
        createWithModelAndSlice name build transition (StageMemoryModel.fromSinglePeak Reduce memoryNeed) elementTransformation (ReduceTo 1UL) []

    let fold<'S,'T> (name: string) (folder: 'T -> 'S -> 'T) (initial: 'T) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'S, 'T> =
        let build () : Pipe<'S,'T> = Pipe.fold name folder initial Streaming
        let transition = ProfileTransition.create Streaming Constant
        createWithModelAndSlice name build transition (StageMemoryModel.fromSinglePeak Reduce memoryNeed) elementTransformation (ReduceTo 1UL) []

    let window (name: string) (winSz: uint) (pad: uint) (zeroMaker: int -> 'T -> 'T) (stride: uint) : Stage<'T, Window<'T>> =
        let pipe = Pipe.window name winSz pad zeroMaker stride
        let transition = ProfileTransition.create Streaming Streaming
        createWithModel name (fun () -> pipe) transition (StageMemoryModel.windowLike winSz stride pad) id []

    let flatten (name: string) : Stage<'T list, 'T> =
        let pipe = Pipe.flatten name 
        let transition = ProfileTransition.create Streaming Streaming
        fromPipe name transition id id pipe

    let flattenWindow (name: string) : Stage<Window<'T>, 'T> =
        let pipe = Pipe.flattenWindow name
        let transition = ProfileTransition.create Streaming Streaming
        fromPipe name transition id id pipe

    let liftUnary<'S,'T> (name: string) (f: 'S -> 'T) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'S, 'T> =
        let build () = Pipe.lift name Streaming f
        let transition = ProfileTransition.create Streaming Streaming
        create name build transition memoryNeed elementTransformation []

    let liftReleaseUnary<'S,'T> (name: string) (release: 'S->unit) (f: 'S -> 'T) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'S, 'T> =
        let build ()= Pipe.liftRelease name Streaming release f
        let transition = ProfileTransition.create Streaming Streaming
        create name build transition memoryNeed elementTransformation []

    let retainWith<'T> (name: string) (ops: ResourceOps<'T>) : Stage<'T, 'T> =
        liftUnary name (ResourceOps.retainAndReturn ops) (fun _ -> 0UL) id

    let releaseWith<'T> (name: string) (ops: ResourceOps<'T>) : Stage<'T, unit> =
        let build () = Pipe.ignore (ResourceOps.release ops)
        let transition = ProfileTransition.create Streaming Unit
        create name build transition id id []

    let liftResourceUnary<'S,'T> (name: string) (ops: ResourceOps<'S>) (f: 'S -> 'T) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'S, 'T> =
        liftReleaseUnary name (ResourceOps.release ops) f memoryNeed elementTransformation

(*
    let liftWindowed<'S,'T when 'S: equality and 'T: equality> (name: string) (window: uint) (pad: uint) (zeroMaker: int->'S->'S) (stride: uint) (emitStart: uint) (emitCount: uint) (f: 'S list -> 'T list) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) : Stage<'S, 'T> =
        let transition = ProfileTransition.create (Window (window,stride,pad,emitStart,emitCount)) Streaming
        let build = Pipe.mapWindowed name window pad zeroMaker stride emitStart emitCount f
        create name build transition memoryNeed elementTransformation
*)

    let tapItStage (name: string) (toString: 'T -> string) : Stage<'T, 'T> =
        liftUnary name (fun x -> printfn "%s" (toString x); x) (fun _ -> 0UL) id

    let tapIt (toString: 'T -> string) : Stage<'T, 'T> =
        tapItStage "tapIt" toString

    let tap (name: string) : Stage<'T, 'T> =
        tapItStage $"tap: {name}" (fun x -> sprintf "[%s] %A" name x)

(*
    let internal tee (op: Stage<'In, 'T>) : Stage<'In, 'T> * Stage<'In, 'T> =
        let leftPipe, rightPipe = Pipe.tee op.Pipe
        let mk name build = 
            create $"{op.Name} - {name}" build op.Transition op.SizeUpdate
        mk "left" leftPipe, mk "right" rightPipe
*)

    let ignore<'T> clean : Stage<'T, unit> =
        let build () = Pipe.ignore clean
        let transition = ProfileTransition.create Streaming Unit
        create "ignore" build transition id id []// check !!!!

    let ignorePairs<'S,'T> (cleanFst, cleanSnd) : Stage<'S*'T, unit> =
        let build () = Pipe.ignorePairs (cleanFst, cleanSnd)
        let transition = ProfileTransition.create Streaming Unit
        create "ignorePairs" build transition id id []// check !!!!

    let consumeWith (name: string) (consume: bool -> int -> 'T -> unit) (memoryNeed: MemoryNeed) : Stage<'T, unit> = 
        let build () = Pipe.consumeWith name consume Streaming
        let transition = ProfileTransition.create Streaming Constant
        createWithModelAndSlice name build transition (StageMemoryModel.fromSinglePeak Iter memoryNeed) id (ReduceTo 0UL) []

    let cast<'S,'T when 'S: equality and 'T: equality> name f (memoryNeed: MemoryNeed) : Stage<'S,'T> = // cast cannot change
        let apply debug input = input |> AsyncSeq.map f 
        let build () = Pipe.create name apply Streaming
        let transition = ProfileTransition.create Streaming Streaming
        create name build transition memoryNeed id []// Type calculation needs to be updated!!!!

type OptimizationCandidate<'S,'T> =
    { Name: string
      Stage: Stage<'S,'T>
      Explanation: string }

type OptimizationDecision =
    { CandidateName: string
      Accepted: bool
      EstimatedMemoryBytes: uint64
      EstimatedMilliseconds: float option
      EstimatedWorkScore: float
      Reason: string }

type OptimizationResult<'S,'T> =
    { Selected: OptimizationCandidate<'S,'T> option
      Decisions: OptimizationDecision list }

module Optimizer =
    let private relativeTolerance = 0.05

    let private workScore pressure =
        pressure.CpuWorkUnits
        + pressure.NativeWorkUnits
        + float pressure.IoReadBytes
        + float pressure.IoWriteBytes
        + float pressure.IoReadOps
        + float pressure.IoWriteOps

    let private nearlyEqual left right =
        let scale = max 1.0 (max (abs left) (abs right))
        abs (left - right) <= scale * relativeTolerance

    let private windowPreference (candidate: OptimizationCandidate<'S,'T>) =
        match candidate.Stage.CostModel.Memory.Evaluation with
        | Windowed(windowSize, _, _) -> windowSize
        | _ -> 0u

    let private compareWindowPreference leftCandidate rightCandidate =
        compare (windowPreference rightCandidate) (windowPreference leftCandidate)

    let private compareAccepted (_leftCandidate, leftDecision) (_rightCandidate, rightDecision) =
        match leftDecision.EstimatedMilliseconds, rightDecision.EstimatedMilliseconds with
        | Some leftMs, Some rightMs -> compare leftMs rightMs
        | Some _, None -> -1
        | None, Some _ -> 1
        | None, None -> compare leftDecision.EstimatedWorkScore rightDecision.EstimatedWorkScore

    let private compareAcceptedWithWindowPreference (leftCandidate, leftDecision) (rightCandidate, rightDecision) =
        match leftDecision.EstimatedMilliseconds, rightDecision.EstimatedMilliseconds with
        | Some leftMs, Some rightMs when nearlyEqual leftMs rightMs ->
            let preference = compareWindowPreference leftCandidate rightCandidate
            if preference <> 0 then preference else compare leftMs rightMs
        | Some _, Some _ ->
            compareAccepted (leftCandidate, leftDecision) (rightCandidate, rightDecision)
        | Some _, None -> -1
        | None, Some _ -> 1
        | None, None when nearlyEqual leftDecision.EstimatedWorkScore rightDecision.EstimatedWorkScore ->
            let preference = compareWindowPreference leftCandidate rightCandidate
            if preference <> 0 then preference else compare leftDecision.EstimatedWorkScore rightDecision.EstimatedWorkScore
        | None, None ->
            compare leftDecision.EstimatedWorkScore rightDecision.EstimatedWorkScore

    let chooseStage (availableMemory: uint64) (inputShape: SingleOrPair) (candidates: OptimizationCandidate<'S,'T> list) : OptimizationResult<'S,'T> =
        let evaluated =
            candidates
            |> List.map (fun candidate ->
                let cost = StageCostModel.estimate candidate.Stage.CostModel inputShape
                let estimatedMemory = cost.Memory.Peak
                let estimatedMilliseconds = StageCostCalibration.estimateMilliseconds cost.Work
                let estimatedWorkScore = workScore cost.Work
                let accepted = estimatedMemory <= availableMemory
                let reason =
                    if accepted then
                        match estimatedMilliseconds with
                        | Some ms -> $"Accepted: estimated {ms:g} ms within {availableMemory} B."
                        | None -> $"Accepted: estimated work score {estimatedWorkScore:g} within {availableMemory} B."
                    else
                        $"Rejected: estimated memory {estimatedMemory} B exceeds available memory {availableMemory} B."

                candidate,
                { CandidateName = candidate.Name
                  Accepted = accepted
                  EstimatedMemoryBytes = estimatedMemory
                  EstimatedMilliseconds = estimatedMilliseconds
                  EstimatedWorkScore = estimatedWorkScore
                  Reason = reason })

        let selected =
            evaluated
            |> List.filter (fun (_, decision) -> decision.Accepted)
            |> List.sortWith compareAcceptedWithWindowPreference
            |> List.tryHead
            |> Option.map fst

        { Selected = selected
          Decisions = evaluated |> List.map snd }

    let chooseStageOrThrow availableMemory inputShape candidates =
        let result = chooseStage availableMemory inputShape candidates
        match result.Selected with
        | Some candidate -> candidate.Stage, result
        | None -> failwith $"No optimization candidate fits within {availableMemory} B."

////////////////////////////////////////////////////////////
// Plan flow controler
type Plan<'S,'T> = { 
    stage          : Stage<'S,'T> option // the function to be applied, when the plan is run
    graph          : PipelineGraph // the analyzable graph built while the DSL is composed
    sourcePeek     : SourcePeek option // source metadata available to future optimizers
    costPeak       : StageCostPressure option // peak stage cost seen while composing the plan
    costObservations: StageCostPressure list // individual stage costs seen while composing the plan
    nElemsPerSlice : SingleOrPair // number of elments (pixels) before transformation - this could be single or pair
    length         : uint64 // length of the sequence, the plan is applied to
    memAvail       : uint64 // memory available for the plan
    memPeak        : uint64 // the plan's estimated peak memory consumption
    debug          : bool
    debugLevel     : uint }

module Plan =
    let private graphOfStage stage =
        stage |> Option.map (fun stg -> stg.Graph) |> Option.defaultValue PipelineGraph.empty

    let private levelOf debug =
        if debug then DebugLevel.current() |> max 1u else 0u

    let create<'S,'T when 'T: equality> (stage: Stage<'S,'T> option) (memAvail : uint64) (memPeak: uint64) (nElemsPerSlice: uint64) (length: uint64) (debug: bool) : Plan<'S, 'T> =
        { stage = stage; graph = graphOfStage stage; sourcePeek = None; costPeak = None; costObservations = []; memAvail = memAvail; memPeak = memPeak; nElemsPerSlice = Single nElemsPerSlice; length = length; debug = debug; debugLevel = levelOf debug }

    let createWrapped<'S,'T when 'T: equality> (stage: Stage<'S,'T> option) (memAvail : uint64) (memPeak: uint64) (nElemsPerSlice: SingleOrPair) (length: uint64) (debug: bool) : Plan<'S, 'T> =
        { stage = stage; graph = graphOfStage stage; sourcePeek = None; costPeak = None; costObservations = []; memAvail = memAvail; memPeak = memPeak; nElemsPerSlice = nElemsPerSlice; length = length; debug = debug; debugLevel = levelOf debug }

    let withSourcePeek (sourcePeek: SourcePeek) (pl: Plan<'S,'T>) =
        { pl with sourcePeek = Some sourcePeek }

    let private mergeCostPeak (current: StageCostPressure option) (candidate: StageCostPressure) =
        match current with
        | None -> Some candidate
        | Some previous ->
            if candidate.Memory.Peak > previous.Memory.Peak then Some candidate else current

    let private workScore pressure =
        pressure.CpuWorkUnits
        + pressure.NativeWorkUnits
        + float pressure.IoReadBytes
        + float pressure.IoWriteBytes
        + float pressure.IoReadOps
        + float pressure.IoWriteOps

    let private trySumEstimatedMilliseconds (observations: StageCostPressure list) =
        observations
        |> List.fold
            (fun state observation ->
                match state, StageCostCalibration.estimateMilliseconds observation.Work with
                | Some total, Some milliseconds -> Some(total + milliseconds)
                | _ -> None)
            (Some 0.0)

    let private totalWorkScore (observations: StageCostPressure list) =
        observations |> List.sumBy (fun observation -> workScore observation.Work)

    let private printOptimizationSummary label (pl: Plan<'S,'T>) =
        if pl.debug && pl.debugLevel >= 1u && pl.debugLevel < 3u then
            let status =
                if pl.memPeak <= pl.memAvail then "accepted" else "exceeds memory limit"
            let timeText =
                match trySumEstimatedMilliseconds pl.costObservations with
                | Some milliseconds -> $", estimated time {milliseconds:g} ms"
                | None -> $", uncalibrated work score {totalWorkScore pl.costObservations:g}"
            let peakStageText =
                match pl.costPeak with
                | Some peak -> $", peak stage {peak.Memory.Peak / 1024UL} KB"
                | None -> ""

            printfn $"[{label}] Optimization {status}: estimated memory peak {pl.memPeak / 1024UL} / {pl.memAvail / 1024UL} KB{peakStageText}{timeText}; {pl.costObservations.Length} cost observations."

    let graph (pl: Plan<'S,'T>) =
        pl.graph

    //////////////////////////////////////////////////
    /// Source type operators
    let source (availableMemory: uint64) : Plan<unit, unit> =
        DebugLevel.set 0u
        create None availableMemory 0UL 0UL 0UL false

    let debug (level: uint) (availableMemory: uint64) : Plan<unit, unit> =
        let level = max 1u level
        DebugLevel.set level
        printfn $"[debug] Preparing plan - {availableMemory} B available, level {level}"
        let result = create None availableMemory 0UL 0UL 0UL true
        printfn $"[debug] Done"
        result

    //////////////////////////////////////////////////////////////
    /// Composition operators
    let composePlan (name: string) (pl: Plan<'a, 'b>) (stage: Stage<'b, 'c>) : Plan<'a, 'c> =
        if pl.debug then printfn $"[{name}] {stage.Name}"

        let stage' = Option.map (fun stg -> Stage.compose stg stage) pl.stage
        let memNeeded = pl.nElemsPerSlice |> stage.MemoryNeed  |> SingleOrPair.sum |> SingleOrPair.fst// Stage.MemoryNeed must be updated as well
        let memPeak = max pl.memPeak memNeeded
        let stageCost = StageCostModel.estimate stage.CostModel pl.nElemsPerSlice
        let costPeak = mergeCostPeak pl.costPeak stageCost
        let costObservations = stageCost :: pl.costObservations
        if (not pl.debug) && memPeak > pl.memAvail then
            failwith $"Out of available memory: {stage.Name} requested {memNeeded} B but have only {pl.memAvail} B"
        let nElemsPerSlice' = SingleOrPair.map stage.ElementTransformation pl.nElemsPerSlice
        let length' =
            SliceCardinality.length pl.length stage.SliceCardinality
            |> Option.defaultValue pl.length
        { createWrapped stage' pl.memAvail memPeak nElemsPerSlice' length' pl.debug with sourcePeek = pl.sourcePeek; costPeak = costPeak; costObservations = costObservations; debugLevel = pl.debugLevel }

    let (>=>) (pl: Plan<'a, 'b>) (stage: Stage<'b, 'c>) : Plan<'a, 'c> =
        composePlan $">=>" pl stage

    let map (name: string) (f: 'U->'V) (pl: Plan<'In,'U>) : Plan<'In,'V> =
        if pl.debug then printfn $"[{name}]"
        let memoryNeed m = 2UL*m // assuming simple transformation
        let elementTransformation = id
        let stage =
            match pl.stage with
            | Some stg -> 
                Stage.map1 name f stg memoryNeed elementTransformation // elementTransformation is unchanged by map per definition 
            | None -> failwith "Plan.map cannot map to empty stage"
        let nElemsPerSlice' = SingleOrPair.map stage.ElementTransformation pl.nElemsPerSlice
        let length' =
            SliceCardinality.length pl.length stage.SliceCardinality
            |> Option.defaultValue pl.length
        let memNeeded = pl.nElemsPerSlice |> stage.MemoryNeed  |> SingleOrPair.sum |> SingleOrPair.fst
        let memPeak = max pl.memPeak memNeeded
        let stageCost = StageCostModel.estimate stage.CostModel pl.nElemsPerSlice
        let costPeak = mergeCostPeak pl.costPeak stageCost
        let costObservations = stageCost :: pl.costObservations
        if (not pl.debug) && memPeak > pl.memAvail then
            failwith $"Out of available memory: {stage.Name} requested {memoryNeed} B but have only {pl.memAvail} B"
        { createWrapped (Some stage) pl.memAvail memPeak nElemsPerSlice' length' pl.debug with sourcePeek = pl.sourcePeek; costPeak = costPeak; costObservations = costObservations; debugLevel = pl.debugLevel }
        
    /// parallel execution of non-synchronised streams
    let internal zipPlan (name: string) (pl1: Plan<'In, 'U>) (pl2: Plan<'In, 'V>) : Plan<'In, ('U * 'V)> =
        match pl1.stage,pl2.stage with
            Some stage1, Some stage2 ->
                let debug = (pl1.debug || pl2.debug)
                if debug then DebugLevel.set (max pl1.debugLevel pl2.debugLevel)
                if debug then printfn $"[{name}] ({stage1.Name}, {stage2.Name})"

                if pl1.length <> pl2.length then
                    failwith $"[{name}] plans to be ziped must be over sequences of equal lengths {pl1.length} <> {pl2.length}"

                let nElemsPerSlice = Pair (SingleOrPair.fst pl1.nElemsPerSlice, SingleOrPair.fst pl2.nElemsPerSlice)  // SingleOrPair.isSingle pl1.nElemsPerSlice && pl1.nElemsPerSlice = pl2.nElemsPerSlice
                let memoryNeed nElemsPerSlice = // Should SingleOrPair be a tree?
                    let nElemsPerSlice1 = SingleOrPair.index1 nElemsPerSlice
                    let nElemsPerSlice2 = SingleOrPair.index2 nElemsPerSlice
                    SingleOrPair.add (nElemsPerSlice1 |> stage1.MemoryNeed) (nElemsPerSlice2 |> stage2.MemoryNeed) // SingleOrPair.add (nElemsPerSlice |> SingleOrPair.index1 |> stage1.MemoryNeed) (nElemsPerSlice |> SingleOrPair.index2 |> stage2.MemoryNeed)
                let elementTransformation = id // Transformation of result equal to any of the input
                let stage =
                    Stage.map2 $"({stage1.Name},{stage2.Name})" debug (fun U V -> (U,V)) stage1 stage2 memoryNeed elementTransformation

                // Check memory constraints for the zipped stream
                let memNeeded = SingleOrPair.add (nElemsPerSlice |> stage1.MemoryNeed) (nElemsPerSlice |> stage2.MemoryNeed)
                let maxMemPeak = max pl1.memPeak pl2.memPeak
                let memPeak = max maxMemPeak (nElemsPerSlice |> memoryNeed |> SingleOrPair.fst)
                let stageCost = StageCostModel.estimate stage.CostModel nElemsPerSlice
                let costPeak =
                    mergeCostPeak pl1.costPeak stageCost
                    |> fun peak ->
                        match pl2.costPeak with
                        | Some rightPeak -> mergeCostPeak peak rightPeak
                        | None -> peak
                let costObservations = stageCost :: (pl1.costObservations @ pl2.costObservations)
                let maxMemAvail = max pl1.memAvail pl2.memAvail // Should we max or sum? What would the most natural usage be for src?
                if (not debug) && (memPeak > maxMemAvail) then
                    failwith $"Out of available memory: {stage.Name} requested {memNeeded|>SingleOrPair.fst}+{memNeeded|>SingleOrPair.snd}={memPeak} B but have only {maxMemAvail} B"
                { createWrapped (Some stage) maxMemAvail memPeak nElemsPerSlice pl1.length debug with sourcePeek = pl1.sourcePeek; costPeak = costPeak; costObservations = costObservations; debugLevel = max pl1.debugLevel pl2.debugLevel }
            | _,_ -> failwith $"[{name}] Cannot zip with an empty plan"

    /// parallel execution of non-synchronised streams
    let zip (pl1: Plan<'In, 'U>) (pl2: Plan<'In, 'V>) : Plan<'In, ('U * 'V)> = zipPlan "zip" pl1 pl2

    /// parallel execution of synchronised streams
    let (>=>>) (pl: Plan<'In, 'S>) (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Plan<'In, 'U * 'V> =
        if stage1.Transition.From <> stage2.Transition.From then 
            failwith $"[>=>>] can only apply stages with the same streaming profile ({stage1.Transition.From} <> {stage2.Transition.From})"

        let memoryNeed nElemsPerSlice = 
            // nElemsPerSlice is always Singel
            SingleOrPair.add (nElemsPerSlice |> stage1.MemoryNeed) (nElemsPerSlice |> stage2.MemoryNeed)

        let elementTransformation = stage1.ElementTransformation 
        let elementTransformation2 = stage2.ElementTransformation 
        let nElems = SingleOrPair.map elementTransformation pl.nElemsPerSlice
        let nElems2 = SingleOrPair.map elementTransformation2 pl.nElemsPerSlice
        if nElems <> nElems2 then
            failwith $"[>=>>] Cannot zip plans with different number of elements {nElems} vs {nElems2}"
        let length = SliceCardinality.length pl.length stage1.SliceCardinality
        let length2 = SliceCardinality.length pl.length stage2.SliceCardinality
        if length <> length2 || stage1.SliceCardinality <> stage2.SliceCardinality then
            failwith $"[>=>>] Cannot synchronize branches with different slice domains {stage1.SliceCardinality} vs {stage2.SliceCardinality}"

        // Combine both stages in a zip-like stage
        let stage = Stage.map2Sync $"({stage1.Name},{stage2.Name})" pl.debug (fun u v -> (u, v)) stage1 stage2 memoryNeed elementTransformation

        composePlan ">=>>" pl stage

    let (>>=>) (pl: Plan<'In,'U*'V>) ((f: 'U -> 'V -> 'W)) : Plan<'In,'W>  = 
        map ">>=>" (fun (u,v) -> f u v) pl
(*
    let (>>=>) (pl: Plan<'In,'U*'V>) (stage: Stage<'U*'V,'W>) : Plan<'In,'W>  = 
        composePlan ">>=>" pl stage
*)

    let (>>=>>) (pl: Plan<'In,'U*'V>) (stage1: Stage<'U,'S>, stage2: Stage<'V,'T>) : Plan<'In,'S*'T> =
        let memoryNeed nElemsPerSlice =
            SingleOrPair.add (nElemsPerSlice |> SingleOrPair.index1 |> stage1.MemoryNeed) (nElemsPerSlice |> SingleOrPair.index2 |> stage2.MemoryNeed)

        let elementTransformation nElems =
            let left = stage1.ElementTransformation nElems
            let right = stage2.ElementTransformation nElems
            if left <> right then
                failwith $"[>>=>>] Cannot zip branch outputs with different element shapes {left} vs {right}"
            left

        if stage1.SliceCardinality <> stage2.SliceCardinality then
            failwith $"[>>=>>] Cannot synchronize branches with different slice domains {stage1.SliceCardinality} vs {stage2.SliceCardinality}"

        let stage = Stage.mapPairSync $"({stage1.Name},{stage2.Name})" pl.debug stage1 stage2 memoryNeed elementTransformation
        composePlan ">>=>>" pl stage

    ///////////////////////////////////////////
    /// sink type operators
    let doCleaning pl = 
        Option.map (fun stg -> stg.Cleaning |> List.rev |> List.iter (fun fct -> fct ())) pl.stage

    let sink (pl: Plan<unit, unit>) : unit =
        if not pl.debug && (pl.memPeak > pl.memAvail) then
            failwith $"Not enough memory for the plan {pl.memPeak} > {pl.memAvail}"
        if pl.debug then printfn $"[sink] Transform plan graph with {pl.graph.Nodes.Length} nodes / {pl.graph.Edges.Length} edges to pipeline"
        let pipeline = Option.map (fun stage -> Stage.toPipe stage ()) pl.stage
        printOptimizationSummary "sink" pl
        if pl.debug then printfn $"[sink] Running pipeline with an estimated {pl.memPeak/1024UL} / {pl.memAvail/1024UL} KB of memory use"
        if pl.debug && pl.debugLevel >= 2u then
            let _, snapshot =
                MemoryProbe.measure 10 (fun () ->
                    Option.map (Pipe.run pl.debug) pipeline |> ignore)
            printfn $"[sink] Process RSS baseline {MemoryProbe.formatBytes snapshot.Baseline}, peak {MemoryProbe.formatBytes snapshot.Peak}, delta {MemoryProbe.formatBytes snapshot.Delta}"
        else
            Option.map (Pipe.run pl.debug) pipeline |> ignore
        if pl.debug then printfn "[sink] Cleaning"
        doCleaning pl |> ignore
        if pl.debug then printfn "[sink] Done"

    let sinkList (plans: Plan<unit, unit> list) : unit = // shape of unit?
        plans |> List.iter sink

    let internal runToScalar (name: string) (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Plan<unit, 'T>) : 'R =
        if pl.memPeak > pl.memAvail then
            failwith $"Not enough memory for the plan {pl.memPeak} > {pl.memAvail}"

        let stage = pl.stage
        let res = 
            match stage with
                Some stg -> 
                    if pl.debug then printfn $"[{name}] Transform plan graph with {pl.graph.Nodes.Length} nodes / {pl.graph.Edges.Length} edges to pipeline"
                    let pipeline = stg.Build ()
                    printOptimizationSummary name pl
                    if pl.debug then printfn $"[{name}] Running pipeline with an estimated {pl.memPeak/1024UL} / {pl.memAvail/1024UL} KB memory use" 
                    if pl.debug && pl.debugLevel >= 2u then
                        let result, snapshot =
                            MemoryProbe.measure 10 (fun () ->
                                AsyncSeq.singleton () |> pipeline.Apply pl.debug |> reducer |> Async.RunSynchronously)
                        printfn $"[{name}] Process RSS baseline {MemoryProbe.formatBytes snapshot.Baseline}, peak {MemoryProbe.formatBytes snapshot.Peak}, delta {MemoryProbe.formatBytes snapshot.Delta}"
                        result
                    else
                        AsyncSeq.singleton () |> pipeline.Apply pl.debug |> reducer |> Async.RunSynchronously
                | _ -> failwith $"[{name}] Plan is empty"
        doCleaning pl |> ignore
        res

    let drainSingle (name: string) (pl: Plan<unit, 'T>) =
        runToScalar name AsyncSeq.toListAsync pl
        |> function
            | [x] -> 
                x
            | []  -> failwith $"[drainSingle] No result from {name}"
            | _   -> failwith $"[drainSingle] Multiple results from {name}, expected one."

    let drainList (name: string) (pl: Plan<unit, 'T>) =
        runToScalar name AsyncSeq.toListAsync pl

    let drainLast (name: string) (pl: Plan<unit, 'T>) =
        runToScalar name AsyncSeq.tryLast pl
        |> function
            | Some x -> x
            | None -> failwith $"[drainLast] No result from {name}"
