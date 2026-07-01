module StackObjects

open System
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Runtime.InteropServices
open System.Text
open FSharp.Control
open SlimPipeline
open StackCore
open StackPoints

type ObjectConnectivity =
    | Six
    | TwentySix

type ObjectBounds =
    { MinX: int
      MaxX: int
      MinY: int
      MaxY: int
      MinZ: int
      MaxZ: int }

type StreamedObject =
    { Label: uint64
      Positions: Position3D<int> list
      Bounds: ObjectBounds
      Size: uint64 }

    member object.Width =
        uint64 (object.Bounds.MaxX - object.Bounds.MinX + 1)

    member object.Height =
        uint64 (object.Bounds.MaxY - object.Bounds.MinY + 1)

    member object.Depth =
        uint64 (object.Bounds.MaxZ - object.Bounds.MinZ + 1)

type ObjectStream<'T> =
    { Objects: StreamedObject list }

let private objectStream<'T> objects : ObjectStream<'T> =
    { Objects = objects }

let private convertFloat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> value =
    Convert.ChangeType(value, typeof<'T>) :?> 'T

let objectSource<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (objects: StreamedObject list)
    (template: Plan<unit, unit>)
    : Plan<unit, ObjectStream<'T>> =
    let sortedObjects =
        objects
        |> List.sortBy (fun object -> object.Bounds.MinZ)
        |> List.toArray

    let stage =
        Stage.init
            "objectSource"
            (uint sortedObjects.Length)
            (fun i -> objectStream<'T> [ sortedObjects[i] ])
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    { Plan.createWithOptimizer
        (Some stage)
        template.memAvail
        0UL
        1UL
        (uint64 sortedObjects.Length)
        template.debug
        template.optimize with
        debugLevel = template.debugLevel
        costDiscrepancy = template.costDiscrepancy
        costFlagPath = template.costFlagPath }

type private SliceComponent =
    { Id: int
      Positions: Position3D<int> list
      Frontier: Set<int * int>
      Bounds: ObjectBounds
      Size: uint64 }

type private ActiveObject =
    { Label: uint64
      PositionsReversed: Position3D<int> list
      Frontier: Set<int * int>
      Bounds: ObjectBounds
      Size: uint64 }

type private GraphNode =
    | Active of uint64
    | Local of int

let private objectPoint x y z =
    { X = x
      Y = y
      Z = z }

let private pointBounds x y z : ObjectBounds =
    { MinX = x
      MaxX = x
      MinY = y
      MaxY = y
      MinZ = z
      MaxZ = z }

let private addPointToBounds (bounds: ObjectBounds) x y z : ObjectBounds =
    { MinX = min bounds.MinX x
      MaxX = max bounds.MaxX x
      MinY = min bounds.MinY y
      MaxY = max bounds.MaxY y
      MinZ = min bounds.MinZ z
      MaxZ = max bounds.MaxZ z }

let private mergeBounds (left: ObjectBounds) (right: ObjectBounds) : ObjectBounds =
    { MinX = min left.MinX right.MinX
      MaxX = max left.MaxX right.MaxX
      MinY = min left.MinY right.MinY
      MaxY = max left.MaxY right.MaxY
      MinZ = min left.MinZ right.MinZ
      MaxZ = max left.MaxZ right.MaxZ }

let private toStreamedObject active =
    { Label = active.Label
      Positions = List.rev active.PositionsReversed
      Bounds = active.Bounds
      Size = active.Size }

let private foreground (value: 'T) =
    Convert.ToDouble(value) <> 0.0

let private xyNeighborOffsets connectivity =
    match connectivity with
    | Six ->
        [| -1, 0; 1, 0; 0, -1; 0, 1 |]
    | TwentySix ->
        [| for dy in -1 .. 1 do
               for dx in -1 .. 1 do
                   if dx <> 0 || dy <> 0 then
                       dx, dy |]

let private zNeighborOffsets connectivity =
    match connectivity with
    | Six -> [| 0, 0 |]
    | TwentySix ->
        [| for dy in -1 .. 1 do
               for dx in -1 .. 1 -> dx, dy |]

let private connectedComponents2D connectivity z width height (isForeground: int -> int -> bool) =
    let visited = Array.zeroCreate<bool> (width * height)
    let offsets = xyNeighborOffsets connectivity
    let components = ResizeArray<SliceComponent>()
    let queue = Queue<int * int>()
    let mutable nextId = 0

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let i = flatIndex2 width x y
            if isForeground x y && not visited[i] then
                visited[i] <- true
                queue.Enqueue(x, y)

                let points = ResizeArray<Position3D<int>>()
                let frontier = HashSet<int * int>()
                let mutable bounds = pointBounds x y z

                while queue.Count > 0 do
                    let cx, cy = queue.Dequeue()
                    points.Add(objectPoint cx cy z)
                    frontier.Add(cx, cy) |> ignore
                    bounds <- addPointToBounds bounds cx cy z

                    for dx, dy in offsets do
                        let nx = cx + dx
                        let ny = cy + dy
                        if nx >= 0 && nx < width && ny >= 0 && ny < height then
                            let ni = flatIndex2 width nx ny
                            if isForeground nx ny && not visited[ni] then
                                visited[ni] <- true
                                queue.Enqueue(nx, ny)

                components.Add
                    { Id = nextId
                      Positions = points |> Seq.toList
                      Frontier = frontier |> Seq.toList |> Set.ofList
                      Bounds = bounds
                      Size = uint64 points.Count }

                nextId <- nextId + 1

    components |> Seq.toList

let private touchesPreviousFront offsets (previousFrontier: Set<int * int>) (currentFrontier: Set<int * int>) =
    currentFrontier
    |> Seq.exists (fun (x, y) ->
        offsets
        |> Array.exists (fun (dx, dy) -> previousFrontier.Contains(x + dx, y + dy)))

let private connectedGroups nodes edges =
    let adjacency =
        nodes
        |> Seq.map (fun node -> node, ResizeArray<GraphNode>())
        |> dict

    for left, right in edges do
        adjacency[left].Add right
        adjacency[right].Add left

    let visited = HashSet<GraphNode>()
    let groups = ResizeArray<GraphNode list>()
    let queue = Queue<GraphNode>()

    for node in nodes do
        if visited.Add node then
            queue.Enqueue node
            let group = ResizeArray<GraphNode>()

            while queue.Count > 0 do
                let current = queue.Dequeue()
                group.Add current

                for next in adjacency[current] do
                    if visited.Add next then
                        queue.Enqueue next

            groups.Add(group |> Seq.toList)

    groups |> Seq.toList

let private mergeActiveAndLocal label activeObjects localComponents =
    let activePoints =
        activeObjects
        |> List.collect _.PositionsReversed

    let localPoints =
        localComponents
        |> List.collect _.Positions

    let bounds =
        let activeBounds = activeObjects |> List.map _.Bounds
        let localBounds = localComponents |> List.map _.Bounds
        (activeBounds @ localBounds)
        |> List.reduce mergeBounds

    let size =
        let activeSize = activeObjects |> List.sumBy _.Size
        let localSize = localComponents |> List.sumBy _.Size
        activeSize + localSize

    { Label = label
      PositionsReversed = (localPoints |> List.rev) @ activePoints
      Frontier =
        localComponents
        |> List.collect (fun objectComponent -> objectComponent.Frontier |> Set.toList)
        |> Set.ofList
      Bounds = bounds
      Size = size }

let private processChunkSlice<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    connectivity
    ((stateNextLabel, active): uint64 * Map<uint64, ActiveObject>)
    z
    (chunk: Chunk<'T>)
    =
    let widthU, heightU, depthU = chunk.Size
    if depthU <> 1UL then
        invalidArg "chunk" $"streamConnectedObjectsChunk expects 2D slice chunks with depth 1, got {chunk.Size}."
    let width = int widthU
    let height = int heightU
    let zOffsets = zNeighborOffsets connectivity
    let pixels = (Chunk.span chunk).ToArray()
    let components =
        connectedComponents2D connectivity z width height (fun x y -> foreground pixels[flatIndex2 width x y])

    let activeNodes = active |> Map.toList |> List.map (fst >> Active)
    let localNodes = components |> List.map (fun objectComponent -> Local objectComponent.Id)
    let componentMap = components |> List.map (fun objectComponent -> objectComponent.Id, objectComponent) |> Map.ofList

    let edges =
        [ for KeyValue(activeLabel, activeObject) in active do
              for objectComponent in components do
                  if touchesPreviousFront zOffsets activeObject.Frontier objectComponent.Frontier then
                      Active activeLabel, Local objectComponent.Id ]

    let groups = connectedGroups (activeNodes @ localNodes) edges
    let mutable nextLabel = stateNextLabel
    let mutable nextActive = Map.empty<uint64, ActiveObject>
    let completed = ResizeArray<StreamedObject>()

    for group in groups do
        let activeLabels =
            group
            |> List.choose (function Active label -> Some label | _ -> None)

        let localIds =
            group
            |> List.choose (function Local id -> Some id | _ -> None)

        match activeLabels, localIds with
        | activeLabels, [] ->
            activeLabels
            |> List.choose (fun label -> active |> Map.tryFind label)
            |> List.iter (toStreamedObject >> completed.Add)
        | activeLabels, localIds ->
            let label =
                match activeLabels with
                | [] ->
                    let label = nextLabel
                    nextLabel <- nextLabel + 1UL
                    label
                | labels ->
                    labels |> List.min

            let activeObjects =
                activeLabels
                |> List.choose (fun label -> active |> Map.tryFind label)

            let localComponents =
                localIds
                |> List.choose (fun id -> componentMap |> Map.tryFind id)

            let merged = mergeActiveAndLocal label activeObjects localComponents
            nextActive <- nextActive |> Map.add label merged

    (nextLabel, nextActive), completed |> Seq.toList

let streamConnectedObjectsChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    connectivity
    : Stage<Chunk<'T>, ObjectStream<uint8>> =
    let name = "streamConnectedObjectsChunk"

    let apply _debug (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            let mutable state = 1UL, Map.empty<uint64, ActiveObject>
            let enumerator = input.GetAsyncEnumerator()
            let mutable more = true
            let mutable z = 0

            while more do
                let! hasNext = enumerator.MoveNextAsync().AsTask() |> Async.AwaitTask

                if hasNext then
                    let chunk = enumerator.Current
                    let nextState, completed =
                        try
                            processChunkSlice connectivity state z chunk
                        finally
                            Chunk.decRef chunk

                    state <- nextState
                    z <- z + 1
                    yield objectStream<uint8> completed
                else
                    more <- false

            let _, active = state
            let finalCompleted =
                active
                |> Map.toList
                |> List.map (snd >> toStreamedObject)

            if finalCompleted.Length > 0 then
                yield objectStream<uint8> finalCompleted
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) id id pipe

let private paintObjectValue width value (buffer: SortedDictionary<int, uint8[]>) (object: StreamedObject) =
    for position in object.Positions do
        match buffer.TryGetValue position.Z with
        | true, pixels ->
            pixels[flatIndex2 width position.X position.Y] <- value
        | false, _ ->
            invalidOp $"Cannot edit completed object label {object.Label}; buffered slice {position.Z} has already been emitted."

let private touchesXYBoundary width height (object: StreamedObject) =
    object.Bounds.MinX <= 0
    || object.Bounds.MinY <= 0
    || object.Bounds.MaxX >= width - 1
    || object.Bounds.MaxY >= height - 1

let private copyBinaryChunkBytes (chunk: Chunk<uint8>) =
    let output = Array.zeroCreate<uint8> chunk.ByteLength
    chunk.Bytes.AsSpan(0, chunk.ByteLength).CopyTo(output.AsSpan())
    output

let private invertedBinaryChunk (chunk: Chunk<uint8>) =
    let output = Chunk.create<uint8> chunk.Size
    try
        let inputPixels = Chunk.span<uint8> chunk
        let outputPixels = Chunk.span<uint8> output
        for i in 0 .. inputPixels.Length - 1 do
            outputPixels[i] <- if inputPixels[i] = 0uy then 1uy else 0uy
        output
    with
    | ex ->
        Chunk.decRef output
        raise ex

let private backgroundBinaryChunk background (chunk: Chunk<uint8>) =
    let output = Chunk.create<uint8> chunk.Size
    try
        let inputPixels = Chunk.span<uint8> chunk
        let outputPixels = Chunk.span<uint8> output
        for i in 0 .. inputPixels.Length - 1 do
            outputPixels[i] <- if inputPixels[i] = background then 1uy else 0uy
        output
    with
    | ex ->
        Chunk.decRef output
        raise ex

let private emitChunkBufferThrough width height cutoff (buffer: SortedDictionary<int, uint8[]>) =
    seq {
        let mutable emitting = true

        while emitting && buffer.Count > 0 do
            let z = buffer.Keys |> Seq.head

            if z <= cutoff then
                let chunk = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
                try
                    buffer[z].AsSpan().CopyTo(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                    buffer.Remove z |> ignore
                    yield chunk
                with
                | ex ->
                    Chunk.decRef chunk
                    raise ex
            else
                emitting <- false
    }

let private bufferedBinaryChunkComponentEdit
    name
    (maximumVolume: uint64)
    connectivity
    targetValue
    componentChunk
    isProtectedComponent
    : Stage<Chunk<uint8>, Chunk<uint8>> =

    if maximumVolume > uint64 Int32.MaxValue then
        invalidArg "maximumVolume" $"{name} maximumVolume must fit in Int32 because slice indices are Int32-backed, got {maximumVolume}."

    let maxRetainedSlices = int maximumVolume

    let couldStillEdit width height firstZ (object: ActiveObject) =
        object.Size <= maximumVolume
        && not (isProtectedComponent width height firstZ None (toStreamedObject object))

    let pruneUneditablePositions width height firstZ active =
        active
        |> Map.map (fun _ object ->
            if couldStillEdit width height firstZ object then
                object
            else
                { object with PositionsReversed = [] })

    let apply _debug (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let buffer = SortedDictionary<int, uint8[]>()
            let mutable state = 1UL, Map.empty<uint64, ActiveObject>
            let mutable width = None
            let mutable height = None
            let mutable firstZ = None
            let mutable z = 0
            let enumerator = input.GetAsyncEnumerator()
            let mutable more = true

            while more do
                let! hasNext = enumerator.MoveNextAsync().AsTask() |> Async.AwaitTask

                if hasNext then
                    let chunk = enumerator.Current
                    let widthU, heightU, depthU = chunk.Size
                    if depthU <> 1UL then
                        invalidArg "chunk" $"{name} expects 2D UInt8 slice chunks with depth 1, got {chunk.Size}."
                    if widthU > uint64 Int32.MaxValue || heightU > uint64 Int32.MaxValue then
                        invalidArg "chunk" $"{name} chunk dimensions must fit in Int32, got {chunk.Size}."

                    let currentWidth = int widthU
                    let currentHeight = int heightU

                    match width, height with
                    | None, None ->
                        width <- Some currentWidth
                        height <- Some currentHeight
                        firstZ <- Some z
                    | Some expectedWidth, Some expectedHeight when expectedWidth = currentWidth && expectedHeight = currentHeight -> ()
                    | _ ->
                        invalidOp $"{name} requires a stream with constant x-y slice size."

                    buffer[z] <- copyBinaryChunkBytes chunk

                    let componentInput = componentChunk chunk
                    let nextState, completed =
                        try
                            processChunkSlice connectivity state z componentInput
                        finally
                            Chunk.decRef componentInput
                            Chunk.decRef chunk

                    state <-
                        let nextLabel, active = nextState
                        nextLabel, pruneUneditablePositions currentWidth currentHeight firstZ active

                    for object in completed do
                        if object.Size <= maximumVolume && not (isProtectedComponent currentWidth currentHeight firstZ None object) then
                            paintObjectValue currentWidth targetValue buffer object

                    let _, active = state
                    let editableActiveMinZ =
                        active
                        |> Map.toSeq
                        |> Seq.choose (fun (_, object) ->
                            if couldStillEdit currentWidth currentHeight firstZ object then
                                Some object.Bounds.MinZ
                            else
                                None)
                        |> Seq.fold (fun current value ->
                            match current with
                            | None -> Some value
                            | Some minimum -> Some (min minimum value)) None

                    let cutoff =
                        match editableActiveMinZ with
                        | Some minActiveZ -> minActiveZ - 1
                        | None -> z

                    for chunk in emitChunkBufferThrough currentWidth currentHeight cutoff buffer do
                        yield chunk

                    z <- z + 1
                else
                    more <- false

            let _, active = state
            let lastZ =
                if buffer.Count = 0 then
                    None
                else
                    Some (buffer.Keys |> Seq.max)

            for object in active |> Map.toList |> List.map (snd >> toStreamedObject) do
                let currentWidth = width |> Option.defaultValue 0
                let currentHeight = height |> Option.defaultValue 0

                if object.Size <= maximumVolume && not (isProtectedComponent currentWidth currentHeight firstZ lastZ object) then
                    paintObjectValue currentWidth targetValue buffer object

            for chunk in emitChunkBufferThrough (width |> Option.defaultValue 0) (height |> Option.defaultValue 0) Int32.MaxValue buffer do
                yield chunk
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    let memoryNeed nPixels =
        nPixels * (uint64 (maxRetainedSlices + 3))

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) memoryNeed id pipe

let private binaryComponentEditChunk
    name
    maximumVolume
    connectivity
    componentValue
    replacementValue
    defaultReplacementValue
    defaultComponentChunk
    isProtectedComponent
    : Stage<Chunk<uint8>, Chunk<uint8>> =
    let targetValue, componentChunk =
        let replacement =
            replacementValue
            |> Option.map convertFloat<uint8>
            |> Option.defaultValue defaultReplacementValue

        match componentValue |> Option.map convertFloat<uint8> with
        | Some componentValue -> replacement, backgroundBinaryChunk componentValue
        | None -> replacement, defaultComponentChunk

    bufferedBinaryChunkComponentEdit
        name
        maximumVolume
        connectivity
        targetValue
        componentChunk
        isProtectedComponent

let removeSmallObjectsChunk maximumVolume connectivity : Stage<Chunk<uint8>, Chunk<uint8>> =
    binaryComponentEditChunk
        "chunkRemoveSmallObjects"
        maximumVolume
        connectivity
        None
        None
        0uy
        Chunk.incRef
        (fun _ _ _ _ _ -> false)

let fillSmallHolesChunk maximumVolume connectivity backgroundValue foregroundValue : Stage<Chunk<uint8>, Chunk<uint8>> =
    let protectExterior width height firstZ lastZ object =
        touchesXYBoundary width height object
        || (firstZ |> Option.exists (fun z -> object.Bounds.MinZ = z))
        || (lastZ |> Option.exists (fun z -> object.Bounds.MaxZ = z))

    binaryComponentEditChunk
        "chunkFillSmallHoles"
        maximumVolume
        connectivity
        backgroundValue
        foregroundValue
        1uy
        invertedBinaryChunk
        protectExterior

let private paintObjectSlice<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: uint)
    (height: uint)
    (background: 'T)
    (foreground: 'T)
    z
    (objects: StreamedObject list)
    =
    if width = 0u then invalidArg (nameof width) "paintObjectsChunk width must be positive."
    if height = 0u then invalidArg (nameof height) "paintObjectsChunk height must be positive."

    let width = int width
    let height = int height
    let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
    let pixels = Chunk.span chunk
    pixels.Fill background

    for object in objects do
        for position in object.Positions do
            if position.Z = z then
                if position.X < 0 || position.X >= width || position.Y < 0 || position.Y >= height then
                    invalidOp $"Object position ({position.X},{position.Y},{position.Z}) is outside the requested paint chunk size {width}x{height}."

                pixels[flatIndex2 width position.X position.Y] <- foreground

    chunk

let paintObjectsChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: uint)
    (height: uint)
    (depth: uint)
    (backgroundValue: double option)
    (foregroundValue: double option)
    : Stage<ObjectStream<'T>, Chunk<'T>> =
    if width = 0u then invalidArg (nameof width) "paintObjectsChunk width must be positive."
    if height = 0u then invalidArg (nameof height) "paintObjectsChunk height must be positive."

    let background = backgroundValue |> Option.defaultValue 0.0 |> convertFloat<'T>
    let foreground = foregroundValue |> Option.defaultValue 1.0 |> convertFloat<'T>

    let apply _debug (input: AsyncSeq<ObjectStream<'T>>) =
        asyncSeq {
            let mutable currentZ = 0
            let mutable active: StreamedObject list = []
            let depth = int depth

            let emitUntil targetZ =
                asyncSeq {
                    let targetZ = min targetZ depth

                    while currentZ < targetZ do
                        active <- active |> List.filter (fun object -> object.Bounds.MaxZ >= currentZ)
                        yield paintObjectSlice<'T> width height background foreground currentZ active
                        currentZ <- currentZ + 1
                }

            for stream in input do
                let orderedBatch =
                    stream.Objects
                    |> List.sortBy (fun object -> object.Bounds.MinZ)

                for object in orderedBatch do
                    if currentZ < depth && object.Bounds.MinZ < depth then
                        yield! emitUntil object.Bounds.MinZ

                    if currentZ < depth && object.Bounds.MaxZ >= currentZ && object.Bounds.MinZ < depth then
                        active <- object :: active

            yield! emitUntil depth
        }

    let memoryNeed nPixels =
        nPixels * 2UL * uint64 (Marshal.SizeOf<'T>())

    Stage.fromAsyncSeq
        "paintObjectsChunk"
        apply
        (ProfileTransition.create Streaming Streaming)
        (StageMemoryModel.fromSinglePeak Map memoryNeed)
        (fun _ -> uint64 depth)
    |> Stage.withSliceCardinality (SliceCardinality.reduceTo (uint64 depth))

let private paintCroppedObjectChunkBatch<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (objects: StreamedObject list)
    =
    let foreground = convertFloat<'T> 1.0

    objects
    |> List.collect (fun object ->
        let width = object.Bounds.MaxX - object.Bounds.MinX + 1
        let height = object.Bounds.MaxY - object.Bounds.MinY + 1

        if width <= 0 || height <= 0 then
            []
        else
            object.Positions
            |> List.groupBy _.Z
            |> List.sortBy fst
            |> List.map (fun (_z, positions) ->
                let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
                chunk.Bytes.AsSpan(0, chunk.ByteLength).Clear()
                let pixels = Chunk.span chunk

                for position in positions do
                    let x = position.X - object.Bounds.MinX
                    let y = position.Y - object.Bounds.MinY

                    if x < 0 || x >= width || y < 0 || y >= height then
                        invalidOp $"Object position ({position.X},{position.Y},{position.Z}) is outside its bounding box."

                    pixels[flatIndex2 width x y] <- foreground

                chunk))

let paintObjectsCroppedChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    : Stage<ObjectStream<'T>, Chunk<'T>> =
    Stage.map "paintObjectsCroppedChunk" (fun _ stream -> paintCroppedObjectChunkBatch<'T> stream.Objects) id id
    --> Stage.flatten "paintObjectsCroppedChunk: flatten"

let private objectFileSuffix (suffix: string) =
    let suffix = if String.IsNullOrWhiteSpace suffix then ".csv" else suffix
    if not (suffix.Equals(".csv", StringComparison.OrdinalIgnoreCase)) then
        failwith $"Unsupported object output format '{suffix}'. Currently supported: .csv."
    suffix

let private objectFileName suffix (object: StreamedObject) =
    sprintf "object_z%010d_z%010d_label%020d%s" object.Bounds.MinZ object.Bounds.MaxZ object.Label suffix

let private objectCsvHeader = "label,x,y,z,size,minX,maxX,minY,maxY,minZ,maxZ"

let private writeObjectCsv (writer: StreamWriter) (object: StreamedObject) =
    writer.WriteLine(objectCsvHeader)

    for position in object.Positions do
        writer.WriteLine(
            String.Join(
                ",",
                [| object.Label.ToString(CultureInfo.InvariantCulture)
                   position.X.ToString(CultureInfo.InvariantCulture)
                   position.Y.ToString(CultureInfo.InvariantCulture)
                   position.Z.ToString(CultureInfo.InvariantCulture)
                   object.Size.ToString(CultureInfo.InvariantCulture)
                   object.Bounds.MinX.ToString(CultureInfo.InvariantCulture)
                   object.Bounds.MaxX.ToString(CultureInfo.InvariantCulture)
                   object.Bounds.MinY.ToString(CultureInfo.InvariantCulture)
                   object.Bounds.MaxY.ToString(CultureInfo.InvariantCulture)
                   object.Bounds.MinZ.ToString(CultureInfo.InvariantCulture)
                   object.Bounds.MaxZ.ToString(CultureInfo.InvariantCulture) |]))

let private parseInvariantUInt64 (text: string) =
    UInt64.Parse(text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture)

let private parseInvariantInt (text: string) =
    Int32.Parse(text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture)

let private readObjectCsvFile path =
    let rows =
        File.ReadLines(path)
        |> Seq.filter (fun line -> not (String.IsNullOrWhiteSpace line))
        |> Seq.toArray

    if rows.Length < 2 then
        invalidOp $"Object CSV file '{path}' did not contain any object rows."

    let parsedRows =
        rows
        |> Seq.skip 1
        |> Seq.map (fun line ->
            let columns = line.Split(',') |> Array.map _.Trim()
            if columns.Length <> 11 then
                invalidOp $"Object CSV rows must have 11 columns, but '{path}' contained '{line}'."

            let label = parseInvariantUInt64 columns[0]
            let position =
                { X = parseInvariantInt columns[1]
                  Y = parseInvariantInt columns[2]
                  Z = parseInvariantInt columns[3] }
            let size = parseInvariantUInt64 columns[4]
            let bounds =
                { MinX = parseInvariantInt columns[5]
                  MaxX = parseInvariantInt columns[6]
                  MinY = parseInvariantInt columns[7]
                  MaxY = parseInvariantInt columns[8]
                  MinZ = parseInvariantInt columns[9]
                  MaxZ = parseInvariantInt columns[10] }
            label, position, size, bounds)
        |> Seq.toList

    let labels = parsedRows |> List.map (fun (label, _, _, _) -> label) |> Set.ofList
    if labels.Count <> 1 then
        invalidOp $"Object CSV file '{path}' contains multiple labels."

    let sizes = parsedRows |> List.map (fun (_, _, size, _) -> size) |> Set.ofList
    if sizes.Count <> 1 then
        invalidOp $"Object CSV file '{path}' contains inconsistent object sizes."

    let bounds = parsedRows |> List.map (fun (_, _, _, bounds) -> bounds) |> Set.ofList
    if bounds.Count <> 1 then
        invalidOp $"Object CSV file '{path}' contains inconsistent object bounds."

    let label, _, size, bounds = parsedRows.Head
    let positions = parsedRows |> List.map (fun (_, position, _, _) -> position)
    if uint64 positions.Length <> size then
        invalidOp $"Object CSV file '{path}' declares size {size}, but contains {positions.Length} positions."

    { Label = label
      Positions = positions
      Bounds = bounds
      Size = size }

let readObjects<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (input: string)
    (suffix: string)
    (template: Plan<unit, unit>)
    : Plan<unit, ObjectStream<'T>> =
    let suffix = objectFileSuffix suffix
    if not (Directory.Exists input) then
        invalidOp $"Object directory '{input}' does not exist."

    let files =
        Directory.GetFiles(input, "*" + suffix, SearchOption.TopDirectoryOnly)
        |> Array.sortBy Path.GetFileName

    let stage =
        Stage.init
            "readObjects"
            (uint files.Length)
            (fun i -> objectStream<'T> [ readObjectCsvFile files[i] ])
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    { Plan.createWithOptimizer
        (Some stage)
        template.memAvail
        0UL
        1UL
        (uint64 files.Length)
        template.debug
        template.optimize with
        debugLevel = template.debugLevel
        costDiscrepancy = template.costDiscrepancy
        costFlagPath = template.costFlagPath }

let writeObjects<'T> (output: string) (suffix: string) : Stage<ObjectStream<'T>, unit> =
    let reducer (debug: bool) (input: AsyncSeq<ObjectStream<'T>>) =
        async {
            let suffix = objectFileSuffix suffix
            if debug then
                printfn $"[writeObjects] Writing objects to {output}"

            Directory.CreateDirectory(output) |> ignore
            for existing in Directory.GetFiles(output, "*" + suffix, SearchOption.TopDirectoryOnly) do
                File.Delete existing

            let mutable fileOrdinal = 0UL

            do!
                input
                |> AsyncSeq.iterAsync (fun stream ->
                    async {
                        for object in stream.Objects do
                            let fileName =
                                let candidate = objectFileName suffix object
                                if File.Exists(Path.Combine(output, candidate)) then
                                    sprintf "object_z%010d_z%010d_label%020d_%020d%s" object.Bounds.MinZ object.Bounds.MaxZ object.Label fileOrdinal suffix
                                else
                                    candidate
                            fileOrdinal <- fileOrdinal + 1UL

                            use writer = new StreamWriter(Path.Combine(output, fileName), false, Encoding.UTF8)
                            writeObjectCsv writer object
                    })
        }

    Stage.reduce $"writeObjects \"{output}\" \"{suffix}\"" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let objectSizes<'T> : Stage<ObjectStream<'T>, uint64 list> =
    Stage.map "objectSizes" (fun _ stream -> stream.Objects |> List.map (fun object -> object.Size)) id id
