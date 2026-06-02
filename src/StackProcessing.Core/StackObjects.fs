module StackObjects

open System
open System.Collections.Generic
open FSharp.Control
open Image.InternalHelpers
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

type ObjectMeasurements =
    { Label: uint64
      Size: uint64
      MinX: int
      MaxX: int
      MinY: int
      MaxY: int
      MinZ: int
      MaxZ: int
      Width: uint64
      Height: uint64
      Depth: uint64 }

type ObjectSizeStats =
    { Count: uint64
      Mean: float
      Variance: float
      Minimum: uint64
      Maximum: uint64 }

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

let private processSlice connectivity ((stateNextLabel, active): uint64 * Map<uint64, ActiveObject>) (image: Image<'T>) =
    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    let z = image.index
    let zOffsets = zNeighborOffsets connectivity
    let pixels = image.toFlatArray()
    // This deliberately uses a small streaming F# flood-fill. If object streaming becomes a bottleneck,
    // compare it with a slab-local implementation that delegates labeling to SimpleITK and keeps this
    // frontier-carry/early-emit logic around the slab labels.
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

let streamConnectedObjects<'T when 'T: equality> connectivity : Stage<Image<'T>, StreamedObject list> =
    let name = "streamConnectedObjects"

    let apply _debug (input: AsyncSeq<Image<'T>>) =
        asyncSeq {
            let mutable state = 1UL, Map.empty<uint64, ActiveObject>
            let enumerator = input.GetAsyncEnumerator()
            let mutable more = true

            while more do
                let! hasNext = enumerator.MoveNextAsync().AsTask() |> Async.AwaitTask

                if hasNext then
                    let image = enumerator.Current
                    let nextState, completed =
                        try
                            processSlice connectivity state image
                        finally
                            image.decRefCount()

                    state <- nextState
                    yield completed
                else
                    more <- false

            let _, active = state
            let finalCompleted =
                active
                |> Map.toList
                |> List.map (snd >> toStreamedObject)

            if finalCompleted.Length > 0 then
                yield finalCompleted
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) id id pipe

let private copyBinaryImage (image: Image<uint8>) =
    let copy = Image<uint8>.ofFlatArray(image.GetSize(), image.toFlatArray())
    copy.index <- image.index
    copy

let private invertedBinaryImage (image: Image<uint8>) =
    let pixels = image.toFlatArray()
    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    let inverted =
        Array.map (fun value -> if value = 0uy then 1uy else 0uy) pixels
        |> fun values -> Image<uint8>.ofFlatArray([ uint width; uint height ], values)
    inverted.index <- image.index
    inverted

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

let private emitBufferedThrough width height cutoff (buffer: SortedDictionary<int, uint8[]>) =
    seq {
        let mutable emitting = true

        while emitting && buffer.Count > 0 do
            let z = buffer.Keys |> Seq.head

            if z <= cutoff then
                let image = Image<uint8>.ofFlatArray([ uint width; uint height ], buffer[z])
                image.index <- z
                buffer.Remove z |> ignore
                yield image
            else
                emitting <- false
    }

let private bufferedBinaryComponentEdit
    name
    (maximumVolume: uint64)
    connectivity
    targetValue
    componentImage
    isProtectedComponent
    : Stage<Image<uint8>, Image<uint8>> =

    let apply _debug (input: AsyncSeq<Image<uint8>>) =
        asyncSeq {
            let buffer = SortedDictionary<int, uint8[]>()
            let mutable state = 1UL, Map.empty<uint64, ActiveObject>
            let mutable width = None
            let mutable height = None
            let mutable firstZ = None
            let enumerator = input.GetAsyncEnumerator()
            let mutable more = true

            while more do
                let! hasNext = enumerator.MoveNextAsync().AsTask() |> Async.AwaitTask

                if hasNext then
                    let image = enumerator.Current
                    let currentWidth = int (image.GetWidth())
                    let currentHeight = int (image.GetHeight())

                    match width, height with
                    | None, None ->
                        width <- Some currentWidth
                        height <- Some currentHeight
                        firstZ <- Some image.index
                    | Some expectedWidth, Some expectedHeight when expectedWidth = currentWidth && expectedHeight = currentHeight -> ()
                    | _ ->
                        invalidOp $"{name} requires a stream with constant x-y slice size."

                    let bufferedIndex = image.index
                    buffer[bufferedIndex] <- image.toFlatArray()

                    let componentInput = componentImage image
                    let nextState, completed =
                        try
                            processSlice connectivity state componentInput
                        finally
                            componentInput.decRefCount()
                            image.decRefCount()

                    state <- nextState

                    for object in completed do
                        if object.Size <= maximumVolume && not (isProtectedComponent currentWidth currentHeight firstZ None object) then
                            paintObjectValue currentWidth targetValue buffer object

                    let _, active = state
                    let cutoff =
                        if active.IsEmpty then
                            bufferedIndex
                        else
                            active
                            |> Map.toSeq
                            |> Seq.map (fun (_, object) -> object.Bounds.MinZ)
                            |> Seq.min
                            |> fun minActiveZ -> minActiveZ - 1

                    for image in emitBufferedThrough currentWidth currentHeight cutoff buffer do
                        yield image
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

            for image in emitBufferedThrough (width |> Option.defaultValue 0) (height |> Option.defaultValue 0) Int32.MaxValue buffer do
                yield image
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) id id pipe

let removeSmallObjects maximumVolume connectivity : Stage<Image<uint8>, Image<uint8>> =
    bufferedBinaryComponentEdit
        "removeSmallObjects"
        maximumVolume
        connectivity
        0uy
        copyBinaryImage
        (fun _ _ _ _ _ -> false)

let fillSmallHoles maximumVolume connectivity : Stage<Image<uint8>, Image<uint8>> =
    let protectExterior width height firstZ lastZ object =
        touchesXYBoundary width height object
        || (firstZ |> Option.exists (fun z -> object.Bounds.MinZ = z))
        || (lastZ |> Option.exists (fun z -> object.Bounds.MaxZ = z))

    bufferedBinaryComponentEdit
        "fillSmallHoles"
        maximumVolume
        connectivity
        1uy
        invertedBinaryImage
        protectExterior

let private paintObjectBatch width height (objects: StreamedObject list) =
    if width = 0u then invalidArg (nameof width) "paintObjects width must be positive."
    if height = 0u then invalidArg (nameof height) "paintObjects height must be positive."

    let width = int width
    let height = int height

    objects
    |> List.collect _.Positions
    |> List.groupBy _.Z
    |> List.sortBy fst
    |> List.map (fun (z, positions) ->
        let pixels = Array.zeroCreate<uint8> (width * height)

        for position in positions do
            if position.X < 0 || position.X >= width || position.Y < 0 || position.Y >= height then
                invalidOp $"Object position ({position.X},{position.Y},{position.Z}) is outside the requested paint image size {width}x{height}."

            pixels[flatIndex2 width position.X position.Y] <- 1uy

        let image = Image<uint8>.ofFlatArray([ uint width; uint height ], pixels)
        image.index <- z
        image)

let paintObjects width height : Stage<StreamedObject list, Image<uint8>> =
    Stage.map "paintObjects" (fun _ -> paintObjectBatch width height) id id
    --> Stage.flatten "paintObjects: flatten"

let private paintObjectCropped (object: StreamedObject) =
    let width = object.Bounds.MaxX - object.Bounds.MinX + 1
    let height = object.Bounds.MaxY - object.Bounds.MinY + 1

    if width <= 0 || height <= 0 then
        []
    else
        object.Positions
        |> List.groupBy _.Z
        |> List.sortBy fst
        |> List.map (fun (z, positions) ->
            let pixels = Array.zeroCreate<uint8> (width * height)

            for position in positions do
                let x = position.X - object.Bounds.MinX
                let y = position.Y - object.Bounds.MinY

                if x < 0 || x >= width || y < 0 || y >= height then
                    invalidOp $"Object position ({position.X},{position.Y},{position.Z}) is outside its bounding box."

                pixels[flatIndex2 width x y] <- 1uy

            let image = Image<uint8>.ofFlatArray([ uint width; uint height ], pixels)
            image.index <- z - object.Bounds.MinZ
            image)

let private paintCroppedObjectBatch (objects: StreamedObject list) =
    objects
    |> List.collect paintObjectCropped

let paintObjectsCropped : Stage<StreamedObject list, Image<uint8>> =
    Stage.map "paintObjectsCropped" (fun _ -> paintCroppedObjectBatch) id id
    --> Stage.flatten "paintObjectsCropped: flatten"

let private measurementsOfObject (object: StreamedObject) =
    { Label = object.Label
      Size = object.Size
      MinX = object.Bounds.MinX
      MaxX = object.Bounds.MaxX
      MinY = object.Bounds.MinY
      MaxY = object.Bounds.MaxY
      MinZ = object.Bounds.MinZ
      MaxZ = object.Bounds.MaxZ
      Width = uint64 (object.Bounds.MaxX - object.Bounds.MinX + 1)
      Height = uint64 (object.Bounds.MaxY - object.Bounds.MinY + 1)
      Depth = uint64 (object.Bounds.MaxZ - object.Bounds.MinZ + 1) }

let measureObjects : Stage<StreamedObject list, ObjectMeasurements list> =
    Stage.map "measureObjects" (fun _ objects -> objects |> List.map measurementsOfObject) id id

let objectSizes : Stage<ObjectMeasurements list, uint64 list> =
    Stage.map "objectSizes" (fun _ measurements -> measurements |> List.map (fun measurement -> measurement.Size)) id id

let private zeroSizeStats =
    { Count = 0UL
      Mean = 0.0
      Variance = 0.0
      Minimum = UInt64.MaxValue
      Maximum = 0UL }

let private addSizeToStats (count, mean, m2, minimum, maximum) (size: uint64) =
    let count' = count + 1UL
    let x = float size
    let delta = x - mean
    let mean' = mean + delta / float count'
    let delta2 = x - mean'
    count', mean', m2 + delta * delta2, min minimum size, max maximum size

let objectSizeStats : Stage<ObjectMeasurements list, ObjectSizeStats> =
    let reducer (_debug: bool) (input: AsyncSeq<ObjectMeasurements list>) =
        async {
            let! count, mean, m2, minimum, maximum =
                input
                |> AsyncSeq.fold
                    (fun state measurements ->
                        measurements
                        |> List.fold (fun state measurement -> addSizeToStats state measurement.Size) state)
                    (0UL, 0.0, 0.0, UInt64.MaxValue, 0UL)

            if count = 0UL then
                return zeroSizeStats
            else
                return
                    { Count = count
                      Mean = mean
                      Variance = if count > 1UL then m2 / float (count - 1UL) else 0.0
                      Minimum = minimum
                      Maximum = maximum }
        }

    Stage.reduce "objectSizeStats" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let histogram binWidth : Stage<uint64 list, Histogram<uint64>> =
    if binWidth = 0UL then invalidArg (nameof binWidth) "histogram binWidth must be positive."

    let addToHistogram histogram (value: uint64) =
        let bin = value / binWidth
        let current = histogram |> Map.tryFind bin |> Option.defaultValue 0UL
        histogram |> Map.add bin (current + 1UL)

    let folder histogram values =
        values |> List.fold addToHistogram histogram

    Stage.fold "histogram" folder Map.empty<uint64, uint64> (fun _ -> 0UL) (fun _ -> 1UL)
    --> Stage.map "histogram: metadata" (fun _ -> Histogram.withFixedWidth binWidth) id id
