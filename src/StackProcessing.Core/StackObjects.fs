module StackObjects

open System
open System.Collections.Generic
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

let private pointBounds x y z =
    { MinX = x
      MaxX = x
      MinY = y
      MaxY = y
      MinZ = z
      MaxZ = z }

let private addPointToBounds bounds x y z =
    { MinX = min bounds.MinX x
      MaxX = max bounds.MaxX x
      MinY = min bounds.MinY y
      MaxY = max bounds.MaxY y
      MinZ = min bounds.MinZ z
      MaxZ = max bounds.MaxZ z }

let private mergeBounds left right =
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
    let visited = Array2D.create width height false
    let offsets = xyNeighborOffsets connectivity
    let components = ResizeArray<SliceComponent>()
    let queue = Queue<int * int>()
    let mutable nextId = 0

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            if isForeground x y && not visited[x, y] then
                visited[x, y] <- true
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
                        if nx >= 0 && nx < width && ny >= 0 && ny < height && isForeground nx ny && not visited[nx, ny] then
                            visited[nx, ny] <- true
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
    // This deliberately uses a small streaming F# flood-fill. If object streaming becomes a bottleneck,
    // compare it with a slab-local implementation that delegates labeling to SimpleITK and keeps this
    // frontier-carry/early-emit logic around the slab labels.
    let components =
        connectedComponents2D connectivity z width height (fun x y -> foreground image[x, y])

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
        let pixels = Array2D.zeroCreate<uint8> width height

        for position in positions do
            if position.X < 0 || position.X >= width || position.Y < 0 || position.Y >= height then
                invalidOp $"Object position ({position.X},{position.Y},{position.Z}) is outside the requested paint image size {width}x{height}."

            pixels[position.X, position.Y] <- 1uy

        let image = Image<uint8>.ofArray2D pixels
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
            let pixels = Array2D.zeroCreate<uint8> width height

            for position in positions do
                let x = position.X - object.Bounds.MinX
                let y = position.Y - object.Bounds.MinY

                if x < 0 || x >= width || y < 0 || y >= height then
                    invalidOp $"Object position ({position.X},{position.Y},{position.Z}) is outside its bounding box."

                pixels[x, y] <- 1uy

            let image = Image<uint8>.ofArray2D pixels
            image.index <- z - object.Bounds.MinZ
            image)

let private paintCroppedObjectBatch (objects: StreamedObject list) =
    objects
    |> List.collect paintObjectCropped

let paintObjectsCropped : Stage<StreamedObject list, Image<uint8>> =
    Stage.map "paintObjectsCropped" (fun _ -> paintCroppedObjectBatch) id id
    --> Stage.flatten "paintObjectsCropped: flatten"
