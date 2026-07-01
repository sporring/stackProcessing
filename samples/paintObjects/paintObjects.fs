// Paint a binary object fixture with enclosed holes for fillSmallHoles.
open System
open StackProcessing
open StackObjects
open StackPoints

let private inRange low high value =
    value >= low && value <= high

let private makeFixtureObjects width height depth : StreamedObject list =
    let makeObject label predicate =
        let positions: Position3D<int> list =
            [ for z in 0 .. depth - 1 do
                  for y in 0 .. height - 1 do
                      for x in 0 .. width - 1 do
                          if predicate x y z then
                              ({ X = x; Y = y; Z = z }: Position3D<int>) ]

        let bounds: ObjectBounds =
            positions
            |> List.fold
                (fun bounds position ->
                    { MinX = min bounds.MinX position.X
                      MaxX = max bounds.MaxX position.X
                      MinY = min bounds.MinY position.Y
                      MaxY = max bounds.MaxY position.Y
                      MinZ = min bounds.MinZ position.Z
                      MaxZ = max bounds.MaxZ position.Z })
                { MinX = Int32.MaxValue
                  MaxX = Int32.MinValue
                  MinY = Int32.MaxValue
                  MaxY = Int32.MinValue
                  MinZ = Int32.MaxValue
                  MaxZ = Int32.MinValue }

        { Label = label
          Positions = positions
          Bounds = bounds
          Size = uint64 positions.Length }

    [ makeObject 1UL (fun x y z -> inRange 6 8 x && inRange 6 8 y && inRange 6 8 z)
      makeObject 2UL (fun x y z -> x = 47 && y = 7 && inRange 4 27 z)
      makeObject 3UL (fun x y z -> inRange 30 43 x && inRange 39 52 y && inRange 20 40 z)
      makeObject 4UL (fun x y z -> x = 10 && y = 23 && inRange 4 43 z) ]

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/paintObjects"

    let width = 64u
    let height = 64u
    let depth = 64u
    let objects = makeFixtureObjects (int width) (int height) (int depth)

    // Make a 3d image from a stream of objects. Used to produce ../data/objects.
    src |> objectSource<uint8> objects
    >=> paintObjects width height depth None (Some 255.0)
    >=> write output ".tiff"
    |> sink

    0
