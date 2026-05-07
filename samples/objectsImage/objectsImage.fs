// Creates a non-trivial binary 3D object image by painting streamed object coordinates.
open System
open System.IO
open SlimPipeline
open StackProcessing

type SyntheticObject =
    | Sphere of cx: float * cy: float * cz: float * radius: float
    | Ellipsoid of cx: float * cy: float * cz: float * rx: float * ry: float * rz: float
    | Box of minX: int * maxX: int * minY: int * maxY: int * minZ: int * maxZ: int

let private insideObject x y z object =
    match object with
    | Sphere(cx, cy, cz, radius) ->
        let dx = (float x - cx) / radius
        let dy = (float y - cy) / radius
        let dz = (float z - cz) / radius
        dx * dx + dy * dy + dz * dz <= 1.0
    | Ellipsoid(cx, cy, cz, rx, ry, rz) ->
        let dx = (float x - cx) / rx
        let dy = (float y - cy) / ry
        let dz = (float z - cz) / rz
        dx * dx + dy * dy + dz * dz <= 1.0
    | Box(minX, maxX, minY, maxY, minZ, maxZ) ->
        x >= minX && x <= maxX && y >= minY && y <= maxY && z >= minZ && z <= maxZ

let private positions width height depth object : Position3D<int> list =
    [ for z in 0 .. depth - 1 do
          for y in 0 .. height - 1 do
              for x in 0 .. width - 1 do
                  if insideObject x y z object then
                      let position: Position3D<int> = { X = x; Y = y; Z = z }
                      position ]

let private boundsOf (positions: Position3D<int> list) : ObjectBounds =
    { MinX = positions |> List.map _.X |> List.min
      MaxX = positions |> List.map _.X |> List.max
      MinY = positions |> List.map _.Y |> List.min
      MaxY = positions |> List.map _.Y |> List.max
      MinZ = positions |> List.map _.Z |> List.min
      MaxZ = positions |> List.map _.Z |> List.max }

let private streamedObject width height depth label object : StreamedObject =
    let positions = positions width height depth object
    { Label = label
      Positions = positions
      Bounds = boundsOf positions
      Size = uint64 positions.Length }

let private objectSource availableMemory (objects: StreamedObject list) : SlimPipeline.Plan<unit, StreamedObject list> =
    let stage =
        Stage.init
            "synthetic streamed objects"
            1u
            (fun _ -> objects)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) availableMemory 0UL 1UL 1UL false

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL

    let width, height, depth, output =
        if args.Length > 0 then
            64u, 64u, 48u, args[0]
        else
            64u, 64u, 48u, "../objectsImage"

    let widthI, heightI, depthI = int width, int height, int depth

    let shapes =
        [ Sphere(14.0, 16.0, 10.0, 6.0)
          Sphere(43.0, 17.0, 17.0, 9.0)
          Ellipsoid(22.0, 43.0, 27.0, 8.0, 5.0, 11.0)
          Box(42, 55, 39, 52, 31, 41)
          Ellipsoid(15.0, 21.0, 38.0, 4.0, 10.0, 5.0)
          Box(60, 60, 60, 60, 0, depthI - 1) ]

    let objects =
        shapes
        |> List.mapi (fun index shape -> streamedObject widthI heightI depthI (uint64 (index + 1)) shape)

    Directory.CreateDirectory(output) |> ignore

    for file in Directory.EnumerateFiles(output, "*.tiff") do
        File.Delete(file)

    objectSource availableMemory objects
    >=> paintObjects width height
    >=> write output ".tiff"
    |> sink

    printfn "Painted %d objects into %s" objects.Length (Path.GetFullPath output)
    0
