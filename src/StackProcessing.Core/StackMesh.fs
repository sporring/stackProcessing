module StackMesh

open System
open System.Globalization
open System.IO
open FSharp.Control
open SlimPipeline
open StackCore

type Point3D =
    { X: float
      Y: float
      Z: float }

type Triangle =
    { A: Point3D
      B: Point3D
      C: Point3D }

type TriangleSet =
    { Triangles: Triangle list }

module TriangleSet =
    let empty = { Triangles = [] }

let private point x y z =
    { X = x; Y = y; Z = z }

let private interpolate iso (p1: Point3D) v1 (p2: Point3D) v2 =
    let denominator = v2 - v1
    let t =
        if abs denominator < 1.0e-12 then
            0.5
        else
            (iso - v1) / denominator
            |> max 0.0
            |> min 1.0
    { X = p1.X + t * (p2.X - p1.X)
      Y = p1.Y + t * (p2.Y - p1.Y)
      Z = p1.Z + t * (p2.Z - p1.Z) }

let private triangulateTetra iso (vertices: (Point3D * float) array) =
    let inside =
        vertices
        |> Array.mapi (fun index (_, value) -> index, value >= iso)

    let insideIndices =
        inside
        |> Array.choose (fun (index, isInside) -> if isInside then Some index else None)

    let outsideIndices =
        inside
        |> Array.choose (fun (index, isInside) -> if isInside then None else Some index)

    let edge a b =
        let p1, v1 = vertices[a]
        let p2, v2 = vertices[b]
        interpolate iso p1 v1 p2 v2

    match insideIndices.Length with
    | 0
    | 4 -> []
    | 1 ->
        let i = insideIndices[0]
        let p0 = edge i outsideIndices[0]
        let p1 = edge i outsideIndices[1]
        let p2 = edge i outsideIndices[2]
        [ { A = p0; B = p1; C = p2 } ]
    | 3 ->
        let o = outsideIndices[0]
        let p0 = edge o insideIndices[0]
        let p1 = edge o insideIndices[1]
        let p2 = edge o insideIndices[2]
        [ { A = p0; B = p2; C = p1 } ]
    | 2 ->
        let i0 = insideIndices[0]
        let i1 = insideIndices[1]
        let o0 = outsideIndices[0]
        let o1 = outsideIndices[1]
        let p00 = edge i0 o0
        let p01 = edge i0 o1
        let p10 = edge i1 o0
        let p11 = edge i1 o1
        [ { A = p00; B = p10; C = p11 }
          { A = p00; B = p11; C = p01 } ]
    | _ -> []

let private cubeTetrahedra =
    [| [| 0; 5; 1; 6 |]
       [| 0; 1; 2; 6 |]
       [| 0; 2; 3; 6 |]
       [| 0; 3; 7; 6 |]
       [| 0; 7; 4; 6 |]
       [| 0; 4; 5; 6 |] |]

let private voxelValue (pixels: 'T[,]) x y =
    pixels[x, y] |> Convert.ToDouble

let private voxelValueOrBackground (pixels: 'T[,]) width height x y =
    if x < 0 || y < 0 || x >= width || y >= height then
        0.0
    else
        voxelValue pixels x y

let private meshBetweenSlices<'T when 'T: equality> surfaceValue (lower: Image<'T>) (upper: Image<'T>) =
    let width = min (int (lower.GetWidth())) (int (upper.GetWidth()))
    let height = min (int (lower.GetHeight())) (int (upper.GetHeight()))
    let lowerZ = float lower.index
    let upperZ = float upper.index
    let lowerPixels = lower.toArray2D()
    let upperPixels = upper.toArray2D()

    let triangles = ResizeArray<Triangle>()

    if width >= 1 && height >= 1 then
        for y in -1 .. height - 1 do
            for x in -1 .. width - 1 do
                let positions =
                    [| point (float x) (float y) lowerZ
                       point (float (x + 1)) (float y) lowerZ
                       point (float (x + 1)) (float (y + 1)) lowerZ
                       point (float x) (float (y + 1)) lowerZ
                       point (float x) (float y) upperZ
                       point (float (x + 1)) (float y) upperZ
                       point (float (x + 1)) (float (y + 1)) upperZ
                       point (float x) (float (y + 1)) upperZ |]

                let values =
                    [| voxelValueOrBackground lowerPixels width height x y
                       voxelValueOrBackground lowerPixels width height (x + 1) y
                       voxelValueOrBackground lowerPixels width height (x + 1) (y + 1)
                       voxelValueOrBackground lowerPixels width height x (y + 1)
                       voxelValueOrBackground upperPixels width height x y
                       voxelValueOrBackground upperPixels width height (x + 1) y
                       voxelValueOrBackground upperPixels width height (x + 1) (y + 1)
                       voxelValueOrBackground upperPixels width height x (y + 1) |]

                let cube = Array.init 8 (fun index -> positions[index], values[index])

                cubeTetrahedra
                |> Array.iter (fun tetra ->
                    tetra
                    |> Array.map (fun index -> cube[index])
                    |> triangulateTetra surfaceValue
                    |> List.iter triangles.Add)

    { Triangles = triangles |> Seq.toList }

let marchingCubes<'T when 'T: equality> surfaceValue : Stage<Image<'T>, TriangleSet> =
    let releaseConsumed (window: Window<Image<'T>>) =
        window.Items
        |> List.take (min (int window.ReleaseCount) window.Items.Length)
        |> List.iter (fun image -> image.decRefCount())

    let mapper (_debug: bool) (window: Window<Image<'T>>) =
        try
            match window.Items with
            | lower :: upper :: _ -> meshBetweenSlices surfaceValue lower upper
            | _ -> TriangleSet.empty
        finally
            releaseConsumed window

    (StackCore.window 2u 1u 1u)
    --> StackCore.mapWindow "marchingCubes" mapper id id

let private validateUnit name value =
    if value <= 0.0 then
        invalidArg name $"{name} must be positive."

let private scalePoint xUnit yUnit zUnit (point: Point3D) =
    { X = point.X * xUnit
      Y = point.Y * yUnit
      Z = point.Z * zUnit }

let private triangleArea xUnit yUnit zUnit (triangle: Triangle) =
    let a = scalePoint xUnit yUnit zUnit triangle.A
    let b = scalePoint xUnit yUnit zUnit triangle.B
    let c = scalePoint xUnit yUnit zUnit triangle.C
    let ux = b.X - a.X
    let uy = b.Y - a.Y
    let uz = b.Z - a.Z
    let vx = c.X - a.X
    let vy = c.Y - a.Y
    let vz = c.Z - a.Z
    let cx = uy * vz - uz * vy
    let cy = uz * vx - ux * vz
    let cz = ux * vy - uy * vx
    0.5 * Math.Sqrt(cx * cx + cy * cy + cz * cz)

let surfaceArea xUnit yUnit zUnit : Stage<TriangleSet, float> =
    validateUnit (nameof xUnit) xUnit
    validateUnit (nameof yUnit) yUnit
    validateUnit (nameof zUnit) zUnit

    let folder area chunk =
        chunk.Triangles
        |> List.sumBy (triangleArea xUnit yUnit zUnit)
        |> (+) area

    Stage.fold "surfaceArea" folder 0.0 id (fun _ -> 1UL)

type MeshFormat =
    | Obj
    | Stl

let private inferFormat (outputPath: string) (format: string) =
    let normalized = format.Trim().TrimStart('.').ToLowerInvariant()
    let fromExtension () =
        Path.GetExtension(outputPath).TrimStart('.').ToLowerInvariant()
    match if String.IsNullOrWhiteSpace normalized || normalized = "auto" then fromExtension() else normalized with
    | "obj" -> Obj
    | "stl" -> Stl
    | "ply" -> failwith "PLY mesh output is paused because ASCII PLY requires vertex and face counts before writing. Use OBJ or STL for streaming mesh output."
    | other -> failwith $"Unsupported mesh format '{other}'. Use obj or stl."

let private meshSuffix meshFormat =
    match meshFormat with
    | Obj -> ".obj"
    | Stl -> ".stl"

let meshFilePath (outputPath: string) (format: string) =
    if Path.HasExtension outputPath then
        outputPath
    else
        outputPath + meshSuffix (inferFormat outputPath format)

let private f (value: float) =
    value.ToString("R", CultureInfo.InvariantCulture)

let private writeObjTriangle (writer: StreamWriter) nextVertex (triangle: Triangle) =
    [ triangle.A; triangle.B; triangle.C ]
    |> List.iter (fun p -> writer.WriteLine($"v {f p.X} {f p.Y} {f p.Z}"))
    writer.WriteLine($"f {nextVertex} {nextVertex + 1} {nextVertex + 2}")
    nextVertex + 3

let private writeStlTriangle (writer: StreamWriter) (triangle: Triangle) =
    writer.WriteLine("  facet normal 0 0 0")
    writer.WriteLine("    outer loop")
    [ triangle.A; triangle.B; triangle.C ]
    |> List.iter (fun p -> writer.WriteLine($"      vertex {f p.X} {f p.Y} {f p.Z}"))
    writer.WriteLine("    endloop")
    writer.WriteLine("  endfacet")

let private writeObjStreaming outputPath (input: AsyncSeq<TriangleSet>) =
    async {
        use writer = new StreamWriter(outputPath, false)
        let mutable nextVertex = 1
        do!
            input
            |> AsyncSeq.iterAsync (fun triangleSet ->
                async {
                    for triangle in triangleSet.Triangles do
                        nextVertex <- writeObjTriangle writer nextVertex triangle
                })
    }

let private writeStlStreaming outputPath (input: AsyncSeq<TriangleSet>) =
    async {
        use writer = new StreamWriter(outputPath, false)
        writer.WriteLine("solid stackProcessing")
        do!
            input
            |> AsyncSeq.iterAsync (fun triangleSet ->
                async {
                    for triangle in triangleSet.Triangles do
                        writeStlTriangle writer triangle
                })
        writer.WriteLine("endsolid stackProcessing")
    }

let writeMesh (outputPath: string) (format: string) : Stage<TriangleSet, unit> =
    let reducer (debug: bool) (input: AsyncSeq<TriangleSet>) =
        async {
            let meshFormat = inferFormat outputPath format
            let outputPath = meshFilePath outputPath format

            let directory = Path.GetDirectoryName(outputPath)
            if not (String.IsNullOrWhiteSpace directory) then
                Directory.CreateDirectory(directory) |> ignore

            if debug then
                printfn $"[writeMesh] Writing {outputPath}"

            match meshFormat with
            | Obj -> do! writeObjStreaming outputPath input
            | Stl -> do! writeStlStreaming outputPath input
        }

    Stage.reduce $"writeMesh \"{outputPath}\"" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)
