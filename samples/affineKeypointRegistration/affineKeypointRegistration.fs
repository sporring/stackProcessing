// Demonstrates painter-generated object images, StackAffineResampler, DoG keypoints,
// point-set registration, and an affine resampling back to the original grid.
open System
open System.IO
open SlimPipeline
open StackProcessing
open TinyLinAlg

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

let private sampleRoot sampleName (args: string[]) =
    if args.Length > 0 then
        let token = args[0]
        if token |> Seq.forall Char.IsDigit then
            "../tmp/{sampleName}{token}"
        else
            token
    else
        "../tmp/{sampleName}"

let private objectSource src (objects: StreamedObject list) : Plan<unit, StreamedObject list> =
    let stage =
        Stage.init
            "synthetic registration objects"
            1u
            (fun _ -> objects)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) src.memAvail 0UL 1UL 1UL src.debug

let private imageSource src (images: Image<float32> list) : Plan<unit, Image<float32>> =
    images |> List.iter _.incRefCount()
    let items = images |> List.toArray
    let stage =
        Stage.init
            "resampled image source"
            (uint items.Length)
            (fun index -> items[index])
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) src.memAvail 0UL 1UL (uint64 items.Length) src.debug

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private identity3 =
    { m00 = 1.0; m01 = 0.0; m02 = 0.0
      m10 = 0.0; m11 = 1.0; m12 = 0.0
      m20 = 0.0; m21 = 0.0; m22 = 1.0 }

let private rotationZ angle =
    let c = Math.Cos angle
    let s = Math.Sin angle
    { m00 = c;  m01 = -s; m02 = 0.0
      m10 = s;  m11 = c;  m12 = 0.0
      m20 = 0.0; m21 = 0.0; m22 = 1.0 }

let private geometry width height depth : StackAffineResampler.ImageGeom =
    { W = int width
      H = int height
      D = int depth
      Origin = v3 0.0 0.0 0.0
      Spacing = v3 1.0 1.0 1.0
      Direction = identity3 }

let private lerp (a: float32) (b: float32) (t: float32) =
    a + (b - a) * t

let private resampleToImages chunkDirectory width height depth transform =
    resampleAffineTrilinearSlices
        chunkDirectory
        ".tiff"
        lerp
        12
        (geometry width height depth)
        (geometry width height depth)
        transform
        0.0f
    |> Seq.map (fun (index, image) ->
        image.index <- index
        image)
    |> Seq.toList

let private strongestKeypoints maxCount (points: CoordinatePoint list) =
    points
    |> List.groupBy (fun point -> int (Math.Round point.X), int (Math.Round point.Y), int (Math.Round point.Z))
    |> List.choose (fun (_, duplicates) -> duplicates |> List.sortByDescending (fun point -> Math.Abs point.Response) |> List.tryHead)
    |> List.sortByDescending (fun point -> Math.Abs point.Response)
    |> List.truncate maxCount

let private detectKeypoints src input =
    src
    |> read<float32> input ".tiff"
    >=> dogKeypoints<float32> 0.8 1.25 4u 0.0005 4u
    |> drainList
    |> List.collect _.Points
    |> strongestKeypoints 40

let private affineError (expected: Affine) (actual: Affine) =
    [ Math.Abs (expected.A.m00 - actual.A.m00)
      Math.Abs (expected.A.m01 - actual.A.m01)
      Math.Abs (expected.A.m02 - actual.A.m02)
      Math.Abs (expected.A.m10 - actual.A.m10)
      Math.Abs (expected.A.m11 - actual.A.m11)
      Math.Abs (expected.A.m12 - actual.A.m12)
      Math.Abs (expected.A.m20 - actual.A.m20)
      Math.Abs (expected.A.m21 - actual.A.m21)
      Math.Abs (expected.A.m22 - actual.A.m22)
      Math.Abs (expected.T.x - actual.T.x)
      Math.Abs (expected.T.y - actual.T.y)
      Math.Abs (expected.T.z - actual.T.z) ]
    |> List.max

let private compareStacks original restored depth =
    let mutable count = 0
    let mutable sumAbs = 0.0
    let mutable maxAbs = 0.0

    for z in 0 .. int depth - 1 do
        let originalImage = Image<float32>.ofFile(Path.Combine(original, sprintf "image_%03d.tiff" z))
        let restoredImage = Image<float32>.ofFile(Path.Combine(restored, sprintf "image_%03d.tiff" z))

        try
            for y in 0 .. int (originalImage.GetHeight()) - 1 do
                for x in 0 .. int (originalImage.GetWidth()) - 1 do
                    let error = Math.Abs (float originalImage[x, y] - float restoredImage[x, y])
                    sumAbs <- sumAbs + error
                    if error > maxAbs then
                        maxAbs <- error
                    count <- count + 1
        finally
            originalImage.decRefCount()
            restoredImage.decRefCount()

    sumAbs / float count, maxAbs

let private printAffine name (affine: Affine) =
    printfn "%s" name
    printfn "  A = [[%.6f %.6f %.6f]; [%.6f %.6f %.6f]; [%.6f %.6f %.6f]]" affine.A.m00 affine.A.m01 affine.A.m02 affine.A.m10 affine.A.m11 affine.A.m12 affine.A.m20 affine.A.m21 affine.A.m22
    printfn "  T = [%.6f %.6f %.6f]" affine.T.x affine.T.y affine.T.z
    printfn "  C = [%.6f %.6f %.6f]" affine.C.x affine.C.y affine.C.z

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args
    let root = sampleRoot "affineKeypointRegistration" args
    let width, height, depth = 64u, 64u, 48u

    let originalStack = Path.Combine(root, "original")
    let originalChunks = Path.Combine(root, "originalChunks")
    let movingStack = Path.Combine(root, "moving")
    let movingChunks = Path.Combine(root, "movingChunks")
    let restoredStack = Path.Combine(root, "restored")

    for path in [ originalStack; originalChunks; movingStack; movingChunks; restoredStack ] do
        clearDirectory path

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

    objectSource src objects
    >=> paintObjects width height
    >=> cast<uint8, float32>
    >=> write originalStack ".tiff"
    >=> writeChunks originalChunks ".tiff" 12u 12u 12u
    >=> ignoreSingles ()
    |> sink

    let knownForward: Affine =
        { A = rotationZ (Math.PI / 36.0)
          T = v3 2.0 -1.5 0.5
          C = v3 ((float width - 1.0) / 2.0) ((float height - 1.0) / 2.0) ((float depth - 1.0) / 2.0) }

    let movingImages = resampleToImages originalChunks width height depth (inverseAffine knownForward)

    imageSource src movingImages
    >=> write movingStack ".tiff"
    >=> writeChunks movingChunks ".tiff" 12u 12u 12u
    >=> ignoreSingles ()
    |> sink

    let fixedKeypoints = detectKeypoints src originalStack
    let movingKeypoints = detectKeypoints src movingStack

    printfn "Detected %d original keypoints and %d transformed keypoints." fixedKeypoints.Length movingKeypoints.Length

    if fixedKeypoints.Length < 4 || movingKeypoints.Length < 4 then
        failwith "The demonstration needs at least four keypoints in each image to estimate an affine transform."

    let registration =
        affineRegistration
            { defaultAffineRegistrationOptions with
                MaxIterations = 80
                InitialLinearStep = 0.02
                InitialTranslationStep = 1.0
                MinStep = 0.001 }
            fixedKeypoints
            movingKeypoints

    printAffine "Known original -> transformed affine:" knownForward
    printAffine "Estimated original -> transformed affine:" registration.InverseTransform
    printfn "Registration EMD distance: %.6f after %d iterations." registration.Distance registration.Iterations
    printfn "Maximum affine parameter error: %.6f" (affineError knownForward registration.InverseTransform)

    let restoredImages = resampleToImages movingChunks width height depth registration.InverseTransform

    imageSource src restoredImages
    >=> write restoredStack ".tiff"
    >=> ignoreSingles ()
    |> sink

    let meanAbs, maxAbs = compareStacks originalStack restoredStack depth
    printfn "Forward/backward resampling mean absolute error: %.8f" meanAbs
    printfn "Forward/backward resampling maximum absolute error: %.8f" maxAbs
    printfn "Wrote original, transformed, and restored stacks under %s" (Path.GetFullPath root)

    movingImages |> List.iter _.decRefCount()
    restoredImages |> List.iter _.decRefCount()
    0
