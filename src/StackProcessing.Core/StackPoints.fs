module StackPoints

open System
open System.Globalization
open System.IO
open System.Text
open FSharp.Control
open Image.InternalHelpers
open SlimPipeline
open StackCore
open TinyLinAlg

type Position3D<'T> =
    { X: 'T
      Y: 'T
      Z: 'T }

type CoordinatePoint =
    { X: float
      Y: float
      Z: float
      Scale: float
      Response: float }

type PointSet =
    { Points: CoordinatePoint list }

module PointSet =
    let empty = { Points = [] }

type VectorizedMatrix =
    { Rows: uint
      Columns: uint
      Values: float list }

let private invariant = CultureInfo.InvariantCulture

let private f (value: float) =
    value.ToString("R", invariant)

let private invariantText (value: 'T) =
    match box value with
    | null -> ""
    | :? IFormattable as formattable -> formattable.ToString(null, invariant)
    | value -> value.ToString()

let private escapeCsv (text: string) =
    if text.Contains(",", StringComparison.Ordinal)
       || text.Contains("\"", StringComparison.Ordinal)
       || text.Contains("\n", StringComparison.Ordinal)
       || text.Contains("\r", StringComparison.Ordinal) then
        "\"" + text.Replace("\"", "\"\"") + "\""
    else
        text

let private parseFloat (text: string) =
    Double.Parse(text.Trim(), NumberStyles.Float ||| NumberStyles.AllowThousands, invariant)

let private outputPathWithSuffix (output: string) (suffix: string) =
    let suffix = if String.IsNullOrWhiteSpace suffix then ".csv" else suffix
    if not (suffix.Equals(".csv", StringComparison.OrdinalIgnoreCase)) then
        failwith $"Unsupported point/matrix output format '{suffix}'. Currently supported: .csv."

    if output.EndsWith(suffix, StringComparison.OrdinalIgnoreCase) then
        output
    else
        output + suffix

let private splitCsvLine (line: string) =
    line.Split(',')
    |> Array.map _.Trim()

let readPointSet (path: string) (pl: Plan<unit, unit>) : Plan<unit, PointSet> =
    let mapper (_idx: int) =
        if pl.debug then
            printfn $"[readPointSet] Reading {path}"

        let lines =
            File.ReadLines(path)
            |> Seq.filter (fun line -> not (String.IsNullOrWhiteSpace line))
            |> Seq.toArray

        let start =
            if lines.Length > 0 && lines[0].ToLowerInvariant().Contains("x") then 1 else 0

        let points =
            lines
            |> Seq.skip start
            |> Seq.map (fun line ->
                let columns = splitCsvLine line
                if columns.Length < 3 then
                    failwith $"Point-set CSV rows must contain at least x,y,z columns, but got '{line}'."

                { X = parseFloat columns[0]
                  Y = parseFloat columns[1]
                  Z = parseFloat columns[2]
                  Scale = if columns.Length > 3 then parseFloat columns[3] else Double.NaN
                  Response = if columns.Length > 4 then parseFloat columns[4] else Double.NaN })
            |> Seq.toList

        { Points = points }

    let stage =
        Stage.init "readPointSet" 1u mapper (ProfileTransition.create Unit Streaming) (fun _ -> 0UL) id
        |> Some

    Plan.createWithOptimizer stage pl.memAvail 0UL 0UL 1UL pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl

let writePointSet (output: string) (suffix: string) : Stage<PointSet, unit> =
    let reducer (debug: bool) (input: AsyncSeq<PointSet>) =
        async {
            let outputPath = outputPathWithSuffix output suffix
            if debug then
                printfn $"[writePointSet] Writing {outputPath}"

            let directory = Path.GetDirectoryName(outputPath)
            if not (String.IsNullOrWhiteSpace directory) then
                Directory.CreateDirectory(directory) |> ignore

            use writer = new StreamWriter(outputPath, false)
            writer.WriteLine("x,y,z,scale,response")

            do!
                input
                |> AsyncSeq.iterAsync (fun chunk ->
                    async {
                        for point in chunk.Points do
                            writer.WriteLine($"{f point.X},{f point.Y},{f point.Z},{f point.Scale},{f point.Response}")
                    })
        }

    Stage.reduce $"writePointSet \"{output}\" \"{suffix}\"" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let writeCSVPointSet (output: string) : Stage<PointSet, unit> =
    writePointSet output ".csv"

let vectorizeMatrix (matrix: float[,]) =
    let rows = matrix.GetLength(0)
    let columns = matrix.GetLength(1)
    { Rows = uint rows
      Columns = uint columns
      Values =
          [ for row in 0 .. rows - 1 do
                for column in 0 .. columns - 1 do
                    matrix[row, column] ] }

let unvectorizeMatrix (matrix: VectorizedMatrix) =
    let rows = int matrix.Rows
    let columns = int matrix.Columns
    if matrix.Values.Length <> rows * columns then
        invalidArg "matrix" $"Vectorized matrix has {matrix.Values.Length} values, but {rows}x{columns} requires {rows * columns}."

    let result = Array2D.zeroCreate<float> rows columns
    matrix.Values
    |> List.iteri (fun index value ->
        let row = index / columns
        let column = index % columns
        result[row, column] <- value)
    result

let writeMatrix (output: string) (suffix: string) : Stage<VectorizedMatrix, unit> =
    let reducer (debug: bool) (input: AsyncSeq<VectorizedMatrix>) =
        async {
            let! matrices = input |> AsyncSeq.toListAsync

            let writeOne (outputPath: string) matrix =
                if debug then
                    printfn $"[writeMatrix] Writing {outputPath}"

                let directory = Path.GetDirectoryName(outputPath)
                if not (String.IsNullOrWhiteSpace directory) then
                    Directory.CreateDirectory(directory) |> ignore

                let values = unvectorizeMatrix matrix
                use writer = new StreamWriter(outputPath, false, Encoding.UTF8)
                for row in 0 .. values.GetLength(0) - 1 do
                    let line =
                        [ for column in 0 .. values.GetLength(1) - 1 ->
                            f values[row, column] ]
                        |> String.concat ","
                    writer.WriteLine(line)

            match matrices with
            | [] -> invalidOp "writeMatrix expected at least one vectorized matrix, but the stream was empty."
            | [ matrix ] ->
                writeOne (outputPathWithSuffix output suffix) matrix
            | _ ->
                let firstPath = outputPathWithSuffix output suffix
                let directory = Path.GetDirectoryName(firstPath)
                let stem = Path.GetFileNameWithoutExtension(firstPath)
                let extension = Path.GetExtension(firstPath)

                matrices
                |> List.iteri (fun index matrix ->
                    let fileName = sprintf "%s_%03d%s" stem index extension
                    let outputPath =
                        if String.IsNullOrWhiteSpace directory then
                            fileName
                        else
                            Path.Combine(directory, fileName)
                    writeOne outputPath matrix)
        }

    Stage.reduce $"writeMatrix \"{output}\" \"{suffix}\"" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let writeCSVMatrix (output: string) : Stage<VectorizedMatrix, unit> =
    writeMatrix output ".csv"

let writeCSVHistogram<'T when 'T: comparison> (output: string) : Stage<Map<'T, uint64>, unit> =
    let reducer (debug: bool) (input: AsyncSeq<Map<'T, uint64>>) =
        async {
            let! histogram =
                input
                |> AsyncSeq.foldAsync
                    (fun state partial ->
                        async {
                            return
                                partial
                                |> Map.fold
                                    (fun histogram key count ->
                                        let current = histogram |> Map.tryFind key |> Option.defaultValue 0UL
                                        histogram |> Map.add key (current + count))
                                    state
                        })
                    Map.empty

            let outputPath = outputPathWithSuffix output ".csv"
            if debug then
                printfn $"[writeHistogram] Writing {outputPath}"

            let directory = Path.GetDirectoryName(outputPath)
            if not (String.IsNullOrWhiteSpace directory) then
                Directory.CreateDirectory(directory) |> ignore

            use writer = new StreamWriter(outputPath, false, Encoding.UTF8)
            writer.WriteLine("key,count")

            for KeyValue(key, count) in histogram do
                writer.WriteLine($"{escapeCsv (invariantText key)},{count.ToString(invariant)}")
        }

    Stage.reduce $"writeCSVHistogram \"{output}\"" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let private validateUnit name value =
    if value <= 0.0 then
        invalidArg name $"{name} must be positive."

let private scaledDistance xUnit yUnit zUnit (a: CoordinatePoint) (b: CoordinatePoint) =
    let dx = (a.X - b.X) * xUnit
    let dy = (a.Y - b.Y) * yUnit
    let dz = (a.Z - b.Z) * zUnit
    Math.Sqrt(dx * dx + dy * dy + dz * dz)

let pointPairDistances xUnit yUnit zUnit : Stage<PointSet, VectorizedMatrix> =
    validateUnit (nameof xUnit) xUnit
    validateUnit (nameof yUnit) yUnit
    validateUnit (nameof zUnit) zUnit

    let reducer (_debug: bool) (input: AsyncSeq<PointSet>) =
        async {
            let! points =
                input
                |> AsyncSeq.foldAsync
                    (fun points pointSet -> async { return (List.rev pointSet.Points) @ points })
                    []

            let points = points |> List.rev |> List.toArray
            let count = points.Length
            let distances = Array2D.zeroCreate<float> count count
            for row in 0 .. count - 1 do
                for column in row + 1 .. count - 1 do
                    let distance = scaledDistance xUnit yUnit zUnit points[row] points[column]
                    distances[row, column] <- distance
                    distances[column, row] <- distance

            return vectorizeMatrix distances
        }

    Stage.reduce "pointPairDistances" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let selectGroupedValueOutput (groupSize: uint) (part: uint) : Stage<'T, 'T> =
    if groupSize = 0u then
        invalidArg "groupSize" "selectGroupedValueOutput: groupSize must be positive."
    if part >= groupSize then
        invalidArg "part" $"selectGroupedValueOutput: part must be smaller than groupSize ({groupSize})."

    Stage.mapi
        "selectGroupedValueOutput"
        (fun _ index value ->
            if uint (index % int64 groupSize) = part then [ value ] else [])
        id
        (fun values -> (values + uint64 groupSize - 1UL) / uint64 groupSize)
    --> StackCore.flattenList ()

let private imageToVolume (images: Image<'T> list) =
    match images with
    | [] -> Array3D.zeroCreate<double> 0 0 0
    | first :: _ ->
        let width = int (first.GetWidth())
        let height = int (first.GetHeight())
        let depth = images.Length
        let volume = Array3D.zeroCreate<double> width height depth

        images
        |> List.iteri (fun z image ->
            let pixels = image.toFlatArray()
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    volume[x, y, z] <- Convert.ToDouble(pixels[flatIndex2 width x y], invariant))

        volume

let private clamp lo hi value =
    if value < lo then lo elif value > hi then hi else value

let private gaussianBlur3D sigma (source: double[,,]) =
    let image = Image<float>.ofArray3D(source, "gaussianBlur3D.input")
    try
        let smoothed = ImageFunctions.smoothingRecursiveGaussian sigma image
        try
            smoothed.toArray3D()
        finally
            smoothed.decRefCount()
    finally
        image.decRefCount()

let private central (source: double[,,]) x y z axis =
    let width = source.GetLength(0)
    let height = source.GetLength(1)
    let depth = source.GetLength(2)

    match axis with
    | 0 -> (source[clamp 0 (width - 1) (x + 1), y, z] - source[clamp 0 (width - 1) (x - 1), y, z]) / 2.0
    | 1 -> (source[x, clamp 0 (height - 1) (y + 1), z] - source[x, clamp 0 (height - 1) (y - 1), z]) / 2.0
    | _ -> (source[x, y, clamp 0 (depth - 1) (z + 1)] - source[x, y, clamp 0 (depth - 1) (z - 1)]) / 2.0

let private second (source: double[,,]) x y z axis =
    let width = source.GetLength(0)
    let height = source.GetLength(1)
    let depth = source.GetLength(2)

    match axis with
    | 0 -> source[clamp 0 (width - 1) (x + 1), y, z] - 2.0 * source[x, y, z] + source[clamp 0 (width - 1) (x - 1), y, z]
    | 1 -> source[x, clamp 0 (height - 1) (y + 1), z] - 2.0 * source[x, y, z] + source[x, clamp 0 (height - 1) (y - 1), z]
    | _ -> source[x, y, clamp 0 (depth - 1) (z + 1)] - 2.0 * source[x, y, z] + source[x, y, clamp 0 (depth - 1) (z - 1)]

let private mixed (source: double[,,]) x y z axisA axisB =
    let shifted da db =
        let sx = if axisA = 0 then x + da elif axisB = 0 then x + db else x
        let sy = if axisA = 1 then y + da elif axisB = 1 then y + db else y
        let sz = if axisA = 2 then z + da elif axisB = 2 then z + db else z
        source[clamp 0 (source.GetLength(0) - 1) sx, clamp 0 (source.GetLength(1) - 1) sy, clamp 0 (source.GetLength(2) - 1) sz]

    (shifted 1 1 - shifted 1 -1 - shifted -1 1 + shifted -1 -1) / 4.0

let private isStrictLocalMaximum (response: double[,,]) x y z =
    let value = response[x, y, z]
    let mutable isMaximum = true

    for dz in -1 .. 1 do
        for dy in -1 .. 1 do
            for dx in -1 .. 1 do
                if dx <> 0 || dy <> 0 || dz <> 0 then
                    if value <= response[x + dx, y + dy, z + dz] then
                        isMaximum <- false

    isMaximum

let private keypointsFromResponse threshold scale (images: Image<'T> list) (window: Window<Image<'T>>) (response: double[,,]) =
    let width = response.GetLength(0)
    let height = response.GetLength(1)
    let depth = response.GetLength(2)
    let targetZ =
        Window.emitItems window
        |> List.map _.index
        |> Set.ofList

    let points = ResizeArray<CoordinatePoint>()

    if width >= 3 && height >= 3 && depth >= 3 then
        for z in 1 .. depth - 2 do
            let sourceZ = images[z].index
            if targetZ.Contains sourceZ then
                for y in 1 .. height - 2 do
                    for x in 1 .. width - 2 do
                        let value = response[x, y, z]
                        if value >= threshold && isStrictLocalMaximum response x y z then
                            points.Add
                                { X = float x
                                  Y = float y
                                  Z = float sourceZ
                                  Scale = scale
                                  Response = value }

    { Points = points |> Seq.toList }

let private releaseConsumed (window: Window<Image<'T>>) =
    window.Items
    |> List.take (min (int window.ReleaseCount) window.Items.Length)
    |> List.iter (fun image -> image.decRefCount())

let private localKeypointStage<'T when 'T: equality> name sigma stride response : Stage<Image<'T>, PointSet> =
    if sigma <= 0.0 then invalidArg "sigma" $"{name} sigma must be positive."
    if stride = 0u then invalidArg "stride" $"{name} stride must be positive."

    let pad = uint (ceil (3.0 * sigma)) + 2u
    let windowSize = stride + 2u * pad

    let mapper (_debug: bool) (window: Window<Image<'T>>) =
        try
            let images = window.Items
            let volume = imageToVolume images
            response images window volume
        finally
            releaseConsumed window

    (StackCore.window windowSize pad stride)
    --> StackCore.mapWindow name mapper id id

let private dogKeypointsInWindow<'T when 'T: equality>
    sigma0
    scaleFactor
    scaleLevels
    contrastThreshold
    (window: Window<Image<'T>>)
    =

    let scaleLevels = int scaleLevels
    if scaleLevels < 4 then
        invalidArg "scaleLevels" "dogKeypoints needs at least 4 Gaussian scale levels, giving at least 3 Difference-of-Gaussian levels."

    let images = window.Items
    let targetZ =
        Window.emitItems window
        |> List.map _.index
        |> Set.ofList

    let volume = imageToVolume images
    let width = volume.GetLength(0)
    let height = volume.GetLength(1)
    let depth = volume.GetLength(2)

    if width < 3 || height < 3 || depth < 3 then
        PointSet.empty
    else
        let sigmas =
            [| for level in 0 .. scaleLevels - 1 -> sigma0 * Math.Pow(scaleFactor, float level) |]

        let blurred =
            sigmas
            |> Array.map (fun sigma -> gaussianBlur3D sigma volume)

        let dogs =
            [| for level in 0 .. scaleLevels - 2 ->
                let lower = blurred[level]
                let upper = blurred[level + 1]
                let dog = Array3D.zeroCreate<double> width height depth
                for z in 0 .. depth - 1 do
                    for y in 0 .. height - 1 do
                        for x in 0 .. width - 1 do
                            dog[x, y, z] <- upper[x, y, z] - lower[x, y, z]
                dog |]

        let points = ResizeArray<CoordinatePoint>()

        for scaleIndex in 0 .. dogs.Length - 1 do
            let dog = dogs[scaleIndex]
            for z in 1 .. depth - 2 do
                let sourceZ = images[z].index
                if targetZ.Contains sourceZ then
                    for y in 1 .. height - 2 do
                        for x in 1 .. width - 2 do
                            let value = dog[x, y, z]
                            if abs value >= contrastThreshold then
                                let mutable greater = true
                                let mutable less = true

                                let scaleNeighborStart = max 0 (scaleIndex - 1)
                                let scaleNeighborStop = min (dogs.Length - 1) (scaleIndex + 1)

                                for neighborScaleIndex in scaleNeighborStart .. scaleNeighborStop do
                                    let neighborDog = dogs[neighborScaleIndex]
                                    for dz in -1 .. 1 do
                                        for dy in -1 .. 1 do
                                            for dx in -1 .. 1 do
                                                if neighborScaleIndex <> scaleIndex || dz <> 0 || dy <> 0 || dx <> 0 then
                                                    let neighbor = neighborDog[x + dx, y + dy, z + dz]
                                                    if value <= neighbor then greater <- false
                                                    if value >= neighbor then less <- false

                                if greater || less then
                                    points.Add
                                        { X = float x
                                          Y = float y
                                          Z = float sourceZ
                                          Scale = sigmas[scaleIndex]
                                          Response = value }

        { Points = points |> Seq.toList }

let dogKeypoints<'T when 'T: equality>
    (sigma0: float)
    (scaleFactor: float)
    (scaleLevels: uint)
    (contrastThreshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    if sigma0 <= 0.0 then invalidArg "sigma0" "dogKeypoints sigma0 must be positive."
    if scaleFactor <= 1.0 then invalidArg "scaleFactor" "dogKeypoints scaleFactor must be greater than 1."
    if scaleLevels < 4u then invalidArg "scaleLevels" "dogKeypoints needs at least 4 Gaussian scale levels."
    if stride = 0u then invalidArg "stride" "dogKeypoints stride must be positive."

    let maxSigma = sigma0 * Math.Pow(scaleFactor, float (scaleLevels - 1u))
    let pad = uint (ceil (3.0 * maxSigma)) + 1u
    let windowSize = stride + 2u * pad

    let releaseConsumed (window: Window<Image<'T>>) =
        window.Items
        |> List.take (min (int window.ReleaseCount) window.Items.Length)
        |> List.iter (fun image -> image.decRefCount())

    let mapper (_debug: bool) (window: Window<Image<'T>>) =
        try
            dogKeypointsInWindow sigma0 scaleFactor scaleLevels contrastThreshold window
        finally
            releaseConsumed window

    (StackCore.window windowSize pad stride)
    --> StackCore.mapWindow "dogKeypoints" mapper id id

let logBlobKeypoints<'T when 'T: equality>
    (sigma: float)
    (threshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    localKeypointStage<'T> "logBlobKeypoints" sigma stride (fun images window volume ->
        let smoothed = gaussianBlur3D sigma volume
        let width = smoothed.GetLength(0)
        let height = smoothed.GetLength(1)
        let depth = smoothed.GetLength(2)
        let response = Array3D.zeroCreate<double> width height depth

        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    let laplacian =
                        second smoothed x y z 0
                        + second smoothed x y z 1
                        + second smoothed x y z 2
                    response[x, y, z] <- sigma * sigma * abs laplacian

        keypointsFromResponse threshold sigma images window response)

let private hessianResponse (responseKind: string) sigma (smoothed: double[,,]) =
    let width = smoothed.GetLength(0)
    let height = smoothed.GetLength(1)
    let depth = smoothed.GetLength(2)
    let response = Array3D.zeroCreate<double> width height depth
    let kind = responseKind.Trim().ToLowerInvariant()

    for z in 0 .. depth - 1 do
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                let hxx = second smoothed x y z 0
                let hyy = second smoothed x y z 1
                let hzz = second smoothed x y z 2
                let hxy = mixed smoothed x y z 0 1
                let hxz = mixed smoothed x y z 0 2
                let hyz = mixed smoothed x y z 1 2
                let eig =
                    symmetricEigen
                        { m00 = hxx; m01 = hxy; m02 = hxz
                          m10 = hxy; m11 = hyy; m12 = hyz
                          m20 = hxz; m21 = hyz; m22 = hzz }
                    |> List.map (fst >> abs)
                    |> List.sortDescending

                let l1 = eig[0]
                let l2 = eig[1]
                let l3 = eig[2]
                let epsilon = 1.0e-12
                let value =
                    match kind with
                    | "tube"
                    | "tubeness"
                    | "vessel"
                    | "vesselness" ->
                        l2 * l3 / (l1 + epsilon)
                    | "sheet"
                    | "sheetness" ->
                        l1 * l3 / (l2 + epsilon)
                    | _ ->
                        l1 * l2 * l3

                response[x, y, z] <- (sigma ** 6.0) * value

    response

let hessianKeypoints<'T when 'T: equality>
    (sigma: float)
    (responseKind: string)
    (threshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    localKeypointStage<'T> "hessianKeypoints" sigma stride (fun images window volume ->
        let smoothed = gaussianBlur3D sigma volume
        let response = hessianResponse responseKind sigma smoothed
        keypointsFromResponse threshold sigma images window response)

let harris3DKeypoints<'T when 'T: equality>
    (sigma: float)
    (rho: float)
    (k: float)
    (threshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    if sigma <= 0.0 then invalidArg "sigma" "harris3DKeypoints sigma must be positive."
    if rho <= 0.0 then invalidArg "rho" "harris3DKeypoints rho must be positive."

    localKeypointStage<'T> "harris3DKeypoints" (max sigma rho) stride (fun images window volume ->
        let smoothed = gaussianBlur3D sigma volume
        let width = smoothed.GetLength(0)
        let height = smoothed.GetLength(1)
        let depth = smoothed.GetLength(2)
        let ixx = Array3D.zeroCreate<double> width height depth
        let iyy = Array3D.zeroCreate<double> width height depth
        let izz = Array3D.zeroCreate<double> width height depth
        let ixy = Array3D.zeroCreate<double> width height depth
        let ixz = Array3D.zeroCreate<double> width height depth
        let iyz = Array3D.zeroCreate<double> width height depth

        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    let gx = central smoothed x y z 0
                    let gy = central smoothed x y z 1
                    let gz = central smoothed x y z 2
                    ixx[x, y, z] <- gx * gx
                    iyy[x, y, z] <- gy * gy
                    izz[x, y, z] <- gz * gz
                    ixy[x, y, z] <- gx * gy
                    ixz[x, y, z] <- gx * gz
                    iyz[x, y, z] <- gy * gz

        let ixx = gaussianBlur3D rho ixx
        let iyy = gaussianBlur3D rho iyy
        let izz = gaussianBlur3D rho izz
        let ixy = gaussianBlur3D rho ixy
        let ixz = gaussianBlur3D rho ixz
        let iyz = gaussianBlur3D rho iyz
        let response = Array3D.zeroCreate<double> width height depth

        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    let m =
                        { m00 = ixx[x, y, z]; m01 = ixy[x, y, z]; m02 = ixz[x, y, z]
                          m10 = ixy[x, y, z]; m11 = iyy[x, y, z]; m12 = iyz[x, y, z]
                          m20 = ixz[x, y, z]; m21 = iyz[x, y, z]; m22 = izz[x, y, z] }
                    let trace = m.m00 + m.m11 + m.m22
                    response[x, y, z] <- det3 m - k * trace * trace * trace

        keypointsFromResponse threshold (max sigma rho) images window response)

let forstner3DKeypoints<'T when 'T: equality>
    (sigma: float)
    (rho: float)
    (threshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    if sigma <= 0.0 then invalidArg "sigma" "forstner3DKeypoints sigma must be positive."
    if rho <= 0.0 then invalidArg "rho" "forstner3DKeypoints rho must be positive."

    localKeypointStage<'T> "forstner3DKeypoints" (max sigma rho) stride (fun images window volume ->
        let smoothed = gaussianBlur3D sigma volume
        let width = smoothed.GetLength(0)
        let height = smoothed.GetLength(1)
        let depth = smoothed.GetLength(2)
        let ixx = Array3D.zeroCreate<double> width height depth
        let iyy = Array3D.zeroCreate<double> width height depth
        let izz = Array3D.zeroCreate<double> width height depth
        let ixy = Array3D.zeroCreate<double> width height depth
        let ixz = Array3D.zeroCreate<double> width height depth
        let iyz = Array3D.zeroCreate<double> width height depth

        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    let gx = central smoothed x y z 0
                    let gy = central smoothed x y z 1
                    let gz = central smoothed x y z 2
                    ixx[x, y, z] <- gx * gx
                    iyy[x, y, z] <- gy * gy
                    izz[x, y, z] <- gz * gz
                    ixy[x, y, z] <- gx * gy
                    ixz[x, y, z] <- gx * gz
                    iyz[x, y, z] <- gy * gz

        let ixx = gaussianBlur3D rho ixx
        let iyy = gaussianBlur3D rho iyy
        let izz = gaussianBlur3D rho izz
        let ixy = gaussianBlur3D rho ixy
        let ixz = gaussianBlur3D rho ixz
        let iyz = gaussianBlur3D rho iyz
        let response = Array3D.zeroCreate<double> width height depth

        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    let m =
                        { m00 = ixx[x, y, z]; m01 = ixy[x, y, z]; m02 = ixz[x, y, z]
                          m10 = ixy[x, y, z]; m11 = iyy[x, y, z]; m12 = iyz[x, y, z]
                          m20 = ixz[x, y, z]; m21 = iyz[x, y, z]; m22 = izz[x, y, z] }
                    let trace = m.m00 + m.m11 + m.m22
                    response[x, y, z] <- if trace <= 1.0e-12 then 0.0 else det3 m / trace

        keypointsFromResponse threshold (max sigma rho) images window response)

let phaseCongruencyKeypoints<'T when 'T: equality>
    (sigma: float)
    (threshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    localKeypointStage<'T> "phaseCongruencyKeypoints" sigma stride (fun images window volume ->
        let smoothed = gaussianBlur3D sigma volume
        let width = smoothed.GetLength(0)
        let height = smoothed.GetLength(1)
        let depth = smoothed.GetLength(2)
        let response = Array3D.zeroCreate<double> width height depth

        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    let gx = central smoothed x y z 0
                    let gy = central smoothed x y z 1
                    let gz = central smoothed x y z 2
                    let gradient = sqrt (gx * gx + gy * gy + gz * gz)
                    let laplacian =
                        second smoothed x y z 0
                        + second smoothed x y z 1
                        + second smoothed x y z 2
                    response[x, y, z] <- abs laplacian / (gradient + 1.0e-6)

        keypointsFromResponse threshold sigma images window response)

let siftKeypoints<'T when 'T: equality>
    (sigma0: float)
    (scaleFactor: float)
    (scaleLevels: uint)
    (contrastThreshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    dogKeypoints<'T> sigma0 scaleFactor scaleLevels contrastThreshold stride
