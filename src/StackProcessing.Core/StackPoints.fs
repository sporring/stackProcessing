module StackPoints

open System
open System.Globalization
open System.IO
open System.Text
open FSharp.Control
open SlimPipeline
open StackCore

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

    Plan.create stage pl.memAvail 0UL 0UL 1UL pl.debug

let writePointSet (output: string) (suffix: string) : Stage<PointSet, unit> =
    let reducer (_debug: bool) (input: AsyncSeq<PointSet>) =
        async {
            let outputPath = outputPathWithSuffix output suffix
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
    let reducer (_debug: bool) (input: AsyncSeq<VectorizedMatrix>) =
        async {
            let! matrices = input |> AsyncSeq.toListAsync

            match matrices with
            | [] -> invalidOp "writeMatrix expected one vectorized matrix, but the stream was empty."
            | [ matrix ] ->
                let outputPath = outputPathWithSuffix output suffix
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
            | _ ->
                invalidOp $"writeMatrix expected one vectorized matrix, but got {matrices.Length}."
        }

    Stage.reduce $"writeMatrix \"{output}\" \"{suffix}\"" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let writeCSVMatrix (output: string) : Stage<VectorizedMatrix, unit> =
    writeMatrix output ".csv"

let writeCSVHistogram<'T when 'T: comparison> (output: string) : Stage<Map<'T, uint64>, unit> =
    let reducer (_debug: bool) (input: AsyncSeq<Map<'T, uint64>>) =
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
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    volume[x, y, z] <- Convert.ToDouble(image[x, y], invariant))

        volume

let private gaussianKernel sigma =
    let sigma = max sigma 0.01
    let radius = max 1 (int (ceil (3.0 * sigma)))
    let kernel =
        [| for offset in -radius .. radius ->
            let x = float offset
            exp (-(x * x) / (2.0 * sigma * sigma)) |]
    let sum = kernel |> Array.sum
    kernel |> Array.map (fun value -> value / sum), radius

let private clamp lo hi value =
    if value < lo then lo elif value > hi then hi else value

let private blur1D axis (kernel: double[]) radius (source: double[,,]) =
    let width = source.GetLength(0)
    let height = source.GetLength(1)
    let depth = source.GetLength(2)
    let result = Array3D.zeroCreate<double> width height depth

    for z in 0 .. depth - 1 do
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                let mutable sum = 0.0
                for k in -radius .. radius do
                    let weight = kernel[k + radius]
                    let sx, sy, sz =
                        match axis with
                        | 0 -> clamp 0 (width - 1) (x + k), y, z
                        | 1 -> x, clamp 0 (height - 1) (y + k), z
                        | _ -> x, y, clamp 0 (depth - 1) (z + k)
                    sum <- sum + weight * source[sx, sy, sz]
                result[x, y, z] <- sum

    result

let private gaussianBlur3D sigma source =
    let kernel, radius = gaussianKernel sigma
    source
    |> blur1D 0 kernel radius
    |> blur1D 1 kernel radius
    |> blur1D 2 kernel radius

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

let siftKeypoints<'T when 'T: equality>
    (sigma0: float)
    (scaleFactor: float)
    (scaleLevels: uint)
    (contrastThreshold: float)
    (stride: uint)
    : Stage<Image<'T>, PointSet> =

    dogKeypoints<'T> sigma0 scaleFactor scaleLevels contrastThreshold stride
