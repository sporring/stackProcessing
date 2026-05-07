module StackPoints

open System
open System.Globalization
open System.IO
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

type PointSetChunk =
    { Points: CoordinatePoint list }

module PointSetChunk =
    let empty = { Points = [] }

let private invariant = CultureInfo.InvariantCulture

let private f (value: float) =
    value.ToString("R", invariant)

let private parseFloat (text: string) =
    Double.Parse(text.Trim(), NumberStyles.Float ||| NumberStyles.AllowThousands, invariant)

let private splitCsvLine (line: string) =
    line.Split(',')
    |> Array.map _.Trim()

let readPointSet (path: string) (pl: Plan<unit, unit>) : Plan<unit, PointSetChunk> =
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

let writePointSet (outputPath: string) : Stage<PointSetChunk, unit> =
    let reducer (_debug: bool) (input: AsyncSeq<PointSetChunk>) =
        async {
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

    Stage.reduce $"writePointSet \"{outputPath}\"" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

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
        PointSetChunk.empty
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
    : Stage<Image<'T>, PointSetChunk> =

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
