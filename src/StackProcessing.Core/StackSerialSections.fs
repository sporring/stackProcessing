module StackSerialSections

open System
open System.Collections.Generic
open System.Globalization
open FSharp.Control
open Image
open SlimPipeline
open StackCore
open StackPoints

[<CLIMutable>]
type SerialSliceTransform =
    { Slice: int
      Matrix: float list list }

[<CLIMutable>]
type SerialSliceManifest =
    { Version: int
      Width: uint
      Height: uint
      Transforms: SerialSliceTransform list }

let private toDouble value =
    Convert.ToDouble(box value, CultureInfo.InvariantCulture)

let private fromDouble<'T> value =
    let t = typeof<'T>
    if t = typeof<float32> then
        box (float32 value) :?> 'T
    elif t = typeof<float> then
        box value :?> 'T
    else
        invalidArg "T" $"Serial floating-point transforms support Float32 and Float64 images, got {t.Name}."

let private validateMatrix matrix =
    if List.length matrix <> 3 || matrix |> List.exists (fun row -> List.length row <> 3) then
        invalidArg "matrix" "Serial slice transforms must be 3x3 homogeneous matrices."

    let last = matrix[2]
    if abs last[0] > 1.0e-12 || abs last[1] > 1.0e-12 || abs (last[2] - 1.0) > 1.0e-12 then
        invalidArg "matrix" "Serial slice transforms must have homogeneous last row [0, 0, 1]."

let private identityMatrix =
    [ [ 1.0; 0.0; 0.0 ]
      [ 0.0; 1.0; 0.0 ]
      [ 0.0; 0.0; 1.0 ] ]

let serialIdentityManifest width height depth =
    if width = 0u then invalidArg "width" "width must be positive."
    if height = 0u then invalidArg "height" "height must be positive."

    { Version = 1
      Width = width
      Height = height
      Transforms =
        [ for z in 0 .. int depth - 1 ->
            { Slice = z
              Matrix = identityMatrix } ] }

let private translationMatrix dx dy =
    [ [ 1.0; 0.0; dx ]
      [ 0.0; 1.0; dy ]
      [ 0.0; 0.0; 1.0 ] ]

let private multiplyMatrix a b =
    validateMatrix a
    validateMatrix b
    [ for r in 0 .. 2 ->
        [ for c in 0 .. 2 ->
            [ 0 .. 2 ] |> List.sumBy (fun k -> a[r][k] * b[k][c]) ] ]

let private invertMatrix matrix =
    validateMatrix matrix
    let a = matrix[0][0]
    let b = matrix[0][1]
    let c = matrix[0][2]
    let d = matrix[1][0]
    let e = matrix[1][1]
    let f = matrix[1][2]
    let det = a * e - b * d
    if abs det < 1.0e-18 then
        invalidOp "Serial slice transform is singular."

    let invDet = 1.0 / det
    let ia = e * invDet
    let ib = -b * invDet
    let id = -d * invDet
    let ie = a * invDet
    [ [ ia; ib; -(ia * c + ib * f) ]
      [ id; ie; -(id * c + ie * f) ]
      [ 0.0; 0.0; 1.0 ] ]

let private transformPoint matrix x y =
    validateMatrix matrix
    matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
    matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]

let private scaleCoordinateMatrix factor =
    [ [ factor; 0.0; 0.0 ]
      [ 0.0; factor; 0.0 ]
      [ 0.0; 0.0; 1.0 ] ]

let private liftMatrixFromDownsampledCoordinates factor matrix =
    multiplyMatrix (scaleCoordinateMatrix factor) (multiplyMatrix matrix (scaleCoordinateMatrix (1.0 / factor)))

let private terms2D order =
    if order < 0 then invalidArg "order" "Polynomial order must be non-negative."
    [ for total in 0 .. order do
        for xPower in 0 .. total do
            let yPower = total - xPower
            xPower, yPower ]

let private norm size value =
    if size <= 1u then 0.0
    else 2.0 * float value / float (size - 1u) - 1.0

let private basis2D terms width height x y =
    let xn = norm width x
    let yn = norm height y
    terms
    |> List.map (fun (xp, yp) -> (xn ** float xp) * (yn ** float yp))

let private coordinatePowers size order =
    Array.init (int size) (fun i ->
        let value = norm size (uint i)
        let powers = Array.zeroCreate<float> (order + 1)
        powers[0] <- 1.0
        for p in 1 .. order do
            powers[p] <- powers[p - 1] * value
        powers)

let private basis2DValues (terms: (int * int)[]) (xPowers: float[][]) (yPowers: float[][]) x y =
    Array.init terms.Length (fun i ->
        let xp, yp = terms[i]
        xPowers[x][xp] * yPowers[y][yp])

let private evalPolynomial2DValues (terms: (int * int)[]) (coefficients: float[]) (xPowers: float[][]) (yPowers: float[][]) x y =
    let mutable sum = 0.0
    for i in 0 .. terms.Length - 1 do
        let xp, yp = terms[i]
        sum <- sum + coefficients[i] * xPowers[x][xp] * yPowers[y][yp]
    sum

let private solveLinearSystem (a: float[,]) (b: float[]) =
    let n = b.Length
    let m = Array2D.copy a
    let rhs = Array.copy b

    for i in 0 .. n - 1 do
        m[i, i] <- m[i, i] + 1.0e-10

    for k in 0 .. n - 1 do
        let mutable pivot = k
        let mutable pivotValue = abs m[k, k]
        for row in k + 1 .. n - 1 do
            let value = abs m[row, k]
            if value > pivotValue then
                pivot <- row
                pivotValue <- value

        if pivotValue < 1.0e-18 then
            invalidOp "Serial polynomial bias fit is singular. Use a lower order or more pixels."

        if pivot <> k then
            for col in k .. n - 1 do
                let tmp = m[k, col]
                m[k, col] <- m[pivot, col]
                m[pivot, col] <- tmp
            let tmp = rhs[k]
            rhs[k] <- rhs[pivot]
            rhs[pivot] <- tmp

        let diag = m[k, k]
        for col in k .. n - 1 do
            m[k, col] <- m[k, col] / diag
        rhs[k] <- rhs[k] / diag

        for row in 0 .. n - 1 do
            if row <> k then
                let factor = m[row, k]
                for col in k .. n - 1 do
                    m[row, col] <- m[row, col] - factor * m[k, col]
                rhs[row] <- rhs[row] - factor * rhs[k]

    rhs

let private fitPolynomial2D order (image: Image<'T>) =
    let width = image.GetWidth()
    let height = image.GetHeight()
    let terms = terms2D order |> List.toArray
    let n = terms.Length
    let normal = Array2D.zeroCreate<float> n n
    let right = Array.zeroCreate<float> n
    let pixels = image.toArray2D()
    let xPowers = coordinatePowers width order
    let yPowers = coordinatePowers height order

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let values = basis2DValues terms xPowers yPowers x y
            let intensity = pixels[x, y] |> toDouble
            for row in 0 .. n - 1 do
                right[row] <- right[row] + values[row] * intensity
                for col in 0 .. n - 1 do
                    normal[row, col] <- normal[row, col] + values[row] * values[col]

    terms, xPowers, yPowers, pixels, solveLinearSystem normal right

let private evalPolynomial2D terms coefficients width height x y =
    basis2D terms width height x y
    |> List.zip (Array.toList coefficients)
    |> List.sumBy (fun (c, b) -> c * b)

let serialPolynomialBiasCorrect<'T when 'T: equality> order : Stage<Image<'T>, Image<'T>> =
    fromDouble<'T> 0.0 |> ignore

    Stage.map
        "serialPolynomialBiasCorrect"
        (fun _ image ->
            try
                let width = image.GetWidth()
                let height = image.GetHeight()
                let terms, xPowers, yPowers, pixels, coefficients = fitPolynomial2D order image
                let output = Array2D.zeroCreate<'T> (int width) (int height)

                for y in 0 .. int height - 1 do
                    for x in 0 .. int width - 1 do
                        let corrected =
                            (pixels[x, y] |> toDouble)
                            - evalPolynomial2DValues terms coefficients xPowers yPowers x y
                        output[x, y] <- fromDouble<'T> corrected

                Image<'T>.ofArray2D(output, "serialPolynomialBiasCorrect", image.index)
            finally
                image.decRefCount())
        id
        id

let private gaussianKernel sigma =
    let radius = max 1 (int (ceil (3.0 * sigma)))
    let weights =
        [| for i in -radius .. radius -> exp (-0.5 * (float i / sigma) ** 2.0) |]
    let sum = Array.sum weights
    radius, weights |> Array.map (fun value -> value / sum)

let private imageToArray (image: Image<'T>) =
    image.toArray2D()
    |> Array2D.map toDouble

let private blur2D sigma (source: double[,]) =
    let radius, kernel = gaussianKernel sigma
    let width = source.GetLength(0)
    let height = source.GetLength(1)
    let tmp = Array2D.zeroCreate<float> width height
    let output = Array2D.zeroCreate<float> width height
    let clamp lo hi value = min hi (max lo value)

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            tmp[x, y] <-
                [ -radius .. radius ]
                |> List.sumBy (fun dx -> kernel[dx + radius] * source[clamp 0 (width - 1) (x + dx), y])

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            output[x, y] <-
                [ -radius .. radius ]
                |> List.sumBy (fun dy -> kernel[dy + radius] * tmp[x, clamp 0 (height - 1) (y + dy)])

    output

let private strictMaximum2D (response: double[,]) x y =
    let value = response[x, y]
    seq {
        for yy in y - 1 .. y + 1 do
            for xx in x - 1 .. x + 1 do
                if xx <> x || yy <> y then
                    response[xx, yy]
    }
    |> Seq.forall (fun neighbor -> value > neighbor)

let serialKeypoints2D<'T when 'T: equality> sigma threshold : Stage<Image<'T>, PointSet> =
    if sigma <= 0.0 then invalidArg "sigma" "serialKeypoints2D sigma must be positive."

    Stage.map
        "serialKeypoints2D"
        (fun _ image ->
            try
                let smoothed = imageToArray image |> blur2D sigma
                let width = smoothed.GetLength(0)
                let height = smoothed.GetLength(1)
                let response = Array2D.zeroCreate<float> width height
                let clamp lo hi value = min hi (max lo value)

                for y in 0 .. height - 1 do
                    for x in 0 .. width - 1 do
                        let laplacian =
                            smoothed[clamp 0 (width - 1) (x + 1), y]
                            + smoothed[clamp 0 (width - 1) (x - 1), y]
                            + smoothed[x, clamp 0 (height - 1) (y + 1)]
                            + smoothed[x, clamp 0 (height - 1) (y - 1)]
                            - 4.0 * smoothed[x, y]
                        response[x, y] <- sigma * sigma * abs laplacian

                let points =
                    [ if width >= 3 && height >= 3 then
                        for y in 1 .. height - 2 do
                            for x in 1 .. width - 2 do
                                let value = response[x, y]
                                if value >= threshold && strictMaximum2D response x y then
                                    { X = float x
                                      Y = float y
                                      Z = float image.index
                                      Scale = sigma
                                      Response = value } ]

                { Points = points }
            finally
                image.decRefCount())
        id
        (fun _ -> 1UL)

let private centroid points =
    match points with
    | [] -> None
    | _ ->
        let n = float (List.length points)
        Some(
            (points |> List.sumBy (fun point -> point.X)) / n,
            (points |> List.sumBy (fun point -> point.Y)) / n)

let serialKeypointsApplyTrans width height : Stage<PointSet, SerialSliceManifest> =
    let reducer (_debug: bool) (input: AsyncSeq<PointSet>) =
        async {
            let! pointSets = input |> AsyncSeq.toListAsync
            let mutable cumulative = identityMatrix
            let mutable previous = None

            let transforms =
                pointSets
                |> List.mapi (fun i pointSet ->
                    match previous, centroid pointSet.Points with
                    | None, current ->
                        previous <- current
                    | Some (px, py), Some (cx, cy) ->
                        let currentToPrevious = translationMatrix (px - cx) (py - cy)
                        cumulative <- multiplyMatrix cumulative currentToPrevious
                        previous <- Some(cx, cy)
                    | _, current ->
                        previous <- current

                    { Slice =
                        pointSet.Points
                        |> List.tryHead
                        |> Option.map (fun p -> int (round p.Z))
                        |> Option.defaultValue i
                      Matrix = cumulative })

            return
                { Version = 1
                  Width = width
                  Height = height
                  Transforms = transforms }
        }

    Stage.reduce "serialKeypointsApplyTrans" reducer Streaming id (fun _ -> 1UL)

let private siftKeypoints2D sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints (image: Image<'T>) =
    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    let scaleLevels = int scaleLevels

    if width < 3 || height < 3 || scaleLevels < 4 || maxKeypoints <= 0 then
        []
    else
        let pixels = imageToArray image
        let sigmas =
            [| for level in 0 .. scaleLevels - 1 -> sigma0 * Math.Pow(scaleFactor, float level) |]

        let blurred = sigmas |> Array.map (fun sigma -> blur2D sigma pixels)
        let dogs =
            [| for level in 0 .. scaleLevels - 2 ->
                let lower = blurred[level]
                let upper = blurred[level + 1]
                let dog = Array2D.zeroCreate<double> width height
                for y in 0 .. height - 1 do
                    for x in 0 .. width - 1 do
                        dog[x, y] <- upper[x, y] - lower[x, y]
                dog |]

        let points = ResizeArray<CoordinatePoint>()

        for scaleIndex in 0 .. dogs.Length - 1 do
            let dog = dogs[scaleIndex]
            for y in 1 .. height - 2 do
                for x in 1 .. width - 2 do
                    let value = dog[x, y]
                    if abs value >= contrastThreshold then
                        let mutable greater = true
                        let mutable less = true
                        let scaleNeighborStart = max 0 (scaleIndex - 1)
                        let scaleNeighborStop = min (dogs.Length - 1) (scaleIndex + 1)

                        for neighborScaleIndex in scaleNeighborStart .. scaleNeighborStop do
                            let neighborDog = dogs[neighborScaleIndex]
                            for dy in -1 .. 1 do
                                for dx in -1 .. 1 do
                                    if neighborScaleIndex <> scaleIndex || dy <> 0 || dx <> 0 then
                                        let neighbor = neighborDog[x + dx, y + dy]
                                        if value <= neighbor then greater <- false
                                        if value >= neighbor then less <- false

                        if greater || less then
                            points.Add
                                { X = float x
                                  Y = float y
                                  Z = float image.index
                                  Scale = sigmas[scaleIndex]
                                  Response = value }

        points
        |> Seq.sortByDescending (fun point -> abs point.Response)
        |> Seq.truncate maxKeypoints
        |> Seq.toList

type private SiftFeature =
    { Point: CoordinatePoint
      Descriptor: float[] }

type private FeatureMatch =
    { Fixed: CoordinatePoint
      Moving: CoordinatePoint }

let private gradientAt (pixels: double[,]) x y =
    let width = pixels.GetLength(0)
    let height = pixels.GetLength(1)
    let clamp lo hi value = min hi (max lo value)
    let gx = pixels[clamp 0 (width - 1) (x + 1), y] - pixels[clamp 0 (width - 1) (x - 1), y]
    let gy = pixels[x, clamp 0 (height - 1) (y + 1)] - pixels[x, clamp 0 (height - 1) (y - 1)]
    gx, gy

let private wrapAngle angle =
    let twoPi = 2.0 * Math.PI
    let mutable wrapped = angle % twoPi
    if wrapped < 0.0 then wrapped <- wrapped + twoPi
    wrapped

let private dominantOrientation orientationBins (pixels: double[,]) (point: CoordinatePoint) =
    let width = pixels.GetLength(0)
    let height = pixels.GetLength(1)
    let bins = Array.zeroCreate<float> orientationBins
    let radius = max 1 (int (round (3.0 * point.Scale)))
    let sigma = max 1.0 (1.5 * point.Scale)
    let cx = int (round point.X)
    let cy = int (round point.Y)

    for y in max 1 (cy - radius) .. min (height - 2) (cy + radius) do
        for x in max 1 (cx - radius) .. min (width - 2) (cx + radius) do
            let gx, gy = gradientAt pixels x y
            let magnitude = sqrt (gx * gx + gy * gy)
            if magnitude > 0.0 then
                let dx = float x - point.X
                let dy = float y - point.Y
                let weight = exp (-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                let bin = int (floor (wrapAngle (atan2 gy gx) / (2.0 * Math.PI) * float orientationBins)) % orientationBins
                bins[bin] <- bins[bin] + weight * magnitude

    let best =
        bins
        |> Array.mapi (fun i value -> i, value)
        |> Array.maxBy snd
        |> fst

    (float best + 0.5) * 2.0 * Math.PI / float orientationBins

let private normalizeDescriptor (descriptor: float[]) =
    let normalize values =
        let norm = sqrt (values |> Array.sumBy (fun value -> value * value))
        if norm <= 1.0e-12 then values else values |> Array.map (fun value -> value / norm)

    let normalized = normalize descriptor
    let clipped = normalized |> Array.map (min 0.2)
    normalize clipped

let private siftDescriptor descriptorSize orientationBins (pixels: double[,]) (point: CoordinatePoint) =
    let width = pixels.GetLength(0)
    let height = pixels.GetLength(1)
    let cellSize = 4.0
    let grid = descriptorSize
    let half = 0.5 * cellSize * float grid
    let orientation = dominantOrientation orientationBins pixels point
    let cosA = cos orientation
    let sinA = sin orientation
    let descriptor = Array.zeroCreate<float> (grid * grid * orientationBins)
    let radius = int (ceil half)
    let cx = int (round point.X)
    let cy = int (round point.Y)
    let sigma = 0.5 * float grid * cellSize

    for y in max 1 (cy - radius) .. min (height - 2) (cy + radius) do
        for x in max 1 (cx - radius) .. min (width - 2) (cx + radius) do
            let rx = float x - point.X
            let ry = float y - point.Y
            let localX = cosA * rx + sinA * ry
            let localY = -sinA * rx + cosA * ry
            if abs localX < half && abs localY < half then
                let cellX = int (floor ((localX + half) / cellSize))
                let cellY = int (floor ((localY + half) / cellSize))
                if cellX >= 0 && cellX < grid && cellY >= 0 && cellY < grid then
                    let gx, gy = gradientAt pixels x y
                    let magnitude = sqrt (gx * gx + gy * gy)
                    if magnitude > 0.0 then
                        let dx = float x - point.X
                        let dy = float y - point.Y
                        let weight = exp (-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                        let angle = wrapAngle ((atan2 gy gx) - orientation)
                        let bin = int (floor (angle / (2.0 * Math.PI) * float orientationBins)) % orientationBins
                        let index = (cellY * grid + cellX) * orientationBins + bin
                        descriptor[index] <- descriptor[index] + weight * magnitude

    normalizeDescriptor descriptor

let private siftFeaturesFromPoints descriptorSize orientationBins pixels points =
    points
    |> List.choose (fun point ->
        let descriptor = siftDescriptor descriptorSize orientationBins pixels point
        if descriptor |> Array.exists (fun value -> value > 0.0) then
            Some { Point = point; Descriptor = descriptor }
        else
            None)

let private descriptorDistanceSquared (a: float[]) (b: float[]) =
    let mutable sum = 0.0
    for i in 0 .. a.Length - 1 do
        let diff = a[i] - b[i]
        sum <- sum + diff * diff
    sum

let private matchSiftFeatures nearestNeighborRatio (fixedFeatures: SiftFeature list) (movingFeatures: SiftFeature list) =
    if fixedFeatures.Length < 2 then
        []
    else
        [ for moving in movingFeatures do
            let distances =
                fixedFeatures
                |> List.map (fun fixedFeature -> descriptorDistanceSquared moving.Descriptor fixedFeature.Descriptor, fixedFeature)
                |> List.sortBy fst

            match distances with
            | (bestDistance, best) :: (nextDistance, _) :: _ when bestDistance <= nearestNeighborRatio * nearestNeighborRatio * nextDistance ->
                { Fixed = best.Point; Moving = moving.Point }
            | _ -> () ]

type private PreparedSerialImage<'T when 'T: equality> =
    { Image: Image<'T>
      Pixels: Lazy<double[,]>
      SsdPixels: Lazy<double[,]>
      DogPoints: Lazy<CoordinatePoint list>
      SiftFeatures: Lazy<SiftFeature list> }

let private estimateSsdTranslationPixels maxShift (fixedPixels: double[,]) (moving: double[,]) =
    let width = fixedPixels.GetLength(0)
    let height = fixedPixels.GetLength(1)
    if moving.GetLength(0) <> width || moving.GetLength(1) <> height then
        invalidOp "serialEstTrans expects all slices to have the same shape."

    let mutable bestDx = 0
    let mutable bestDy = 0
    let mutable bestScore = Double.PositiveInfinity

    for dy in -maxShift .. maxShift do
        for dx in -maxShift .. maxShift do
            let mutable score = 0.0
            let mutable count = 0
            for y in 0 .. height - 1 do
                let my = y - dy
                if my >= 0 && my < height then
                    for x in 0 .. width - 1 do
                        let mx = x - dx
                        if mx >= 0 && mx < width then
                            let diff = fixedPixels[x, y] - moving[mx, my]
                            score <- score + diff * diff
                            count <- count + 1

            if count > 0 then
                let normalized = score / float count
                if normalized < bestScore then
                    bestScore <- normalized
                    bestDx <- dx
                    bestDy <- dy

    float bestDx, float bestDy

let private sampleDoubleBilinear background (pixels: double[,]) x y =
    let width = Array2D.length1 pixels
    let height = Array2D.length2 pixels
    if x < 0.0 || y < 0.0 || x > float (width - 1) || y > float (height - 1) then
        background
    else
        let x0 = int (floor x)
        let y0 = int (floor y)
        let x1 = min (width - 1) (x0 + 1)
        let y1 = min (height - 1) (y0 + 1)
        let tx = x - float x0
        let ty = y - float y0
        (1.0 - tx) * (1.0 - ty) * pixels[x0, y0]
        + tx * (1.0 - ty) * pixels[x1, y0]
        + (1.0 - tx) * ty * pixels[x0, y1]
        + tx * ty * pixels[x1, y1]

let private matrixFromAffineParameters (parameters: float[]) =
    [ [ parameters[0]; parameters[1]; parameters[2] ]
      [ parameters[3]; parameters[4]; parameters[5] ]
      [ 0.0; 0.0; 1.0 ] ]

let private affineParametersFromMatrix matrix =
    validateMatrix matrix
    [| matrix[0][0]; matrix[0][1]; matrix[0][2]; matrix[1][0]; matrix[1][1]; matrix[1][2] |]

let private downsampleSmoothed factor (pixels: double[,]) =
    let width = pixels.GetLength(0)
    let height = pixels.GetLength(1)
    let outputWidth = max 3 (int (ceil (float width / factor)))
    let outputHeight = max 3 (int (ceil (float height / factor)))
    let smoothed = blur2D (max 0.5 (0.5 * factor)) pixels
    let output = Array2D.zeroCreate<float> outputWidth outputHeight

    for y in 0 .. outputHeight - 1 do
        for x in 0 .. outputWidth - 1 do
            output[x, y] <- sampleDoubleBilinear 0.0 smoothed (float x * factor) (float y * factor)

    output

let private estimateSsdAffineMatrixAtResolution maxShift maxIterations initialLinearStep initialTranslationStep minStep stepShrink pixelFraction initialMatrix (fixedPixels: double[,]) (movingPixels: double[,]) =
    let width = fixedPixels.GetLength(0)
    let height = fixedPixels.GetLength(1)
    if movingPixels.GetLength(0) <> width || movingPixels.GetLength(1) <> height then
        invalidOp "serialEstTrans expects all slices to have the same shape."

    let mutable bestParameters =
        match initialMatrix with
        | Some matrix -> affineParametersFromMatrix matrix
        | None ->
            let initialDx, initialDy = estimateSsdTranslationPixels maxShift fixedPixels movingPixels
            [| 1.0; 0.0; initialDx; 0.0; 1.0; initialDy |]

    let steps = [| initialLinearStep; initialLinearStep; initialTranslationStep; initialLinearStep; initialLinearStep; initialTranslationStep |]
    let sampleStride = max 1 (int (round (sqrt (1.0 / pixelFraction))))

    let objective parameters =
        let matrix = matrixFromAffineParameters parameters
        try
            let inverse = invertMatrix matrix
            let mutable score = 0.0
            let mutable count = 0

            for y in 0 .. sampleStride .. height - 1 do
                for x in 0 .. sampleStride .. width - 1 do
                    let movingX, movingY = transformPoint inverse (float x) (float y)
                    if movingX >= 0.0 && movingY >= 0.0 && movingX <= float (width - 1) && movingY <= float (height - 1) then
                        let diff = fixedPixels[x, y] - sampleDoubleBilinear 0.0 movingPixels movingX movingY
                        score <- score + diff * diff
                        count <- count + 1

            if count = 0 then Double.PositiveInfinity else score / float count
        with _ ->
            Double.PositiveInfinity

    let mutable bestScore = objective bestParameters
    let mutable iteration = 0

    while iteration < maxIterations && (steps |> Array.max) > minStep do
        let mutable improved = false

        for parameterIndex in 0 .. bestParameters.Length - 1 do
            for direction in [ 1.0; -1.0 ] do
                let candidate = Array.copy bestParameters
                candidate[parameterIndex] <- candidate[parameterIndex] + direction * steps[parameterIndex]

                if abs candidate[2] <= float maxShift && abs candidate[5] <= float maxShift then
                    let score = objective candidate
                    if score < bestScore then
                        bestParameters <- candidate
                        bestScore <- score
                        improved <- true

        if not improved then
            for index in 0 .. steps.Length - 1 do
                steps[index] <- steps[index] * stepShrink

        iteration <- iteration + 1

    matrixFromAffineParameters bestParameters

let private estimateSsdAffineMatrixFromPrepared maxShift maxIterations initialLinearStep initialTranslationStep minStep stepShrink pixelFraction factor (fixedImage: PreparedSerialImage<'T>) (movingImage: PreparedSerialImage<'T>) =
    let fixedPixels = fixedImage.Pixels.Value
    let movingPixels = movingImage.Pixels.Value
    let width = fixedPixels.GetLength(0)
    let height = fixedPixels.GetLength(1)
    if movingPixels.GetLength(0) <> width || movingPixels.GetLength(1) <> height then
        invalidOp "serialEstTrans expects all slices to have the same shape."

    if factor <= 1.0 || width < int (3.0 * factor) || height < int (3.0 * factor) then
        estimateSsdAffineMatrixAtResolution maxShift maxIterations initialLinearStep initialTranslationStep minStep stepShrink pixelFraction None fixedPixels movingPixels
    else
        let lowMaxShift = max 1 (int (ceil (float maxShift / factor)))
        let lowMatrix =
            estimateSsdAffineMatrixAtResolution
                lowMaxShift
                maxIterations
                initialLinearStep
                (max 1.0 (initialTranslationStep / factor))
                minStep
                stepShrink
                pixelFraction
                None
                fixedImage.SsdPixels.Value
                movingImage.SsdPixels.Value
        let fullInitial = liftMatrixFromDownsampledCoordinates factor lowMatrix

        estimateSsdAffineMatrixAtResolution
            maxShift
            (max 8 (maxIterations / 4))
            initialLinearStep
            initialTranslationStep
            minStep
            stepShrink
            pixelFraction
            (Some fullInitial)
            fixedPixels
            movingPixels

let private estimateKeypointTranslation maxShift matchTolerance fixedPoints movingPoints =
    if List.isEmpty fixedPoints || List.isEmpty movingPoints then
        0.0, 0.0
    else
        let votes = Dictionary<int * int, ResizeArray<float * float>>()

        for fixedPoint in fixedPoints do
            for movingPoint in movingPoints do
                let dx = fixedPoint.X - movingPoint.X
                let dy = fixedPoint.Y - movingPoint.Y
                if abs dx <= float maxShift && abs dy <= float maxShift then
                    let key = int (round dx), int (round dy)
                    let mutable bucket = Unchecked.defaultof<ResizeArray<float * float>>
                    if not (votes.TryGetValue(key, &bucket)) then
                        bucket <- ResizeArray<float * float>()
                        votes[key] <- bucket
                    bucket.Add(dx, dy)

        if votes.Count = 0 then
            0.0, 0.0
        else
            let scoreBucket (KeyValue((binDx, binDy), candidates)) =
                let close =
                    candidates
                    |> Seq.filter (fun (dx, dy) ->
                        let distance = sqrt ((dx - float binDx) ** 2.0 + (dy - float binDy) ** 2.0)
                        distance <= matchTolerance)
                    |> Seq.toArray

                if close.Length = 0 then
                    0, Double.PositiveInfinity, float binDx, float binDy
                else
                    let meanDx = close |> Array.averageBy fst
                    let meanDy = close |> Array.averageBy snd
                    let residual =
                        close
                        |> Array.averageBy (fun (dx, dy) -> (dx - meanDx) ** 2.0 + (dy - meanDy) ** 2.0)
                    close.Length, residual, meanDx, meanDy

            let _, _, dx, dy =
                votes
                |> Seq.map scoreBucket
                |> Seq.sortBy (fun (count, residual, _, _) -> -count, residual)
                |> Seq.head

            dx, dy

let private dogMatchesNearTranslation matchTolerance dx dy fixedPoints movingPoints : StackRansac.PointMatch2D list =
    movingPoints
    |> List.choose (fun moving ->
        fixedPoints
        |> List.map (fun fixedPoint ->
            let predictedX = moving.X + dx
            let predictedY = moving.Y + dy
            let error = sqrt ((fixedPoint.X - predictedX) ** 2.0 + (fixedPoint.Y - predictedY) ** 2.0)
            fixedPoint, error)
        |> List.sortBy snd
        |> function
            | (fixedPoint, error) :: _ when error <= matchTolerance ->
                Some
                    { StackRansac.FixedX = fixedPoint.X
                      FixedY = fixedPoint.Y
                      MovingX = moving.X
                      MovingY = moving.Y }
            | _ -> None)

let private estimateSiftAffineMatrixFromPrepared maxShift nearestNeighborRatio ransacIterations ransacMaxError ransacMinInlierRatio fixedImage movingImage =
    let matches =
        matchSiftFeatures nearestNeighborRatio fixedImage.SiftFeatures.Value movingImage.SiftFeatures.Value
        |> List.filter (fun matchItem ->
            abs (matchItem.Fixed.X - matchItem.Moving.X) <= float maxShift
            && abs (matchItem.Fixed.Y - matchItem.Moving.Y) <= float maxShift)

    let pointMatches: StackRansac.PointMatch2D list =
        matches
        |> List.map (fun matchItem ->
            { FixedX = matchItem.Fixed.X
              FixedY = matchItem.Fixed.Y
              MovingX = matchItem.Moving.X
              MovingY = matchItem.Moving.Y })

    match StackRansac.affine2DRansac ransacIterations ransacMaxError ransacMinInlierRatio 12345 pointMatches with
    | Some matrix -> matrix
    | None ->
        let dx, dy = estimateSsdTranslationPixels maxShift fixedImage.Pixels.Value movingImage.Pixels.Value
        translationMatrix dx dy

let private singleSliceManifest width height transform =
    { Version = 1
      Width = width
      Height = height
      Transforms = [ transform ] }

let serialEstTrans<'T when 'T: equality> searchRadius (method: string) scale pixelFraction : Stage<Image<'T>, Image<'T> * SerialSliceManifest> =
    if searchRadius < 0 then invalidArg "searchRadius" "searchRadius must be non-negative."
    let methodName = method.Trim().ToLowerInvariant()
    if methodName <> "dogaffine" && methodName <> "ssdaffine" && methodName <> "siftaffine" then
        invalidArg "method" "serialEstTrans method must be dogAffine, siftAffine, or SSDAffine."
    if scale <= 0.0 then invalidArg "scale" "serialEstTrans scale must be positive."
    if pixelFraction <= 0.0 || pixelFraction > 1.0 then invalidArg "pixelFraction" "serialEstTrans pixelFraction must be in (0, 1]."

    let sigma0 = scale
    let scaleFactor = sqrt 2.0
    let scaleLevels = 4u
    let contrastThreshold = 0.03
    let maxKeypoints = 50u
    let matchTolerance = max 1.5 (2.0 * scale)
    let maxIterations = 60
    let initialLinearStep = 0.05
    let initialTranslationStep = max 1.0 scale
    let minStep = 0.0001
    let stepShrink = 0.5
    let descriptorSize = 4u
    let orientationBins = 8u
    let nearestNeighborRatio = 0.8
    let ransacIterations = 200
    let ransacMaxError = max 2.0 (2.0 * scale)
    let ransacMinInlierRatio = 0.05
    let ssdResolutionScale = max 1.0 (round scale)

    let prepareImage image =
        let pixels = lazy (imageToArray image)
        let dogPoints =
            lazy (siftKeypoints2D sigma0 scaleFactor scaleLevels contrastThreshold (int maxKeypoints) image)

        { Image = image
          Pixels = pixels
          SsdPixels =
              lazy
                  (if ssdResolutionScale <= 1.0 then
                       pixels.Value
                   else
                       downsampleSmoothed ssdResolutionScale pixels.Value)
          DogPoints = dogPoints
          SiftFeatures =
              lazy (siftFeaturesFromPoints (int descriptorSize) (int orientationBins) pixels.Value dogPoints.Value) }

    let apply (_debug: bool) (input: AsyncSeq<Image<'T>>) =
        asyncSeq {
            let mutable width = None
            let mutable height = None
            let mutable cumulative = identityMatrix
            let mutable previous = None

            try
                for image in input do
                    match width, height with
                    | None, None ->
                        width <- Some(image.GetWidth())
                        height <- Some(image.GetHeight())
                    | Some expectedWidth, Some expectedHeight ->
                        if image.GetWidth() <> expectedWidth || image.GetHeight() <> expectedHeight then
                            invalidOp "serialEstTrans expects all slices to have the same shape."
                    | _ ->
                        invalidOp "serialEstTrans has inconsistent cached image dimensions."

                    let currentPrepared = prepareImage image

                    match previous with
                    | None -> ()
                    | Some previousImage ->
                        let pairwiseTransform =
                            match methodName with
                            | "ssdaffine" ->
                                estimateSsdAffineMatrixFromPrepared
                                    searchRadius
                                    maxIterations
                                    initialLinearStep
                                    initialTranslationStep
                                    minStep
                                    stepShrink
                                    pixelFraction
                                    ssdResolutionScale
                                    previousImage
                                    currentPrepared
                            | "siftaffine" ->
                                estimateSiftAffineMatrixFromPrepared
                                    searchRadius
                                    nearestNeighborRatio
                                    ransacIterations
                                    ransacMaxError
                                    ransacMinInlierRatio
                                    previousImage
                                    currentPrepared
                            | _ ->
                                let dx, dy =
                                    estimateKeypointTranslation
                                        searchRadius
                                        matchTolerance
                                        previousImage.DogPoints.Value
                                        currentPrepared.DogPoints.Value

                                if previousImage.DogPoints.Value.Length < 3 || currentPrepared.DogPoints.Value.Length < 3 then
                                    translationMatrix dx dy
                                else
                                    let matches =
                                        dogMatchesNearTranslation
                                            matchTolerance
                                            dx
                                            dy
                                            previousImage.DogPoints.Value
                                            currentPrepared.DogPoints.Value

                                    match StackRansac.affine2DRansac ransacIterations ransacMaxError ransacMinInlierRatio 12345 matches with
                                    | Some matrix -> matrix
                                    | None -> translationMatrix dx dy
                        cumulative <- multiplyMatrix cumulative pairwiseTransform
                        previousImage.Image.decRefCount()

                    image.incRefCount()
                    previous <- Some currentPrepared

                    let transform =
                        { Slice = image.index
                          Matrix = cumulative }

                    yield image, singleSliceManifest (Option.get width) (Option.get height) transform
            finally
                previous |> Option.iter (fun prepared -> prepared.Image.decRefCount())
        }

    let transition = ProfileTransition.create Streaming Streaming
    let pipe = { Name = "serialEstTrans"; Apply = apply; Profile = Streaming }
    Stage.fromPipe "serialEstTrans" transition id id pipe

let private transformForSlice manifest slice =
    manifest.Transforms
    |> List.tryFind (fun transform -> transform.Slice = slice)
    |> Option.orElseWith (fun () -> manifest.Transforms |> List.tryItem slice)
    |> Option.map _.Matrix
    |> Option.defaultValue identityMatrix

let private sampleBilinear background (pixels: 'T[,]) x y =
    let width = Array2D.length1 pixels
    let height = Array2D.length2 pixels
    if x < 0.0 || y < 0.0 || x > float (width - 1) || y > float (height - 1) then
        background
    else
        let x0 = int (floor x)
        let y0 = int (floor y)
        let x1 = min (width - 1) (x0 + 1)
        let y1 = min (height - 1) (y0 + 1)
        let tx = x - float x0
        let ty = y - float y0
        let v00 = pixels[x0, y0] |> toDouble
        let v10 = pixels[x1, y0] |> toDouble
        let v01 = pixels[x0, y1] |> toDouble
        let v11 = pixels[x1, y1] |> toDouble
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11

let private manifestBounds manifest =
    let width = float manifest.Width
    let height = float manifest.Height
    let corners = [ 0.0, 0.0; width - 1.0, 0.0; 0.0, height - 1.0; width - 1.0, height - 1.0 ]
    let transformed =
        manifest.Transforms
        |> List.collect (fun transform ->
            corners |> List.map (fun (x, y) -> transformPoint transform.Matrix x y))

    let xs = transformed |> List.map fst
    let ys = transformed |> List.map snd
    let minX = floor (List.min xs)
    let minY = floor (List.min ys)
    let maxX = ceil (List.max xs)
    let maxY = ceil (List.max ys)
    minX, minY, uint (maxX - minX + 1.0), uint (maxY - minY + 1.0)

let private applyManifestSlice manifest expand background (image: Image<'T>) =
    try
        let minX, minY, outputWidth, outputHeight =
            if expand then manifestBounds manifest
            else 0.0, 0.0, manifest.Width, manifest.Height

        let matrix = transformForSlice manifest image.index
        let inverse = invertMatrix matrix
        let output = Array2D.zeroCreate<'T> (int outputWidth) (int outputHeight)
        let pixels = image.toArray2D()

        for y in 0u .. outputHeight - 1u do
            for x in 0u .. outputWidth - 1u do
                let referenceX = float x + minX
                let referenceY = float y + minY
                let inputX, inputY = transformPoint inverse referenceX referenceY
                output[int x, int y] <- sampleBilinear background pixels inputX inputY |> fromDouble<'T>

        Image<'T>.ofArray2D(output, "serialApplyTrans", image.index)
    finally
        image.decRefCount()

let serialApplyTrans<'T when 'T: equality> background : Stage<Image<'T> * SerialSliceManifest, Image<'T>> =
    Stage.map "serialApplyTrans" (fun _ (image, manifest) -> applyManifestSlice manifest false background image) id id

let serialTransImage<'T when 'T: equality> : Stage<Image<'T> * SerialSliceManifest, Image<'T>> =
    Stage.map "serialTransImage" (fun _ (image, _manifest) -> image) id id

let serialTransManifest<'T when 'T: equality> : Stage<Image<'T> * SerialSliceManifest, SerialSliceManifest> =
    Stage.map
        "serialTransManifest"
        (fun _ (image, manifest) ->
            image.decRefCount()
            manifest)
        id
        id

let serialApplyManifestInBoundingBox<'T when 'T: equality> manifest background : Stage<Image<'T>, Image<'T>> =
    Stage.map "serialApplyManifestInBoundingBox" (fun _ image -> applyManifestSlice manifest true background image) id id
