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

let private estimateSsdTranslation maxShift (fixedImage: Image<'T>) (movingImage: Image<'T>) =
    let width = int (fixedImage.GetWidth())
    let height = int (fixedImage.GetHeight())
    if movingImage.GetWidth() <> uint width || movingImage.GetHeight() <> uint height then
        invalidOp "serialEstTrans expects all slices to have the same shape."

    let fixedPixels = imageToArray fixedImage
    let moving = imageToArray movingImage
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

let private estimateKeypointTranslation maxShift sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints matchTolerance (fixedImage: Image<'T>) (movingImage: Image<'T>) =
    let width = int (fixedImage.GetWidth())
    let height = int (fixedImage.GetHeight())
    if movingImage.GetWidth() <> uint width || movingImage.GetHeight() <> uint height then
        invalidOp "serialEstTrans expects all slices to have the same shape."

    let fixedPoints = siftKeypoints2D sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints fixedImage
    let movingPoints = siftKeypoints2D sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints movingImage

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

let private toRegistrationPoint (point: CoordinatePoint) =
    { point with Z = 0.0 }

let private affineToSerialMatrix transform =
    let matrix = StackRegistration.affineToMatrix transform |> unvectorizeMatrix

    [ [ matrix[0, 0]; matrix[0, 1]; matrix[0, 3] ]
      [ matrix[1, 0]; matrix[1, 1]; matrix[1, 3] ]
      [ 0.0; 0.0; 1.0 ] ]

let private estimateAffineMatrix maxShift sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints matchTolerance registrationOptions fixedImage movingImage =
    let dx, dy =
        estimateKeypointTranslation
            maxShift
            sigma0
            scaleFactor
            scaleLevels
            contrastThreshold
            maxKeypoints
            matchTolerance
            fixedImage
            movingImage

    let fixedPoints = siftKeypoints2D sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints fixedImage
    let movingPoints = siftKeypoints2D sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints movingImage

    if fixedPoints.Length < 3 || movingPoints.Length < 3 then
        translationMatrix dx dy
    else
        let nearTranslation point =
            movingPoints
            |> List.exists (fun moving ->
                abs ((point.X - moving.X) - dx) <= matchTolerance
                && abs ((point.Y - moving.Y) - dy) <= matchTolerance)

        let fixedCandidates =
            fixedPoints
            |> List.filter nearTranslation
            |> function
                | [] -> fixedPoints
                | points -> points

        let movingCandidates =
            movingPoints
            |> List.filter (fun moving ->
                fixedPoints
                |> List.exists (fun fixedPoint ->
                    abs ((fixedPoint.X - moving.X) - dx) <= matchTolerance
                    && abs ((fixedPoint.Y - moving.Y) - dy) <= matchTolerance))
            |> function
                | [] -> movingPoints
                | points -> points

        try
            let result =
                StackRegistration.affineRegistration
                    registrationOptions
                    (fixedCandidates |> List.map toRegistrationPoint)
                    (movingCandidates |> List.map toRegistrationPoint)

            affineToSerialMatrix result.Transform
        with _ ->
            translationMatrix dx dy

let private singleSliceManifest width height transform =
    { Version = 1
      Width = width
      Height = height
      Transforms = [ transform ] }

let serialEstTrans<'T when 'T: equality> maxShift (method: string) sigma0 scaleFactor scaleLevels contrastThreshold maxKeypoints matchTolerance maxIterations initialLinearStep initialTranslationStep minStep stepShrink : Stage<Image<'T>, Image<'T> * SerialSliceManifest> =
    if maxShift < 0 then invalidArg "maxShift" "maxShift must be non-negative."
    let methodName = method.Trim().ToLowerInvariant()
    if methodName <> "siftaffine" && methodName <> "sift" && methodName <> "ssdtranslation" && methodName <> "ssd" then
        invalidArg "method" "serialEstTrans method must be SiftAffine or SSDTranslation."
    if sigma0 <= 0.0 then invalidArg "sigma0" "serialEstTrans sigma0 must be positive."
    if scaleFactor <= 1.0 then invalidArg "scaleFactor" "serialEstTrans scaleFactor must be greater than 1."
    if scaleLevels < 4u then invalidArg "scaleLevels" "serialEstTrans needs at least 4 Gaussian scale levels."
    if maxKeypoints = 0u then invalidArg "maxKeypoints" "serialEstTrans maxKeypoints must be positive."
    if matchTolerance <= 0.0 then invalidArg "matchTolerance" "serialEstTrans matchTolerance must be positive."
    if maxIterations <= 0 then invalidArg "maxIterations" "serialEstTrans maxIterations must be positive."
    if initialLinearStep <= 0.0 then invalidArg "initialLinearStep" "serialEstTrans initialLinearStep must be positive."
    if initialTranslationStep <= 0.0 then invalidArg "initialTranslationStep" "serialEstTrans initialTranslationStep must be positive."
    if minStep <= 0.0 then invalidArg "minStep" "serialEstTrans minStep must be positive."
    if stepShrink <= 0.0 || stepShrink >= 1.0 then invalidArg "stepShrink" "serialEstTrans stepShrink must be between 0 and 1."
    let registrationOptions =
        { StackRegistration.defaultAffineRegistrationOptions with
            MaxIterations = maxIterations
            InitialLinearStep = initialLinearStep
            InitialTranslationStep = initialTranslationStep
            MinStep = minStep
            StepShrink = stepShrink }

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

                    match previous with
                    | None -> ()
                    | Some previousImage ->
                        let pairwiseTransform =
                            match methodName with
                            | "ssd"
                            | "ssdtranslation" ->
                                let dx, dy = estimateSsdTranslation maxShift previousImage image
                                translationMatrix dx dy
                            | _ ->
                                estimateAffineMatrix
                                    maxShift
                                    sigma0
                                    scaleFactor
                                    scaleLevels
                                    contrastThreshold
                                    (int maxKeypoints)
                                    matchTolerance
                                    registrationOptions
                                    previousImage
                                    image
                        cumulative <- multiplyMatrix cumulative pairwiseTransform
                        previousImage.decRefCount()

                    image.incRefCount()
                    previous <- Some image

                    let transform =
                        { Slice = image.index
                          Matrix = cumulative }

                    yield image, singleSliceManifest (Option.get width) (Option.get height) transform
            finally
                previous |> Option.iter (fun image -> image.decRefCount())
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
