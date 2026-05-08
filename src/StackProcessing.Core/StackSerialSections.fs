module StackSerialSections

open System
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
    let terms = terms2D order
    let n = terms.Length
    let normal = Array2D.zeroCreate<float> n n
    let right = Array.zeroCreate<float> n

    for y in 0u .. height - 1u do
        for x in 0u .. width - 1u do
            let values = basis2D terms width height x y |> List.toArray
            let intensity = image.Get [ x; y ] |> toDouble
            for row in 0 .. n - 1 do
                right[row] <- right[row] + values[row] * intensity
                for col in 0 .. n - 1 do
                    normal[row, col] <- normal[row, col] + values[row] * values[col]

    terms, solveLinearSystem normal right

let private evalPolynomial2D terms coefficients width height x y =
    basis2D terms width height x y
    |> List.zip (Array.toList coefficients)
    |> List.sumBy (fun (c, b) -> c * b)

let serialPolynomialBiasCorrect<'T when 'T: equality> order : Stage<Image<'T>, Image<float>> =
    Stage.map
        "serialPolynomialBiasCorrect"
        (fun _ image ->
            try
                let width = image.GetWidth()
                let height = image.GetHeight()
                let terms, coefficients = fitPolynomial2D order image
                let output = Array2D.zeroCreate<float> (int width) (int height)

                for y in 0u .. height - 1u do
                    for x in 0u .. width - 1u do
                        let corrected =
                            (image.Get [ x; y ] |> toDouble)
                            - evalPolynomial2D terms coefficients width height x y
                        output[int x, int y] <- corrected

                Image<float>.ofArray2D(output, "serialPolynomialBiasCorrect", image.index)
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
    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    Array2D.init width height (fun x y -> image.Get [ uint x; uint y ] |> toDouble)

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

let serialKeypointTranslationManifest width height : Stage<PointSet, SerialSliceManifest> =
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

    Stage.reduce "serialKeypointTranslationManifest" reducer Streaming id (fun _ -> 1UL)

let private estimateTranslation maxShift (fixedImage: Image<'T>) (movingImage: Image<'T>) =
    let width = int (fixedImage.GetWidth())
    let height = int (fixedImage.GetHeight())
    if movingImage.GetWidth() <> uint width || movingImage.GetHeight() <> uint height then
        invalidOp "serialImageTranslationManifest expects all slices to have the same shape."

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

let serialImageTranslationManifest<'T when 'T: equality> maxShift : Stage<Image<'T>, SerialSliceManifest> =
    if maxShift < 0 then invalidArg "maxShift" "maxShift must be non-negative."

    let reducer (_debug: bool) (input: AsyncSeq<Image<'T>>) =
        async {
            let! images = input |> AsyncSeq.toListAsync
            match images with
            | [] -> return serialIdentityManifest 1u 1u 0u
            | first :: _ ->
                let width = first.GetWidth()
                let height = first.GetHeight()
                let mutable cumulative = identityMatrix
                let mutable previous = None

                let transforms =
                    images
                    |> List.map (fun image ->
                        try
                            match previous with
                            | None -> ()
                            | Some previousImage ->
                                let dx, dy = estimateTranslation maxShift previousImage image
                                cumulative <- multiplyMatrix cumulative (translationMatrix dx dy)

                            previous |> Option.iter (fun old -> old.decRefCount())
                            image.incRefCount()
                            previous <- Some image
                            { Slice = image.index
                              Matrix = cumulative }
                        with
                        | ex ->
                            previous |> Option.iter (fun old -> old.decRefCount())
                            images |> List.iter (fun img -> img.decRefCount())
                            raise ex)

                previous |> Option.iter (fun old -> old.decRefCount())
                images |> List.iter (fun image -> image.decRefCount())
                return
                    { Version = 1
                      Width = width
                      Height = height
                      Transforms = transforms }
        }

    Stage.reduce "serialImageTranslationManifest" reducer Streaming id (fun _ -> 1UL)

let private transformForSlice manifest slice =
    manifest.Transforms
    |> List.tryFind (fun transform -> transform.Slice = slice)
    |> Option.orElseWith (fun () -> manifest.Transforms |> List.tryItem slice)
    |> Option.map _.Matrix
    |> Option.defaultValue identityMatrix

let private sampleBilinear background (image: Image<'T>) x y =
    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    if x < 0.0 || y < 0.0 || x > float (width - 1) || y > float (height - 1) then
        background
    else
        let x0 = int (floor x)
        let y0 = int (floor y)
        let x1 = min (width - 1) (x0 + 1)
        let y1 = min (height - 1) (y0 + 1)
        let tx = x - float x0
        let ty = y - float y0
        let v00 = image.Get [ uint x0; uint y0 ] |> toDouble
        let v10 = image.Get [ uint x1; uint y0 ] |> toDouble
        let v01 = image.Get [ uint x0; uint y1 ] |> toDouble
        let v11 = image.Get [ uint x1; uint y1 ] |> toDouble
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
        let output = Array2D.zeroCreate<float> (int outputWidth) (int outputHeight)

        for y in 0u .. outputHeight - 1u do
            for x in 0u .. outputWidth - 1u do
                let referenceX = float x + minX
                let referenceY = float y + minY
                let inputX, inputY = transformPoint inverse referenceX referenceY
                output[int x, int y] <- sampleBilinear background image inputX inputY

        Image<float>.ofArray2D(output, "serialApplyManifest", image.index)
    finally
        image.decRefCount()

let serialApplyManifest<'T when 'T: equality> manifest background : Stage<Image<'T>, Image<float>> =
    Stage.map "serialApplyManifest" (fun _ image -> applyManifestSlice manifest false background image) id id

let serialApplyManifestInBoundingBox<'T when 'T: equality> manifest background : Stage<Image<'T>, Image<float>> =
    Stage.map "serialApplyManifestInBoundingBox" (fun _ image -> applyManifestSlice manifest true background image) id id
