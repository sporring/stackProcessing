module StackSerialSections

open System
open System.Collections.Generic
open System.Globalization
open FSharp.Control
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

[<CLIMutable>]
type SerialVolumeGeometry =
    { Version: int
      MinX: float
      MinY: float
      Width: uint
      Height: uint
      Depth: uint }

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

let private singleSliceManifest width height transform =
    { Version = 1
      Width = width
      Height = height
      Transforms = [ transform ] }

let private transformForSlice manifest slice =
    manifest.Transforms
    |> List.tryFind (fun transform -> transform.Slice = slice)
    |> Option.orElseWith (fun () -> manifest.Transforms |> List.tryItem slice)
    |> Option.map _.Matrix
    |> Option.defaultValue identityMatrix

let private manifestBounds (manifest: SerialSliceManifest) =
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

let private manifestGeometry (manifest: SerialSliceManifest) =
    let minX, minY, width, height = manifestBounds manifest
    { Version = 1
      MinX = minX
      MinY = minY
      Width = width
      Height = height
      Depth = uint manifest.Transforms.Length }

let private geometryFromManifestSlice (manifest: SerialSliceManifest) =
    { Version = 1
      MinX = 0.0
      MinY = 0.0
      Width = manifest.Width
      Height = manifest.Height
      Depth = 1u }

let private chunkShape2D name (chunk: Chunk<'T>) =
    let width64, height64, depth64 = chunk.Size
    if depth64 <> 1UL then
        invalidArg "chunk" $"{name} expects 2D slice chunks with depth 1, got {chunk.Size}."
    if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
        invalidArg "chunk" $"{name} dimensions must fit Int32, got {chunk.Size}."
    int width64, int height64

let private estimateTranslationSsd<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    searchRadius
    pixelFraction
    (fixedChunk: Chunk<'T>)
    (movingChunk: Chunk<'T>) =
    let width, height = chunkShape2D "chunkSerialEstTrans" fixedChunk
    let movingWidth, movingHeight = chunkShape2D "chunkSerialEstTrans" movingChunk
    if movingWidth <> width || movingHeight <> height then
        invalidOp "chunkSerialEstTrans expects all slices to have the same shape."

    let fixedPixels = Chunk.span fixedChunk
    let movingPixels = Chunk.span movingChunk
    let radius = int searchRadius
    let step = max 1 (int (round (1.0 / max 1.0e-6 pixelFraction)))
    let mutable bestDx = 0
    let mutable bestDy = 0
    let mutable bestScore = Double.PositiveInfinity

    for dy in -radius .. radius do
        for dx in -radius .. radius do
            let mutable score = 0.0
            let mutable count = 0
            for y in 0 .. step .. height - 1 do
                let my = y - dy
                if my >= 0 && my < height then
                    for x in 0 .. step .. width - 1 do
                        let mx = x - dx
                        if mx >= 0 && mx < width then
                            let a = toDouble fixedPixels[Chunk.toIndex width height x y 0]
                            let b = toDouble movingPixels[Chunk.toIndex width height mx my 0]
                            let d = a - b
                            score <- score + d * d
                            count <- count + 1
            if count > 0 then
                let normalized = score / float count
                if normalized < bestScore then
                    bestScore <- normalized
                    bestDx <- dx
                    bestDy <- dy

    float bestDx, float bestDy

let serialEstTransChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    searchRadius
    (_method: string)
    (_scale: float)
    pixelFraction
    : Stage<Chunk<'T>, Chunk<'T> * SerialSliceManifest> =
    if searchRadius < 0 then invalidArg "searchRadius" "chunkSerialEstTrans searchRadius must be non-negative."
    if pixelFraction <= 0.0 || pixelFraction > 1.0 then invalidArg "pixelFraction" "chunkSerialEstTrans pixelFraction must be in (0, 1]."

    let mutable previous: Chunk<'T> option = None
    let mutable sliceIndex = 0

    let mapper _debug (chunk: Chunk<'T>) =
        let width, height = chunkShape2D "chunkSerialEstTrans" chunk
        let matrix =
            match previous with
            | None -> identityMatrix
            | Some fixedChunk ->
                let dx, dy = estimateTranslationSsd searchRadius pixelFraction fixedChunk chunk
                translationMatrix dx dy

        previous |> Option.iter Chunk.decRef
        Chunk.incRef chunk |> ignore
        previous <- Some chunk

        let manifest =
            { Version = 1
              Width = uint width
              Height = uint height
              Transforms =
                [ { Slice = sliceIndex
                    Matrix = matrix } ] }

        sliceIndex <- sliceIndex + 1
        chunk, manifest

    Stage.map "chunkSerialEstTrans" mapper id id

let serialEstBoundingBoxChunk<'T when 'T: equality> : Stage<Chunk<'T> * SerialSliceManifest, SerialVolumeGeometry> =
    let reducer (_debug: bool) (input: AsyncSeq<Chunk<'T> * SerialSliceManifest>) =
        async {
            let mutable minX = Double.PositiveInfinity
            let mutable minY = Double.PositiveInfinity
            let mutable maxX = Double.NegativeInfinity
            let mutable maxY = Double.NegativeInfinity
            let mutable depth = 0u

            let includePoint x y =
                minX <- min minX x
                minY <- min minY y
                maxX <- max maxX x
                maxY <- max maxY y

            for chunk, manifest in input do
                try
                    let matrix =
                        match manifest.Transforms with
                        | transform :: _ -> transform.Matrix
                        | [] -> identityMatrix
                    let width = float manifest.Width
                    let height = float manifest.Height
                    let corners = [ 0.0, 0.0; width - 1.0, 0.0; 0.0, height - 1.0; width - 1.0, height - 1.0 ]
                    for x, y in corners do
                        let tx, ty = transformPoint matrix x y
                        includePoint tx ty
                    depth <- depth + 1u
                finally
                    Chunk.decRef chunk

            if depth = 0u then
                return { Version = 1; MinX = 0.0; MinY = 0.0; Width = 0u; Height = 0u; Depth = 0u }
            else
                let minX = floor minX
                let minY = floor minY
                let maxX = ceil maxX
                let maxY = ceil maxY
                return
                    { Version = 1
                      MinX = minX
                      MinY = minY
                      Width = uint (maxX - minX + 1.0)
                      Height = uint (maxY - minY + 1.0)
                      Depth = depth }
        }

    Stage.reduce "chunkSerialEstBoundingBox" reducer Streaming id (fun _ -> 1UL)

let private applyManifestChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (manifest: SerialSliceManifest)
    (geometry: SerialVolumeGeometry option)
    background
    (chunk: Chunk<'T>) =
    try
        let inputWidth, inputHeight = chunkShape2D "chunkSerialApplyTrans" chunk
        let geometry = geometry |> Option.defaultValue (geometryFromManifestSlice manifest)
        let matrix =
            match manifest.Transforms with
            | transform :: _ -> transform.Matrix
            | [] -> identityMatrix
        let inverse = invertMatrix matrix
        let output = Chunk.create<'T> (uint64 geometry.Width, uint64 geometry.Height, 1UL)
        try
            let inputPixels = Chunk.span chunk
            let outputPixels = Chunk.span output
            let outputWidth = int geometry.Width
            let outputHeight = int geometry.Height
            for y in 0 .. outputHeight - 1 do
                for x in 0 .. outputWidth - 1 do
                    let worldX = float x + geometry.MinX
                    let worldY = float y + geometry.MinY
                    let sx, sy = transformPoint inverse worldX worldY
                    let ix = int (round sx)
                    let iy = int (round sy)
                    let value =
                        if ix < 0 || iy < 0 || ix >= inputWidth || iy >= inputHeight then
                            background
                        else
                            inputPixels[Chunk.toIndex inputWidth inputHeight ix iy 0]
                    outputPixels[Chunk.toIndex outputWidth outputHeight x y 0] <- value
            output
        with
        | _ ->
            Chunk.decRef output
            reraise()
    finally
        Chunk.decRef chunk

let serialApplyTransChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    background
    geometry
    : Stage<Chunk<'T> * SerialSliceManifest, Chunk<'T>> =
    Stage.map "chunkSerialApplyTrans" (fun _ (chunk, manifest) -> applyManifestChunk manifest geometry background chunk) id id

let serialTransChunk<'T when 'T: equality> : Stage<Chunk<'T> * SerialSliceManifest, Chunk<'T>> =
    Stage.map "chunkSerialTransChunk" (fun _ (chunk, _manifest) -> chunk) id id

let serialTransManifestChunk<'T when 'T: equality> : Stage<Chunk<'T> * SerialSliceManifest, SerialSliceManifest> =
    Stage.map
        "chunkSerialTransManifest"
        (fun _ (chunk, manifest) ->
            Chunk.decRef chunk
            manifest)
        id
        id

let serialApplyManifestInBoundingBoxChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    manifest
    background
    : Stage<Chunk<'T>, Chunk<'T>> =
    Stage.map "chunkSerialApplyManifestInBoundingBox" (fun _ chunk -> applyManifestChunk manifest (Some(manifestGeometry manifest)) background chunk) id id
