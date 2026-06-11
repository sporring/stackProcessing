module StackBias

open System
open System.Globalization
open System.Runtime.InteropServices
open FSharp.Control
open Image
open Image.InternalHelpers
open SlimPipeline
open StackCore

type BiasPolynomialTerm =
    { XPower: int
      YPower: int
      ZPower: int }

type BiasPolynomialModel =
    { Order: int
      Width: uint
      Height: uint
      Depth: uint
      Terms: BiasPolynomialTerm list
      Coefficients: float list }

let private toDouble value =
    Convert.ToDouble(box value, CultureInfo.InvariantCulture)

let private polynomialTerms order =
    if order < 0 then invalidArg "order" "Polynomial order must be non-negative."

    [ for total in 0 .. order do
        for xPower in 0 .. total do
            for yPower in 0 .. total - xPower do
                let zPower = total - xPower - yPower
                { XPower = xPower
                  YPower = yPower
                  ZPower = zPower } ]

let private normalizeCoordinate size value =
    if size <= 1u then 0.0
    else 2.0 * float value / float (size - 1u) - 1.0

let private coordinatePowers size order =
    Array.init (int size) (fun i ->
        let value = normalizeCoordinate size (uint i)
        let powers = Array.zeroCreate<float> (order + 1)
        powers[0] <- 1.0

        for p in 1 .. order do
            powers[p] <- powers[p - 1] * value

        powers)

let private basis terms width height depth x y z =
    let xn = normalizeCoordinate width x
    let yn = normalizeCoordinate height y
    let zn = normalizeCoordinate depth z

    terms
    |> List.map (fun term ->
        (xn ** float term.XPower)
        * (yn ** float term.YPower)
        * (zn ** float term.ZPower))

let private basisValues
    (terms: BiasPolynomialTerm[])
    (xPowers: float[][])
    (yPowers: float[][])
    (zPowers: float[][])
    x
    y
    z =

    Array.init terms.Length (fun i ->
        let term = terms[i]
        xPowers[x][term.XPower] * yPowers[y][term.YPower] * zPowers[z][term.ZPower])

let private evaluate model x y z =
    basis model.Terms model.Width model.Height model.Depth x y z
    |> List.zip model.Coefficients
    |> List.sumBy (fun (coefficient, value) -> coefficient * value)

let private evaluateValues
    (terms: BiasPolynomialTerm[])
    (coefficients: float[])
    (xPowers: float[][])
    (yPowers: float[][])
    (zPowers: float[][])
    x
    y
    z =

    let mutable sum = 0.0

    for i in 0 .. terms.Length - 1 do
        let term = terms[i]
        sum <- sum + coefficients[i] * xPowers[x][term.XPower] * yPowers[y][term.YPower] * zPowers[z][term.ZPower]

    sum

let private solveLinearSystem (a: float[,]) (b: float[]) =
    let n = b.Length
    let m = Array2D.copy a
    let rhs = Array.copy b
    let ridge = 1.0e-10

    for i in 0 .. n - 1 do
        m[i, i] <- m[i, i] + ridge

    for k in 0 .. n - 1 do
        let mutable pivot = k
        let mutable pivotValue = abs m[k, k]

        for row in k + 1 .. n - 1 do
            let value = abs m[row, k]
            if value > pivotValue then
                pivot <- row
                pivotValue <- value

        if pivotValue < 1.0e-18 then
            invalidOp "Bias polynomial fit is singular. Use a lower order or more sampled pixels."

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
                if factor <> 0.0 then
                    for col in k .. n - 1 do
                        m[row, col] <- m[row, col] - factor * m[k, col]
                    rhs[row] <- rhs[row] - factor * rhs[k]

    rhs

type private BiasFitState =
    { Order: int
      Depth: uint
      Terms: BiasPolynomialTerm list
      Normal: float[,]
      Right: float[]
      mutable Width: uint option
      mutable Height: uint option
      mutable Count: uint64 }

let private emptyFitState order depth =
    if depth = 0u then invalidArg "depth" "Bias model depth must be positive."

    let terms = polynomialTerms order
    let n = terms.Length
    { Order = order
      Depth = depth
      Terms = terms
      Normal = Array2D.zeroCreate n n
      Right = Array.zeroCreate n
      Width = None
      Height = None
      Count = 0UL }

let private observePixelValues state (values: float[]) value =
    for row in 0 .. values.Length - 1 do
        state.Right[row] <- state.Right[row] + values[row] * value
        for col in 0 .. values.Length - 1 do
            state.Normal[row, col] <- state.Normal[row, col] + values[row] * values[col]

    state.Count <- state.Count + 1UL

let private ensureShape state (image: Image<'T>) =
    let width = image.GetWidth()
    let height = image.GetHeight()

    match state.Width, state.Height with
    | None, None ->
        state.Width <- Some width
        state.Height <- Some height
    | Some expectedWidth, Some expectedHeight when expectedWidth = width && expectedHeight = height -> ()
    | Some expectedWidth, Some expectedHeight ->
        invalidOp $"Bias model fit expected {expectedWidth}x{expectedHeight} slices, got {width}x{height} at slice {image.index}."
    | _ ->
        invalidOp "Bias model fit has inconsistent shape state."

    width, height

let private ensureChunkShape state (chunk: Chunk<'T>) =
    let width, height, chunkDepth = chunk.Size
    if chunkDepth <> 1UL then
        invalidOp $"Bias model fit expects 2D slice chunks with depth 1, got {chunk.Size}."

    let width = uint width
    let height = uint height

    match state.Width, state.Height with
    | None, None ->
        state.Width <- Some width
        state.Height <- Some height
    | Some expectedWidth, Some expectedHeight when expectedWidth = width && expectedHeight = height -> ()
    | Some expectedWidth, Some expectedHeight ->
        invalidOp $"Bias model fit expected {expectedWidth}x{expectedHeight} chunks, got {width}x{height}."
    | _ ->
        invalidOp "Bias model fit has inconsistent shape state."

    width, height

let private addImageObservations state (image: Image<'T>) mask =
    try
        let width, height = ensureShape state image
        let z = uint (max 0 image.index)

        if z >= state.Depth then
            invalidOp $"Bias model fit got slice index {image.index}, outside declared depth {state.Depth}."

        let pixels = image.toFlatArray()
        let maskPixels = mask |> Option.map (fun (maskImage: Image<uint8>) -> maskImage.toFlatArray())
        let terms = state.Terms |> List.toArray
        let xPowers = coordinatePowers width state.Order
        let yPowers = coordinatePowers height state.Order
        let zPowers = coordinatePowers state.Depth state.Order
        let zIndex = int z

        for y in 0 .. int height - 1 do
            for x in 0 .. int width - 1 do
                let includePixel =
                    match maskPixels with
                    | None -> true
                    | Some maskValues -> maskValues[flatIndex2 (int width) x y] <> 0uy

                if includePixel then
                    let values = basisValues terms xPowers yPowers zPowers x y zIndex
                    observePixelValues state values (pixels[flatIndex2 (int width) x y] |> toDouble)
    finally
        image.decRefCount()
        mask |> Option.iter (fun (maskImage: Image<uint8>) -> maskImage.decRefCount())

    state

let private addChunkObservationsAt<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    z
    (state: BiasFitState)
    (chunk: Chunk<'T>)
    (mask: Chunk<uint8> option)
    =
    try
        let width, height = ensureChunkShape state chunk
        if z >= state.Depth then
            invalidOp $"Bias model fit got slice index {z}, outside declared depth {state.Depth}."

        mask
        |> Option.iter (fun maskChunk ->
            let maskWidth, maskHeight, maskDepth = maskChunk.Size
            if maskDepth <> 1UL || maskWidth <> uint64 width || maskHeight <> uint64 height then
                invalidOp $"fitBiasModelChunkMasked expects image and mask chunks with the same 2D shape at slice {z}.")

        let widthI = int width
        let heightI = int height
        let pixels = Chunk.span<'T> chunk
        let terms = state.Terms |> List.toArray
        let xPowers = coordinatePowers width state.Order
        let yPowers = coordinatePowers height state.Order
        let zPowers = coordinatePowers state.Depth state.Order
        let zIndex = int z

        match mask with
        | None ->
            for y in 0 .. heightI - 1 do
                for x in 0 .. widthI - 1 do
                    let values = basisValues terms xPowers yPowers zPowers x y zIndex
                    observePixelValues state values (pixels[flatIndex2 widthI x y] |> toDouble)
        | Some maskChunk ->
            let maskPixels = Chunk.span<uint8> maskChunk
            for y in 0 .. heightI - 1 do
                for x in 0 .. widthI - 1 do
                    let i = flatIndex2 widthI x y
                    if maskPixels[i] <> 0uy then
                        let values = basisValues terms xPowers yPowers zPowers x y zIndex
                        observePixelValues state values (pixels[i] |> toDouble)
    finally
        Chunk.decRef chunk
        mask |> Option.iter Chunk.decRef

    state

let private stateToModel state =
    if state.Count = 0UL then
        invalidOp "Bias polynomial fit did not observe any pixels. Check the input stream or mask."

    let coefficients = solveLinearSystem state.Normal state.Right |> Array.toList

    { Order = state.Order
      Width = state.Width |> Option.defaultValue 0u
      Height = state.Height |> Option.defaultValue 0u
      Depth = state.Depth
      Terms = state.Terms
      Coefficients = coefficients }

let fitBiasModel<'T when 'T: equality> order depth : Stage<Image<'T>, BiasPolynomialModel> =
    let folder (state: BiasFitState) (image: Image<'T>) =
        addImageObservations state image None

    Stage.fold "fitBiasModel" folder (emptyFitState order depth) id (fun _ -> 1UL)
    --> Stage.map "fitBiasModel: solve" (fun _ state -> stateToModel state) id id

let fitBiasModelMasked<'T when 'T: equality> order depth : Stage<Image<'T> * Image<uint8>, BiasPolynomialModel> =
    let folder (state: BiasFitState) ((image, mask): Image<'T> * Image<uint8>) =
        let imageWidth = image.GetWidth()
        let imageHeight = image.GetHeight()
        if mask.GetWidth() <> imageWidth || mask.GetHeight() <> imageHeight then
            invalidOp $"fitBiasModelMasked expects image and mask slices with the same shape at slice {image.index}."

        addImageObservations state image (Some mask)

    Stage.fold "fitBiasModelMasked" folder (emptyFitState order depth) id (fun _ -> 1UL)
    --> Stage.map "fitBiasModelMasked: solve" (fun _ state -> stateToModel state) id id

let fitBiasModelChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    order
    depth
    : Stage<Chunk<'T>, BiasPolynomialModel> =
    let name = $"fitBiasModelChunk.{typeof<'T>.Name}"
    let memoryNeed n = n * uint64 (Marshal.SizeOf<'T>())
    let apply _debug (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            let! chunks = input |> AsyncSeq.toListAsync
            let state = emptyFitState order depth
            chunks
            |> List.iteri (fun z chunk ->
                addChunkObservationsAt (uint z) state chunk None |> ignore)
            yield stateToModel state
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) memoryNeed (fun _ -> 1UL) pipe
    |> Stage.withSliceCardinality (SliceCardinality.reduceTo 1UL)

let fitBiasModelChunkMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    order
    depth
    : Stage<Chunk<'T> * Chunk<uint8>, BiasPolynomialModel> =
    let name = $"fitBiasModelChunkMasked.{typeof<'T>.Name}"
    let memoryNeed n = n * uint64 (Marshal.SizeOf<'T>() + Marshal.SizeOf<uint8>())
    let apply _debug (input: AsyncSeq<Chunk<'T> * Chunk<uint8>>) =
        asyncSeq {
            let! chunks = input |> AsyncSeq.toListAsync
            let state = emptyFitState order depth
            chunks
            |> List.iteri (fun z (chunk, mask) ->
                addChunkObservationsAt (uint z) state chunk (Some mask) |> ignore)
            yield stateToModel state
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) memoryNeed (fun _ -> 1UL) pipe
    |> Stage.withSliceCardinality (SliceCardinality.reduceTo 1UL)

let private correctedImage (model: BiasPolynomialModel) (image: Image<'T>) mask =
    try
        let width = image.GetWidth()
        let height = image.GetHeight()

        if model.Width <> width || model.Height <> height then
            invalidOp $"Bias correction model expects {model.Width}x{model.Height} slices, got {width}x{height} at slice {image.index}."

        let z = uint (max 0 image.index)
        if z >= model.Depth then
            invalidOp $"Bias correction got slice index {image.index}, outside model depth {model.Depth}."

        let widthI = int width
        let heightI = int height
        let output = Array.zeroCreate<float> (widthI * heightI)
        let pixels = image.toFlatArray()
        let maskPixels = mask |> Option.map (fun (maskImage: Image<uint8>) -> maskImage.toFlatArray())
        let terms = model.Terms |> List.toArray
        let coefficients = model.Coefficients |> List.toArray
        let xPowers = coordinatePowers width model.Order
        let yPowers = coordinatePowers height model.Order
        let zPowers = coordinatePowers model.Depth model.Order
        let zIndex = int z

        for y in 0 .. heightI - 1 do
            for x in 0 .. widthI - 1 do
                let i = flatIndex2 widthI x y
                let input = pixels[i] |> toDouble
                let includePixel =
                    match maskPixels with
                    | None -> true
                    | Some maskValues -> maskValues[i] <> 0uy
                let value =
                    if includePixel then input - evaluateValues terms coefficients xPowers yPowers zPowers x y zIndex
                    else input
                output[i] <- value

        Image<float>.ofFlatArray([ width; height ], output, "correctBias", image.index)
    finally
        image.decRefCount()
        mask |> Option.iter (fun (maskImage: Image<uint8>) -> maskImage.decRefCount())

let private correctedChunkAt<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    z
    (model: BiasPolynomialModel)
    (chunk: Chunk<'T>)
    (mask: Chunk<uint8> option)
    : Chunk<float>
    =
    try
        let width, height, depth = chunk.Size
        if depth <> 1UL then
            invalidOp $"Bias correction expects 2D slice chunks with depth 1, got {chunk.Size}."
        if model.Width <> uint width || model.Height <> uint height then
            invalidOp $"Bias correction model expects {model.Width}x{model.Height} chunks, got {width}x{height} at slice {z}."
        if z >= model.Depth then
            invalidOp $"Bias correction got slice index {z}, outside model depth {model.Depth}."

        mask
        |> Option.iter (fun maskChunk ->
            let maskWidth, maskHeight, maskDepth = maskChunk.Size
            if maskDepth <> 1UL || maskWidth <> width || maskHeight <> height then
                invalidOp $"correctBiasChunkMasked expects image and mask chunks with the same 2D shape at slice {z}.")

        let widthI = int width
        let heightI = int height
        let output = Chunk.create<float> (width, height, 1UL)

        try
            let inputPixels = Chunk.span<'T> chunk
            let outputPixels = Chunk.span<float> output
            let terms = model.Terms |> List.toArray
            let coefficients = model.Coefficients |> List.toArray
            let xPowers = coordinatePowers model.Width model.Order
            let yPowers = coordinatePowers model.Height model.Order
            let zPowers = coordinatePowers model.Depth model.Order
            let zIndex = int z

            match mask with
            | None ->
                for y in 0 .. heightI - 1 do
                    for x in 0 .. widthI - 1 do
                        let i = flatIndex2 widthI x y
                        let input = inputPixels[i] |> toDouble
                        outputPixels[i] <- input - evaluateValues terms coefficients xPowers yPowers zPowers x y zIndex
            | Some maskChunk ->
                let maskPixels = Chunk.span<uint8> maskChunk
                for y in 0 .. heightI - 1 do
                    for x in 0 .. widthI - 1 do
                        let i = flatIndex2 widthI x y
                        let input = inputPixels[i] |> toDouble
                        outputPixels[i] <-
                            if maskPixels[i] <> 0uy then input - evaluateValues terms coefficients xPowers yPowers zPowers x y zIndex
                            else input

            output
        with
        | _ ->
            Chunk.decRef output
            reraise()
    finally
        Chunk.decRef chunk
        mask |> Option.iter Chunk.decRef

let correctBias<'T when 'T: equality> model : Stage<Image<'T>, Image<float>> =
    Stage.map "correctBias" (fun _ image -> correctedImage model image None) id id

let correctBiasMasked<'T when 'T: equality> model : Stage<Image<'T> * Image<uint8>, Image<float>> =
    Stage.map "correctBiasMasked" (fun _ (image, mask) -> correctedImage model image (Some mask)) id id

let correctBiasChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    model
    : Stage<Chunk<'T>, Chunk<float>> =
    let name = $"correctBiasChunk.{typeof<'T>.Name}"
    let memoryNeed n = n * uint64 (Marshal.SizeOf<'T>() + Marshal.SizeOf<float>())
    let apply _debug (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            let mutable z = 0u
            for chunk in input do
                let output = correctedChunkAt z model chunk None
                z <- z + 1u
                yield output
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) memoryNeed id pipe

let correctBiasChunkMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    model
    : Stage<Chunk<'T> * Chunk<uint8>, Chunk<float>> =
    let name = $"correctBiasChunkMasked.{typeof<'T>.Name}"
    let memoryNeed n = n * uint64 (Marshal.SizeOf<'T>() + Marshal.SizeOf<uint8>() + Marshal.SizeOf<float>())
    let apply _debug (input: AsyncSeq<Chunk<'T> * Chunk<uint8>>) =
        asyncSeq {
            let mutable z = 0u
            for chunk, mask in input do
                let output = correctedChunkAt z model chunk (Some mask)
                z <- z + 1u
                yield output
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) memoryNeed id pipe
