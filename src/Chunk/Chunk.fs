module Chunk

open System
open System.Buffers
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

type ChunkLayout =
    { VolumeSize: uint64 * uint64 * uint64
      ChunkSize: uint64 * uint64 * uint64
      ChunkCounts: int * int * int
      PixelType: string
      Components: uint }

type ChunkIndex = int * int * int

// Equality is identity-like: chunks are owned storage handles, not structural values.
[<CustomEquality; NoComparison>]
type Chunk<'T when 'T: equality> =
    { Size: uint64 * uint64 * uint64
      Bytes: byte[]
      ByteLength: int
      Release: unit -> unit
      RefCount: int ref }

    override this.Equals(other) =
        match other with
        | :? Chunk<'T> as other -> obj.ReferenceEquals(this.RefCount, other.RefCount)
        | _ -> false

    override this.GetHashCode() =
        RuntimeHelpers.GetHashCode(this.RefCount)

type VectorChunk<'T when 'T: equality> =
    { SpatialSize: uint64 * uint64 * uint64
      Components: uint32
      Chunk: Chunk<'T> }

type HistogramBinning =
    | FixedEdges of firstLeftEdge: float * lastLeftEdge: float * bins: uint32
    | FixedWidth of binWidth: uint64

type Histogram<'T when 'T: comparison> =
    { Counts: Map<'T, uint64>
      Binning: HistogramBinning option }

module NativeSp =
    [<Literal>]
    let LibraryPath = "spnth"

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_inplace")>]
    extern int fftwfComplexXYInplace(
        nativeint interleaved,
        int width,
        int height,
        int inverse)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_z_inplace")>]
    extern int fftwfComplexZInplace(
        nativeint interleaved,
        int width,
        int height,
        int depth,
        int inverse)

    let ensureAvailable () =
        let mutable handle = nativeint 0
        let searchPath = Nullable(DllImportSearchPath.AssemblyDirectory)
        if NativeLibrary.TryLoad(LibraryPath, typeof<ChunkLayout>.Assembly, searchPath, &handle) then
            NativeLibrary.Free(handle)
        else
            invalidOp "Native StackProcessing helper 'spnth' was not found. Build it with native/StackProcessing.NativeMedian/build.sh so the platform library is placed in the solution lib directory and copied to the application output."

    let checkStatus operation status =
        if status <> 0 then
            invalidOp $"{operation} failed in native helper with status {status}."

module Histogram =
    let ofMap counts =
        { Counts = counts
          Binning = None }

    let withFixedEdges firstLeftEdge lastLeftEdge bins counts =
        { Counts = counts
          Binning = Some(FixedEdges(firstLeftEdge, lastLeftEdge, bins)) }

    let withFixedWidth binWidth counts =
        { Counts = counts
          Binning = Some(FixedWidth binWidth) }

type DenseUInt32UnionFind(initialCapacity: int) =
    let mutable parent = Array.zeroCreate<uint32> (max 2 initialCapacity)
    let mutable rank = Array.zeroCreate<byte> parent.Length

    member private _.Ensure(label: uint32) =
        let index = int label
        if index >= parent.Length then
            let mutable newLength = parent.Length * 2
            while index >= newLength do
                newLength <- newLength * 2
            Array.Resize(&parent, newLength)
            Array.Resize(&rank, newLength)

    member this.Add(label: uint32) =
        this.Ensure label
        let index = int label
        if parent[index] = 0u then
            parent[index] <- label

    member this.Find(label: uint32) =
        this.Ensure label
        let mutable current = label
        let mutable currentIndex = int current
        if parent[currentIndex] = 0u then
            parent[currentIndex] <- current
        while parent[currentIndex] <> current do
            current <- parent[currentIndex]
            currentIndex <- int current
        let root = current
        let mutable node = label
        let mutable nodeIndex = int node
        while parent[nodeIndex] <> root do
            let next = parent[nodeIndex]
            parent[nodeIndex] <- root
            node <- next
            nodeIndex <- int node
        root

    member this.Union(left: uint32, right: uint32) =
        let leftRoot = this.Find left
        let rightRoot = this.Find right
        if leftRoot <> rightRoot then
            let leftIndex = int leftRoot
            let rightIndex = int rightRoot
            let leftRank = rank[leftIndex]
            let rightRank = rank[rightIndex]
            if leftRank < rightRank then
                parent[leftIndex] <- rightRoot
            elif leftRank > rightRank then
                parent[rightIndex] <- leftRoot
            elif leftRoot < rightRoot then
                parent[rightIndex] <- leftRoot
                rank[leftIndex] <- leftRank + 1uy
            else
                parent[leftIndex] <- rightRoot
                rank[rightIndex] <- rightRank + 1uy

    member private this.RootWithoutCompression(label: uint32) =
        this.Ensure label
        let mutable current = label
        let mutable currentIndex = int current
        if parent[currentIndex] = 0u then
            parent[currentIndex] <- current
        while parent[currentIndex] <> current do
            current <- parent[currentIndex]
            currentIndex <- int current
        current

    member this.UnionWithoutCompression(left: uint32, right: uint32) =
        let leftRoot = this.RootWithoutCompression left
        let rightRoot = this.RootWithoutCompression right
        if leftRoot <> rightRoot then
            let leftIndex = int leftRoot
            let rightIndex = int rightRoot
            let leftRank = rank[leftIndex]
            let rightRank = rank[rightIndex]
            if leftRank < rightRank then
                parent[leftIndex] <- rightRoot
            elif leftRank > rightRank then
                parent[rightIndex] <- leftRoot
            elif leftRoot < rightRoot then
                parent[rightIndex] <- leftRoot
                rank[leftIndex] <- leftRank + 1uy
            else
                parent[leftIndex] <- rightRoot
                rank[rightIndex] <- rightRank + 1uy

let create<'T when 'T: equality> size : Chunk<'T> =
    let width, height, depth = size
    let expected = width * height * depth * (Unsafe.SizeOf<'T>() |> uint64)
    if expected > uint64 Int32.MaxValue then
        invalidArg "size" $"Chunk byte buffer length must fit in Int32 for ArrayPool<byte>; got {expected}."
    let bytes = ArrayPool<byte>.Shared.Rent(int expected)
    { Size = size
      Bytes = bytes
      ByteLength = int expected
      Release = fun () -> ArrayPool<byte>.Shared.Return(bytes)
      RefCount = ref 1 }

let span<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    MemoryMarshal.Cast<byte, 'T>(chunk.Bytes.AsSpan(0, chunk.ByteLength))

let incRef chunk =
    lock chunk.RefCount (fun () ->
        if chunk.RefCount.Value <= 0 then
            invalidOp "Cannot increment a released chunk."
        chunk.RefCount.Value <- chunk.RefCount.Value + 1)
    chunk

let decRef chunk =
    let shouldRelease =
        lock chunk.RefCount (fun () ->
            chunk.RefCount.Value <- chunk.RefCount.Value - 1
            if chunk.RefCount.Value = 0 then
                true
            elif chunk.RefCount.Value < 0 then
                invalidOp "Chunk.decRef called after the chunk was already released."
            else
                false)
    if shouldRelease then
        chunk.Release()

let inline toIndex (width: int) (height: int) (x: int) (y: int) (z: int) =
    (z * height + y) * width + x

let inline ofIndex (width: int) (height: int) (index: int) =
    let plane = width * height
    let z = index / plane
    let remainder = index - z * plane
    let y = remainder / width
    let x = remainder - y * width
    x, y, z

let iter (f: 'T -> unit) (chunk: Chunk<'T>) =
    let inputSpan = span<'T> chunk
    let mutable i = 0
    while i < inputSpan.Length do
        f inputSpan[i]
        i <- i + 1

let iteri (f: int -> 'T -> unit) (chunk: Chunk<'T>) =
    let inputSpan = span<'T> chunk
    let mutable i = 0
    while i < inputSpan.Length do
        f i inputSpan[i]
        i <- i + 1

let mapInto (f: 'T -> 'U) (input: Chunk<'T>) (output: Chunk<'U>) =
    let inputSpan = span<'T> input
    let outputSpan = span<'U> output
    if outputSpan.Length < inputSpan.Length then
        invalidArg "output" $"Chunk.mapInto output has capacity for {outputSpan.Length} {typeof<'U>.Name} elements, expected at least {inputSpan.Length}."

    let mutable i = 0
    while i < inputSpan.Length do
        outputSpan[i] <- f inputSpan[i]
        i <- i + 1

let map (f: 'T -> 'U) (chunk: Chunk<'T>) =
    let output = create<'U> chunk.Size
    try
        mapInto f chunk output
        output
    with
    | _ ->
        decRef output
        reraise()

let mapi (f: int -> 'T -> 'U) (chunk: Chunk<'T>) =
    let inputSpan = span<'T> chunk
    let output = create<'U> chunk.Size
    try
        let outputSpan = span<'U> output
        let mutable i = 0
        while i < inputSpan.Length do
            outputSpan[i] <- f i inputSpan[i]
            i <- i + 1
        output
    with
    | _ ->
        decRef output
        reraise()

let fold (folder: 'State -> 'T -> 'State) (state: 'State) (chunk: Chunk<'T>) =
    let inputSpan = span<'T> chunk
    let mutable acc = state
    let mutable i = 0
    while i < inputSpan.Length do
        acc <- folder acc inputSpan[i]
        i <- i + 1
    acc

let foldi (folder: 'State -> int -> 'T -> 'State) (state: 'State) (chunk: Chunk<'T>) =
    let inputSpan = span<'T> chunk
    let mutable acc = state
    let mutable i = 0
    while i < inputSpan.Length do
        acc <- folder acc i inputSpan[i]
        i <- i + 1
    acc

let private checkedIntDimension name value =
    if value > uint64 Int32.MaxValue then
        invalidArg name $"Chunk dimension must fit in Int32 for managed indexing, got {value}."
    int value

let private checkedComponents components =
    if components = 0u then
        invalidArg "components" "Vector chunks require at least one component."
    if uint64 components > uint64 Int32.MaxValue then
        invalidArg "components" $"Vector chunk component count must fit in Int32, got {components}."
    int components

let private spatialCount (width: int) (height: int) (depth: int) =
    width * height * depth

let private vectorStorageSize (width, height, depth) components =
    let c = uint64 components
    if width > UInt64.MaxValue / c then
        invalidArg "components" $"Vector chunk storage width overflow for width {width} and components {components}."
    width * c, height, depth

let private validateVectorStorage name (vector: VectorChunk<'T>) =
    let expectedSize = vectorStorageSize vector.SpatialSize vector.Components
    if vector.Chunk.Size <> expectedSize then
        invalidArg name $"Vector chunk storage size {vector.Chunk.Size} does not match spatial size {vector.SpatialSize} with {vector.Components} components."

let private vectorFlatIndex spatialIndex components componentIndex =
    spatialIndex * components + componentIndex

let private createVectorChunk<'T when 'T: equality> spatialSize components =
    { SpatialSize = spatialSize
      Components = components
      Chunk = create<'T> (vectorStorageSize spatialSize components) }

let vectorSpan<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    span<'T> vector.Chunk

let toVectorImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (components: Chunk<'T> list) =
    match components with
    | [] -> invalidArg "components" "toVectorImage requires at least one scalar chunk."
    | first :: rest ->
        let spatialSize = first.Size
        rest
        |> List.iteri (fun i chunk ->
            if chunk.Size <> spatialSize then
                invalidArg "components" $"toVectorImage expects all chunks to have size {spatialSize}, got {chunk.Size} at component {i + 1}.")

        let componentCount = uint32 components.Length
        let output = createVectorChunk<'T> spatialSize componentCount
        try
            let outputPixels = vectorSpan output
            let width, height, depth = spatialSize
            let count = spatialCount (checkedIntDimension "width" width) (checkedIntDimension "height" height) (checkedIntDimension "depth" depth)
            let componentCountI = int componentCount

            let componentChunks = components |> List.toArray
            for c in 0 .. componentCountI - 1 do
                let chunk = componentChunks[c]
                let inputPixels = span<'T> chunk
                for i in 0 .. count - 1 do
                    outputPixels[vectorFlatIndex i componentCountI c] <- inputPixels[i]

            output
        with
        | _ ->
            decRef output.Chunk
            reraise()

let vectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (componentId: uint) (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    let selectedComponent = int componentId
    let components = checkedComponents vector.Components
    if selectedComponent < 0 || selectedComponent >= components then
        invalidArg "componentId" $"vectorElement: component {componentId} is outside the available component range 0..{components - 1}."

    let output = create<'T> vector.SpatialSize
    try
        let inputPixels = vectorSpan vector
        let outputPixels = span<'T> output
        for i in 0 .. outputPixels.Length - 1 do
            outputPixels[i] <- inputPixels[vectorFlatIndex i components selectedComponent]
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (firstComponent: uint) (componentCount: uint) (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    if componentCount = 0u then
        invalidArg "componentCount" "vectorRange needs at least one component."
    if firstComponent + componentCount > vector.Components then
        invalidArg "componentCount" $"vectorRange requested {firstComponent}..{firstComponent + componentCount - 1u}, but vector has {vector.Components} components."

    let inputComponents = checkedComponents vector.Components
    let outputComponents = checkedComponents componentCount
    let first = int firstComponent
    let output = createVectorChunk<'T> vector.SpatialSize componentCount
    try
        let inputPixels = vectorSpan vector
        let outputPixels = vectorSpan output
        let spatialCount = outputPixels.Length / outputComponents

        for i in 0 .. spatialCount - 1 do
            for c in 0 .. outputComponents - 1 do
                outputPixels[vectorFlatIndex i outputComponents c] <- inputPixels[vectorFlatIndex i inputComponents (first + c)]

        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let appendVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (vector: VectorChunk<'T>) (element: Chunk<'T>) =
    validateVectorStorage "vector" vector
    if element.Size <> vector.SpatialSize then
        invalidArg "element" $"appendVectorElement: chunk sizes differ: {vector.SpatialSize} vs {element.Size}."

    let oldComponents = checkedComponents vector.Components
    let newComponents = vector.Components + 1u
    let output = createVectorChunk<'T> vector.SpatialSize newComponents
    try
        let inputPixels = vectorSpan vector
        let elementPixels = span<'T> element
        let outputPixels = vectorSpan output
        let newComponentsI = int newComponents

        for i in 0 .. elementPixels.Length - 1 do
            for c in 0 .. oldComponents - 1 do
                outputPixels[vectorFlatIndex i newComponentsI c] <- inputPixels[vectorFlatIndex i oldComponents c]
            outputPixels[vectorFlatIndex i newComponentsI oldComponents] <- elementPixels[i]

        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let mapVectorElements (f: float -> float) (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    let output = createVectorChunk<float> vector.SpatialSize vector.Components
    try
        let inputPixels = vectorSpan vector
        let outputPixels = vectorSpan output
        for i in 0 .. inputPixels.Length - 1 do
            outputPixels[i] <- f inputPixels[i]
        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let vector3ToColor inputMinimum inputMaximum (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    if inputMaximum <= inputMinimum then
        invalidArg "inputMaximum" "vector3ToColor input maximum must be greater than input minimum."
    if vector.Components <> 3u then
        invalidArg "vector" $"vector3ToColor expects 3 components, got {vector.Components}."

    let output = createVectorChunk<uint8> vector.SpatialSize 3u
    try
        let inputPixels = vectorSpan vector
        let outputPixels = vectorSpan output
        let scale = 255.0 / (inputMaximum - inputMinimum)
        for i in 0 .. inputPixels.Length - 1 do
            let value = (inputPixels[i] - inputMinimum) * scale
            outputPixels[i] <- byte (max 0.0 (min 255.0 (Math.Round value)))
        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let colorToVector3 outputMinimum outputMaximum (vector: VectorChunk<uint8>) =
    validateVectorStorage "vector" vector
    if outputMaximum <= outputMinimum then
        invalidArg "outputMaximum" "colorToVector3 output maximum must be greater than output minimum."
    if vector.Components <> 3u then
        invalidArg "vector" $"colorToVector3 expects 3 components, got {vector.Components}."

    let output = createVectorChunk<float> vector.SpatialSize 3u
    try
        let inputPixels = vectorSpan vector
        let outputPixels = vectorSpan output
        let scale = (outputMaximum - outputMinimum) / 255.0
        for i in 0 .. inputPixels.Length - 1 do
            outputPixels[i] <- outputMinimum + float inputPixels[i] * scale
        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let private ensureMatchingVectorChunks name (a: VectorChunk<float>) (b: VectorChunk<float>) =
    validateVectorStorage "a" a
    validateVectorStorage "b" b
    if a.SpatialSize <> b.SpatialSize then
        invalidArg "b" $"{name}: spatial sizes differ: {a.SpatialSize} vs {b.SpatialSize}."
    if a.Components <> b.Components then
        invalidArg "b" $"{name}: component counts differ: {a.Components} vs {b.Components}."

let vectorDot (a: VectorChunk<float>) (b: VectorChunk<float>) =
    ensureMatchingVectorChunks "vectorDot" a b
    let components = checkedComponents a.Components
    let output = create<float> a.SpatialSize
    try
        let aPixels = vectorSpan a
        let bPixels = vectorSpan b
        let outputPixels = span<float> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable sum = 0.0
            for c in 0 .. components - 1 do
                let index = vectorFlatIndex i components c
                sum <- sum + aPixels[index] * bPixels[index]
            outputPixels[i] <- sum
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorMagnitude (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    let components = checkedComponents vector.Components
    let output = create<float> vector.SpatialSize
    try
        let inputPixels = vectorSpan vector
        let outputPixels = span<float> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable normSquared = 0.0
            for c in 0 .. components - 1 do
                let value = inputPixels[vectorFlatIndex i components c]
                normSquared <- normSquared + value * value
            outputPixels[i] <- sqrt normSquared
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorCross3D (a: VectorChunk<float>) (b: VectorChunk<float>) =
    ensureMatchingVectorChunks "vectorCross3D" a b
    if a.Components <> 3u then
        invalidArg "a" $"vectorCross3D: expected 3-component vector chunks, got {a.Components} components."

    let output = createVectorChunk<float> a.SpatialSize 3u
    try
        let aPixels = vectorSpan a
        let bPixels = vectorSpan b
        let outputPixels = vectorSpan output
        for i in 0 .. (outputPixels.Length / 3) - 1 do
            let ax = aPixels[vectorFlatIndex i 3 0]
            let ay = aPixels[vectorFlatIndex i 3 1]
            let az = aPixels[vectorFlatIndex i 3 2]
            let bx = bPixels[vectorFlatIndex i 3 0]
            let by = bPixels[vectorFlatIndex i 3 1]
            let bz = bPixels[vectorFlatIndex i 3 2]
            outputPixels[vectorFlatIndex i 3 0] <- ay * bz - az * by
            outputPixels[vectorFlatIndex i 3 1] <- az * bx - ax * bz
            outputPixels[vectorFlatIndex i 3 2] <- ax * by - ay * bx
        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let vectorAngleTo (reference: float list) (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    let components = checkedComponents vector.Components
    if reference.Length <> components then
        invalidArg "reference" $"vectorAngleTo: reference vector has {reference.Length} components, vector chunk has {vector.Components}."

    let referenceValues = reference |> List.toArray
    let referenceNorm = referenceValues |> Array.sumBy (fun value -> value * value) |> sqrt
    if referenceNorm < 1e-18 then
        invalidArg "reference" "vectorAngleTo: reference vector must be non-zero."

    let output = create<float> vector.SpatialSize
    try
        let inputPixels = vectorSpan vector
        let outputPixels = span<float> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable dot = 0.0
            let mutable normSquared = 0.0
            for c in 0 .. components - 1 do
                let value = inputPixels[vectorFlatIndex i components c]
                dot <- dot + value * referenceValues[c]
                normSquared <- normSquared + value * value
            let valueNorm = sqrt normSquared
            outputPixels[i] <-
                if valueNorm < 1e-18 then
                    Double.NaN
                else
                    dot / (valueNorm * referenceNorm)
                    |> max -1.0
                    |> min 1.0
                    |> acos
        output
    with
    | _ ->
        decRef output
        reraise()

let private ensureMatchingVectorChunksFloat32 name (a: VectorChunk<float32>) (b: VectorChunk<float32>) =
    validateVectorStorage "a" a
    validateVectorStorage "b" b
    if a.SpatialSize <> b.SpatialSize then
        invalidArg "b" $"{name}: spatial sizes differ: {a.SpatialSize} vs {b.SpatialSize}."
    if a.Components <> b.Components then
        invalidArg "b" $"{name}: component counts differ: {a.Components} vs {b.Components}."

let mapVectorElementsFloat32 (f: float32 -> float32) (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    let output = createVectorChunk<float32> vector.SpatialSize vector.Components
    try
        let inputPixels = vectorSpan vector
        let outputPixels = vectorSpan output
        for i in 0 .. inputPixels.Length - 1 do
            outputPixels[i] <- f inputPixels[i]
        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let vectorDotFloat32 (a: VectorChunk<float32>) (b: VectorChunk<float32>) =
    ensureMatchingVectorChunksFloat32 "vectorDotFloat32" a b
    let components = checkedComponents a.Components
    let output = create<float32> a.SpatialSize
    try
        let aPixels = vectorSpan a
        let bPixels = vectorSpan b
        let outputPixels = span<float32> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable sum = 0.0f
            for c in 0 .. components - 1 do
                let index = vectorFlatIndex i components c
                sum <- sum + aPixels[index] * bPixels[index]
            outputPixels[i] <- sum
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorMagnitudeFloat32 (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    let components = checkedComponents vector.Components
    let output = create<float32> vector.SpatialSize
    try
        let inputPixels = vectorSpan vector
        let outputPixels = span<float32> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable normSquared = 0.0f
            for c in 0 .. components - 1 do
                let value = inputPixels[vectorFlatIndex i components c]
                normSquared <- normSquared + value * value
            outputPixels[i] <- sqrt normSquared
        output
    with
    | _ ->
        decRef output
        reraise()

let vector3ToColorFloat32 inputMinimum inputMaximum (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    if inputMaximum <= inputMinimum then
        invalidArg "inputMaximum" "vector3ToColorFloat32 input maximum must be greater than input minimum."
    if vector.Components <> 3u then
        invalidArg "vector" $"vector3ToColorFloat32 expects 3 components, got {vector.Components}."

    let output = createVectorChunk<uint8> vector.SpatialSize 3u
    try
        let inputPixels = vectorSpan vector
        let outputPixels = vectorSpan output
        let scale = 255.0f / (inputMaximum - inputMinimum)
        for i in 0 .. inputPixels.Length - 1 do
            let value = (inputPixels[i] - inputMinimum) * scale
            outputPixels[i] <- byte (max 0.0f (min 255.0f (MathF.Round value)))
        output
    with
    | _ ->
        decRef output.Chunk
        reraise()

let vectorAngleToFloat32 (reference: float32 list) (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    let components = checkedComponents vector.Components
    if reference.Length <> components then
        invalidArg "reference" $"vectorAngleToFloat32: reference vector has {reference.Length} components, vector chunk has {vector.Components}."

    let referenceValues = reference |> List.toArray
    let referenceNorm = referenceValues |> Array.sumBy (fun value -> value * value) |> sqrt
    if referenceNorm < 1e-18f then
        invalidArg "reference" "vectorAngleToFloat32: reference vector must be non-zero."

    let output = create<float32> vector.SpatialSize
    try
        let inputPixels = vectorSpan vector
        let outputPixels = span<float32> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable dot = 0.0f
            let mutable normSquared = 0.0f
            for c in 0 .. components - 1 do
                let value = inputPixels[vectorFlatIndex i components c]
                dot <- dot + value * referenceValues[c]
                normSquared <- normSquared + value * value
            let valueNorm = sqrt normSquared
            outputPixels[i] <-
                if valueNorm < 1e-18f then
                    Single.NaN
                else
                    dot / (valueNorm * referenceNorm)
                    |> max -1.0f
                    |> min 1.0f
                    |> acos
        output
    with
    | _ ->
        decRef output
        reraise()
