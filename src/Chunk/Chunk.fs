module Chunk

open System
open System.Buffers
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

type ChunkStats =
    { Created: int64
      Released: int64
      IncRef: int64
      DecRef: int64
      Live: int64
      PeakLive: int64
      LiveBytes: int64
      PeakLiveBytes: int64 }

module ChunkStats =
    let private gate = obj()
    let mutable private created = 0L
    let mutable private released = 0L
    let mutable private incRef = 0L
    let mutable private decRef = 0L
    let mutable private live = 0L
    let mutable private peakLive = 0L
    let mutable private liveBytes = 0L
    let mutable private peakLiveBytes = 0L
    let mutable private debugLevel = 0u

    let private mib bytes =
        float bytes / (1024.0 * 1024.0)

    let private chunkId (refCount: int ref) =
        RuntimeHelpers.GetHashCode(refCount)

    let private printDebugMessage event typeName byteLength (refCount: int ref) currentRefCount =
        if debugLevel >= 2u then
            printfn
                "%8d KB / %8d KB %3d / %3d Chunks %s chunk=%08x type=%s bytes=%d refCount=%d"
                (liveBytes / 1024L)
                (peakLiveBytes / 1024L)
                live
                peakLive
                event
                (chunkId refCount)
                typeName
                byteLength
                currentRefCount

    let setDebugLevel level =
        lock gate (fun () ->
            debugLevel <- level)

    let reset () =
        lock gate (fun () ->
            created <- 0L
            released <- 0L
            incRef <- 0L
            decRef <- 0L
            live <- 0L
            peakLive <- 0L
            liveBytes <- 0L
            peakLiveBytes <- 0L)

    let recordCreate typeName byteLength refCount =
        lock gate (fun () ->
            created <- created + 1L
            live <- live + 1L
            liveBytes <- liveBytes + int64 byteLength
            peakLive <- max peakLive live
            peakLiveBytes <- max peakLiveBytes liveBytes
            printDebugMessage "Created ArrayPool buffer for" typeName byteLength refCount refCount.Value)

    let recordIncRef typeName byteLength refCount currentRefCount =
        lock gate (fun () ->
            incRef <- incRef + 1L
            //printDebugMessage "Increased reference to" typeName byteLength refCount currentRefCount
            )

    let recordDecRef typeName byteLength refCount currentRefCount =
        lock gate (fun () ->
            decRef <- decRef + 1L
            //printDebugMessage "Decreased reference to" typeName byteLength refCount currentRefCount
            )

    let recordRelease typeName byteLength refCount =
        lock gate (fun () ->
            released <- released + 1L
            live <- live - 1L
            liveBytes <- liveBytes - int64 byteLength
            printDebugMessage "Returned ArrayPool buffer for" typeName byteLength refCount 0)

    let snapshot () =
        lock gate (fun () ->
            { Created = created
              Released = released
              IncRef = incRef
              DecRef = decRef
              Live = live
              PeakLive = peakLive
              LiveBytes = liveBytes
              PeakLiveBytes = peakLiveBytes })

    let format (stats: ChunkStats) =
        $"created={stats.Created}, released={stats.Released}, live={stats.Live}, peakLive={stats.PeakLive}, incRef={stats.IncRef}, decRef={stats.DecRef}, liveBytes={stats.LiveBytes} ({mib stats.LiveBytes:F1} MiB), peakLiveBytes={stats.PeakLiveBytes} ({mib stats.PeakLiveBytes:F1} MiB)"

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
      Index: ChunkIndex option
      SourceDepth: uint option
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

type LocatedChunk<'T when 'T: equality> =
    { Index: ChunkIndex
      Layout: ChunkLayout
      Chunk: Chunk<'T> }

type EncodedLocatedChunk =
    { Index: ChunkIndex
      Layout: ChunkLayout
      Payload: ReadOnlyMemory<byte> option }

type VectorChunk<'T when 'T: equality> =
    { SpatialSize: uint64 * uint64 * uint64
      Components: Chunk<'T>[] }

type SpectralLayout =
    | FullComplex64Interleaved
    | HermitianPackedComplex64Interleaved of packedAxis: int * realSize: uint64

type SpectralChunk =
    { LogicalSize: uint64 * uint64 * uint64
      Layout: SpectralLayout
      Chunk: Chunk<float32> }

type HistogramBinning =
    | FixedEdges of firstLeftEdge: float * lastLeftEdge: float * bins: uint32

type Histogram<'T when 'T: comparison> =
    { Counts: Map<'T, uint64>
      Binning: HistogramBinning option }

module NativeSp =
    [<Literal>]
    let LibraryPath = "lowlevel"

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_inplace")>]
    extern int fftwfComplexXYInplace(
        nativeint interleaved,
        int width,
        int height,
        int inverse)

    [<DllImport(LibraryPath, EntryPoint = "sp_inv_fftwf_complex_xy_inplace")>]
    extern int invFftwfComplexXYInplace(
        nativeint interleaved,
        int width,
        int height)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_real_xy_to_complex")>]
    extern int fftwfRealXYToComplex(
        nativeint real,
        nativeint interleaved,
        int width,
        int height)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_real_xy_plan_create")>]
    extern nativeint fftwfRealXYPlanCreate(
        nativeint real,
        nativeint interleaved,
        int width,
        int height)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_real_xy_plan_execute")>]
    extern int fftwfRealXYPlanExecute(
        nativeint plan,
        nativeint real,
        nativeint interleaved)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_real_xy_plan_destroy")>]
    extern void fftwfRealXYPlanDestroy(
        nativeint plan)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_to_real_plan_create")>]
    extern nativeint fftwfComplexXYToRealPlanCreate(
        nativeint interleaved,
        nativeint real,
        int width,
        int height)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_to_real_plan_execute")>]
    extern int fftwfComplexXYToRealPlanExecute(
        nativeint plan,
        nativeint interleaved,
        nativeint real)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_to_real_plan_destroy")>]
    extern void fftwfComplexXYToRealPlanDestroy(
        nativeint plan)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_plan_create")>]
    extern nativeint fftwfComplexXYPlanCreate(
        nativeint interleaved,
        int width,
        int height,
        int inverse)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_plan_execute")>]
    extern int fftwfComplexXYPlanExecute(
        nativeint plan,
        nativeint interleaved)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_xy_plan_destroy")>]
    extern void fftwfComplexXYPlanDestroy(
        nativeint plan)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_z_inplace")>]
    extern int fftwfComplexZInplace(
        nativeint interleaved,
        int width,
        int height,
        int depth,
        int inverse)

    [<DllImport(LibraryPath, EntryPoint = "sp_inv_fftwf_complex_z_inplace")>]
    extern int invFftwfComplexZInplace(
        nativeint interleaved,
        int width,
        int height,
        int depth)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_real_z_to_complex")>]
    extern int fftwfRealZToComplex(
        nativeint real,
        nativeint interleaved,
        int width,
        int height,
        int depth)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_z_plan_create")>]
    extern nativeint fftwfComplexZPlanCreate(
        nativeint interleaved,
        int width,
        int height,
        int depth,
        int inverse)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_z_plan_execute")>]
    extern int fftwfComplexZPlanExecute(
        nativeint plan,
        nativeint interleaved)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_z_plan_destroy")>]
    extern void fftwfComplexZPlanDestroy(
        nativeint plan)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_complex_3d_inplace")>]
    extern int fftwfComplex3DInplace(
        nativeint interleaved,
        int width,
        int height,
        int depth,
        int inverse)

    [<DllImport(LibraryPath, EntryPoint = "sp_inv_fftwf_complex_3d_inplace")>]
    extern int invFftwfComplex3DInplace(
        nativeint interleaved,
        int width,
        int height,
        int depth)

    [<DllImport(LibraryPath, EntryPoint = "sp_fftwf_real_3d_to_complex")>]
    extern int fftwfReal3DToComplex(
        nativeint real,
        nativeint interleaved,
        int width,
        int height,
        int depth)

    let ensureAvailable () =
        let mutable handle = nativeint 0
        let searchPath = Nullable(DllImportSearchPath.AssemblyDirectory)
        if NativeLibrary.TryLoad(LibraryPath, typeof<ChunkLayout>.Assembly, searchPath, &handle) then
            NativeLibrary.Free(handle)
        else
            invalidOp "Native StackProcessing helper 'lowlevel' was not found. Build it with lowlevel/build.sh or lowlevel/build.ps1 so the platform library is placed in the solution lib directory and copied to the application output."

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
    let refCount = ref 1
    ChunkStats.recordCreate typeof<'T>.Name (int expected) refCount
    { Size = size
      Index = None
      SourceDepth = None
      Bytes = bytes
      ByteLength = int expected
      Release = fun () ->
          ChunkStats.recordRelease typeof<'T>.Name (int expected) refCount
          ArrayPool<byte>.Shared.Return(bytes)
      RefCount = refCount }

let withIndex index (chunk: Chunk<'T>) =
    { chunk with Index = Some index }

let withSourceDepth sourceDepth (chunk: Chunk<'T>) =
    { chunk with SourceDepth = Some sourceDepth }

let withIndexOption index (chunk: Chunk<'T>) =
    { chunk with Index = index }

let withSameIndex (source: Chunk<'S>) (target: Chunk<'T>) =
    { target with
        Index = source.Index
        SourceDepth = source.SourceDepth }

let span<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    MemoryMarshal.Cast<byte, 'T>(chunk.Bytes.AsSpan(0, chunk.ByteLength))

let incRef<'T when 'T: equality> (chunk: Chunk<'T>) =
    let mutable currentRefCount = 0
    lock chunk.RefCount (fun () ->
        if chunk.RefCount.Value <= 0 then
            invalidOp "Cannot increment a released chunk."
        chunk.RefCount.Value <- chunk.RefCount.Value + 1
        currentRefCount <- chunk.RefCount.Value)
    ChunkStats.recordIncRef typeof<'T>.Name chunk.ByteLength chunk.RefCount currentRefCount
    chunk

let decRef<'T when 'T: equality> (chunk: Chunk<'T>) =
    let mutable currentRefCount = 0
    let shouldRelease =
        lock chunk.RefCount (fun () ->
            chunk.RefCount.Value <- chunk.RefCount.Value - 1
            currentRefCount <- chunk.RefCount.Value
            if chunk.RefCount.Value = 0 then
                true
            elif chunk.RefCount.Value < 0 then
                invalidOp "Chunk.decRef called after the chunk was already released."
            else
                false)
    ChunkStats.recordDecRef typeof<'T>.Name chunk.ByteLength chunk.RefCount currentRefCount
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

let vectorComponentCount (vector: VectorChunk<'T>) =
    uint32 vector.Components.Length

let private validateVectorStorage name (vector: VectorChunk<'T>) =
    if vector.Components.Length = 0 then
        invalidArg name "Vector chunk requires at least one component."
    vector.Components
    |> Array.iteri (fun i chunk ->
        if chunk.Size <> vector.SpatialSize then
            invalidArg name $"Vector component {i} size {chunk.Size} does not match spatial size {vector.SpatialSize}.")

let private vectorFlatIndex spatialIndex components componentIndex =
    spatialIndex * components + componentIndex

let private createVectorChunk<'T when 'T: equality> spatialSize components : VectorChunk<'T> =
    let componentCount = checkedComponents components
    { SpatialSize = spatialSize
      Components = Array.init componentCount (fun _ -> create<'T> spatialSize) }

let incRefVector<'T when 'T: equality> (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    vector.Components |> Array.iter (incRef >> ignore)
    vector

let decRefVector<'T when 'T: equality> (vector: VectorChunk<'T>) =
    vector.Components |> Array.iter decRef

let vectorComponent<'T when 'T: equality> (componentId: uint) (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    let selectedComponent = int componentId
    if selectedComponent < 0 || selectedComponent >= vector.Components.Length then
        invalidArg "componentId" $"vectorComponent: component {componentId} is outside the available component range 0..{vector.Components.Length - 1}."
    incRef vector.Components[selectedComponent]

let ofChunk<'T when 'T: equality> (chunk: Chunk<'T>) : VectorChunk<'T> =
    { SpatialSize = chunk.Size
      Components = [| incRef chunk |] }

let private ofChunkWithComponents<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> components (chunk: Chunk<'T>) : VectorChunk<'T> =
    let componentCount = checkedComponents components
    let width, height, depth = chunk.Size
    if width % uint64 componentCount <> 0UL then
        invalidArg "chunk" $"ofChunk expected width {width} to be divisible by component count {components}."
    let spatialSize = width / uint64 componentCount, height, depth
    let output = createVectorChunk<'T> spatialSize components
    try
        let inputPixels = span<'T> chunk
        let spatialWidth, spatialHeight, spatialDepth = spatialSize
        let count =
            spatialCount
                (checkedIntDimension "width" spatialWidth)
                (checkedIntDimension "height" spatialHeight)
                (checkedIntDimension "depth" spatialDepth)
        for c in 0 .. componentCount - 1 do
            let outputPixels = span<'T> output.Components[c]
            for i in 0 .. count - 1 do
                outputPixels[i] <- inputPixels[vectorFlatIndex i componentCount c]
        output
    with
    | _ ->
        decRefVector output
        reraise()

let toChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    let components = vector.Components.Length
    let output = create<'T> (vectorStorageSize vector.SpatialSize (uint32 components))
    try
        let outputPixels = span<'T> output
        for c in 0 .. components - 1 do
            let inputPixels = span<'T> vector.Components[c]
            for i in 0 .. inputPixels.Length - 1 do
                outputPixels[vectorFlatIndex i components c] <- inputPixels[i]
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorSpan<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (_vector: VectorChunk<'T>) =
    invalidOp "Chunk.vectorSpan is only valid for the old packed VectorChunk representation. Use VectorChunk components directly or VectorChunk.toChunk at packing boundaries."

let toVectorImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (components: Chunk<'T> list) =
    match components with
    | [] -> invalidArg "components" "toVectorImage requires at least one scalar chunk."
    | first :: rest ->
        let spatialSize = first.Size
        rest
        |> List.iteri (fun i chunk ->
            if chunk.Size <> spatialSize then
                invalidArg "components" $"toVectorImage expects all chunks to have size {spatialSize}, got {chunk.Size} at component {i + 1}.")

        { SpatialSize = spatialSize
          Components = components |> List.map incRef |> List.toArray }

let vectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (componentId: uint) (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    let selectedComponent = int componentId
    let components = vector.Components.Length
    if selectedComponent < 0 || selectedComponent >= components then
        invalidArg "componentId" $"vectorElement: component {componentId} is outside the available component range 0..{components - 1}."
    incRef vector.Components[selectedComponent]

let vectorRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (firstComponent: uint) (componentCount: uint) (vector: VectorChunk<'T>) =
    validateVectorStorage "vector" vector
    if componentCount = 0u then
        invalidArg "componentCount" "vectorRange needs at least one component."
    let availableComponents = vectorComponentCount vector
    if firstComponent + componentCount > availableComponents then
        invalidArg "componentCount" $"vectorRange requested {firstComponent}..{firstComponent + componentCount - 1u}, but vector has {availableComponents} components."

    let outputComponents = checkedComponents componentCount
    let first = int firstComponent
    { SpatialSize = vector.SpatialSize
      Components = Array.init outputComponents (fun c -> incRef vector.Components[first + c]) }

let appendVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (vector: VectorChunk<'T>) (element: Chunk<'T>) =
    validateVectorStorage "vector" vector
    if element.Size <> vector.SpatialSize then
        invalidArg "element" $"appendVectorElement: chunk sizes differ: {vector.SpatialSize} vs {element.Size}."

    { SpatialSize = vector.SpatialSize
      Components = Array.append (vector.Components |> Array.map incRef) [| incRef element |] }

let mapVectorElements (f: float -> float) (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    let output = createVectorChunk<float> vector.SpatialSize (vectorComponentCount vector)
    try
        for c in 0 .. vector.Components.Length - 1 do
            let inputPixels = span<float> vector.Components[c]
            let outputPixels = span<float> output.Components[c]
            for i in 0 .. inputPixels.Length - 1 do
                outputPixels[i] <- f inputPixels[i]
        output
    with
    | _ ->
        decRefVector output
        reraise()

let private intensityStretchVectorFloat64
    inputMinimum
    inputMaximum
    outputMinimum
    outputMaximum
    (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    if inputMaximum <= inputMinimum then
        invalidArg "inputMaximum" "intensityStretch input maximum must be greater than input minimum."

    let output = createVectorChunk<float> vector.SpatialSize (vectorComponentCount vector)
    try
        let scale = (outputMaximum - outputMinimum) / (inputMaximum - inputMinimum)
        for c in 0 .. vector.Components.Length - 1 do
            let inputPixels = span<float> vector.Components[c]
            let outputPixels = span<float> output.Components[c]
            for i in 0 .. inputPixels.Length - 1 do
                outputPixels[i] <- outputMinimum + (inputPixels[i] - inputMinimum) * scale
        output
    with
    | _ ->
        decRefVector output
        reraise()

let private intensityStretchVectorFloat32
    inputMinimum
    inputMaximum
    outputMinimum
    outputMaximum
    (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    if inputMaximum <= inputMinimum then
        invalidArg "inputMaximum" "intensityStretch input maximum must be greater than input minimum."

    let output = createVectorChunk<float32> vector.SpatialSize (vectorComponentCount vector)
    try
        let scale = (outputMaximum - outputMinimum) / (inputMaximum - inputMinimum)
        for c in 0 .. vector.Components.Length - 1 do
            let inputPixels = span<float32> vector.Components[c]
            let outputPixels = span<float32> output.Components[c]
            for i in 0 .. inputPixels.Length - 1 do
                outputPixels[i] <- outputMinimum + (inputPixels[i] - inputMinimum) * scale
        output
    with
    | _ ->
        decRefVector output
        reraise()

let intensityStretchVector<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> inputMinimum inputMaximum outputMinimum outputMaximum (vector: VectorChunk<'T>) =
    if typeof<'T> = typeof<float> then
        let inputMinimum = Convert.ToDouble(box inputMinimum)
        let inputMaximum = Convert.ToDouble(box inputMaximum)
        let outputMinimum = Convert.ToDouble(box outputMinimum)
        let outputMaximum = Convert.ToDouble(box outputMaximum)
        let vector = unbox<VectorChunk<float>> (box vector)
        unbox<VectorChunk<'T>> (box (intensityStretchVectorFloat64 inputMinimum inputMaximum outputMinimum outputMaximum vector))
    elif typeof<'T> = typeof<float32> then
        let inputMinimum = Convert.ToSingle(box inputMinimum)
        let inputMaximum = Convert.ToSingle(box inputMaximum)
        let outputMinimum = Convert.ToSingle(box outputMinimum)
        let outputMaximum = Convert.ToSingle(box outputMaximum)
        let vector = unbox<VectorChunk<float32>> (box vector)
        unbox<VectorChunk<'T>> (box (intensityStretchVectorFloat32 inputMinimum inputMaximum outputMinimum outputMaximum vector))
    else
        invalidArg "T" $"intensityStretch supports float and float32 vector chunks, got {typeof<'T>.Name}."

let colorToVector3 outputMinimum outputMaximum (vector: VectorChunk<uint8>) =
    validateVectorStorage "vector" vector
    if outputMaximum <= outputMinimum then
        invalidArg "outputMaximum" "colorToVector3 output maximum must be greater than output minimum."
    if vectorComponentCount vector <> 3u then
        invalidArg "vector" $"colorToVector3 expects 3 components, got {vectorComponentCount vector}."

    let output = createVectorChunk<float> vector.SpatialSize 3u
    try
        let scale = (outputMaximum - outputMinimum) / 255.0
        for c in 0 .. 2 do
            let inputPixels = span<uint8> vector.Components[c]
            let outputPixels = span<float> output.Components[c]
            for i in 0 .. inputPixels.Length - 1 do
                outputPixels[i] <- outputMinimum + float inputPixels[i] * scale
        output
    with
    | _ ->
        decRefVector output
        reraise()

let private ensureMatchingVectorChunks name (a: VectorChunk<float>) (b: VectorChunk<float>) =
    validateVectorStorage "a" a
    validateVectorStorage "b" b
    if a.SpatialSize <> b.SpatialSize then
        invalidArg "b" $"{name}: spatial sizes differ: {a.SpatialSize} vs {b.SpatialSize}."
    if vectorComponentCount a <> vectorComponentCount b then
        invalidArg "b" $"{name}: component counts differ: {vectorComponentCount a} vs {vectorComponentCount b}."

let vectorDot (a: VectorChunk<float>) (b: VectorChunk<float>) =
    ensureMatchingVectorChunks "vectorDot" a b
    let components = a.Components.Length
    let output = create<float> a.SpatialSize
    try
        let outputPixels = span<float> output
        outputPixels.Clear()
        for c in 0 .. components - 1 do
            let aPixels = span<float> a.Components[c]
            let bPixels = span<float> b.Components[c]
            for i in 0 .. outputPixels.Length - 1 do
                outputPixels[i] <- outputPixels[i] + aPixels[i] * bPixels[i]
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorMagnitude (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    let components = vector.Components.Length
    let output = create<float> vector.SpatialSize
    try
        let outputPixels = span<float> output
        outputPixels.Clear()
        for c in 0 .. components - 1 do
            let inputPixels = span<float> vector.Components[c]
            for i in 0 .. outputPixels.Length - 1 do
                let value = inputPixels[i]
                outputPixels[i] <- outputPixels[i] + value * value
        for i in 0 .. outputPixels.Length - 1 do
            outputPixels[i] <- sqrt outputPixels[i]
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorCross3D (a: VectorChunk<float>) (b: VectorChunk<float>) =
    ensureMatchingVectorChunks "vectorCross3D" a b
    if vectorComponentCount a <> 3u then
        invalidArg "a" $"vectorCross3D: expected 3-component vector chunks, got {vectorComponentCount a} components."

    let output = createVectorChunk<float> a.SpatialSize 3u
    try
        let axSpan = span<float> a.Components[0]
        let aySpan = span<float> a.Components[1]
        let azSpan = span<float> a.Components[2]
        let bxSpan = span<float> b.Components[0]
        let bySpan = span<float> b.Components[1]
        let bzSpan = span<float> b.Components[2]
        let oxSpan = span<float> output.Components[0]
        let oySpan = span<float> output.Components[1]
        let ozSpan = span<float> output.Components[2]
        for i in 0 .. oxSpan.Length - 1 do
            let ax = axSpan[i]
            let ay = aySpan[i]
            let az = azSpan[i]
            let bx = bxSpan[i]
            let by = bySpan[i]
            let bz = bzSpan[i]
            oxSpan[i] <- ay * bz - az * by
            oySpan[i] <- az * bx - ax * bz
            ozSpan[i] <- ax * by - ay * bx
        output
    with
    | _ ->
        decRefVector output
        reraise()

let vectorAngleTo (reference: float list) (vector: VectorChunk<float>) =
    validateVectorStorage "vector" vector
    let components = vector.Components.Length
    if reference.Length <> components then
        invalidArg "reference" $"vectorAngleTo: reference vector has {reference.Length} components, vector chunk has {vectorComponentCount vector}."

    let referenceValues = reference |> List.toArray
    let referenceNorm = referenceValues |> Array.sumBy (fun value -> value * value) |> sqrt
    if referenceNorm < 1e-18 then
        invalidArg "reference" "vectorAngleTo: reference vector must be non-zero."

    let output = create<float> vector.SpatialSize
    try
        let outputPixels = span<float> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable dot = 0.0
            let mutable normSquared = 0.0
            for c in 0 .. components - 1 do
                let inputPixels = span<float> vector.Components[c]
                let value = inputPixels[i]
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
    if vectorComponentCount a <> vectorComponentCount b then
        invalidArg "b" $"{name}: component counts differ: {vectorComponentCount a} vs {vectorComponentCount b}."

let mapVectorElementsFloat32 (f: float32 -> float32) (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    let output = createVectorChunk<float32> vector.SpatialSize (vectorComponentCount vector)
    try
        for c in 0 .. vector.Components.Length - 1 do
            let inputPixels = span<float32> vector.Components[c]
            let outputPixels = span<float32> output.Components[c]
            for i in 0 .. inputPixels.Length - 1 do
                outputPixels[i] <- f inputPixels[i]
        output
    with
    | _ ->
        decRefVector output
        reraise()

let vectorDotFloat32 (a: VectorChunk<float32>) (b: VectorChunk<float32>) =
    ensureMatchingVectorChunksFloat32 "vectorDotFloat32" a b
    let components = a.Components.Length
    let output = create<float32> a.SpatialSize
    try
        let outputPixels = span<float32> output
        outputPixels.Clear()
        for c in 0 .. components - 1 do
            let aPixels = span<float32> a.Components[c]
            let bPixels = span<float32> b.Components[c]
            for i in 0 .. outputPixels.Length - 1 do
                outputPixels[i] <- outputPixels[i] + aPixels[i] * bPixels[i]
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorMagnitudeFloat32 (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    let components = vector.Components.Length
    let output = create<float32> vector.SpatialSize
    try
        let outputPixels = span<float32> output
        outputPixels.Clear()
        for c in 0 .. components - 1 do
            let inputPixels = span<float32> vector.Components[c]
            for i in 0 .. outputPixels.Length - 1 do
                let value = inputPixels[i]
                outputPixels[i] <- outputPixels[i] + value * value
        for i in 0 .. outputPixels.Length - 1 do
            outputPixels[i] <- sqrt outputPixels[i]
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorMagnitudeSquaredFloat32 (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    let components = vector.Components.Length
    let output = create<float32> vector.SpatialSize
    try
        let outputPixels = span<float32> output
        outputPixels.Clear()
        for c in 0 .. components - 1 do
            let inputPixels = span<float32> vector.Components[c]
            for i in 0 .. outputPixels.Length - 1 do
                let value = inputPixels[i]
                outputPixels[i] <- outputPixels[i] + value * value
        output
    with
    | _ ->
        decRef output
        reraise()

let vectorAngleToFloat32 (reference: float32 list) (vector: VectorChunk<float32>) =
    validateVectorStorage "vector" vector
    let components = vector.Components.Length
    if reference.Length <> components then
        invalidArg "reference" $"vectorAngleToFloat32: reference vector has {reference.Length} components, vector chunk has {vectorComponentCount vector}."

    let referenceValues = reference |> List.toArray
    let referenceNorm = referenceValues |> Array.sumBy (fun value -> value * value) |> sqrt
    if referenceNorm < 1e-18f then
        invalidArg "reference" "vectorAngleToFloat32: reference vector must be non-zero."

    let output = create<float32> vector.SpatialSize
    try
        let outputPixels = span<float32> output
        for i in 0 .. outputPixels.Length - 1 do
            let mutable dot = 0.0f
            let mutable normSquared = 0.0f
            for c in 0 .. components - 1 do
                let inputPixels = span<float32> vector.Components[c]
                let value = inputPixels[i]
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
