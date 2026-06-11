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
