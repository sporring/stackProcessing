module SmartImagePipeline

open System
open System.IO
open Plotly.NET
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open itk.simple

// --- Types ---
/// <summary>
/// Represents a slice of a stack of 2d images. 
/// </summary>
type ImageSlice = {
    Index: uint
    Image: Image
}

/// <summary>
/// Represents memory usage strategies during image processing.
/// </summary>
type MemoryProfile =
    | Streaming // Slice by slice independently
    | Sliding of uint // Sliding window of slices of depth
    | Buffered // All slices of depth

    /// <summary>
    /// Estimates the amount of memory required for the given dimensions.
    /// </summary>
    /// <param name="width">Image width in pixels</param>
    /// <param name="height">Image height in pixels</param>
    /// <param name="depth">Number of image slices</param>
    member this.EstimateUsage (width: uint) (height: uint) (depth: uint) : uint64 =
        let pixelSize = 1UL // Assume 1 byte per pixel for UInt8
        let sliceBytes = (uint64 width) * (uint64 height) * pixelSize
        match this with
            | Streaming -> sliceBytes
            | Sliding windowSize -> sliceBytes * uint64 windowSize
            | Buffered -> sliceBytes * uint64 depth

    /// <summary>
    /// Determines whether buffering is required based on available memory and image size.
    /// </summary>
    /// <param name="availableMemory">Memory available for processing</param>
    /// <param name="width">Image width</param>
    /// <param name="height">Image height</param>
    /// <param name="depth">Image depth</param>
    member this.RequiresBuffering (availableMemory: uint64) (width: uint) (height: uint) (depth: uint) : bool =
        let usage = this.EstimateUsage width height depth
        usage > availableMemory

/// <summary>
/// Represents a configurable image processing step that operates on image slices.
/// </summary>
type SmartProcessor = {
    /// <summary>
    /// Defines the memory strategy used when applying this processor (e.g., streaming, buffered).
    /// </summary>
    Profile: MemoryProfile

    /// <summary>
    /// The function that processes a stream of image slices and returns a transformed stream.
    /// </summary>
    Apply: AsyncSeq<ImageSlice> -> AsyncSeq<ImageSlice>
}

// --- AsyncSeq extensions ---
module AsyncSeqExtensions =
    /// <summary>
    /// Attempts to retrieve the item at the specified index from an asynchronous sequence.
    /// </summary>
    /// <param name="n">The zero-based index of the item to retrieve.</param>
    /// <param name="source">The asynchronous sequence to retrieve the item from.</param>
    /// <returns>
    /// An asynchronous computation that yields <c>Some(item)</c> if found, or <c>None</c> if the index is out of bounds.
    /// </returns>
    let tryItem (n: int) (source: AsyncSeq<'T>) : Async<'T option> =
        async {
            let mutable i = 0
            let mutable result = None
            use enumerator = source.GetEnumerator()
            let rec loop () =
                async {
                    match! enumerator.MoveNext() with
                        | Some item ->
                            if i = n then
                                result <- Some item
                            else
                                i <- i + 1
                                return! loop ()
                        | None -> return ()
                }
            do! loop ()
            return result
        }

    /// <summary>
    /// Creates a sliding window over an asynchronous sequence, returning lists of elements of the specified window size.
    /// </summary>
    /// <param name="windowSize">The number of elements in each window. Must be greater than 0.</param>
    /// <param name="source">The asynchronous sequence to create windows from.</param>
    /// <returns>
    /// An asynchronous sequence of lists, where each list represents a window of elements from the source sequence.
    /// </returns>
    /// <exception cref="System.ArgumentException">
    /// Thrown if <paramref name="windowSize"/> is less than or equal to 0.
    /// </exception>
    let windowed (windowSize: int) (source: AsyncSeq<'T>) : AsyncSeq<'T list> =
        if windowSize <= 0 then
            invalidArg "windowSize" "Must be greater than 0"
        asyncSeq {
            let queue = System.Collections.Generic.Queue<'T>()
            for awaitElem in source do
                queue.Enqueue awaitElem
                if queue.Count = windowSize then
                    yield queue |> Seq.toList
                    queue.Dequeue() |> ignore
        }

    /// <summary>
    /// Splits an asynchronous sequence into fixed-size chunks.
    /// </summary>
    /// <param name="chunkSize">The number of elements per chunk. Must be greater than 0.</param>
    /// <param name="source">The asynchronous sequence to divide into chunks.</param>
    /// <returns>
    /// An asynchronous sequence of lists, where each list contains up to <paramref name="chunkSize"/> elements.
    /// The last chunk may be smaller if the total number of elements is not a multiple of the chunk size.
    /// </returns>
    /// <exception cref="System.ArgumentException">
    /// Thrown if <paramref name="chunkSize"/> is less than or equal to 0.
    /// </exception>
    let chunkBySize (chunkSize: int) (source: AsyncSeq<'T>) : AsyncSeq<'T list> =
        if chunkSize <= 0 then
            invalidArg "chunkSize" "Chunk size must be greater than 0"
        asyncSeq {
            let buffer = ResizeArray<'T>(chunkSize)
            for awaitElem in source do
                buffer.Add(awaitElem)
                if buffer.Count = chunkSize then
                    yield buffer |> Seq.toList
                    buffer.Clear()
            // yield remaining items
            if buffer.Count > 0 then
                yield buffer |> Seq.toList
        }

// --- I/O Utilities ---
module IO =
    /// <summary>
    /// Gets the number of TIFF image slices in a specified directory, representing the image volume depth.
    /// </summary>
    /// <param name="inputDir">The directory path to search for TIFF files.</param>
    /// <returns>
    /// The number of TIFF files in the directory, returned as an unsigned integer (uint).
    /// </returns>
    let getDepth (inputDir: string) : uint =
        let files = Directory.GetFiles(inputDir, "*.tiff") |> Array.sort
        files.Length |> uint

    /// <summary>
    /// Asynchronously reads TIFF image slices from a directory and returns them as an asynchronous sequence of <c>ImageSlice</c>.
    /// </summary>
    /// <param name="inputDir">The directory containing TIFF image files.</param>
    /// <returns>
    /// An asynchronous sequence of <c>ImageSlice</c> values, where each slice includes an index and its corresponding image.
    /// </returns>
    /// <remarks>
    /// TIFF files are sorted by filename and loaded using <c>ImageFileReader</c>. Each read is performed asynchronously.
    /// </remarks>
    let readSlicesAsync (inputDir: string) : AsyncSeq<ImageSlice> =
        Directory.GetFiles(inputDir, "*.tiff") |> Array.sort
        |> Array.mapi (fun i filePath ->
            async {
                printfn "[Read] Loading slice %d from %s" i filePath
                let reader = new ImageFileReader()
                reader.SetFileName(filePath)
                let img = reader.Execute()
                return { Index = uint i; Image = img }
            })
        |> Seq.ofArray
        |> AsyncSeq.ofSeqAsync

    /// <summary>
    /// Asynchronously writes a sequence of <c>ImageSlice</c> objects to TIFF files in the specified output directory.
    /// </summary>
    /// <param name="outputDir">The directory where TIFF files will be saved. Created if it doesn't exist.</param>
    /// <param name="slices">An asynchronous sequence of image slices to write.</param>
    /// <returns>
    /// An asynchronous operation that writes each image slice to disk as a separate TIFF file named <c>slice_###.tiff</c>.
    /// </returns>
    /// <remarks>
    /// Each slice is saved using its index to construct a zero-padded filename. Writes occur in the order received.
    /// </remarks>
    let writeSlicesAsync (outputDir: string) (slices: AsyncSeq<ImageSlice>) =
        if not (Directory.Exists(outputDir)) then
            Directory.CreateDirectory(outputDir) |> ignore
        slices
        |> AsyncSeq.iterAsync (fun slice ->
            async {
                let path = Path.Combine(outputDir, sprintf "slice_%03d.tiff" slice.Index)
                let writer = new ImageFileWriter()
                writer.SetFileName(path)
                writer.Execute(slice.Image)
                printfn "[Write] Saved slice %d to %s" slice.Index path
            })

    /// <summary>
    /// Synchronously writes a sequence of <c>ImageSlice</c> objects to TIFF files by blocking on the asynchronous writer.
    /// </summary>
    /// <param name="path">The output directory where TIFF files will be saved.</param>
    /// <param name="stream">An asynchronous sequence of image slices to write.</param>
    /// <returns>
    /// Unit. This function blocks until all slices are written to disk.
    /// </returns>
    /// <remarks>
    /// Internally calls <see cref="writeSlicesAsync" /> and blocks using <c>Async.RunSynchronously</c>.
    /// Useful for scenarios where synchronous behavior is preferred.
    /// </remarks>
    let writeSlicesSync path stream =
        writeSlicesAsync path stream |> Async.RunSynchronously

// --- Pipeline computation expression ---
/// <summary>
/// Provides computation expression support for building memory-aware image processing pipelines.
/// Automatically inserts disk-based buffering when memory limits are exceeded.
/// </summary>
/// <param name="availableMemory">Maximum memory available for processing.</param>
/// <param name="width">Width of each image slice.</param>
/// <param name="height">Height of each image slice.</param>
/// <param name="depth">Total number of image slices.</param>
type PipelineBuilder(availableMemory: uint64, width: uint, height: uint, depth: uint) =
    /// <summary>
    /// Chains two <c>SmartProcessor</c> instances, optionally inserting intermediate disk I/O
    /// if the combined profile exceeds available memory.
    /// </summary>
    /// <param name="p">The input processor.</param>
    /// <param name="f">A function that transforms the processor.</param>
    /// <returns>A new <c>SmartProcessor</c> that includes buffering if needed.</returns>
    member _.Bind(p: SmartProcessor, f: SmartProcessor -> SmartProcessor) : SmartProcessor =
        let composed = f p
        let combinedProfile = composed.Profile
        if combinedProfile.RequiresBuffering availableMemory  width  height depth then
            printfn "[Memory] Exceeded memory limits. Splitting pipeline."
            let tempDir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName())
            Directory.CreateDirectory(tempDir) |> ignore
            let intermediate = fun input ->
                // Step 1: Write intermediate slices to disk
                IO.writeSlicesAsync tempDir input
                |> Async.RunSynchronously
                // Step 2: Read them back for next stage
                IO.readSlicesAsync tempDir
            { Profile = composed.Profile; Apply = composed.Apply << intermediate } // The profile needs to be reset here. How to do that?
        else
            composed

    /// <summary>
    /// Wraps a processor value for use in the pipeline computation expression.
    /// </summary>
    member _.Return(p: SmartProcessor) = p

    /// <summary>
    /// Allows returning a processor directly from another computation expression.
    /// </summary>
    member _.ReturnFrom(p: SmartProcessor) = p

    /// <summary>
    /// Provides a default identity processor using streaming as the memory profile.
    /// </summary>
    member _.Zero() = { Profile = Streaming; Apply = id }

/// <summary>
/// Combines two <c>SmartProcessor</c> instances into one by composing their memory profiles and transformation functions.
/// </summary>
/// <param name="p1">The first processor to apply.</param>
/// <param name="p2">The second processor to apply.</param>
/// <returns>
/// A new <c>SmartProcessor</c> whose memory profile is the more restrictive of the two,
/// and whose apply function is the composition of the two processors' apply functions.
/// </returns>
/// <remarks>
/// Memory profile resolution follows these rules:
/// <list type="bullet">
/// <item><description><c>Streaming + Streaming = Streaming</c></description></item>
/// <item><description><c>Streaming + Sliding(sz) = Sliding(sz)</c></description></item>
/// <item><description><c>Sliding(sz1) + Sliding(sz2) = Sliding(max sz1 sz2)</c></description></item>
/// <item><description>Any combination involving <c>Buffered</c> results in <c>Buffered</c>.</description></item>
/// </list>
/// </remarks>
let (>>=>) (p1: SmartProcessor) (p2: SmartProcessor) : SmartProcessor =
    {
        Profile = 
            match p1.Profile, p2.Profile with
                | Streaming, Streaming -> Streaming
                | Sliding sz, Streaming
                | Streaming, Sliding sz -> Sliding sz
                | Sliding sz1, Sliding sz2 -> Sliding (max sz1 sz2)
                | Streaming, Buffered
                | Buffered, Streaming
                | Sliding _, Buffered
                | Buffered, Sliding _
                | Buffered, Buffered -> Buffered
        Apply = fun input -> input |> p1.Apply |> p2.Apply
    }

/// <summary>
/// Initializes a memory-aware pipeline builder with the specified processing constraints.
/// </summary>
/// <param name="availableMemory">The total memory available for the pipeline in bytes.</param>
/// <param name="width">The width of each image slice.</param>
/// <param name="height">The height of each image slice.</param>
/// <param name="depth">The total number of image slices (volume depth).</param>
/// <returns>
/// A <c>PipelineBuilder</c> instance that supports memory-constrained pipeline composition using computation expressions.
/// </returns>
let pipeline availableMemory width height depth = PipelineBuilder(availableMemory, width, height, depth)

// --- Processing Utilities ---
module Processing =
    /// <summary>
    /// Retrieves the intensity value of a voxel at the specified 3D coordinates from an image.
    /// </summary>
    /// <param name="image">The 3D image to access.</param>
    /// <param name="x">The x-coordinate (column) of the voxel.</param>
    /// <param name="y">The y-coordinate (row) of the voxel.</param>
    /// <param name="z">The z-coordinate (slice index) of the voxel.</param>
    /// <returns>The intensity value at the specified location as a byte.</returns>
    let getPixel3D (image: Image) (x: uint) (y: uint) (z: uint) : byte =
        image.GetPixelAsUInt8(new VectorUInt32([| x; y; z |]))

    /// <summary>
    /// Extracts a 2D slice from a 3D image at the given z-index.
    /// </summary>
    /// <param name="image">The 3D image volume to slice.</param>
    /// <param name="z">The slice index along the z-axis.</param>
    /// <returns>A 2D image representing the extracted slice.</returns>
    /// <remarks>
    /// Uses <c>ExtractImageFilter</c> with a fixed size in the z-dimension of 1.
    /// </remarks>
    let extractSlice (image: Image) (z: uint) : Image =
        let width = image.GetWidth() |> uint
        let height = image.GetHeight() |> uint
        let extractor = new ExtractImageFilter()
        extractor.SetSize(new VectorUInt32([| width; height; 1u |]))
        extractor.SetIndex(new VectorInt32([| 0; 0; int z |]))
        extractor.Execute(image)

    /// <summary>
    /// Stacks a list of 2D image slices into a single 3D volume.
    /// </summary>
    /// <param name="slices">A non-empty list of <c>ImageSlice</c> objects to stack.</param>
    /// <returns>
    /// A new <c>ImageSlice</c> containing a 3D volume built from the input slices, using the index of the first slice.
    /// </returns>
    /// <exception cref="System.ArgumentException">
    /// Thrown if the input list is empty.
    /// </exception>
    /// <remarks>
    /// This function assumes all input images are aligned and have matching dimensions.
    /// </remarks>
    let stack (slices: ImageSlice list) : ImageSlice =
        let idx = slices[0].Index
        let images = List.map (fun s -> s.Image) slices
        let joiner = new JoinSeriesImageFilter()
        let volume = joiner.Execute(new VectorOfImage(images))
        { Index = idx; Image = volume }

    /// <summary>
    /// Decomposes a 3D volume <c>ImageSlice</c> into an asynchronous sequence of 2D slices.
    /// </summary>
    /// <param name="slices">An <c>ImageSlice</c> containing a 3D volume image.</param>
    /// <returns>
    /// An <c>AsyncSeq</c> of 2D image slices extracted from the volume, preserving spatial order.
    /// </returns>
    /// <remarks>
    /// Each extracted slice is assigned a sequential index based on the base index of the input volume.
    /// </remarks>
    let unstack (slices: ImageSlice) : AsyncSeq<ImageSlice> =
        let baseIndex = slices.Index
        let volume = slices.Image
        let size = volume.GetSize()
        let width, height, depth = size.[0], size.[1], size.[2]
        let extractor = new ExtractImageFilter()
        extractor.SetSize(new VectorUInt32([| width; height; 0u |]))
        Seq.init (int depth) (fun z ->
            extractor.SetIndex(new VectorInt32([| 0; 0; z |]))
            let slice = extractor.Execute(volume)
            { Index = baseIndex + uint z; Image = slice })
        |> AsyncSeq.ofSeq

    /// <summary>
    /// Multiplies the pixel values of two image slices element-wise.
    /// </summary>
    /// <param name="s1">The first image slice.</param>
    /// <param name="s2">The second image slice.</param>
    /// <returns>
    /// A new <c>ImageSlice</c> containing the result of the element-wise multiplication.
    /// </returns>
    /// <remarks>
    /// Assumes both input slices have the same dimensions.
    /// </remarks>
    let multiplySlices (s1:ImageSlice) (s2:ImageSlice) : ImageSlice =
        let filter = new MultiplyImageFilter()
        { s1 with Image = filter.Execute (s1.Image, s2.Image) }

    /// <summary>
    /// Adds the pixel values of two image slices element-wise.
    /// </summary>
    /// <param name="s1">The first image slice.</param>
    /// <param name="s2">The second image slice.</param>
    /// <returns>
    /// A new <c>ImageSlice</c> containing the sum of the pixel values.
    /// </returns>
    /// <remarks>
    /// Assumes both input slices are aligned and dimensionally compatible.
    /// </remarks>
    let addSlices (s1:ImageSlice) (s2:ImageSlice) : ImageSlice =
        let filter = new AddImageFilter()
        { s1 with Image = filter.Execute (s1.Image, s2.Image) }

    /// <summary>
    /// Applies a linear transformation to the pixel values of an image slice using shift and scale.
    /// </summary>
    /// <param name="delta">The value to shift (add to) each pixel.</param>
    /// <param name="factor">The scale multiplier for each pixel.</param>
    /// <param name="slice">The input image slice to transform.</param>
    /// <returns>
    /// A new <c>ImageSlice</c> with transformed pixel values.
    /// </returns>
    let shiftScaleSlice (delta: float) (factor: float) (slice:ImageSlice) : ImageSlice =
        let filter = new ShiftScaleImageFilter ()
        filter.SetScale factor
        filter.SetShift delta
        { slice with Image = filter.Execute slice.Image }

    /// <summary>
    /// Applies additive Gaussian noise to an image slice.
    /// </summary>
    /// <param name="mean">The mean of the Gaussian noise.</param>
    /// <param name="stddev">The standard deviation of the Gaussian noise.</param>
    /// <param name="slice">The input image slice to which noise will be added.</param>
    /// <returns>
    /// A new <c>ImageSlice</c> with Gaussian noise applied.
    /// </returns>
    let additiveGaussianNoiseSlice (mean: float) (stddev: float) (slice: ImageSlice) : ImageSlice =
        let filter = new AdditiveGaussianNoiseImageFilter()
        filter.SetMean(mean)
        filter.SetStandardDeviation(stddev)
        { slice with Image = filter.Execute(slice.Image) }

    /// <summary>
    /// Applies a binary threshold to an image slice, setting pixel values outside the range to zero.
    /// </summary>
    /// <param name="lower">The lower threshold value (inclusive).</param>
    /// <param name="upper">The upper threshold value (inclusive).</param>
    /// <returns>
    /// A function that transforms an <c>ImageSlice</c> by applying the binary threshold.
    /// </returns>
    /// <remarks>
    /// Pixels with values between <paramref name="lower"/> and <paramref name="upper"/> are preserved;
    /// others are set to zero or the background.
    /// </remarks>
    let thresholdSlice (lower: float) (upper: float) : ImageSlice -> ImageSlice =
        fun slice ->
            let filter = new BinaryThresholdImageFilter()
            filter.SetLowerThreshold lower
            filter.SetUpperThreshold upper
            { slice with Image = filter.Execute slice.Image }

    /// <summary>
    /// Applies a 3D discrete Gaussian filter to a volumetric slice and extracts a representative 2D image from the center slice.
    /// </summary>
    /// <param name="sigma">The standard deviation (blur amount) of the Gaussian kernel.</param>
    /// <param name="depth">The depth of the slice stack to simulate a 3D convolution.</param>
    /// <returns>
    /// A function that transforms an <c>ImageSlice</c> by applying 3D Gaussian smoothing and extracting the central slice.
    /// </returns>
    /// <remarks>
    /// The center slice is computed as <c>depth / 2</c>. This is a pseudo-3D convolution assuming a volume around the input slice.
    /// </remarks>
    let convolve3DGaussianSlice (sigma: float) (depth: uint): ImageSlice -> ImageSlice =
        fun slice ->
            let filter = new DiscreteGaussianImageFilter()
            filter.SetVariance (sigma * sigma)
            let volume = filter.Execute slice.Image
            let image = extractSlice volume (depth/2u)
            { slice with Image = image }

    /// <summary>
    /// Combines two asynchronous sequences of image slices element-wise using a user-defined function.
    /// </summary>
    /// <param name="combine">A function that defines how to combine two <c>ImageSlice</c> values.</param>
    /// <param name="a">The first asynchronous sequence of image slices.</param>
    /// <param name="b">The second asynchronous sequence of image slices.</param>
    /// <returns>
    /// An asynchronous sequence where each element is the result of applying <paramref name="combine"/> to corresponding elements of <paramref name="a"/> and <paramref name="b"/>.
    /// </returns>
    /// <remarks>
    /// The resulting sequence ends when the shorter of the two input sequences ends.
    /// </remarks>
    let zipJoin (combine: ImageSlice -> ImageSlice -> ImageSlice) 
                (a: AsyncSeq<ImageSlice>) 
                (b: AsyncSeq<ImageSlice>) : AsyncSeq<ImageSlice> =
        AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> combine x y)

    /// <summary>
    /// Creates a <c>SmartProcessor</c> that adds each input slice to a corresponding slice from another asynchronous sequence.
    /// </summary>
    /// <param name="other">An asynchronous sequence of image slices to add to the input stream.</param>
    /// <returns>
    /// A <c>SmartProcessor</c> that performs element-wise addition of image slices using the <c>addSlices</c> function.
    /// </returns>
    /// <remarks>
    /// Assumes both sequences are aligned in order and size.
    /// </remarks>
    let addTo (other: AsyncSeq<ImageSlice>) : SmartProcessor =
        {   Profile = Streaming
            Apply = fun input ->
                zipJoin addSlices input other
        }

    /// <summary>
    /// Creates a <c>SmartProcessor</c> that multiplies each input slice with a corresponding slice from another asynchronous sequence.
    /// </summary>
    /// <param name="other">An asynchronous sequence of image slices to multiply with the input stream.</param>
    /// <returns>
    /// A <c>SmartProcessor</c> that performs element-wise multiplication using the <c>multiplySlices</c> function.
    /// </returns>
    /// <remarks>
    /// Assumes both sequences are aligned in order and size.
    /// </remarks>
    let multiplyWith (other: AsyncSeq<ImageSlice>) : SmartProcessor =
        {
            Profile = Streaming
            Apply = fun input ->
                zipJoin multiplySlices input other
        }

    /// <summary>
    /// Applies a transformation function to each image slice in a streaming fashion, with error handling and logging.
    /// </summary>
    /// <param name="label">A label used for logging errors during slice processing.</param>
    /// <param name="profile">The memory profile to associate with the resulting processor (typically <c>Streaming</c>).</param>
    /// <param name="f">A function that transforms a single <c>ImageSlice</c>.</param>
    /// <returns>
    /// A <c>SmartProcessor</c> that applies <paramref name="f"/> to each image slice individually.
    /// </returns>
    /// <remarks>
    /// If a transformation fails, the original slice is returned and the error is logged with the specified label.
    /// </remarks>
    let mapSlices (label: string) (profile: MemoryProfile) (f: ImageSlice -> ImageSlice) : SmartProcessor =
        {
            Profile = profile
            Apply = fun input ->
                input
                |> AsyncSeq.map (fun slice ->
                    try
                        f slice
                    with ex ->
                        printfn "[Error] %s failed on slice %d: %s" label slice.Index ex.Message
                        slice)
        }

    /// <summary>
    /// Applies a transformation to overlapping windows of slices by stacking them into a temporary volume,
    /// then applying the function to the stacked image.
    /// </summary>
    /// <param name="label">A label used for error reporting and logging.</param>
    /// <param name="depth">The size of the sliding window.</param>
    /// <param name="f">A function that operates on a stacked <c>ImageSlice</c> (representing a volume).</param>
    /// <returns>
    /// A <c>SmartProcessor</c> using a <c>Sliding</c> memory profile that applies <paramref name="f"/> to each stacked window.
    /// </returns>
    /// <remarks>
    /// If processing fails for a window, the stacked input slice is returned as-is and the error is logged.
    /// </remarks>
    let mapSlicesWindowed (label: string) (depth: uint) (f: ImageSlice -> ImageSlice) : SmartProcessor =
        {
            Profile = Sliding depth
            Apply = fun input ->
                AsyncSeqExtensions.windowed (int depth) input
                |> AsyncSeq.map (fun window ->
                    let stacked = stack window
                    try
                        f stacked
                    with ex ->
                        printfn "[Error] %s failed on windowed slice %d: %s" label stacked.Index ex.Message
                        stacked)
        }

    /// <summary>
    /// Applies a transformation function to fixed-size chunks of image slices. Each chunk is stacked into a volume before processing.
    /// </summary>
    /// <param name="label">A label for error reporting in case the chunk fails to process.</param>
    /// <param name="chunkSize">The number of slices per chunk.</param>
    /// <param name="baseIndex">The index assigned to the stacked chunk before unstacking.</param>
    /// <param name="f">The function to apply to the stacked volume slice.</param>
    /// <returns>
    /// A <c>SmartProcessor</c> using a <c>Sliding</c> memory profile that applies the transformation to each chunked volume.
    /// </returns>
    /// <remarks>
    /// If processing fails for a chunk, the original unmodified chunk is returned as a fallback.
    /// </remarks>
    let mapSlicesChunked (label: string) (chunkSize: uint) (baseIndex: uint) (f: ImageSlice -> ImageSlice) : SmartProcessor =
        {
            Profile = Sliding chunkSize
            Apply = fun input ->
                AsyncSeqExtensions.chunkBySize (int chunkSize) input
                |> AsyncSeq.collect (fun chunk ->
                    try
                        let volume = stack chunk
                        let result = f { volume with Index = baseIndex }
                        unstack result
                    with ex ->
                        printfn "[Error] %s failed on chunk starting at %d: %s" label chunk.[0].Index ex.Message
                        AsyncSeq.ofSeq chunk) // fallback
        }

    /// <summary>
    /// Generates a constant-valued volume of image slices.
    /// </summary>
    /// <param name="value">The constant pixel value to assign.</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="depth">Number of slices to generate.</param>
    let constant (value: byte) (width: uint) (height: uint) (depth: uint) : AsyncSeq<ImageSlice> =
        AsyncSeq.init (int depth) (fun i ->
            let image = new Image(width, height, PixelIDValueEnum.sitkUInt8)
            { Index = uint i; Image = image } |> shiftScaleSlice (float value) 1.0)

    /// <summary>
    /// Adds Gaussian noise to each image slice.
    /// </summary>
    /// <param name="mean">Mean of the Gaussian distribution.</param>
    /// <param name="stddev">Standard deviation of the Gaussian distribution.</param>
    let additiveGaussianNoise (mean: float) (stddev: float) : SmartProcessor =
        mapSlices "Additive Gaussian noise" Streaming (additiveGaussianNoiseSlice mean stddev)

    /// <summary>
    /// Applies a shift and scale transformation to pixel values in each image slice.
    /// </summary>
    /// <param name="delta">The amount to shift pixel values.</param>
    /// <param name="factor">The scale factor for pixel values.</param>
    let shiftScale (delta: float) (factor: float) : SmartProcessor =
        mapSlices "Shift Scale" Streaming (shiftScaleSlice delta factor)

    /// <summary>
    /// Creates a <c>SmartProcessor</c> that applies binary thresholding to each image slice.
    /// </summary>
    /// <param name="lower">The lower threshold value (inclusive).</param>
    /// <param name="upper">The upper threshold value (inclusive).</param>
    /// <returns>
    /// A <c>SmartProcessor</c> using a <c>Streaming</c> profile that thresholds each slice using the specified bounds.
    /// </returns>
    /// <remarks>
    /// Pixel values between <paramref name="lower"/> and <paramref name="upper"/> are preserved, others are suppressed.
    /// </remarks>
    let threshold (lower: float) (upper: float) : SmartProcessor =
        mapSlices "Threshold" Streaming (thresholdSlice lower upper)

    /// <summary>
    /// Creates a <c>SmartProcessor</c> that applies 3D Gaussian smoothing to a sliding window of image slices.
    /// </summary>
    /// <param name="sigma">The standard deviation of the Gaussian kernel.</param>
    /// <returns>
    /// A <c>SmartProcessor</c> using a <c>Sliding</c> profile that stacks slices into a volume and smooths them with a 3D Gaussian filter.
    /// </returns>
    /// <remarks>
    /// The window depth is automatically computed as <c>1 + 2 * round(sigma)</c>.
    /// Only the center slice of the smoothed volume is returned per window.
    /// </remarks>
    let convolve3DGaussian (sigma: float) : SmartProcessor =
        let depth = 1u + 2u * uint (0.5 + sigma)
        mapSlicesWindowed "3D Gaussian" depth (convolve3DGaussianSlice sigma depth)

// --- Visualization ---
module Visualization =
    /// <summary>
    /// Visualizes a specific image slice from an asynchronous sequence by displaying it as a heatmap.
    /// </summary>
    /// <param name="idx">The zero-based index of the slice to display.</param>
    /// <param name="slices">The asynchronous sequence of <c>ImageSlice</c> objects to retrieve from.</param>
    /// <returns>
    /// An asynchronous workflow that attempts to retrieve the specified slice and, if found, displays it as a heatmap using Plotly.
    /// </returns>
    /// <remarks>
    /// - Uses <c>tryItem</c> to safely access the slice by index.
    /// - Outputs slice dimensions to the console.
    /// - Uses <c>Processing.getPixel3D</c> to extract pixel data for visualization.
    /// </remarks>
    let showSliceAtIndex (idx: int) (slices: AsyncSeq<ImageSlice>) : Async<unit> =
        async {
            let! maybeSlice =
                slices
                |> AsyncSeqExtensions.tryItem idx

            match maybeSlice with
            | Some slice ->
                let image = slice.Image
                let width = image.GetWidth() |> int
                let height = image.GetHeight() |> int
                printfn "Showing image slice %d which has size %d x %d" idx width height
                let data =
                    Seq.init height (fun y ->
                        Seq.init width (fun x ->
                            Processing.getPixel3D image (uint x) (uint y) 0u |> float))
                data |> Chart.Heatmap |> Chart.show
            | None ->
                printfn "[Visualization] No slice found at index %d." idx
        }
