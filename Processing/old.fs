module Processing

open System
open System.IO
open FSharp.Control
open System.Collections.Generic
open System.Collections.Concurrent
open AsyncSeqExtensions
open StackPipeline
open Slice

// --- Types ---
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
type StackProcessor<'S,'T> = {
    /// <summary>
    /// A name of this processor (e.g., "Thresholding").
    /// </summary>
    Name: string

    /// <summary>
    /// Defines the memory strategy used when applying this processor (e.g., streaming, buffered).
    /// </summary>
    Profile: MemoryProfile

    /// <summary>
    /// The function that processes a stream of image slices and returns a transformed stream.
    /// </summary>
    Apply: AsyncSeq<'S> -> AsyncSeq<'T>
}

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
    /// Chains two <c>StackProcessor</c> instances, optionally inserting intermediate disk I/O
    /// if the combined profile exceeds available memory.
    /// </summary>
    /// <param name="p">The input processor.</param>
    /// <param name="f">A function that transforms the processor.</param>
    /// <returns>A new <c>StackProcessor</c> that includes buffering if needed.</returns>
    member _.Bind(p: StackProcessor<'S,'T>, f: StackProcessor<'S,'T> -> StackProcessor<'S,'T>) : StackProcessor<'S,'T> =
        let composed = f p
        let combinedProfile = composed.Profile
        if combinedProfile.RequiresBuffering availableMemory  width  height depth then
            printfn "[Memory] Exceeded memory limits. Splitting pipeline."
            let tempDir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName())
            Directory.CreateDirectory(tempDir) |> ignore
            let intermediate = fun input ->
                // Step 1: Write intermediate slices to disk
                writeSlicesAsync tempDir input
                |> Async.RunSynchronously
                // Step 2: Read them back for next stage
                readSlices tempDir

            { Name = $"{composed.Name} {p.Name}"; Profile = Streaming; Apply = composed.Apply << intermediate } // The profile needs to be reset here. How to do that?
        else
            composed

    /// <summary>
    /// Wraps a processor value for use in the pipeline computation expression.
    /// </summary>
    member _.Return(p: StackProcessor<'S,'T>) = p

    /// <summary>
    /// Allows returning a processor directly from another computation expression.
    /// </summary>
    member _.ReturnFrom(p: StackProcessor<'S,'T>) = p

    /// <summary>
    /// Provides a default identity processor using streaming as the memory profile.
    /// </summary>
    member _.Zero() = { Name=""; Profile = Streaming; Apply = id }

/// <summary>
/// Combines two <c>StackProcessor</c> instances into one by composing their memory profiles and transformation functions.
/// </summary>
/// <param name="p1">The first processor to apply.</param>
/// <param name="p2">The second processor to apply.</param>
/// <returns>
/// A new <c>StackProcessor</c> whose memory profile is the more restrictive of the two,
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
let (>>=>) (p1: StackProcessor<'S,'T>) (p2: StackProcessor<'T,'U>) : StackProcessor<'S,'U> =
    {
        Name = $"{p2.Name} {p1.Name}"; 
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
    /// Decomposes a 3D volume <c>Slice<'T></c> into an asynchronous sequence of 2D slices.
    /// </summary>
    /// <param name="slices">An <c>Slice<'T></c> containing a 3D volume image.</param>
    /// <returns>
    /// An <c>AsyncSeq</c> of 2D image slices extracted from the volume, preserving spatial order.
    /// </returns>
    /// <remarks>
    /// Each extracted slice is assigned a sequential index based on the base index of the input volume.
    /// </remarks>
    let unstack (slices: Slice<'T>) : AsyncSeq<Slice<'T>> =
        let baseIndex = slices.Index
        let volume = slices.Image
        let size = volume.GetSize()
        let width, height, depth = size.[0], size.[1], size.[2]
        Seq.init (int depth) (fun z -> extractSlice (baseIndex+(uint z)) slices)
        |> AsyncSeq.ofSeq

    /// <summary>
    /// Creates a <c>StackProcessor</c> that adds each input slice to a corresponding slice from another asynchronous sequence.
    /// </summary>
    /// <param name="other">An asynchronous sequence of image slices to add to the input stream.</param>
    /// <returns>
    /// A <c>StackProcessor</c> that performs element-wise addition of image slices using the <c>addSlices</c> function.
    /// </returns>
    /// <remarks>
    /// Assumes both sequences are aligned in order and size.
    /// </remarks>
    let addTo (other: AsyncSeq<Slice<'T>>) : StackProcessor<Slice<'T> ,Slice<'T>> =
        {   
            Name = "AddTo"
            Profile = Streaming
            Apply = fun input ->
                zipJoin addSlices input other
        }

    /// <summary>
    /// Creates a <c>StackProcessor</c> that multiplies each input slice with a corresponding slice from another asynchronous sequence.
    /// </summary>
    /// <param name="other">An asynchronous sequence of image slices to multiply with the input stream.</param>
    /// <returns>
    /// A <c>StackProcessor</c> that performs element-wise multiplication using the <c>multiplySlices</c> function.
    /// </returns>
    /// <remarks>
    /// Assumes both sequences are aligned in order and size.
    /// </remarks>
    let multiplyWith (other: AsyncSeq<Slice<'T>>) : StackProcessor<Slice<'T> ,Slice<'T>> =
        printfn "[multiplyWith]"
        {
            Name = "multiplyWith"
            Profile = Streaming
            Apply = fun input ->
                printfn "[multiplyWith]"
                zipJoin multiplySlices input other
        }

    /// <summary>
    /// Applies a transformation function to each image slice in a streaming fashion, with error handling and logging.
    /// </summary>
    /// <param name="label">A label used for logging errors during slice processing.</param>
    /// <param name="profile">The memory profile to associate with the resulting processor (typically <c>Streaming</c>).</param>
    /// <param name="f">A function that transforms a single <c>Slice<'T></c>.</param>
    /// <returns>
    /// A <c>StackProcessor</c> that applies <paramref name="f"/> to each image slice individually.
    /// </returns>
    /// <remarks>
    /// If a transformation fails, the original slice is returned and the error is logged with the specified label.
    /// </remarks>
    let mapSlices (label: string) (profile: MemoryProfile) (f: 'S -> 'T) : StackProcessor<'S,'T> =
        {
            Name = "mapSlices"; 
            Profile = profile
            Apply = fun input ->
                input
                |> AsyncSeq.map (fun slice -> f slice)
        }

    /// <summary>
    /// Applies a transformation to overlapping windows of slices by stacking them into a temporary volume,
    /// then applying the function to the stacked image.
    /// </summary>
    /// <param name="label">A label used for error reporting and logging.</param>
    /// <param name="depth">The size of the sliding window.</param>
    /// <param name="f">A function that operates on a stacked <c>Slice<'T></c> (representing a volume).</param>
    /// <returns>
    /// A <c>StackProcessor</c> using a <c>Sliding</c> memory profile that applies <paramref name="f"/> to each stacked window.
    /// </returns>
    /// <remarks>
    /// If processing fails for a window, the stacked input slice is returned as-is and the error is logged.
    /// </remarks>
    let mapSlicesWindowed (label: string) (depth: uint) (f: Slice<'T> -> Slice<'T>) : StackProcessor<Slice<'T>,Slice<'T>> =
        { // Due to stack, this is StackProcessor<Slice<'T>,Slice<'T>>
            Name = "mapSlicesWindowed"; 
            Profile = Sliding depth
            Apply = fun input ->
                AsyncSeqExtensions.windowed (int depth) input
                |> AsyncSeq.map (fun window -> window |> stack |> f)
        }

    /// <summary>
    /// Applies a transformation function to fixed-size chunks of image slices. Each chunk is stacked into a volume before processing.
    /// </summary>
    /// <param name="label">A label for error reporting in case the chunk fails to process.</param>
    /// <param name="chunkSize">The number of slices per chunk.</param>
    /// <param name="baseIndex">The index assigned to the stacked chunk before unstacking.</param>
    /// <param name="f">The function to apply to the stacked volume slice.</param>
    /// <returns>
    /// A <c>StackProcessor</c> using a <c>Sliding</c> memory profile that applies the transformation to each chunked volume.
    /// </returns>
    /// <remarks>
    /// If processing fails for a chunk, the original unmodified chunk is returned as a fallback.
    /// </remarks>
    let mapSlicesChunked (label: string) (chunkSize: uint) (baseIndex: uint) (f: Slice<'T> -> Slice<'T>) : StackProcessor<Slice<'T>,Slice<'T>> =
        { // Due to stack, this is StackProcessor<Slice<'T>,Slice<'T>>
            Name = "mapSlicesChunked"; 
            Profile = Sliding chunkSize
            Apply = fun input ->
                AsyncSeqExtensions.chunkBySize (int chunkSize) input
                |> AsyncSeq.collect (fun chunk ->
                        let volume = stack chunk
                        let result = f { volume with Index = baseIndex }
                        unstack result)
        }

    /// Lifts a reducer into a processor that emits a single output item as a stream.
    let fromReducer (name: string) (profile: MemoryProfile) (reducer: AsyncSeq<'In> -> Async<'Out>) : StackProcessor<'In, 'Out> =
        {
            Name = name
            Profile = profile
            Apply = fun input ->
                reducer input |> ofAsync
        }

    /// <summary>
    /// Creates a <c>StackProcessor</c> that adds a constant value to each pixel in the image slice.
    /// </summary>
    /// <param name="value">The constant value to add.</param>
    /// <returns>A <c>StackProcessor</c> that adds <paramref name="value"/> to each pixel.</returns>
    let add (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[add]"
        mapSlices "Add" Streaming (addConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that adds corresponding pixels from a second image to the input image slice.
    /// </summary>
    /// <param name="image">The image to add to the input slice.</param>
    /// <returns>A <c>StackProcessor</c> that adds corresponding pixels from <paramref name="image"/>.</returns>
    let addWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[addWithImage]"
        mapSlices "AddImage" Streaming (addImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that subtracts a constant value from each pixel in the image slice.
    /// </summary>
    /// <param name="value">The constant value to subtract.</param>
    /// <returns>A <c>StackProcessor</c> that subtracts <paramref name="value"/> from each pixel.</returns>
    let subtract (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[subtract]"
        mapSlices "Subtract" Streaming (subtractConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that subtracts corresponding pixels in a second image from the input image slice.
    /// </summary>
    /// <param name="image">The image to subtract from the input slice.</param>
    /// <returns>A <c>StackProcessor</c> that subtracts corresponding pixels from <paramref name="image"/>.</returns>
    let subtractWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[subtractWithImage]"
        mapSlices "SubtractImage" Streaming (subtractImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that divides each pixel in the image slice by a constant value.
    /// </summary>
    /// <param name="value">The constant divisor.</param>
    /// <returns>A <c>StackProcessor</c> that divides each pixel by <paramref name="value"/>.</returns>
    let divide (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[divide]"
        mapSlices "Divide" Streaming (divideConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that divides each pixel in the image slice by the corresponding pixel in a second image.
    /// </summary>
    /// <param name="image">The divisor image.</param>
    /// <returns>A <c>StackProcessor</c> that divides pixel-wise by <paramref name="image"/>.</returns>
    let divideWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[divideWithImage]"
        mapSlices "DivideImage" Streaming (divideImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that multiplies each pixel in the image slice by a constant value.
    /// </summary>
    /// <param name="value">The constant multiplier.</param>
    /// <returns>A <c>StackProcessor</c> that multiplies each pixel by <paramref name="value"/>.</returns>
    let multiply (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[multiply]"
        mapSlices "Multiply" Streaming (multiplyConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that multiplies each pixel in the image slice by the corresponding pixel in another image.
    /// </summary>
    /// <param name="image">The second image to multiply with.</param>
    /// <returns>A <c>StackProcessor</c> that performs pixel-wise multiplication with <paramref name="image"/>.</returns>
    let multiplyWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[multiplyWithImage]"
        mapSlices "MultiplyImage" Streaming (multiplyImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that computes the modulus (remainder) of each pixel with respect to a constant value.
    /// </summary>
    let modulus (value: uint) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[modulus]"
        mapSlices "Modulus" Streaming (modulusConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that computes the pixel-wise modulus (remainder) with another image.
    /// </summary>
    let modulusWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[modulusWithImage]"
        mapSlices "ModulusImage" Streaming (modulusImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that raises each pixel to a constant power.
    /// </summary>
    let pow (exponent: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[pow]"
        mapSlices "Power" Streaming (powConst exponent)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that raises each pixel to the power of the corresponding pixel in another image.
    /// </summary>
    let powWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[powWithImage]"
        mapSlices "PowerImage" Streaming (powImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels greater than or equal to a constant value.
    /// </summary>
    /// <param name="value">The comparison constant.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels ≥ <paramref name="value"/>.</returns>
    let greaterEqual (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[greaterEqual]"
        mapSlices "GreaterEqual" Streaming (greaterEqualConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that compares each pixel to the corresponding pixel in a reference image, marking those greater than or equal.
    /// </summary>
    /// <param name="image">The reference image.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels where input ≥ reference.</returns>
    let greaterEqualWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[greaterEqualWithImage]"
        mapSlices "GreaterEqualImage" Streaming (greaterEqualImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels strictly greater than a constant value.
    /// </summary>
    /// <param name="value">The comparison constant.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels &gt; <paramref name="value"/>.</returns>
    let greater (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[greater]"
        mapSlices "Greater" Streaming (greaterConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that compares each pixel to a reference image, marking those strictly greater.
    /// </summary>
    /// <param name="image">The reference image.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels where input &gt; reference.</returns>
    let greaterWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[greaterWithImage]"
        mapSlices "GreaterImage" Streaming (greaterImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels equal to a constant value.
    /// </summary>
    /// <param name="value">The value to test for equality.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels == <paramref name="value"/>.</returns>
    let equal (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[equal]"
        mapSlices "Equal" Streaming (equalConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels equal to corresponding pixels in a reference image.
    /// </summary>
    /// <param name="image">The reference image.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels where input == reference.</returns>
    let equalWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[equalWithImage]"
        mapSlices "EqualImage" Streaming (equalImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels not equal to a constant value.
    /// </summary>
    /// <param name="value">The value to compare against.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels != <paramref name="value"/>.</returns>
    let notEqual (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[notEqual]"
        mapSlices "NotEqual" Streaming (notEqualConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels not equal to the corresponding pixels in a reference image.
    /// </summary>
    /// <param name="image">The reference image.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels where input != reference.</returns>
    let notEqualWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[notEqualWithImage]"
        mapSlices "NotEqualImage" Streaming (notEqualImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels strictly less than a constant value.
    /// </summary>
    /// <param name="value">The comparison constant.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels &lt; <paramref name="value"/>.</returns>
    let less (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[less]"
        mapSlices "Less" Streaming (lessConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels less than the corresponding pixel in a reference image.
    /// </summary>
    /// <param name="image">The reference image.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels where input &lt; reference.</returns>
    let lessWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[lessWithImage]"
        mapSlices "LessImage" Streaming (lessImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels less than or equal to a constant value.
    /// </summary>
    /// <param name="value">The comparison constant.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels ≤ <paramref name="value"/>.</returns>
    let lessEqual (value: float) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[lessEqual]"
        mapSlices "LessEqual" Streaming (lessEqualConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that marks pixels less than or equal to the corresponding pixels in a reference image.
    /// </summary>
    /// <param name="image">The reference image.</param>
    /// <returns>A <c>StackProcessor</c> marking pixels where input ≤ reference.</returns>
    let lessEqualWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[lessEqualWithImage]"
        mapSlices "LessEqualImage" Streaming (lessEqualImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a bitwise AND between each pixel and a constant integer.
    /// </summary>
    /// <param name="value">The constant value to AND with each pixel.</param>
    let bitwiseAnd (value: int) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseAnd]"
        mapSlices "BitwiseAnd" Streaming (bitwiseAndConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a pixel-wise bitwise AND between two images.
    /// </summary>
    /// <param name="image">The second image to AND with each slice.</param>
    let bitwiseAndWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseAndWithImage]"
        mapSlices "BitwiseAndImage" Streaming (bitwiseAndImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a bitwise OR between each pixel and a constant integer.
    /// </summary>
    /// <param name="value">The constant value to OR with each pixel.</param>
    let bitwiseOr (value: int) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseOr]"
        mapSlices "BitwiseOr" Streaming (bitwiseOrConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a pixel-wise bitwise OR between two images.
    /// </summary>
    /// <param name="image">The second image to OR with each slice.</param>
    let bitwiseOrWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseOrWithImage]"
        mapSlices "BitwiseOrImage" Streaming (bitwiseOrImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a bitwise XOR between each pixel and a constant integer.
    /// </summary>
    /// <param name="value">The constant value to XOR with each pixel.</param>
    let bitwiseXor (value: int) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseXor]"
        mapSlices "BitwiseXor" Streaming (bitwiseXorConst value)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a pixel-wise bitwise XOR between two images.
    /// </summary>
    /// <param name="image">The second image to XOR with each slice.</param>
    let bitwiseXorWithImage (image: Image) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseXorWithImage]"
        mapSlices "BitwiseXorImage" Streaming (bitwiseXorImage image)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a bitwise NOT (~) operation by inverting pixel intensity up to a maximum.
    /// </summary>
    /// <param name="maximum">The maximum pixel value (e.g., 255 for 8-bit).</param>
    let bitwiseNot (maximum: int) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseNot]"
        mapSlices "BitwiseNot" Streaming (bitwiseNot maximum)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that performs a left bitwise shift by multiplying each pixel by 2ⁿ.
    /// </summary>
    /// <param name="shiftBits">The number of bits to shift left.</param>
    let bitwiseLeftShift (shiftBits: int) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseLeftShift]"
        mapSlices "BitwiseLeftShift" Streaming (bitwiseLeftShift shiftBits)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that simulates a right bitwise shift by dividing each pixel by 2ⁿ.
    /// Works for unsigned integer types where division is equivalent to logical right shift.
    /// </summary>
    /// <param name="shiftBits">The number of bits to shift right.</param>
    let bitwiseRightShift (shiftBits: int) : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[bitwiseRightShift]"
        mapSlices "BitwiseRightShift" Streaming (bitwiseRightShift shiftBits)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that applies the absolute value function to each pixel.
    /// </summary>
    /// <returns>A <c>StackProcessor</c> that sets each pixel to <c>abs(pixel)</c>.</returns>
    let abs : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[abs]"
        mapSlices "Abs" Streaming absImage

    /// <summary>
    /// Creates a <c>StackProcessor</c> that applies the square root function to each pixel.
    /// </summary>
    /// <returns>A <c>StackProcessor</c> that sets each pixel to <c>sqrt(pixel)</c>.</returns>
    /// <remarks>
    /// Input pixel values must be non-negative. Negative inputs result in NaN.
    /// </remarks>
    let sqrt : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[sqrt]"
        mapSlices "Sqrt" Streaming sqrtImage

    /// <summary>
    /// Creates a <c>StackProcessor</c> that applies the natural logarithm to each pixel.
    /// </summary>
    /// <returns>A <c>StackProcessor</c> that sets each pixel to <c>log(pixel)</c>.</returns>
    /// <remarks>
    /// Input pixel values must be positive. Zero or negative values will result in -inf or NaN.
    /// </remarks>
    let log : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[log]"
        mapSlices "Log" Streaming logImage

    /// <summary>
    /// Creates a <c>StackProcessor</c> that applies the exponential function (e<sup>x</sup>) to each pixel.
    /// </summary>
    /// <returns>A <c>StackProcessor</c> that sets each pixel to <c>exp(pixel)</c>.</returns>
    /// <remarks>
    /// May overflow for large pixel values depending on the image type.
    /// </remarks>
    let exp : StackProcessor<Slice<'T>, Slice<'T>> =
        printfn "[exp]"
        mapSlices "Exp" Streaming expImage

    /// <summary>
    /// Computes a cumulative histogram from a sequence of image slices, each assumed to be of byte (UInt8) type.
    /// </summary>
    /// <param name="slices">An asynchronous sequence of <c>Slice<'T></c> values.</param>
    /// <returns>
    /// An asynchronous computation that yields a 256-bin histogram, where each bin counts the total occurrences
    /// of that intensity across all slices.
    /// </returns>
    /// <remarks>
    /// Each slice is processed independently using <see cref="sliceHistogram" />, and the results are aggregated.
    /// </remarks>
    let histogram : StackProcessor<Slice<'T>, Vector<int>> =
        let histogramReducer (slices: AsyncSeq<Slice<'T>>) =
            slices
            |> AsyncSeq.map histogramSlice
            |> AsyncSeqExtensions.fold Vector.add (Vector.zero 0 256)
        fromReducer "Histogram" Streaming histogramReducer

    /// <summary>
    /// Generates a constant-valued volume of image slices.
    /// </summary>
    /// <param name="value">The constant pixel value to assign.</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="depth">Number of slices to generate.</param>
    let constant (value: byte) (width: uint) (height: uint) (depth: uint) : AsyncSeq<Slice<'T>> =
        printfn "[constant]"
        AsyncSeq.init (int depth) (fun i ->
            let image = new Image(width, height, PixelIDValueEnum.sitkUInt8)
            { Index = uint i; Image = image } |> shiftScaleSlice (float value) 1.0)

    /// <summary>
    /// Adds Gaussian noise to each image slice.
    /// </summary>
    /// <param name="mean">Mean of the Gaussian distribution.</param>
    /// <param name="stddev">Standard deviation of the Gaussian distribution.</param>
    let additiveGaussianNoise (mean: float) (stddev: float) : StackProcessor<Slice<'T> ,Slice<'T>> =
        printfn "[additiveGaussianNoise]"
        mapSlices "Additive Gaussian noise" Streaming (additiveGaussianNoiseSlice mean stddev)

    /// <summary>
    /// Applies a shift and scale transformation to pixel values in each image slice.
    /// </summary>
    /// <param name="delta">The amount to shift pixel values.</param>
    /// <param name="factor">The scale factor for pixel values.</param>
    let shiftScale (delta: float) (factor: float) : StackProcessor<Slice<'T> ,Slice<'T>> =
        printfn "[shiftScale]"
        mapSlices "Shift Scale" Streaming (shiftScaleSlice delta factor)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that applies binary thresholding to each image slice.
    /// </summary>
    /// <param name="lower">The lower threshold value (inclusive).</param>
    /// <param name="upper">The upper threshold value (inclusive).</param>
    /// <returns>
    /// A <c>StackProcessor</c> using a <c>Streaming</c> profile that thresholds each slice using the specified bounds.
    /// </returns>
    /// <remarks>
    /// Pixel values between <paramref name="lower"/> and <paramref name="upper"/> are preserved, others are suppressed.
    /// </remarks>
    let threshold (lower: float) (upper: float) : StackProcessor<Slice<'T> ,Slice<'T>> =
        printfn "[threshold]"
        mapSlices "Threshold" Streaming (thresholdSlice lower upper)

    /// <summary>
    /// Creates a <c>StackProcessor</c> that applies 3D Gaussian smoothing to a sliding window of image slices.
    /// </summary>
    /// <param name="sigma">The standard deviation of the Gaussian kernel.</param>
    /// <returns>
    /// A <c>StackProcessor</c> using a <c>Sliding</c> profile that stacks slices into a volume and smooths them with a 3D Gaussian filter.
    /// </returns>
    /// <remarks>
    /// The window depth is automatically computed as <c>1 + 2 * round(sigma)</c>.
    /// Only the center slice of the smoothed volume is returned per window.
    /// </remarks>
    let convolve3DGaussian (sigma: float) : StackProcessor<Slice<'T> ,Slice<'T>> =
        printfn "[convolve3DGaussian]"
        let depth = 1u + 2u * uint (0.5 + sigma)
        mapSlicesWindowed "3D Gaussian" depth (convolve3DGaussianSlice sigma depth)
