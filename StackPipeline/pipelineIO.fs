module pipelineIO

open System.IO
open FSharp.Control
open CoreTypes
open Vector
open Plotly.NET
open AsyncSeqExtensions
open itk.simple
open simpleITKWrappers

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
/// Returns the dimensions (width, height, depth) of a 3D image represented by TIFF slices in a directory.
/// </summary>
/// <param name="inputDir">The directory containing TIFF image files.</param>
/// <returns>
/// A tuple of (width, height, depth) where depth is the number of TIFF files.
/// </returns>
let getVolumeSize (inputDir: string) : uint * uint * uint =
    let files = Directory.GetFiles(inputDir, "*.tiff") |> Array.sort
    if files.Length = 0 then
        failwithf "No TIFF files found in directory: %s" inputDir

    let reader = new ImageFileReader()
    reader.SetFileName(files.[0])
    let img = reader.Execute()
    let size = img.GetSize()
    let width = size.[0] |> uint
    let height = size.[1] |> uint
    let depth = files.Length |> uint

    (width, height, depth)


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
let readSlices (inputDir: string) : AsyncSeq<ImageSlice> =
    printfn "[readSlices]"
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
/// Asynchronously reads TIFF image slices from a directory and returns them as an asynchronous sequence of <c>ImageSlice</c>.
/// </summary>
/// <param name="inputDir">The directory containing TIFF image files.</param>
/// <returns>
/// An asynchronous sequence of <c>ImageSlice</c> values, where each slice includes an index and its corresponding image.
/// </returns>
/// <remarks>
/// TIFF files are sorted by filename and loaded using <c>ImageFileReader</c>. Each read is performed asynchronously.
/// </remarks>
let readRandomSlices (count: uint) (inputDir: string) : AsyncSeq<ImageSlice> =
    printfn "[readRandomSlices]"
    Directory.GetFiles(inputDir, "*.tiff") |> Array.randomChoices (int count)
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
let writeSlices path stream =
    printfn "[writeSlices]"
    writeSlicesAsync path stream |> Async.RunSynchronously

/// <summary>
/// Prints the contents of a vector to the terminal as a comma-separated list enclosed in square brackets.
/// </summary>
/// <param name="vector">
/// An asynchronous sequence of <c>Vector&lt;int&gt;</c> objects to be printed. Typically this contains a single vector,
/// such as the result of a histogram processor wrapped in an <c>AsyncSeq</c>.
/// </param>
/// <returns>
/// Unit. This function runs synchronously and prints the vector contents to standard output.
/// </returns>
/// <remarks>
/// This function consumes the asynchronous sequence and prints each <c>Vector&lt;int&gt;</c> inline in a compact format.
/// Intended for simple debugging or inspection purposes.
/// </remarks>
/// <example>
/// <code>
/// let output = pipeline.Apply input
/// printVector output  // Prints: [0, 3, 7, 2, 0, ...]
/// </code>
/// </example>
let printVector (vector: AsyncSeq<Vector<int>>) =
    vector
    |> AsyncSeq.iterAsync (fun (Vector values) ->
        async {
            printf "["
            values
            |> Array.iter (fun v -> printf "%d, " v)
            printf "]"
        })
    |> Async.RunSynchronously

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
let showSliceAtIndex (idx: uint) (slices: AsyncSeq<ImageSlice>) : Async<unit> =
    async {
        let! maybeSlice =
            slices
            |> AsyncSeqExtensions.tryItem (int idx)

        match maybeSlice with
        | Some slice ->
            let image = slice.Image
            let width = image.GetWidth() |> int
            let height = image.GetHeight() |> int
            printfn "Showing image slice %d which has size %d x %d" idx width height
            let data =
                Seq.init height (fun y ->
                    Seq.init width (fun x ->
                        getPixel3D image (uint x) (uint y) 0u |> float))
            data |> Chart.Heatmap |> Chart.show
        | None ->
            printfn "[Visualization] No slice found at index %d." idx
    }

/// <summary>
/// Plots the contents of a vector as a line chart using Plotly.NET.
/// </summary>
/// <param name="vectorSeq">
/// An asynchronous sequence of <c>Vector&lt;int&gt;</c> objects to plot. Typically this contains a single vector,
/// such as the result of a histogram processor wrapped in an <c>AsyncSeq</c>.
/// </param>
/// <returns>
/// Unit. This function runs synchronously and opens a Plotly.NET interactive window displaying the chart.
/// </returns>
/// <remarks>
/// The vector values are treated as y-axis values, with automatically generated zero-based bin indices on the x-axis.
/// This function is useful for visualizing histograms or intensity distributions derived from image slices.
/// </remarks>
/// <example>
/// <code>
/// let output = pipeline.Apply input
/// plotVector output  // Displays a line chart of histogram frequencies
/// </code>
/// </example>
let plotVector (vectorSeq: AsyncSeq<Vector<int>>) =
    vectorSeq
    |> AsyncSeq.iterAsync (fun (Vector.Vector values) ->
        async {
            let bins = Array.mapi (fun i v -> i) values
            Chart.Line(bins, values) |> Chart.show
        })
    |> Async.RunSynchronously


    