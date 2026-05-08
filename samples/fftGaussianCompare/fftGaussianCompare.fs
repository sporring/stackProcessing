open System
open System.IO
open Image
open StackProcessing

let private sampleRoot () =
    Path.Combine(Path.GetTempPath(), $"stackprocessing-fft-gaussian-{Guid.NewGuid():N}")

let private resetDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private writeSlices path (slices: Image<float> list) =
    resetDirectory path
    slices
    |> List.iteri (fun index slice ->
        slice.toFile(Path.Combine(path, sprintf "image_%03d.mha" index)))

let private circularShiftRealVolume (shift: uint list) (volume: Image<float>) =
    let size = volume.GetSize()
    let shifted = new Image<float>(size, 1u, "circularShiftRealVolume", volume.index, true)

    size
    |> Image.InternalHelpers.flatIndices
    |> Seq.iter (fun src ->
        let dst =
            src
            |> List.mapi (fun axis value -> (value + shift[axis]) % size[axis])

        shifted.Set dst (volume.Get src))

    shifted

let private maxAbsDifference (expected: Image<float> list) (actual: Image<float> list) =
    if expected.Length <> actual.Length then
        invalidArg "actual" $"Expected {expected.Length} slices, got {actual.Length}."

    (expected, actual)
    ||> List.map2 (fun left right ->
        if left.GetSize() <> right.GetSize() then
            invalidArg "actual" $"Slice sizes differ: {left.GetSize()} vs {right.GetSize()}."

        let mutable maxError = 0.0
        for y in 0 .. int (left.GetHeight()) - 1 do
            for x in 0 .. int (left.GetWidth()) - 1 do
                maxError <- max maxError (Math.Abs(left[x, y] - right[x, y]))
        maxError)
    |> List.max

let private disposeImages images =
    images |> List.iter (fun (image: Image<'T>) -> image.decRefCount())

[<EntryPoint>]
let main _ =
    let availableMemory = 512UL * 1024UL * 1024UL
    let size = 5
    let sigma = 1.25
    let chunkX, chunkY, chunkZ = 2u, 2u, 2u
    let suffix = ".mha"
    let root = sampleRoot ()
    let inputDir = Path.Combine(root, "input")
    let kernelDir = Path.Combine(root, "kernel")

    try
        let inputSlices =
            [ for z in 0 .. size - 1 ->
                Image<float>.ofArray2D (
                    Array2D.init size size (fun x y ->
                        if x = 0 && y = 0 && z = 0 then 1.0 else 0.0)) ]

        let inputVolume = ImageFunctions.stack inputSlices
        let centeredKernel: Image<float> = ImageFunctions.gauss 3u sigma (Some (uint size))
        let originKernel = circularShiftRealVolume [ uint (size / 2); uint (size / 2); uint (size / 2) ] centeredKernel
        let originKernelSlices = ImageFunctions.unstack 2u originKernel

        writeSlices inputDir inputSlices
        writeSlices kernelDir originKernelSlices

        let referenceVolume =
            ImageFunctions.discreteGaussian
                3u
                sigma
                (Some (uint size))
                (Some ImageFunctions.Same)
                (Some ImageFunctions.PerodicPad)
                inputVolume

        let referenceSlices = ImageFunctions.unstack 2u referenceVolume

        let inputFft =
            source availableMemory
            |> read<float> inputDir suffix
            >=> FFT<float> chunkX chunkY chunkZ

        let kernelFft =
            source availableMemory
            |> read<float> kernelDir suffix
            >=> FFT<float> chunkX chunkY chunkZ

        let fftConvolutionSlices =
            zip inputFft kernelFft
            >>=> mulPair
            >=> invFFT chunkX chunkY chunkZ
            |> drainList

        let maxError = maxAbsDifference referenceSlices fftConvolutionSlices

        printfn "FFT Gaussian convolution comparison"
        printfn "  volume: %dx%dx%d" size size size
        printfn "  sigma: %.3f" sigma
        printfn "  chunks: %u x %u x %u" chunkX chunkY chunkZ
        printfn "  max |periodic Gaussian smoothing - Re(invFFT(FFT(I) * FFT(G)))| = %.12g" maxError

        disposeImages fftConvolutionSlices
        disposeImages referenceSlices
        referenceVolume.decRefCount()
        disposeImages originKernelSlices
        originKernel.decRefCount()
        centeredKernel.decRefCount()
        inputVolume.decRefCount()
        disposeImages inputSlices
        0
    finally
        if Directory.Exists root then
            Directory.Delete(root, true)
