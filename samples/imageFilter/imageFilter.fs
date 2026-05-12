// Edge, smoothing, histogram equalization, gradient, and PCA examples.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, outputRoot =
        match args with
        | [| input; outputRoot |] -> input, outputRoot
        | [| input |] -> input, "../tmp/imageFilters"
        | _ -> "../data/volume", "../tmp/imageFilters"

    let histogram =
        src
        |> readRandom<uint8> 8u input ".tiff"
        >=> imHistogram ()
        |> drain

    src
    |> readRange<uint8> "0" 1 "31" input ".tiff"
    >=> histogramEqualization histogram
    >=> cast<float, uint8>
    >=> write (outputRoot + "/histogramEqualization") ".tiff"
    |> sink

    src
    |> readRange<float> "0" 1 "31" input ".tiff"
    >=> smoothWMedian<float> 1u 5u
    >=> smoothWBilateral<float> 1.5 30.0 5u
    >=> gradientMagnitude<float> 5u
    >=> sobelEdge<float> 5u
    >=> laplacian<float> 5u
    >=> intensityStretch<float> 0.0 255.0 0.0 255.0
    >=> cast<float, uint8>
    >=> write (outputRoot + "/edges") ".tiff"
    |> sink

    src
    |> readRange<float> "0" 1 "31" input ".tiff"
    >=> gradient 1u (Some 5u)
    >=> PCA 3u
    >=> selectGroupedOutput 4u 1u
    >=> vector3ToColor -1.0 1.0
    >=> write (outputRoot + "/pcaGradientDirection") ".tiff"
    |> sink

    0
