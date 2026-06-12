// Estimate an Otsu threshold from sampled slices, then threshold the full stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/otsuThreshold"
        | _ -> "../data/volume", "../tmp/otsuThreshold"

    let thresholdValue =
        src
        |> readRandom<uint8> 16u input ".tiff"
        >=> imageHistogram<uint8> ()
        |> drain
        |> otsuThresholdFromHistogram

    src
    |> read<uint8> input ".tiff"
    >=> thresholdRange<uint8> thresholdValue 255.0
    >=> write output ".tiff"
    |> sink

    printfn "Estimated Otsu threshold: %.6f" thresholdValue
    0
