// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input =
        match args with
        | [| input |] -> input
        | _ -> "../data/volume"

    let maxSlices = 16u
    let confidence = 0.95
    let targetError = 0.02

    // Randomly select slices to estimate the histogram with a maxium error of 0.02 at 95% confidence using the DKW inequality and a holdout set to validate the estimate.
    let estimate =
        src
        |> histogramEstimate<uint8> maxSlices input ".tiff" "DKWAndHoldout" confidence targetError
        |> drain

    printfn $"max slices: {maxSlices}"
    printfn $"slices read: {estimate.SlicesRead}"
    printfn $"samples in estimate half: {estimate.Samples}"
    printfn $"confidence: {estimate.Confidence}"
    printfn $"target CDF error: {targetError}"
    printfn $"DKW CDF half-width: {estimate.CdfHalfWidth}"
    printfn $"holdout max CDF delta: {estimate.HoldoutMaxCdfDelta}"

    showHistogramWithLabels "Estimated histogram" "gray values" "sample count" estimate.Histogram
    0
