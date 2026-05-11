// Estimate a volume histogram from a random subset of slices.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input =
        match args with
        | [| input |] -> input
        | _ -> "../data/volume"

    let estimate =
        src
        |> estimateHistogram<float> 16u input ".tiff" 4u "DKWAndHoldout" 0.95
        |> drain

    printfn "Histogram estimate: samples=%u confidence=%.3f CDF half-width=%.6f holdout delta=%.6f"
        estimate.Samples
        estimate.Confidence
        estimate.CdfHalfWidth
        estimate.HoldoutMaxCdfDelta

    0
