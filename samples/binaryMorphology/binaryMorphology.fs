// Small binary morphology pipeline on the rotating boxes mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/binaryMorphology"
        | _ -> "../data/rotatingBoxes", "../tmp/binaryMorphology"

    src
    |> read<uint8> input ".tiff"
    >=> dilate 2u
    >=> opening 2u
    >=> binaryMedian 1u 5u
    >=> binaryContour false 5u
    >=> fillSmallHoles 128UL ObjectConnectivity.TwentySix
    >=> write output ".tiff"
    |> sink

    0
