// Fill small enclosed background regions in a UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/fillSmallHoles"
        | _ -> "../data/objects", "../tmp/fillSmallHoles"

    // fill any hole which is 32 voxels in volume or smaller.
    src
    |> read<uint8> input ".tiff"
    >=> intensityStretch 0.0 255.0 255.0 0.0
    >=> fillSmallHoles 32UL ObjectConnectivity.TwentySix (Some 0.0) (Some 255.0)
    >=> write output ".tiff"
    |> sink

    0
