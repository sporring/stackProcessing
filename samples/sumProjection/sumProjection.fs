// Reduces a 3D image stack to one 2D summed projection image.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/sumProjection"
        | _ -> "../data/rotatingBoxes", "../tmp/sumProjection"

    src
    |> read<uint8> input ".tiff"
    >=> sumProjection<uint8> "Identity"
    >=> cast<float32, uint8>
    >=> write output ".tiff"
    |> sink

    printfn "Wrote summed projection to %s" output
    0
