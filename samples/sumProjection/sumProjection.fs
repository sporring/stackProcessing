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
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkSumProjection<uint8> "Identity"
    >=> chunkCast<float, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    printfn "Wrote summed projection to %s" output
    0
