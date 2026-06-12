// Repaint connected objects from a binary stack into an object-label image.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/objectsImage"
        | _ -> "../data/rotatingBoxes", "../tmp/objectsImage"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> streamConnectedObjectsChunk<uint8> ObjectConnectivity.TwentySix
    >=> paintObjectsChunk 512u 384u
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
