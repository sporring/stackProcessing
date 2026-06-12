// Binary contour extracts foreground boundaries from a UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/binaryContour"
        | _ -> "../data/rotatingBoxes", "../tmp/binaryContour"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkBinaryContourZonohedral false
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
