// Binary median smooths a UInt8 mask with a local majority vote.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/binaryMedian"
        | _ -> "../data/rotatingBoxes", "../tmp/binaryMedian"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkMedianNativeNthElementUInt8ParallelCollect 3 4
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
