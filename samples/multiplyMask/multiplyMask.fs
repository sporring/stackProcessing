// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input, mask, output =
        if arg.Length > 0 then
            "../data/rotatingBoxes", "../data/rotatingBoxes", "../tmp/multiplyMask"
        else
            "../data/rotatingBoxes", "../data/rotatingBoxes", "../tmp/multiplyMask"

    let imageMaker =
        src
        |> read<uint8> input ".tiff"
    let maskMaker =
        src
        |> read<uint8> mask ".tiff"

    // Demonstrate the merging of two sources.
    (imageMaker, maskMaker) ||> zip
    >=> mulPair
    >=> write output ".tiff"
    |> sink

    0
