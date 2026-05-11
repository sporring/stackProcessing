// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,mask,output = 
        if arg.Length > 0 then
            "../data/volume", "../data/rotatingBoxes", "../tmp/multiplyMask"
        else
            "../data/volume", "../data/rotatingBoxes", "../tmp/multiplyMask"

    let imageMaker =
        src
        |> read<uint8> input ".tiff"
    let maskMaker =
        src
        |> read<uint8> mask ".tiff"

    (imageMaker, maskMaker) ||> zip
    >=> tap "[tab] For mul2"
    >>=> mulPair
    >=> tap "[tab] For write"
    >=> write output ".tiff"
    |> sink

    0
