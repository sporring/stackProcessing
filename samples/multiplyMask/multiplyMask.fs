// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,mask,output = 
        if arg.Length > 0 then
            $"../image{arg[0]}", $"../mask{arg[0]}", $"../result{arg[0]}"
        else
            "../image18", "../mask18", "../result18"

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
