// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory/2UL
        else
            source availableMemory/2UL

    let imageMaker =
        src
        |> read<uint8> "image" ".tiff"
    let maskMaker =
        src
        |> read<uint8> "mask" ".tiff"

    (imageMaker, maskMaker) ||> zip
    >=> tap "[tab] For mul2"
    >>=> mul2
    >=> tap "[tab] For write"
    >=> write "result" ".tif"
    |> sink

    0
