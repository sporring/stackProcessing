// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            debug availableMemory
        else
            source availableMemory
    let input,mask,output = 
        if arg.Length > 1 then
            $"image{arg[1]}", $"mask{arg[1]}", $"result{arg[1]}"
        else
            "image18", "mask18", "result18"

    let imageMaker =
        src
        |> read<uint8> ("../"+input) ".tiff"
    let maskMaker =
        src
        |> read<uint8> ("../"+mask) ".tiff"

    (imageMaker, maskMaker) ||> zip
    >=> tap "[tab] For mul2"
    >>=> mul2
    >=> tap "[tab] For write"
    >=> write ("../"+output) ".tiff"
    |> sink

    0
