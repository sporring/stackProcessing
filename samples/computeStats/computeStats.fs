// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input = 
        if arg.Length > 0 then
            $"../image{arg[0]}"
        else
            "../image18"

    src
    |> read<uint8> input ".tiff"
    >=> computeStats () --> print ()
    |> sink

    0
