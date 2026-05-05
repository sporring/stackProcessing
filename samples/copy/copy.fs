// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            $"../image{arg[0]}", $"../mask{arg[0]}"
        else
            "../image18", "../mask18"

    src
    |> read<uint8> input ".tiff"
    //|> getFilenames (input) ".tiff" Array.sort
    //>=> readFiles<uint8>
    >=> write output ".tiff"
    |> sink

    0
