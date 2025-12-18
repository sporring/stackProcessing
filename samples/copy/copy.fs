// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            debug availableMemory
        else
            source availableMemory
    let input,output = 
        if arg.Length > 1 then
            $"image{arg[1]}", $"mask{arg[1]}"
        else
            "image18", "mask18"

    src
    |> read<uint8> ("../"+input) ".tiff"
    //|> getFilenames ("../"+input) ".tiff" Array.sort
    //>=> readFiles<uint8>
    >=> write ("../"+output) ".tiff"
    |> sink

    0
