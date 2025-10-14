// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory
    let input,output = 
        if arg.Length > 1 then
            $"image{arg[1]}", $"result{arg[1]}"
        else
            "image18", "result18"

    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> computeStats () --> print ()
    |> sink

    0
