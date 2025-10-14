// To run, remember to:
// export DYLD_LIBRARY_PATH=../Core/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 1.0

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
    |> read<float> ("../"+input) ".tiff"
    >=> sqrt
    >=> convGauss sigma 
    >=> sqrt
    >=> convGauss sigma 
    >=> cast<float,uint8>
    >=> write ("../"+output) ".tiff"
    |> sink


    0
