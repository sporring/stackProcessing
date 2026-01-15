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
            $"image{arg[1]}", $"result{arg[1]}"
        else
            "image18", "result18"

    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> permuteAxes (0u,1u,2u) 64u
    >=> write ("../"+output+"012") ".tiff"
    |> sink

    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> permuteAxes (0u,2u,1u) 64u
    >=> write ("../"+output+"021") ".tiff"
    |> sink
    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> permuteAxes (1u,0u,2u) 64u
    >=> write ("../"+output+"102") ".tiff"
    |> sink
    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> permuteAxes (1u,2u,0u) 64u
    >=> write ("../"+output+"120") ".tiff"
    |> sink
    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> permuteAxes (2u,0u,1u) 64u
    >=> write ("../"+output+"201") ".tiff"
    |> sink
    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> permuteAxes (2u,1u,0u) 64u
    >=> write ("../"+output+"210") ".tiff"
    |> sink
(*
    src
    |> read<uint8> ("../"+output) ".tiff"
    >=> permuteAxes (1u,2u,0u) 64u
    >=> write ("../"+input+"b") ".tiff"
    |> sink
*)
    0
