// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/transpose"
        else
            "../data/volume", "../tmp/transpose"

    let fname021 = output+"021"
    deleteIfExists fname021
    src
    |> read<uint8> input ".tiff"
    >=> tapIt (fun elm -> $"Read {elm}")
    >=> permuteAxes<uint8> [| 0; 2; 1 |]
    >=> write fname021 ".tiff"
    |> sink

    let fname021021 = output + "021021"
    deleteIfExists fname021021
    src
    |> read<uint8> fname021 ".tiff"
    >=> permuteAxes<uint8> [| 0; 2; 1 |]
    >=> write fname021021 ".tiff"
    |> sink

    0
