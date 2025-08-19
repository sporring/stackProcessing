// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory
    let width, height, depth, input,output = 
        if arg.Length > 1 then
            let n = (int arg[1]) / 3 |> pown 2 |> uint 
            n, n, n, $"image{arg[1]}", $"result{arg[1]}"
        else
            64u, 64u, 64u, "image18", "result18"

    let maskMaker = 
        src
        |>  zero<uint8> width height depth
        >=> imageAddScalar 1uy
        >=> imageMulScalar 2uy

    let readMaker =
        src
        |> zero<uint8> width height depth

    (readMaker, maskMaker) ||> zip 
    >>=> mul2
    >=> write output ".tiff"
    |> sink

    0
