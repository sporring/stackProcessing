// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let w, h, d = 1024u, 1024u, 1024u
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory

    let maskMaker = 
        src
        |>  zero<uint8> w h d
        >=> imageAddScalar 1uy
        >=> imageMulScalar 2uy

    let readMaker =
        src
        |> zero<uint8> w h d

    (readMaker, maskMaker) ||> zip 
    >>=> mul2
    >=> write "result" ".tif"
    |> sink

    0
