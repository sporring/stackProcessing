// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let trg = "result"
    let w, h, d = 1024u, 1024u, 1024u
    let mem = 2UL * 1024UL * 1024UL *1024UL

    let maskMaker = 
        debug mem
        |>  zero<uint8> w h d
        >=> imageAddScalar 1uy
        >=> imageMulScalar 2uy

    let readMaker =
        debug mem
        |> zero<uint8> w h d

    (readMaker, maskMaker) ||> zip 
    >>=> mul2
    >=> write "result" ".tif"
    |> sink

    0
