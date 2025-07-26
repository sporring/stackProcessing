// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    let trg = "result"
    let w, h, d = 100u, 120u, 20u
    let mem = 1024UL * 1024UL

    let maskMaker = 
        source mem
        |>  create<uint8> w h d
        >=> sliceAddScalar 1uy
        >=> sliceMulScalar 2uy

    let readMaker =
        source mem
        |> create<uint8> w h d

    zipWith Slice.mul readMaker maskMaker
    >=> write "result" ".tif"
    |> sink

    0
