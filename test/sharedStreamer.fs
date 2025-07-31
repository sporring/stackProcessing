// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    let readMaker = 
        source mem
        |> readAs<uint8> "image" ".tiff"

    readMaker 
    >=>> (sliceAddScalar 1uy, sliceAddScalar 2uy)
    >>=> Slice.mul
    >=> ignoreAll ()
    |> sink

    0
