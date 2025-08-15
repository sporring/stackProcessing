// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

    debug availableMemory
    |> read<uint8> "image" ".tiff"
    >=> write "result" ".tif"
    |> sink

    0
