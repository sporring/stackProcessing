// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example

    let imageMaker =
        source availableMemory
        |> readAs<uint8> "image" ".tiff"
    let maskMaker =
        source availableMemory
        |> readAs<uint8> "mask" ".tiff"

    zipWith Slice.mul imageMaker maskMaker
    >=> write "result" ".tif"
    |> sink

    0
