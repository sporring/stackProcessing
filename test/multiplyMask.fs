// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example

    let imageMaker =
        debug availableMemory
        |> read<uint8> "image" ".tiff"
    let maskMaker =
        source availableMemory
        |> read<uint8> "mask" ".tiff"

    (imageMaker, maskMaker) ||> zip
    >>=> Slice.mul
    >=> write "result" ".tif"
    |> sink

    0
