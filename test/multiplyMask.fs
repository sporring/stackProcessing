// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example

    let imageMaker =
        debug (availableMemory/2UL)
        |> read<uint8> "image" ".tiff"
    let maskMaker =
        debug (availableMemory/2UL)
        |> read<uint8> "mask" ".tiff"

    (imageMaker, maskMaker) ||> zip
    >>=> mul2
    //>=> tap "Slice.mul"
    >=> write "result" ".tif"
    |> sink

    0
