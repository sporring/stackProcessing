// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    let src = "image"
    let trg = "result"
    let width, height, depth = getStackSize src ".tiff"
    let availableMemory = 1024UL * 1024UL // 1MB for example

    let imageMaker =
        source<Slice<uint8>> availableMemory
        |> read "image" ".tiff"
    let maskMaker =
        source<Slice<uint8>> availableMemory
        |> read "mask" ".tiff"

    zipWith Slice.mul imageMaker maskMaker
    >=> write "result" ".tif"
    |> sink

    0
