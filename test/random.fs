// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let trg = "image"
    let width, height, depth = 64u, 64u, 20u
    let availableMemory = 1024UL * 1024UL // 1MB for example

    debug availableMemory
    |> zero<uint8> width height depth
    >=> addNormalNoise 128.0 50.0
    >=> write trg ".tiff"
    |> sink

    0
