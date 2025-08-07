// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let trg = "result"
    let width, height, depth = 64u, 64u, 8u
    let availableMemory = 1024UL * 1024UL // 1MB for example

    debug availableMemory
    |> zero<float> width height depth
    >=> addNormalNoise 128.0 50.0
    >=> convGauss 2.0
    >=> cast<float,uint8>
    >=> threshold 128.0 infinity
    >=> sliceMulScalar 255uy
    >=> write "result" ".tif"
    |> sink

    0
