// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let width, height, depth = 64u, 64u, 20u
    let availableMemory = 1024UL * 1024UL // 1MB for example

    debug availableMemory
        |> zero<float> width height depth
        >=> addNormalNoise 128.0 50.0
        >=> threshold 128.0 infinity
        >=> cast<float,uint8>
        >=> connectedComponents 5u
        >=> cast<uint64,uint16>
        // Tiff supporst uint8, int8, uint16, int16, and float32
        >=> write "result" ".tif"
        |> sink

    0
