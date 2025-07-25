// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let trg = "result"
    let width, height, depth = 64u, 64u, 20u
    let availableMemory = 1024UL * 1024UL // 1MB for example
    source<Slice<uint8>> availableMemory
        |> create width height depth
        >=> addNormalNoise 128.0 50.0
        >=> threshold 128.0 infinity
        >=> erode 1u 
        >=> dilate 1u
        >=> opening 1u
        >=> closing 1u
        >=> write "result" ".tif"
        |> sink

    0
