// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let width, height, depth = 64u, 64u, 20u
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let radius = 1u

    source availableMemory
        |> createAs<uint8> width height depth
        >=> addNormalNoise 128.0 50.0
        >=> threshold 128.0 infinity
        >=> erode radius 
        >=> dilate radius
        >=> opening radius
        >=> closing radius
        >=> write "result" ".tif"
        |> sink

    0
