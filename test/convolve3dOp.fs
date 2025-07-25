// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    let src = "image"
    let trg = "result"
    let width, height, depth = getStackSize src ".tiff"
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let sigma = 1.0

    sourceOp<Slice<float>> availableMemory
    |> readOp "image" ".tiff"
    >>=> convGaussOp 1.0 None
    >>=> castFloatToUInt8Op
    >>=> writeOp "result" ".tif"
    |> sinkOp

    0
