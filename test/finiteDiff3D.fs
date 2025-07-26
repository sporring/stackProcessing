// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline
open Slice

[<EntryPoint>]
let main _ =
    printfn "Setting up finite difference filter"
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let sigma = 2.0

    let diffFilter = Slice.finiteDiffFilter3D 1u 2u;

    source availableMemory
    |> readAs<float> "image" ".tiff"
    >=> convGauss sigma None
    >=> Pipeline.conv diffFilter // fix naming shadowing!
    >=> tap "tap: convolution"
    >=> Pipeline.castFloatToUInt8 // fix naming shadowing!
    >=> write "result" ".tif"
    |> sink

    0
