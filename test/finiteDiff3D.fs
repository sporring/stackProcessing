// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    printfn "Setting up finite difference filter"
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let sigma = 1.0

    source availableMemory
    |> readAs<float> "image" ".tiff"
    >=> convGauss sigma
    >=> tap "tap: convGauss"
    >=> finiteDiff 1u 2u
    >=> tap "tap: finiteDiff"
    >=> cast<float,uint8>
    >=> write "result" ".tif"
    |> sink

    0
