// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    printfn "Setting up finite difference filter"
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 1.0

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory

    src
    |> read<float> "image" ".tiff"
    >=> tap "tap: For finiteDiff"
    >=> finiteDiff sigma 1u 2u
    >=> tap "tap: For cast"
    >=> cast<float,uint8>
    >=> tap "tap: For write"
    >=> write "result" ".tif"
    |> sink

    0
