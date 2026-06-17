// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    printfn "Setting up finite difference filter"
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let src, arg = commandLineSource availableMemory arg
    let input, output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/finiteDiff3D"
        else
            "../data/volume", "../tmp/finiteDiff3D"

    src
    |> read<float32> input ".tiff"
    >=> tap "tap: For finiteDiff"
    >=> finiteDiffZ<float32> 2u 4
    >=> tap "tap: For cast"
    >=> cast<float32,uint8>
    >=> tap "tap: For write"
    >=> write output ".tiff"
    |> sink

    0
