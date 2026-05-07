// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    printfn "Setting up finite difference filter"
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 1.0

    let src, arg = commandLineSource availableMemory arg
    let input, output = 
        if arg.Length > 0 then
            $"../image{arg[0]}", $"../result{arg[0]}"
        else
            "../image18", "../result18"

    src
    |> read<float> input ".tiff"
    >=> tap "tap: For finiteDiff"
    >=> finiteDiff sigma 2u 1u
    >=> tap "tap: For cast"
    >=> cast<float,uint8>
    >=> tap "tap: For write"
    >=> write output ".tiff"
    |> sink

    0
