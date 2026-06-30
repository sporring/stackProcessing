// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/sharedImbalancedStreamer"
        else
            "../data/volume", "../tmp/sharedImbalancedStreamer"


    let readMaker = 
        src
        |> read<float32> input ".tiff"
        // >=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "For >=>>"
    >=>> (addScalar 1.0, gaussianFilter 3.0 (Some 7u))
    >=> tap "For >>=>"
    >=> mulPair
    >=> tap "For cast"
    >=> cast<_, int8>
    >=> tap "For write"
    >=> write output ".tiff"
    //>>=> ignorePairs ()
    |> sink

    0
