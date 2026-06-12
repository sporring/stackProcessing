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
        |> readChunkSlices<float32> input ".tiff"
        // >=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "For >=>>"
    >=>> (chunkImageAddScalar 1.0f, gaussianFilterNativeParallelCollect<float32> 3.0 3 4)
    >=> tap "For >>=>"
    >=> chunkMulPair<float32>
    >=> tap "For cast"
    >=> chunkCast<float32,int8>
    >=> tap "For write"
    >=> writeChunkSlices output ".tiff"
    //>>=> ignorePairs ()
    |> sink

    0
