// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/sharedStreamer"
        else
            "../data/volume", "../tmp/sharedStreamer"


    let readMaker = 
        src
        |> readChunkSlices<uint8> input ".tiff"
        //>=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "cast"
    >=> chunkCast<uint8,float32>
    >=> tap "fan out"
    >=>> (chunkImageAddScalar 1.0f, chunkImageAddScalar 2.0f)
    >=> tap "fan in"
    >=> chunkMulPair<float32>
    >=> tap "cast"
    >=> chunkCast<float32,int8>
    >=> tap "write"
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
