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
        |> read<uint8> input ".tiff"
        //>=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "cast"
    >=> cast<uint8,float32>
    >=> tap "fan out"
    >=>> (addScalar 1.0f, addScalar 2.0f)
    >=> tap "fan in"
    >=> mulPair<float32>
    >=> tap "cast"
    >=> cast<float32,int8>
    >=> tap "write"
    >=> write output ".tiff"
    |> sink

    0
