// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/result"
        else
            "../data/volume", "../tmp/result"


    let readMaker = 
        src
        |> read<uint8> input ".tiff"
        //>=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "cast"
    >=> cast<uint8,float>
    >=> tap "fan out"
    >=>> (imageAddScalar 1.0, imageAddScalar 2.0)
    >=> tap "fan in"
    >>=> mulPair
    >=> tap "cast"
    >=> cast<float,int8>
    >=> tap "write"
    >=> write output ".tiff"
    |> sink

    0
