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
        |> read<float> input ".tiff"
        // >=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "For >=>>"
    >=>> (imageAddScalar 1.0, smoothWGauss 1.0 None None None)
    >=> tap "For >>=>"
    >>=> mulPair
    >=> tap "For cast"
    >=> cast<float,int8>
    >=> tap "For write"
    >=> write output ".tiff"
    //>>=> ignorePairs ()
    |> sink

    0
