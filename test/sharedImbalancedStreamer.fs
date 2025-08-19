// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory

    let readMaker = 
        src
        |> read<float> "image" ".tiff"
        // >=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "For >=>>"
    >=>> (imageAddScalar 1.0, convGauss 1.0)
    >=> tap "For >>=>"
    >>=> mul2
    >=> tap "For cast"
    >=> cast<float,int8>
    >=> tap "For write"
    >=> write "result" ".tiff"
    //>>=> ignorePairs ()
    |> sink

    0
