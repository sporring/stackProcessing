// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    let readMaker = 
        debug mem
        |> read<uint8> "image" ".tiff"
//        >=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> tap "cast"
    >=> cast<uint8,float>
    >=> tap "fan out"
    >=>> (imageAddScalar 1.0, imageAddScalar 2.0)
    >=> tap "fan in"
    >>=> mul2
    >=> tap "cast"
    >=> cast<float,int8>
    >=> tap "write"
    >=> write "result" ".tiff"
    |> sink

    0
