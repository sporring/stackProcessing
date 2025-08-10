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
    >=> cast<uint8,float>
    >=> tap "cast"
    >=>> (imageAddScalar 1.0, imageAddScalar 2.0)
    >=> tap "fan out"
    >>=> mul2
    >=> tap "fan in"
    >=> cast<float,int8>
    >=> tap "cast"
    >=> write "result" ".tiff"
    >=> tap "write"
    |> sink

    0
