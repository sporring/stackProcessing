// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    let readMaker = 
        debug mem
        |> readAs<uint8> "image" ".tiff"
//        >=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    readMaker 
    >=> cast<uint8,float>
//    >=> tapIt (fun s -> $"[cast<uint8,float>] {s.Index} -> Image {s.Image}")
    >=>> (sliceAddScalar 1.0, sliceAddScalar 2.0)
    >>=> Slice.mul
    >=> cast<float,int>
//    >=> tapIt (fun s -> $"[cast<float,int>] {s.Index} -> Image {s.Image}")
    >=> ignoreAll ()
//    >=> tapIt (fun s -> $"[ignoreAll] {s}")
    |> sink

    0
