// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let mem = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    debug mem
    |> read<uint8> "image" ".tiff"
    >=> computeStats () --> print ()
    |> sink

    0
