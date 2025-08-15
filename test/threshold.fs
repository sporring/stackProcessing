// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let trg = "result"
    let width, height, depth = 1024u, 1024u, 1024u
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

    debug availableMemory
    |> zero<int8> width height depth
    >=> cast<int8,float>
    >=> addNormalNoise 128.0 50.0
    >=> threshold 128.0 infinity
    >=> imageMulScalar 255.0
    >=> cast<float,int8>
    //>=> ignoreImages ()
    >=> write "result" ".tif"
    |> sink

    0
