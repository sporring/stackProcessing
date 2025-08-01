// To run, remember to:
// export DYLD_LIBRARY_PATH=$(pwd)/lib 
open StackProcessing

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let sigma = 1.0

    debug availableMemory
    |> readAs<float> "image" ".tiff"
    >=> convGauss sigma
    >=> cast<float,uint8>
    >=> write "result" ".tif"
    |> sink

    0
