// To run, remember to:
// export DYLD_LIBRARY_PATH=$(pwd)/lib 
open StackProcessing

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let sigma = 1.0

    debug availableMemory
    |> read<float> "image" ".tiff"
    >=> discreteGaussian sigma None None (Some 7u) // What's the rule for ligning up winSz and stream size, such that the output is also stream size?
    //>=> convGauss sigma
    >=> cast<float,uint8>
    >=> write "result" ".tif"
    //>=> ignoreImages ()
    |> sink

    0
