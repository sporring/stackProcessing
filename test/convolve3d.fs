// To run, remember to:
// export DYLD_LIBRARY_PATH=$(pwd)/lib 
open StackProcessing

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let sigma = 1.0

    debug availableMemory
    |> read<float> "image" ".tiff"
    // sigma = 1 => pad=2, depth = 22 => integer solution for number of strides when:
    // windowSize = 1, 6, 15, or 26, => n = 21, 10, 1, or 0
    >=> discreteGaussian sigma None None (Some 26u) 
    //>=> convGauss sigma
    >=> cast<float,uint8>
    >=> write "result" ".tif"
    //>=> ignoreImages ()
    |> sink

    0
