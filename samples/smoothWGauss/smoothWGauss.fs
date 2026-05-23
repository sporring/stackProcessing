// To run, remember to:
// export DYLD_LIBRARY_PATH=$(pwd)/lib 
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 3.0

    let src, arg = commandLineSource availableMemory arg
    let input, output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/smoothWGauss"
        else
            "../data/volume", "../tmp/smoothWGauss"

    src
    |> read<float> input ".tiff"
    >=> smoothWGauss sigma None None None 
    >=> cast<float,uint8>
    >=> write output ".tiff"
    //>=> ignoreImages ()
    |> sink

    0
