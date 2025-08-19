// To run, remember to:
// export DYLD_LIBRARY_PATH=$(pwd)/lib 
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 1.0

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory
    let input,output = 
        if arg.Length > 1 then
            $"image{arg[1]}", $"result{arg[1]}"
        else
            "image18", "result18"

    src
    |> read<float> input ".tiff"
    // sigma = 1 => pad=2, depth = 22 => integer solution for number of strides when:
    // windowSize = 1, 6, 15, or 26, => n = 21, 10, 1, or 0
    >=> discreteGaussian sigma None None (Some 15u) 
    //>=> convGauss sigma
    >=> cast<float,uint8>
    >=> write output ".tiff"
    //>=> ignoreImages ()
    |> sink

    0
