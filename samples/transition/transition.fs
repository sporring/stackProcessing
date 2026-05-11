// To run, remember to:
// export DYLD_LIBRARY_PATH=../Core/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 1.0

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/result"
        else
            "../data/volume", "../tmp/result"

    src
    |> read<float> input ".tiff"
    >=> sqrt
    >=> smoothWGauss sigma None None None 
    >=> sqrt
    >=> smoothWGauss sigma None None None 
    >=> cast<float,uint8>
    >=> write output ".tiff"
    |> sink


    0
