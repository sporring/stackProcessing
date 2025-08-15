// To run, remember to:
// export DYLD_LIBRARY_PATH=../Core/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 1.0

    debug availableMemory
    |> read<float> "image" ".tiff"
    >=> sqrt
    >=> convGauss sigma 
    >=> sqrt
    >=> convGauss sigma 
    >=> cast<float,uint8>
    >=> write "result" ".tif"
    |> sink


    0
