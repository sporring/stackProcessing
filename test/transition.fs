// To run, remember to:
// export DYLD_LIBRARY_PATH=../Core/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let sigma = 1.0

    source availableMemory
    |> readAs<float> "image" ".tiff"
    >=> sqrtFloat
    >=> convGauss sigma None
    >=> sqrtFloat
    >=> convGauss sigma None
    >=> castFloatToUInt8
    >=> write "result" ".tif"
    |> sink


    0
