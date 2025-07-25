// To run, remember to:
// export DYLD_LIBRARY_PATH=../Core/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    let src = "image"
    let trg = "result"
    let width, height, depth =  getStackSize src ".tiff"
    let availableMemory = 1024UL * 1024UL // 1MB for example

    let someKernel = Slice.gauss 3u 1.3 None

    source<Slice<float>> availableMemory
    |> read "image" ".tiff"
    >=> sqrtFloat
    >=> convGauss 1.0 None
    >=> sqrtFloat
    >=> convGauss 1.0 None
    >=> castFloatToUInt8
    >=> write "result" ".tif"
    |> sink


    0
