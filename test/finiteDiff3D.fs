// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline

[<EntryPoint>]
let main _ =
    let src = "image"
    let trg = "result"
    let width, height, depth = getStackSize src ".tiff"
    let availableMemory = 1024UL * 1024UL // 1MB for example

    let sigma = 1.0
    let kernelSize = 1u + 2u * uint (0.5 + sigma);
    let windowSize = kernelSize+2u
    printfn "Setting up mask pipeline"
    let gaussMaker = 
        source<Slice<float>> availableMemory
        |> gaussSource 2.0 None
    let diffMaker = 
        source<Slice<float>> availableMemory
        |> finiteDiffFilter3D 1u 2u
        >=> tap "tap: dx2Maker"
    let dx2Maker = zipWith Slice.conv gaussMaker diffMaker >=> tap "tap: dx2Maker"

    let imageMaker =
        source<Slice<float>> availableMemory
        |> read "image" ".tiff"
    zipWith Slice.conv imageMaker dx2Maker
    >=> castFloatToUInt8 
    >=> write "result" ".tif"
    |> sink

    0
