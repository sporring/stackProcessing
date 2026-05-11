// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    printfn "Setting up pipeline"
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 1MB for example
    let radius = 1u

    let src, arg = commandLineSource availableMemory arg
    let width, height, depth, output = 
        if arg.Length > 0 then
            let n = (int arg[0]) / 3 |> pown 2 |> uint 
            n, n, n, "../tmp/erode"
        else
            64u, 64u, 64u, "../tmp/erode"

    src
    |> zero<uint8> width height depth
    >=> addNormalNoise 128.0 50.0
    >=> threshold 128.0 infinity
    >=> erode radius 
    //>=> dilate radius
    //>=> opening radius
    //>=> closing radius
    >=> write output ".tiff"
    |> sink

    0
