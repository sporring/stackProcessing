// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    printfn "Setting up pipeline"
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 1MB for example
    let radius = 1u

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            debug availableMemory
        else
            source availableMemory
    let width, height, depth, input,output = 
        if arg.Length > 1 then
            let n = (int arg[1]) / 3 |> pown 2 |> uint 
            n, n, n, $"image{arg[1]}", $"result{arg[1]}"
        else
            64u, 64u, 64u, "image18", "result18"

    src
    |> zero<uint8> width height depth
    >=> addNormalNoise 128.0 50.0
    >=> threshold 128.0 infinity
    >=> erode radius 
    //>=> dilate radius
    //>=> opening radius
    //>=> closing radius
    >=> write ("../"+output) ".tiff"
    |> sink

    0
