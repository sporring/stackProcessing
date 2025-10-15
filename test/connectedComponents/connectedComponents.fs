// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

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
    |> zero<float> width height depth
    >=> addNormalNoise 128.0 50.0
    >=> threshold 128.0 infinity
    >=> connectedComponents (1024u/8u)
    >=> cast<uint64,uint16>
    // Tiff supporst uint8, int8, uint16, int16, and float32
    >=> write ("../"+output) ".tiff"
    |> sink

    0
