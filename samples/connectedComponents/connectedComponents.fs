// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            debug availableMemory
        else
            source availableMemory
    let width, height, depth, input, output = 
        if arg.Length > 1 then
            let n = (int arg[1]) / 3 |> pown 2 |> uint 
            n, n, n, $"image{arg[1]}", $"result{arg[1]}"
        else
            64u, 64u, 64u, "../image18", "../result18"
    let tmp = "tmp"
    let tmpSuffix = ".mha"

    let wsz = (depth/8u)

    let transTbl =
        src
        |> read<uint8> input ".tiff"
        >=> threshold 128.0 infinity
        >=> connectedComponents wsz
        >=> teeFst (writeChunkSlices tmp tmpSuffix wsz)
        >=> makeConnectedComponentTranslationTable wsz
        |> drain
    printfn "Translation Table drain:\n%A" transTbl

    src
    |> read<uint64> tmp tmpSuffix
    >=> updateConnectedComponents wsz transTbl
    >=> cast<uint64,uint8>
    >=> write output ".tiff"
    |> sink

    0
