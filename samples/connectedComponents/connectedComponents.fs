// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

    let src, arg = commandLineSource availableMemory arg
    let width, height, depth, input, output = 
        if arg.Length > 0 then
            let n = (int arg[0]) / 3 |> pown 2 |> uint 
            n, n, n, "../data/rotatingBoxes", "../tmp/result"
        else
            64u, 64u, 64u, "../data/rotatingBoxes", "../tmp/result"
    let tmp = "../tmp/connectedComponents-labels"
    let suffix = ".tiff"
    let tmpSuffix = ".mha"

    let wsz = (depth/8u)

    let transTbl =
        src
        |> read<uint8> input suffix
        >=> connectedComponents wsz
        >=> teeFst (writeSlabSlices tmp tmpSuffix wsz)
        >=> makeConnectedComponentTranslationTable wsz
        |> drain
    printfn "Translation Table drain:\n%A" transTbl

    src
    |> read<uint64> tmp tmpSuffix
    >=> updateConnectedComponents wsz transTbl
    >=> cast<uint64,uint8>
    >=> write output suffix
    |> sink

    0
