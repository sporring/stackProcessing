// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let width, height, depth, output = 
        if arg.Length > 0 then
            let n = (int arg[0]) / 3 |> pown 2 |> uint 
            n, n, n, "../tmp/multiplyConstant"
        else
            64u, 64u, 64u, "../tmp/multiplyConstant"

    let maskMaker = 
        src
        |> zero<uint8> width height depth
        >=> addScalar 1.0
        >=> mulScalar 2.0

    let imageMaker =
        src
        |> zero<uint8> width height depth

    (imageMaker, maskMaker) ||> zip
    >=> mulPair
    >=> write output ".tiff"
    |> sink

    0
