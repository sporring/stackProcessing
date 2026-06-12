// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open System
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

    let src, arg = commandLineSource availableMemory arg
    let width, height, depth, input, output = 
        if arg.Length > 0 then
            let n = (int arg[0]) / 3 |> pown 2 |> uint 
            n, n, n, "../data/rotatingBoxes", "../tmp/connectedComponents"
        else
            64u, 64u, 64u, "../data/rotatingBoxes", "../tmp/connectedComponents"
    let suffix = ".tiff"

    let windowSize = max 1 (int (depth / 8u))
    let workers = max 1 (min Environment.ProcessorCount windowSize)

    src
    |> readChunkSlices<uint8> input suffix
    >=> chunkConnectedComponentsSauf3DUInt8UInt32ParallelCollect windowSize workers
    >=> chunkCast<uint32,uint8>
    >=> writeChunkSlices output suffix
    |> sink

    0
