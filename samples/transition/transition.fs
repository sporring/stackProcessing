// To run, remember to:
// export DYLD_LIBRARY_PATH=../Core/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let sigma = 3.0

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/transition"
        else
            "../data/volume", "../tmp/transition"

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> chunkSqrtFloat32
    >=> gaussianFilterNativeParallelCollect<float32> sigma 3 4
    >=> chunkSqrtFloat32
    >=> gaussianFilterNativeParallelCollect<float32> sigma 3 4
    >=> chunkCast<float32,uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink


    0
