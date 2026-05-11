// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/chunks"
        else
            "../data/volume", "../tmp/chunks"

    deleteIfExists output
    src
    |> read<uint8> input ".tiff"
    //|> getFilenames (input) ".tiff" Array.sort
    //>=> readFiles<uint8>
    >=> writeChunks output ".tiff" 12u 13u 14u
    >=> ignoreSingles ()
    |> sink

    let chunkInfo = getChunkInfo output ".tiff"
    printfn $"Wrote chunks: chunks={chunkInfo.chunks} size={chunkInfo.size} componentType={chunkInfo.topLeftInfo.componentType}"

    let output2 = "../tmp/volume-copy"
    deleteIfExists output2
    src |> readSlab<uint8> output ".tiff"
    >=> write output2 ".tiff"
    |> sink

    0
