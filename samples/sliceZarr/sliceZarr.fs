// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL

    let src, arg = commandLineSource availableMemory arg
    let input, output =
        match arg with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/chunks.zarr"
        | _ -> "../data/volume", "../tmp/chunks.zarr"

    deleteIfExists output
    src
    |> readRange<uint8> 0u 1 31u input ".tiff"
    >=> writeZarr output "image" 32u 12u 13u 14u 1.0 1.0 1.0 0
    |> sink

    let chunkInfo = getZarrInfo output 0 0
    printfn $"Wrote Zarr: chunks={chunkInfo.chunks} size={chunkInfo.size} componentType={chunkInfo.componentType}"

    let output2 = "../tmp/chunk-zarr-copy"
    deleteIfExists output2
    src
    |> readZarrRange<uint8> 0u 1 31u output 0 0 0 0 0
    >=> write output2 ".tiff"
    |> sink

    0
