// Computes the Chunk structure tensor eigensystem and writes the x/color component
// of the first eigenvector as a scalar image stack.
open StackProcessing
open System.IO

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/structureTensorEigenvector0Color"
        | _ -> "../data/volume", "../tmp/structureTensorEigenvector0Color"

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> chunkStructureTensorNativeParallelCollect 1.0 7 0.0 0 4
    >=> chunkVectorRange<float32> 3u 3u
    >=> chunkVector3ToColorFloat32 -1.0f 1.0f
    >=> chunkVectorElement<uint8> 0u
    >=> writeChunkSlices output ".tiff"
    |> sink

    printfn "Wrote the first structure-tensor eigenvector x/color component to %s" (Path.GetFullPath output)
    0
