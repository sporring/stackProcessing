// Computes the Chunk structure tensor eigensystem and writes the first eigenvector
// as RGB color chunk slices.
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
    >=> chunkStructureTensorNativeParallelCollect 1.0 7 1.0 7 4
    >=> chunkVectorRange<float32> 3u 3u
    >=> chunkVector3ToColorFloat32 -1.0f 1.0f
    >=> writeColorChunkSlices output ".tiff"
    |> sink

    printfn "Wrote the first structure-tensor eigenvector color slices to %s" (Path.GetFullPath output)
    0
