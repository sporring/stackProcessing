// Computes the Chunk structure tensor, decomposes it, and writes the first
// eigenvector as RGB color chunk slices.
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
    |> read<float32> input ".tiff"
    >=> structureTensor 1.0 7 1.0 7 4
    >=> symmetricTensorEigenvector 0u
    >=> vector3ToColorFloat32 -1.0f 1.0f
    >=> writeColor output ".tiff"
    |> sink

    printfn "Wrote the first structure-tensor eigenvector color slices to %s" (Path.GetFullPath output)
    0
