// Computes the structure tensor eigensystem and writes the first eigenvector as an RGB image stack.
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
    |> readRange<float> "0" 1 "31" input ".tiff"
    >=> structureTensor 1.0 0.0
    >=> vectorRange<float> 3u 3u
    >=> vector3ToColor -1.0 1.0
    >=> write output ".tiff"
    |> sink

    printfn "Wrote the first structure-tensor eigenvector as RGB color to %s" (Path.GetFullPath output)
    0
