open Image
open StackProcessing
open System
open System.IO

let private makeRampSlice z =
    Array2D.init 5 5 (fun x y ->
        float x + 0.25 * float y + 0.0 * float z)
    |> Image<float>.ofArray2D

let private disposeImages images =
    images |> List.iter (fun (image: Image<'T>) -> image.decRefCount())

let private writeInput directory =
    deleteIfExists directory
    Directory.CreateDirectory directory |> ignore

    [ 0 .. 4 ]
    |> List.iter (fun z ->
        let slice = makeRampSlice z
        try
            slice.toFile(Path.Combine(directory, $"image_{z:D3}.mha"))
        finally
            slice.decRefCount())

let private runPart inputDir name part =
    let result =
        source (64UL * 1024UL * 1024UL)
        |> read<float> inputDir ".mha"
        >=> structureTensor 0.0 0.0
        >=> selectGroupedOutput 4u part
        |> drainList

    try
        printfn "%s" name
        printfn "  slices: %d" result.Length
        printfn "  center pixel in middle slice: %A" result[2].[2, 2]
    finally
        disposeImages result

[<EntryPoint>]
let main _ =
    let inputDir = Path.Combine(Path.GetTempPath(), $"stackprocessing-structure-tensor-{Guid.NewGuid():N}")
    writeInput inputDir

    printfn "Structure tensor sample"
    printfn "Input: f(x,y,z) = x + 0.25 y on a 5x5x5 volume"
    printfn "Output convention: eigenvalues, eigenvector 0, eigenvector 1, eigenvector 2"

    try
        runPart inputDir "Eigenvalues" 0u
        runPart inputDir "Eigenvector 0" 1u
        runPart inputDir "Eigenvector 1" 2u
        runPart inputDir "Eigenvector 2" 3u
    finally
        deleteIfExists inputDir

    0
