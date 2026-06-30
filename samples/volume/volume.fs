// Measure foreground volume from a thresholded volume file using Chunk stages.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 1UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input =
        match args with
        | [| input |] -> input
        | _ -> "../data/volumedata.tif"

    let voxelVolume =
        src
        |> readVolume<uint8> (volumeFilePath input ".tiff")
        >=> thresholdRange 1 255
        >=> volume 1.0 1.0 1.0
        |> drain

    printfn "Foreground volume: %.3f" voxelVolume

    0
