// Measure mask volume directly and surface area through marching cubes.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input =
        match args with
        | [| input |] -> input
        | _ -> "../data/rotatingBoxes"

    let voxelVolume =
        src
        |> read<uint8> input ".tiff"
        >=> imageDivScalar<uint8> 255uy
        >=> volume 1.0 1.0 1.0
        |> drain

    let area =
        src
        |> read<uint8> input ".tiff"
        >=> imageDivScalar<uint8> 255uy
        >=> marchingCubes<uint8> 0.5
        >=> surfaceArea 1.0 1.0 1.0
        |> drain

    printfn "Foreground volume: %.3f" voxelVolume
    printfn "Surface area: %.3f" area
    0
