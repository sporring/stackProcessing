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

    // Calculate the volume and surface area of the foreground mask. This also demonstrates the synchroneous fan-out operator >=>>. Since marchingCubes requires 2 neighbouring slices, but volume only needs a single slice, the pipeline is delayed by 1 to ensure that input and output is synchroneously served.
    let area, volume =
        src
        |> read<uint8> input ".tiff"
        >=> threshold 1.0
        >=>> (marchingCubes 127.5 --> objectSurfaceArea 1.0 1.0 1.0, objectVolume 1.0 1.0 1.0)
        |> drain

    printfn "Foreground volume: %.3f" volume
    printfn "Surface area: %.3f" area

    0
