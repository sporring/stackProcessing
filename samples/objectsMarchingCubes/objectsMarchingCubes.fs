// Runs marching cubes on the binary object image and writes a mesh file.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/surface"
        | _ -> "../data/rotatingBoxes", "../tmp/surface"

    // Runs marching cubes on the binary object image and writes a mesh file viewable e.g. with the meshview app.
    src
    |> read<uint8> input ".tiff"
    >=> marchingCubes 0.5
    >=> writeMesh output ".obj"
    |> sink

    0
