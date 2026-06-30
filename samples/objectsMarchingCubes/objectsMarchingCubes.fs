// Runs marching cubes on the binary object image and writes a mesh file.
open System.IO
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

    let directory = Path.GetDirectoryName(output)
    if not (System.String.IsNullOrWhiteSpace directory) then
        Directory.CreateDirectory(directory) |> ignore

    src
    |> read<uint8> input ".tiff"
    >=> marchingCubes 0.5
    >=> writeMesh output ".obj"
    |> sink

    printfn "Wrote mesh to %s" (Path.GetFullPath(meshFilePath output ".obj"))
    0
