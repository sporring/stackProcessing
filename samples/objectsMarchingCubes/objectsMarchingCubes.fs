// Runs marching cubes on the synthetic object image produced by samples/objectsImage.
open System.IO
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL

    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../objectsMesh/surface.obj"
        | _ -> "../objectsImage", "../objectsMesh/surface.obj"

    let directory = Path.GetDirectoryName(output)
    if not (System.String.IsNullOrWhiteSpace directory) then
        Directory.CreateDirectory(directory) |> ignore

    src
    |> read<uint8> input ".tiff"
    >=> marchingCubes<uint8> 0.5
    >=> writeMesh output "auto"
    |> sink

    printfn "Wrote mesh to %s" (Path.GetFullPath output)
    0
