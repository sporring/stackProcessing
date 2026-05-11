// Runs marching cubes on the synthetic object image produced by samples/objectsImage.
open System.IO
open StackProcessing

let private samplePath fallback (args: string[]) =
    if args.Length > 0 then
        let token = args[0]
        if token |> Seq.forall System.Char.IsDigit then
            fallback
        else
            token
    else
        fallback

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> samplePath "../data/rotatingBoxes" [| input |], output
        | [| _ |] -> samplePath "../data/rotatingBoxes" args, "../tmp/objectsMesh/surface.obj"
        | _ -> "../data/rotatingBoxes", "../tmp/objectsMesh/surface.obj"

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
