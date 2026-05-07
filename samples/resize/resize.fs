// Demonstrates axis-aligned resize to an explicit x/y/z size.
open System
open System.IO
open SlimPipeline
open StackProcessing

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private syntheticSlice width height index =
    let image = Image<float32>([width; height], 1u, $"resize-source[{index}]", index)
    let cx = float32 width * 0.5f
    let cy = float32 height * 0.5f

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let dx = float32 x - cx
            let dy = float32 y - cy
            let radial = 120.0f * float32 (Math.Exp (float (-(dx * dx + dy * dy) / 90.0f)))
            image[x, y] <- radial + float32 x + 3.0f * float32 y + 12.0f * float32 index

    image

let private syntheticSource availableMemory width height depth =
    let stage =
        Stage.init
            "resize synthetic source"
            depth
            (syntheticSlice width height)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) availableMemory 0UL (uint64 width * uint64 height) (uint64 depth) false

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let root = if args.Length > 0 then args[0] else "../resize"
    let original = Path.Combine(root, "original")
    let resizedLinear = Path.Combine(root, "resized-linear")
    let resizedNearest = Path.Combine(root, "resized-nearest")

    for path in [ original; resizedLinear; resizedNearest ] do
        clearDirectory path

    syntheticSource availableMemory 32u 32u 12u
    >=> write original ".tiff"
    |> sink

    syntheticSource availableMemory 32u 32u 12u
    |> resize<float32> 64u 48u 20u "Linear"
    >=> write resizedLinear ".tiff"
    |> sink

    syntheticSource availableMemory 32u 32u 12u
    |> resize<float32> 20u 20u 8u "NearestNeighbor"
    >=> write resizedNearest ".tiff"
    |> sink

    printfn "Wrote original and explicit-size resized stacks under %s" (Path.GetFullPath root)
    0
