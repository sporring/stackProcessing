// Demonstrates the LMIP pattern: histogram -> moments threshold estimate -> standard threshold.
open System
open System.IO
open SlimPipeline
open StackProcessing

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private syntheticSlice width height index =
    let image = Image<float32>([width; height], 1u, $"moments-source[{index}]", index)
    let cx = float width * 0.5 + Math.Sin(float index * 0.3) * 5.0
    let cy = float height * 0.5

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let dx = float x - cx
            let dy = float y - cy
            let inside = dx * dx / 300.0 + dy * dy / 180.0 <= 1.0
            image[x, y] <- if inside then 12.0f else 2.0f

    image

let private syntheticSource availableMemory width height depth =
    let stage =
        Stage.init
            "moments synthetic source"
            depth
            (syntheticSlice width height)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) availableMemory 0UL (uint64 width * uint64 height) (uint64 depth) false

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let root = if args.Length > 0 then args[0] else "../momentsThreshold"
    let input = Path.Combine(root, "input")
    let output = Path.Combine(root, "output")

    for path in [ input; output ] do
        clearDirectory path

    syntheticSource availableMemory 64u 64u 24u
    >=> write input ".tiff"
    |> sink

    let thresholdValue =
        source availableMemory
        |> readRandom<float32> 8u input ".tiff"
        >=> histogram ()
        |> drain
        |> momentsThresholdFromHistogram

    source availableMemory
    |> read<float32> input ".tiff"
    >=> threshold thresholdValue infinity
    >=> write output ".tiff"
    |> sink

    printfn "Estimated moments threshold: %.6f" thresholdValue
    printfn "Wrote input and thresholded stacks under %s" (Path.GetFullPath root)
    0
