// Demonstrates the LMIP pattern: histogram -> Otsu threshold estimate -> standard threshold.
open System
open System.IO
open SlimPipeline
open StackProcessing

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private syntheticSlice width height index =
    let image = Image<float32>([width; height], 1u, $"otsu-source[{index}]", index)

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let foreground = x > int width / 2 + int index % 5 - 2
            image[x, y] <- if foreground then 10.0f else 0.0f

    image

let private syntheticSource availableMemory width height depth =
    let stage =
        Stage.init
            "otsu synthetic source"
            depth
            (syntheticSlice width height)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) availableMemory 0UL (uint64 width * uint64 height) (uint64 depth) false

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let root = if args.Length > 0 then args[0] else "../otsuThreshold"
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
        |> otsuThresholdFromHistogram

    source availableMemory
    |> read<float32> input ".tiff"
    >=> threshold thresholdValue infinity
    >=> write output ".tiff"
    |> sink

    printfn "Estimated Otsu threshold: %.6f" thresholdValue
    printfn "Wrote input and thresholded stacks under %s" (Path.GetFullPath root)
    0
