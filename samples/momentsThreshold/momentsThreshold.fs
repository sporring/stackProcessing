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

let private sampleRoot sampleName (args: string[]) =
    if args.Length > 0 then
        let token = args[0]
        if token |> Seq.forall Char.IsDigit then
            "../tmp/{sampleName}{token}"
        else
            token
    else
        "../tmp/{sampleName}"

let private syntheticSource src width height depth =
    let stage =
        Stage.init
            "moments synthetic source"
            depth
            (syntheticSlice width height)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) src.memAvail 0UL (uint64 width * uint64 height) (uint64 depth) src.debug

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args
    let root = sampleRoot "momentsThreshold" args
    let input = Path.Combine(root, "input")
    let output = Path.Combine(root, "output")

    for path in [ input; output ] do
        clearDirectory path

    syntheticSource src 64u 64u 24u
    >=> write input ".tiff"
    |> sink

    let thresholdValue =
        src
        |> readRandom<float32> 8u input ".tiff"
        >=> imHistogram ()
        |> drain
        |> momentsThresholdFromHistogram

    src
    |> read<float32> input ".tiff"
    >=> threshold thresholdValue infinity
    >=> write output ".tiff"
    |> sink

    printfn "Estimated moments threshold: %.6f" thresholdValue
    printfn "Wrote input and thresholded stacks under %s" (Path.GetFullPath root)
    0
