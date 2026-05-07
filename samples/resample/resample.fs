// Demonstrates axis-aligned resampling by x/y/z scale factors.
open System
open System.IO
open SlimPipeline
open StackProcessing

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private syntheticSlice width height index =
    let image = Image<float32>([width; height], 1u, $"resample-source[{index}]", index)
    let cx = float32 width * 0.45f
    let cy = float32 height * 0.55f

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let dx = float32 x - cx
            let dy = float32 y - cy
            let wave =
                30.0f * float32 (Math.Sin (float32 x / 4.0f |> float))
                + 20.0f * float32 (Math.Cos (float32 y / 5.0f |> float))
            let spot = 150.0f * float32 (Math.Exp (float (-(dx * dx + dy * dy) / 110.0f)))
            image[x, y] <- spot + wave + 10.0f * float32 index

    image

let private sampleRoot sampleName (args: string[]) =
    if args.Length > 0 then
        let token = args[0]
        if token |> Seq.forall Char.IsDigit then
            $"../{sampleName}{token}"
        else
            token
    else
        $"../{sampleName}"

let private syntheticSource src width height depth =
    let stage =
        Stage.init
            "resample synthetic source"
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
    let root = sampleRoot "resample" args
    let original = Path.Combine(root, "original")
    let upsampled = Path.Combine(root, "resampled-up-linear")
    let anisotropic = Path.Combine(root, "resampled-anisotropic-nearest")

    for path in [ original; upsampled; anisotropic ] do
        clearDirectory path

    syntheticSource src 32u 32u 12u
    >=> write original ".tiff"
    |> sink

    syntheticSource src 32u 32u 12u
    |> resample<float32> 1.5 1.5 2.0 "Linear"
    >=> write upsampled ".tiff"
    |> sink

    syntheticSource src 32u 32u 12u
    |> resample<float32> 0.75 1.25 0.5 "NearestNeighbor"
    >=> write anisotropic ".tiff"
    |> sink

    printfn "Wrote original and factor-resampled stacks under %s" (Path.GetFullPath root)
    0
