// Demonstrates reducing a 3D stack to one 2D summed projection image.
open System
open System.IO
open SlimPipeline
open StackProcessing

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private syntheticSlice width height depth index =
    let image = Image<float>([width; height], 1u, $"projection-source[{index}]", index)
    let z = float index
    let depthF = float depth

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let xf = float x
            let yf = float y
            let ridge =
                let center = 10.0 + 42.0 * z / max 1.0 (depthF - 1.0)
                let dx = xf - center
                let dy = yf - 22.0
                120.0 * Math.Exp(-(dx * dx + dy * dy) / 80.0)

            let sphere =
                let dx = xf - 42.0
                let dy = yf - 41.0
                let dz = z - 16.0
                if dx * dx + dy * dy + dz * dz <= 9.0 * 9.0 then 90.0 else 0.0

            let background = 8.0 + 0.15 * xf + 0.1 * yf
            image[x, y] <- background + ridge + sphere

    image

let private syntheticSource availableMemory width height depth =
    let stage =
        Stage.init
            "sum projection synthetic source"
            depth
            (syntheticSlice width height depth)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) availableMemory 0UL (uint64 width * uint64 height) (uint64 depth) false

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let root = if args.Length > 0 then args[0] else "../sumProjection"
    let input = Path.Combine(root, "input")
    let projection = Path.Combine(root, "projection")
    let logProjection = Path.Combine(root, "projection-log1pabs")

    for path in [ input; projection; logProjection ] do
        clearDirectory path

    let width, height, depth = 64u, 64u, 32u

    syntheticSource availableMemory width height depth
    >=> write input ".tiff"
    |> sink

    syntheticSource availableMemory width height depth
    >=> sumProjection<float> "Identity"
    >=> write projection ".mha"
    |> sink

    syntheticSource availableMemory width height depth
    >=> sumProjection<float> "Log1pAbs"
    >=> write logProjection ".mha"
    |> sink

    printfn "Wrote source slices and two summed projection images under %s" (Path.GetFullPath root)
    printfn "The projection folders contain one image each: image_000.mha."
    0
