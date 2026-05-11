// Demonstrates streaming band-limited signed distance maps.
open System
open System.IO
open SlimPipeline
open StackProcessing

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private syntheticSlice width height depth index =
    let image = Image<uint8>([width; height], 1u, $"signed-distance-source[{index}]", index)
    let cx = float width * 0.5
    let cy = float height * 0.5
    let cz = float depth * 0.5
    let radius = float (min width height) * 0.23

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let dx = float x - cx
            let dy = float y - cy
            let dz = float index - cz
            image[x, y] <- if dx * dx + dy * dy + dz * dz <= radius * radius then 1uy else 0uy

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
            "signed distance synthetic source"
            depth
            (syntheticSlice width height depth)
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) src.memAvail 0UL (uint64 width * uint64 height) (uint64 depth) src.debug

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args
    let root = sampleRoot "signedDistanceBand" args
    let mask = Path.Combine(root, "mask")
    let distance = Path.Combine(root, "distance-band")

    for path in [ mask; distance ] do
        clearDirectory path

    let width, height, depth = 64u, 64u, 32u
    let bandRadius = 6u
    let stride = 4u

    syntheticSource src width height depth
    >=> write mask ".tiff"
    |> sink

    syntheticSource src width height depth
    >=> signedDistanceBand bandRadius stride
    >=> write distance ".mha"
    |> sink

    printfn "Wrote binary mask and signed distance band under %s" (Path.GetFullPath root)
    printfn "Values outside +/- %u pixels from the object boundary are NaN." bandRadius
    0
