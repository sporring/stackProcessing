// Demonstrates streaming removal of small connected foreground objects.
open System
open System.IO
open SlimPipeline
open StackProcessing

let private clearDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private inSphere x y z cx cy cz radius =
    let dx = float x - cx
    let dy = float y - cy
    let dz = float z - cz
    dx * dx + dy * dy + dz * dz <= radius * radius

let private isSmallSpeck x y z =
    (x = 7 && y = 8 && z = 3)
    || (x = 22 && y = 9 && (z = 5 || z = 6))
    || (x = 48 && y = 42 && z >= 10 && z <= 12)
    || ((x = 12 || x = 13) && y = 50 && z = 18)

let private syntheticSlice width height depth index =
    let image = Image<uint8>([width; height], 1u, $"small-object-source[{index}]", index)

    for y in 0 .. int height - 1 do
        for x in 0 .. int width - 1 do
            let z = int index
            let largeObject =
                inSphere x y z 20.0 21.0 9.0 7.0
                || inSphere x y z 43.0 39.0 18.0 9.0
                || (x >= 34 && x <= 45 && y >= 10 && y <= 20 && z >= 23 && z <= 29)

            image[x, y] <- if largeObject || isSmallSpeck x y z then 1uy else 0uy

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
            "remove small objects synthetic source"
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
    let root = sampleRoot "removeSmallObjects" args
    let input = Path.Combine(root, "input")
    let cleaned = Path.Combine(root, "cleaned")

    for path in [ input; cleaned ] do
        clearDirectory path

    let width, height, depth = 64u, 64u, 32u
    let maximumVolume = 4UL

    syntheticSource src width height depth
    >=> write input ".tiff"
    |> sink

    syntheticSource src width height depth
    >=> removeSmallObjects maximumVolume ObjectConnectivity.Six
    >=> write cleaned ".tiff"
    |> sink

    printfn "Wrote input and cleaned binary stacks under %s" (Path.GetFullPath root)
    printfn "Foreground components with volume <= %d voxels were removed from the cleaned stack." maximumVolume
    0
