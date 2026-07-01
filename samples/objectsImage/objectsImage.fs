// Stream connected objects from a binary stack to disk.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/objectsImage"
        | _ -> "../data/objects", "../tmp/objectsImage"

    let info = getImageInfo input ".tiff"
    let size = info.size
    let width,height,depth = size[0],size[1],size[2]
    let objectsDirectory = output + "/objects"
    let imageDirectory = output + "/image"

    // Stream connected objects from a binary stack to disk as a single pass.
    src
    |> read<uint8> input ".tiff"
    >=> streamConnectedObjects ObjectConnectivity.TwentySix
    >=> writeObjects objectsDirectory ".csv"
    |> sink

    // Paint the connected objects to an image stack and write to disk. paintObject requires the objects to be ordered by first z-value. Therefore, readObjects reads in lexicographical order of the objects filenames.
    src
    |> readObjects<uint8> objectsDirectory ".csv"
    >=> paintObjects width height depth None (Some 255.0)
    >=> write imageDirectory ".tiff"
    |> sink

    0
