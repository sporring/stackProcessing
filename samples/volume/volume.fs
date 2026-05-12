// Resize the sample volume file to a small image stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 1UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | _ -> "../data/volumedata.tif", "../tmp/volume"

    src
    |> readVolume<uint8> (volumeFilePath input ".tiff")
    |> resize<uint8> 64u 64u 64u "Linear"
    >=> write output ".tiff"
    |> sink

    0
