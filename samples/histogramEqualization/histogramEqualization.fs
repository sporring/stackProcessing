// Equalize stack intensities from a histogram estimated from random slices.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/histogramEqualization"
        | _ -> "../data/volume", "../tmp/histogramEqualization"

    let histogram =
        src
        |> readChunkSlicesRandom<uint8> 8u input ".tiff"
        >=> chunkHistogramDense<uint8> ()
        |> drain

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkHistogramEqualization<uint8> (histogram :> obj)
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
