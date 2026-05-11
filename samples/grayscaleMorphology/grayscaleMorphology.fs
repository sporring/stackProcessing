// Grayscale morphology filters for local bright/dark structure.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/grayscaleMorphology"
        | _ -> "../data/volume", "../tmp/grayscaleMorphology"

    src
    |> read<uint8> input ".tiff"
    >=> grayscaleErode<uint8> 1u 5u
    >=> grayscaleDilate<uint8> 1u 5u
    >=> grayscaleOpening<uint8> 1u 5u
    >=> grayscaleClosing<uint8> 1u 5u
    >=> whiteTopHat<uint8> 1u 5u
    >=> blackTopHat<uint8> 1u 5u
    >=> morphologicalGradient<uint8> 1u 5u
    >=> write output ".tiff"
    |> sink

    0
