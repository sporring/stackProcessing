// Compare synthetic noise sources and slice-wise noise additions.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let outputRoot =
        match args with
        | [| outputRoot |] -> outputRoot
        | _ -> "../tmp/noiseVariants"

    src
    |> saltAndPepperNoise<uint8> 64u 64u 16u 0.02
    >=> write (outputRoot + "/saltAndPepper") ".tiff"
    |> sink

    src
    |> shotNoise<float> 64u 64u 16u 2.0
    >=> cast<float, uint8>
    >=> write (outputRoot + "/shot") ".tiff"
    |> sink

    src
    |> speckleNoise<float> 64u 64u 16u 0.5
    >=> cast<float, uint8>
    >=> write (outputRoot + "/speckle") ".tiff"
    |> sink

    src
    |> zero<uint8> 64u 64u 16u
    >=> addSaltAndPepperNoise 0.02
    >=> write (outputRoot + "/addedSaltAndPepper") ".tiff"
    |> sink

    src
    |> zero<float> 64u 64u 16u
    >=> addShotNoise 2.0
    >=> addSpeckleNoise 0.5
    >=> cast<float, uint8>
    >=> write (outputRoot + "/addedShotSpeckle") ".tiff"
    |> sink

    0
