// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main args =
    printfn "Setting up pipeline"
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 1MB for example
    let src, args = commandLineSource availableMemory args
 
    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/erode"
        | _ -> "../data/rotatingBoxes", "../tmp/erode"

    // Erode a binary image with a sphere of radius 2. Stretch intensities for easy viewing.
    src
    |> read<uint8> input ".tiff"
    >=> binaryErode 2u
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    0
