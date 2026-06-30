// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume","../tmp/normalize"
        else
            "../data/volume","../tmp/normalize"

    let Float640 = 255.0
    let ImageStats0 =
        src
        |> readRandom<float32> 20u input ".tiff"
        >=> computeStats<float32> ()
        |> drain
    let Float641 = (ImageStats0.Max - ImageStats0.Min)
    let Float642 = (Float640 / Float641)

    src
    |> read<float32> input ".tiff"
    >=> subScalar ImageStats0.Min
    >=> mulScalar Float642
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
