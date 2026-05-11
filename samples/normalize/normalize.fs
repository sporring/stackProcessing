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
        |> readRandom<float> 20u input ".tiff"
        >=> computeStats ()
        |> drain
    let Float641 = (ImageStats0.Max - ImageStats0.Min)
    let Float642 = (Float640 / Float641)

    src
    |> read<float> input ".tiff"
    >=> imageSubScalar ImageStats0.Min
    >=> imageMulScalar Float642
    >=> cast<float,uint8>
    >=> write output ".tiff"
    |> sink

    0
