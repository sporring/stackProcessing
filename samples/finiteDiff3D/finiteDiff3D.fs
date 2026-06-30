// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example
    let src, args = commandLineSource availableMemory args
    let input, output = 
        if args.Length > 0 then
            "../data/objects", "../tmp/finiteDiff3D"
        else
            "../data/objects", "../tmp/finiteDiff3D"

    // read an image and calculate the derivative in the z-direction
    src
    |> read<float32> input ".tiff"
    >=> finiteDiffZ 2u
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
