// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let width, height, depth = 1024u, 1024u, 1024u
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory

    src
    |> zero<uint8> width height depth
    >=> addNormalNoise 128.0 50.0
    >=> write "image" ".tiff"
    |> sink

    0
