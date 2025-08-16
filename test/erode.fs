// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let width, height, depth = 1024u, 1024u, 1024u
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 1MB for example
    let radius = 1u

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory

    src
    |> zero<uint8> width height depth
    >=> addNormalNoise 128.0 50.0
    >=> threshold 128.0 infinity
    >=> erode radius 
    //>=> dilate radius
    //>=> opening radius
    //>=> closing radius
    >=> write "result" ".tif"
    |> sink

    0
