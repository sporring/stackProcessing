// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing
open Plotly.NET

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            debug availableMemory
        else
            source availableMemory
    let input = 
        if arg.Length > 1 then
            $"image{arg[1]}"
        else
            "../image18"

    // Plotly.Net plot function
    let plt (I: Image<uint8>) = 
        let img = ImageFunctions.toSeqSeq I
        Chart.Heatmap(img)
        |> Chart.withTitle "An Image"
        |> Chart.show

    src
    |> readRandom<uint8> 1u input ".tiff"
    >=> show plt 
    |> sink

    0
