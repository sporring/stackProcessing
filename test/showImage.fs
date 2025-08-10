// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing
open Image
open ImageFunctions
open Plotly.NET

[<EntryPoint>]
let main _ =
    let availableMemory = 1024UL * 1024UL // 1MB for example

    // Plotly.Net plot function
    let plt (I: Image<uint8>) = 
        let img = ImageFunctions.toSeqSeq I
        Chart.Heatmap(img)
        |> Chart.withTitle "An Image"
        |> Chart.show

    debug availableMemory
    |> readRandom<uint8> 1u "image" ".tiff"
    >=> show plt 
    |> sink

    0
