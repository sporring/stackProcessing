// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline
open Slice
open Plotly.NET

[<EntryPoint>]
let main _ =
    let src = "image"
    let trg = "result"
    let width, height, depth = getStackSize src ".tiff"
    let availableMemory = 1024UL * 1024UL // 1MB for example

    // Plotly.Net plot function
    let plt (slice: Slice<uint8>) = 
        let img = toSeqSeq slice
        Chart.Heatmap(img)
        |> Chart.withTitle "An Image"
        |> Chart.show

    source<Slice<uint8>> availableMemory
        |> readRandom 1u "image" ".tiff"
        >=> show plt 
        |> sink

    0
