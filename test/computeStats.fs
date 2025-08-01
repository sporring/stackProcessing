// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing
open Plotly.NET

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    // Plotly.Net plot function
    let plt (x:float list) (y:float list) = 
         Chart.Column(values = y, Keys = x)
        |> Chart.withTitle "Histogram"
        |> Chart.withXAxisStyle ("gray values")
        |> Chart.withYAxisStyle ("count")
        |> Chart.show

    debug mem
    |> readAs<uint8> "image" ".tiff"
    >=> computeStats () --> print ()
    |> sink

    0
