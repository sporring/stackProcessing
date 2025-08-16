// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing
open Plotly.NET

[<EntryPoint>]
let main _ =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            Image.Image<_>.setDebug true; 
            debug availableMemory
        else
            source availableMemory

    // Plotly.Net plot function
    let plt (x:float list) (y:float list) = 
         Chart.Column(values = y, Keys = x)
        |> Chart.withTitle "Histogram"
        |> Chart.withXAxisStyle ("gray values")
        |> Chart.withYAxisStyle ("count")
        |> Chart.show

    src
    |> read<uint8> "image" ".tiff" 
    >=> histogram ()
    >=> map2pairs --> pairs2floats --> plot plt
    |> sink

    0
