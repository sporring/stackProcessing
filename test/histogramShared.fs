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

    let histogramMaker = 
        source mem
        |> read<uint8> "image" ".tiff"
        >=> histogram () --> map2pairs --> pairs2floats
    histogramMaker
    >=>> (print (),plot plt)
    //>=>> (tap "left", tap "right")
    >>=> fun _ _ -> ()
//    >>=> ignorePairs ()
    |> sink

    0
