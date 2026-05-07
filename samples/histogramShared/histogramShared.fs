// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing
open Plotly.NET

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input = 
        if arg.Length > 0 then
            $"../image{arg[0]}"
        else
            "../image18"

    // Plotly.Net plot function
    let plt (x:float list) (y:float list) = 
         Chart.Column(values = y, Keys = x)
        |> Chart.withTitle "Histogram"
        |> Chart.withXAxisStyle ("gray values")
        |> Chart.withYAxisStyle ("count")
        |> Chart.show

    let histogramMaker = 
        src
        |> read<uint8> input ".tiff"
        >=> histogram () --> map2pairs --> pairs2floats
    histogramMaker
    >=>> (print (),plot plt)
    //>=>> (tap "left", tap "right")
    >>=> fun _ _ -> ()
//    >>=> ignorePairs ()
    |> sink

    0
