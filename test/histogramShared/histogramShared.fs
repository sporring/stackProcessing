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
    let input,output = 
        if arg.Length > 1 then
            $"image{arg[1]}", $"result{arg[1]}"
        else
            "image18", "result18"

    // Plotly.Net plot function
    let plt (x:float list) (y:float list) = 
         Chart.Column(values = y, Keys = x)
        |> Chart.withTitle "Histogram"
        |> Chart.withXAxisStyle ("gray values")
        |> Chart.withYAxisStyle ("count")
        |> Chart.show

    let histogramMaker = 
        src
        |> read<uint8> ("../"+input) ".tiff"
        >=> histogram () --> map2pairs --> pairs2floats
    histogramMaker
    >=>> (print (),plot plt)
    //>=>> (tap "left", tap "right")
    >>=> fun _ _ -> ()
//    >>=> ignorePairs ()
    |> sink

    0
