// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline
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

    let readMaker = 
        source mem
        |> readAs<uint8> "image" ".tiff"
    let plotHist = map2pairs --> pairs2floats --> plot plt

    readMaker 
    >=>> (histogram --> plotHist, castUInt8ToFloat --> convGauss 1.0 None --> castFloatToUInt8 --> histogram --> plotHist)
    >>=> combineIgnore
    |> sink

    0
