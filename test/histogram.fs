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

    let readHistogramMaker = 
        source mem
        |> readAs<uint8> "image" ".tiff"
        >=> histogram

    let left, right = tee readHistogramMaker
    let path2 = right >=> map2pairs >=> pairs2floats
    // compile time analysis:
    [path2 >=> tap "path2" >=> plot plt; left >=> tap "left" >=> print ()] |> sinkList
    // runtime analysis:
    // zipWith (fun _ _ -> ()) (path2 >=> plot plt) (left >=> print ()) |> sink

    0
