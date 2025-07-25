// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline
open Slice

[<EntryPoint>]
let main _ =
    let src = "image"
    let trg = "result"
    let mem = 1024UL * 1024UL // 1MB for example

    let normalizeWith (stats: ImageStats) (slice: Slice<float>) =
        sliceDivScalar (sliceSubScalar slice stats.Mean) stats.Std

    let readMaker =
        source<Slice<float>> mem
        |> read "image" ".tiff"

    let stats = 
        readMaker >=> computeStats
        |> cacheScalar "stats"

    stats >=> print |> sink

    zipWith normalizeWith stats readMaker
    >=> computeStats >=> print 
    |> sink

    0
