// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open Pipeline
open Slice

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    let normalizeWith (stats: ImageStats) (slice: Slice<float>) =
        sliceDivScalar (sliceSubScalar slice stats.Mean) stats.Std

    let readMaker =
        source mem
        |> readAs<float> "image" ".tiff"

    let stats = 
        readMaker 
        >=> Pipeline.computeStats // fix the naming conflict!!!
        |> drainSingle
    printfn "%A" stats

    let normalizeWithOp = liftUnary (normalizeWith stats)
    let updatedStats = 
        readMaker
        >=> normalizeWithOp
        >=> Pipeline.computeStats 
        |> drainSingle
    printfn "%A" updatedStats

    0
