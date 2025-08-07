// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing
open Slice

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    let normalizeWith (stats: ImageStats) (slice: Slice<float>) =
        sliceDivScalar (sliceSubScalar slice stats.Mean) stats.Std

    let readMaker =
        debug mem
        |> read<float> "image" ".tiff"

    let stats = 
        readMaker 
        >=> StackProcessing.computeStats () // fix the naming conflict!!!
        |> drainSingle
    printfn "%A" stats

    let normalizeWithOp = liftUnary (normalizeWith stats) id id
    let updatedStats = 
        readMaker
        >=> normalizeWithOp
        >=> StackProcessing.computeStats ()
        |> drainSingle 
    printfn "%A" updatedStats

    0
