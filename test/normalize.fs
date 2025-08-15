// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing
open ImageFunctions

[<EntryPoint>]
let main _ =
    let mem = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

    let readMaker =
        debug mem
        |> read<float> "image" ".tiff"

    let stats = 
        readMaker 
        >=> StackProcessing.computeStats () // fix the naming conflict!!!
        |> drainSingle
    printfn "%A" stats

    let normalizeWith (stats: ImageStats) (image: Image<float>) =
        let J = imageSubScalar image stats.Mean;
        let K = imageDivScalar J stats.Std
        J.decRefCount()
        K

    // normalizeWith can release image and normalizeWithOp use liftUnary. This saves storing 1 image per iteration
    let normalizeWithOp = liftUnaryReleaseAfter "normalizeWithOp" (normalizeWith stats) id id

    let updatedStats = 
        readMaker
        >=> normalizeWithOp
        >=> StackProcessing.computeStats ()
        |> drainSingle 
    printfn "%A" updatedStats

    0
