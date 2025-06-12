// To run, remember to:
// export DYLD_LIBRARY_PATH=./SmartImagePipeline/lib:$(pwd)/bin/Debug/net8.0
open SmartImagePipeline
open Processing
open pipelineIO

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let pipeline = histogram

    printfn "Applying pipeline to an image"
    let input = readRandomSlices 2u "image"
    let output = pipeline.Apply input 

    printVector output
    plotVector output

    0
