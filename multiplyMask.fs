// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackPipeline
open Processing
open pipelineIO

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let mask = readSlices "mask"
    let pipeline = multiplyWith mask

    printfn "Applying pipeline to an image"
    let input = readSlices "image"
    let output = pipeline.Apply input

    printfn "Running pipeline and writing to disk"
    writeSlices "result" output

    0
