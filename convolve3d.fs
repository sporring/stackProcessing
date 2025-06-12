// To run, remember to:
// export DYLD_LIBRARY_PATH=./SmartImagePipeline/lib:$(pwd)/bin/Debug/net8.0
open SmartImagePipeline
open Processing
open pipelineIO

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let pipeline = convolve3DGaussian 1.0

    printfn "Applying pipeline to an image"
    let input = readSlices "image"
    let output = pipeline.Apply input

    printfn "Running pipeline and writing to disk"
    writeSlices "result" output

    0
