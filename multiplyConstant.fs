// To run, remember to:
// export DYLD_LIBRARY_PATH=./SmartImagePipeline/lib:$(pwd)/bin/Debug/net8.0
open SmartImagePipeline
open Processing
open pipelineIO

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let src = "image"
    let trg = "result"
    let width, height, depth = getVolumeSize src // 64u, 64u, 8u
    let mask = constant 1uy width height depth
    let pipeline = multiplyWith mask

    printfn "Applying pipeline to InputTest image"
    let input = readSlices "image"
    let output = pipeline.Apply input

    printfn "Running pipeline and writing to disk"
    writeSlices "result" output

    0
