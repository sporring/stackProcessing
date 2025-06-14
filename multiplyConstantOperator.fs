// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackPipeline
open Processing
open pipelineIO

[<EntryPoint>]
let main _ =
    printfn "Setting up pipeline"
    let trg = "result"
    let width, height, depth = 64u, 64u, 8u
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let composed =
        pipeline availableMemory width height depth {
            return
                additiveGaussianNoise 0.0 50.0
                >>=> threshold 100.0 255.0
                >>=> shiftScale 0.0 255.0
                >>=> convolve3DGaussian 1.0
        }

    printfn "Applying pipeline to InputTest image"
    let input = constant 100uy width height depth
    let result = composed.Apply input

    printfn "Running pipeline and writing to disk"
    writeSlices trg result

    0
