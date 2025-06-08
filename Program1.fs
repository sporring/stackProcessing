// Program.fs
// To run, remember to:
// export DYLD_LIBRARY_PATH=./lib:$(pwd)/bin/Debug/net8.0
open SmartImagePipeline
open Processing
open IO
open Visualization

[<EntryPoint>]
let main _ =
    // --- Example Usage ---
    let width, height, depth = 64u, 64u, 8u
    let availableMemory = 1024UL * 1024UL // 1MB for example
    let input = constant 100uy width height depth

    let composed =
        pipeline availableMemory width height depth {
            return
                additiveGaussianNoise 0.0 50.0
                >>=> threshold 100.0 255.0
                >>=> shiftScale 0.0 255.0
                >>=> convolve3DGaussian 1.0
//                >>=> floodFill3D 32u 32u 4u 100.0 255.0
        }

    let result = composed.Apply input // AsyncSeq<ImageSlice>

    result |> showSliceAtIndex 0 |> ignore

    writeSlicesAsync "output" result |> Async.RunSynchronously

    0
