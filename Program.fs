module Program
// Program.fs
// To run, remember to:
// export DYLD_LIBRARY_PATH=./lib:$(pwd)/bin/Debug/net8.0
open SmartImagePipeline
open Processing
open IO
open Visualization

[<EntryPoint>]
let main _ =
    let width, height, depth = 64u, 64u, 10u
    let inputStream = IO.readSlicesAsync "inputTest"

    // Create a constant mask of 1s (effectively no-op) or 0s (masking)
    let maskStream = Processing.constant 1uy width height depth

    let processor = multiplyWith maskStream
    let outputStream = processor.Apply inputStream

    IO.writeSlicesSync "output" outputStream

    0
