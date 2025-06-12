// To run, remember to:
// export DYLD_LIBRARY_PATH=./SmartImagePipeline/lib:$(pwd)/bin/Debug/net8.0
open SmartImagePipeline
open Processing
open pipelineIO

[<EntryPoint>]
let main _ =
    let outputStream = readRandomSlices 2u "image"
    showSliceAtIndex 0u outputStream |> Async.RunSynchronously

    0
