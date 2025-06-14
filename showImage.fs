// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackPipeline
open Processing
open pipelineIO

[<EntryPoint>]
let main _ =
    let outputStream = readRandomSlices 2u "image"
    showSliceAtIndex 0u outputStream |> Async.RunSynchronously

    0
