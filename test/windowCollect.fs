// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    debug mem
    |> read<uint8> "image" ".tiff"
    >=> tap "For window"
    >=> window 3u 0u 1u // There is trickery going on here, since ignoreImages disposes of all, but window resuse all but the first image
    >=> tap "For map"
    >=> map (fun lst -> [lst[0]])
    >=> collect ()
    >=> tap "For ignoreImages"
    >=> ignoreImages ()
    |> sink

    0
