// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main _ =
    let mem = 1024UL * 1024UL // 1MB for example

    let readMaker = 
        debug mem
        |> read<float> "image" ".tiff"
//        >=> tapIt (fun s -> $"[readAs] {s.Index} -> Image {s.Image}")

    let zeroMaker i = id
    let winSz,pad,stride = 5u,2u,1u
    let delay (stg:Stage<'S,'T>): Stage<'S,'T> = 
        stg |> promoteStreamingToSliding "testing" winSz pad stride 0u 1u

    readMaker 
    >=> tap "For >=>>"
    //>=>> (delay (idOp<Image<float>> ()), delay (idOp<Image<float>> ()))
    //>=>> (delay (idOp<Image<float>> ()), convGauss 1.0)
    >=>> (idOp<Image<float>> (), convGauss 1.0)
    >=> tap "For >>=>"
(*
    >>=> mul2
    >=> tap "cast"
    >=> cast<float,int8>
    >=> tap "write"
    >=> write "result" ".tiff"
*)
    >>=> fun _ _ -> ()
    |> sink

    0
