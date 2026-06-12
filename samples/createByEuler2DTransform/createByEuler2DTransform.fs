// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =

    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let width, height, depth, output = 
        if arg.Length > 0 then
            let n = (int arg[0]) / 3 |> pown 2 |> uint 
            n, n, n, "../tmp/rotatingBoxes"
        else
            64u, 64u, 64u, "../tmp/rotatingBoxes"

    let polygon : Polygon2D =
        [ { X = 24.0; Y = 24.0 }
          { X = 40.0; Y = 24.0 }
          { X = 40.0; Y = 40.0 }
          { X = 24.0; Y = 40.0 } ]

    let mask = chunkPolygonMask width height polygon

    let movingMask transformName =
        src
        |> chunkRepeat mask 1u
        >=> chunkCreateByEuler2DTransformFromChunk<uint8> depth (chunkEuler2DTransformPath width height depth transformName)

    let diagonal =
        movingMask "Diagonal"

    let topDown =
        movingMask "TopDown"

    let antiDiagonal =
        movingMask "AntiDiagonal"

    (
        (diagonal, topDown) ||> zip >=> chunkMaxOfPair<uint8> >=> tap "first",
        antiDiagonal >=> tap "second"
    ) ||> zip >=> chunkMaxOfPair<uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
