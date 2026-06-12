// Create a 2D polygon mask and repeat it into a small 3D stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/polygonMask"

    let polygon : Polygon2D =
        [ { X = 16.0; Y = 12.0 }
          { X = 50.0; Y = 18.0 }
          { X = 44.0; Y = 50.0 }
          { X = 12.0; Y = 42.0 } ]

    let mask = chunkPolygonMask 64u 64u polygon

    src
    |> chunkRepeat mask 16u
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
