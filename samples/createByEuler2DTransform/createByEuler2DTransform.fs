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

    // Create square as a mask image from a polygon
    let polygon : Polygon2D =
        [ { X = 24.0; Y = 24.0 }
          { X = 40.0; Y = 24.0 }
          { X = 40.0; Y = 40.0 }
          { X = 24.0; Y = 40.0 } ]
    let mask = polygonMask width height polygon

    let euler2DTransformPath (transformName: string) (i: uint) (chunk: StackCore.Chunk<uint8>) =
        let centerX = float width / 2.0 - 0.5
        let centerY = float height / 2.0 - 0.5
        let dx = float i
        let angle = 2.0 * System.Math.PI * float i / float depth

        let rotation, translation =
            match transformName with
            | "AntiDiagonal" -> (centerX, centerY, angle), (float width - dx - centerX, dx - centerY)
            | "TopDown" ->      (centerX, centerY, angle), (0.0, dx - centerY)
            | _ ->              (centerX, centerY, angle), (0.0, 0.0)
        euler2DTransform<uint8> rotation translation chunk

    // Create 3 sources that are squares which move in different ways.
    let movingMask transformName =
        src 
        |> repeat mask depth
        >=> mapi (euler2DTransformPath transformName)
    let diagonal =     movingMask "Diagonal"
    let topDown =      movingMask "TopDown"
    let antiDiagonal = movingMask "AntiDiagonal"

    zip (zip diagonal topDown >>=> maxOfPair<uint8>) antiDiagonal
    >>=> maxOfPair<uint8>
    >=> write output ".tiff"
    |> sink

    0
