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

    // Create square as a mask image from a polygon in the middle of a slice
    let polygon : Polygon2D =
        let n = min width height
        let lower = n/2u - n/8u |> float
        let upper = n/2u + n/8u |> float
        [ { X = lower; Y = lower }
          { X = upper; Y = lower }
          { X = upper; Y = upper }
          { X = lower; Y = upper } ]
    let mask = polygonMask width height polygon

    // Make a function for transforming masks slices in interesting way
    let euler2DTransformPath (transformName: string) (i: uint) (chunk: StackCore.Chunk<uint8>) =
        let centerX = float width / 2.0 - 0.5
        let centerY = float height / 2.0 - 0.5
        let dx = float i
        let angle = 2.0 * System.Math.PI * float i / float depth

        let rotation, translation =
            match transformName with
            | "AntiDiagonal" -> (centerX, centerY, angle), (float width - dx - centerX, dx - centerY)
            | "TopDown" ->      (centerX, centerY, angle), (0.0, dx - centerY)
            | _ ->              (centerX, centerY, angle), (-centerX, -centerY)
        euler2DTransform<uint8> rotation translation chunk

    // Create 3 sources that are squares which move in different ways.
    let movingMask transformName =
        src 
        |> repeat mask depth
        >=> mapi (euler2DTransformPath transformName)
    let diagonal =     movingMask "Diagonal"
    let topDown =      movingMask "TopDown"
    let antiDiagonal = movingMask "AntiDiagonal"

    // Max of a tripple fan-out
    zip (zip diagonal topDown >>=> maxOfPair) antiDiagonal
    >>=> maxOfPair
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    0
