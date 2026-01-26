// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =

    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            debug availableMemory
        else
            source availableMemory
    let width, height, depth, output = 
        if arg.Length > 1 then
            let n = (int arg[1]) / 3 |> pown 2 |> uint 
            n, n, n, $"image{arg[1]}"
        else
            64u, 64u, 64u, "../image18"

    let boxSz = 16;

    let img = Image<uint8>([width;height])
    for i in [0..boxSz-1] do
        for j in [0..boxSz-1] do
            img[i,j] <- 255uy

    let transFctDiag (i:uint) : (float*float*float)*(float*float) =
        let dx = float i
        let a = 2.0*3.141592*(float i)/(float depth)
        let offset = (float boxSz)/2.0-0.5
        //(offset,offset,a),(dx-offset,dx-offset)
        (offset,offset,a),(0.0,0.0)

    let transFctAntiDiag (i:uint) : (float*float*float)*(float*float) =
        let dx = float i
        let a = 2.0*3.141592*(float i)/(float depth)
        let offset = (float boxSz)/2.0-0.5
        (offset,offset,a),(float(width)-dx-offset,dx-offset)

    let transFctTopDown (i:uint) : (float*float*float)*(float*float) =
        let dx = float i
        let a = 2.0*3.141592*(float i)/(float depth)
        let offset = (float boxSz)/2.0-0.5
        (offset,offset,a),(float width/2.0-offset,dx-offset)

    let diagonal =
        src
        |> createByEuler2DTransform<uint8> img depth transFctDiag

    let topDown =
        src
        |> createByEuler2DTransform<uint8> img depth transFctTopDown

    let antiDiagonal =
        src
        |> createByEuler2DTransform<uint8> img depth transFctAntiDiag

    (
        (diagonal, topDown) ||> zip >>=> maxOfPair >=> tap "first",
        antiDiagonal >=> tap "second"
    ) ||> zip >>=> maxOfPair
    >=> write output ".tiff"
    |> sink

    0
