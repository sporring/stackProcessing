#I "../../src/StackProcessing.Probe/bin/Debug/net10.0"
#I "../../src/StackProcessing/bin/Debug/net10.0"

#r "FSharp.Control.AsyncSeq.dll"
#r "SimpleITKCSharpManaged.dll"
#r "AsyncSeqExtensions.dll"
#r "TinyLinAlg.dll"
#r "Image.dll"
#r "SlimPipeline.dll"
#r "StackProcessing.Cost.dll"
#r "StackProcessing.Core.dll"
#r "StackProcessing.dll"

open System
open System.Diagnostics
open Image

let width = 1024
let height = 1024
let depth = 32
let repeats = 3

let time name f =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let sw = Stopwatch.StartNew()
    let checksum = f ()
    sw.Stop()
    printfn "%-28s %8.1f ms checksum=%0.1f" name sw.Elapsed.TotalMilliseconds checksum

let releaseAll (images: Image<float> list) =
    images |> List.iter (fun image -> image.decRefCount())

let currentLoop mapper =
    [ for z in 0 .. depth - 1 ->
        let values = Array2D.zeroCreate<float> width height
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                values[x, y] <- mapper x y z
        Image<float>.ofArray2D(values, "currentLoop", z) ]

let zeroMApi mapper =
    [ for z in 0 .. depth - 1 ->
        let zero = new Image<float>([ uint width; uint height ], 1u, "zero", z)
        let image = Image.mapi (fun index _ -> mapper (int index[0]) (int index[1]) z) zero
        zero.decRefCount()
        image ]

let constantArrayZ () =
    [ for z in 0 .. depth - 1 ->
        Array2D.create width height (float z)
        |> fun values -> Image<float>.ofArray2D(values, $"coordinateZ[{z}]", z) ]

let copyBaseX () =
    let values = Array2D.init width height (fun x _ -> float x)
    let baseImage = Image<float>.ofArray2D(values, "coordinateX.base", 0)
    let images =
        [ for z in 0 .. depth - 1 ->
            baseImage.copy($"coordinateX[{z}]", z) ]
    baseImage.decRefCount()
    images

let copyBaseArrayX () =
    let values = Array2D.init width height (fun x _ -> float x)
    [ for z in 0 .. depth - 1 ->
        Array2D.copy values
        |> fun slice -> Image<float>.ofArray2D(slice, $"coordinateX[{z}]", z) ]

let copyBaseY () =
    let values = Array2D.init width height (fun _ y -> float y)
    let baseImage = Image<float>.ofArray2D(values, "coordinateY.base", 0)
    let images =
        [ for z in 0 .. depth - 1 ->
            baseImage.copy($"coordinateY[{z}]", z) ]
    baseImage.decRefCount()
    images

let copyBaseArrayY () =
    let values = Array2D.init width height (fun _ y -> float y)
    [ for z in 0 .. depth - 1 ->
        Array2D.copy values
        |> fun slice -> Image<float>.ofArray2D(slice, $"coordinateY[{z}]", z) ]

let checksumAndRelease (images: Image<float> list) =
    let checksum =
        images
        |> List.mapi (fun z image ->
            image.Get [ uint (width - 1); uint (height - 1) ] + float z)
        |> List.sum
    releaseAll images
    checksum

let run name f =
    for i in 1 .. repeats do
        time $"{name} #{i}" (fun () -> f () |> checksumAndRelease)

printfn "Coordinate source construction benchmark: %dx%dx%d, repeats=%d" width height depth repeats
run "X current loop" (fun () -> currentLoop (fun x _ _ -> float x))
run "X copy base" copyBaseX
run "X copy base array" copyBaseArrayX
run "Y current loop" (fun () -> currentLoop (fun _ y _ -> float y))
run "Y copy base" copyBaseY
run "Y copy base array" copyBaseArrayY
run "Z current loop" (fun () -> currentLoop (fun _ _ z -> float z))
run "Z Array2D.create" constantArrayZ
run "Z zero mapi" (fun () -> zeroMApi (fun _ _ z -> float z))
