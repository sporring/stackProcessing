// Resample a small chunked volume with affine trilinear interpolation.
open System.IO
open StackProcessing
open TinyLinAlg

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/resampleAffineTrilinearSlices"
        | _ -> "../data/volume", "../tmp/resampleAffineTrilinearSlices"

    let chunks = "../tmp/resampleAffineTrilinearSlicesChunks"
    deleteIfExists chunks
    deleteIfExists output

    src
    |> readRange<uint8> 0u 1 15u input ".tiff"
    >=> writeChunks chunks ".tiff" 16u 16u 16u
    >=> ignoreSingles ()
    |> sink

    let identity =
        { m00 = 1.0; m01 = 0.0; m02 = 0.0
          m10 = 0.0; m11 = 1.0; m12 = 0.0
          m20 = 0.0; m21 = 0.0; m22 = 1.0 }

    let inputGeometry : StackAffineResampler.ImageGeom =
        { W = 64; H = 64; D = 16
          Origin = v3 0.0 0.0 0.0
          Spacing = v3 1.0 1.0 1.0
          Direction = identity }

    let outputGeometry =
        { inputGeometry with D = 8 }

    let affine =
        { A = identity
          T = v3 4.0 -2.0 0.0
          C = v3 32.0 32.0 4.0 }

    let lerp a b t = a + (b - a) * t

    Directory.CreateDirectory(output) |> ignore

    for index, image in resampleAffineTrilinearSlices chunks ".tiff" lerp 16 inputGeometry outputGeometry affine 0.0f do
        image.toFile(Path.Combine(output, sprintf "image_%03d.tiff" index))
        image.decRefCount()

    0
