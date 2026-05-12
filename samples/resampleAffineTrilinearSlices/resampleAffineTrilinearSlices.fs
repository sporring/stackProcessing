// Run the chunked affine trilinear resampler directly and save TIFF slices.
open System.IO
open StackProcessing
open TinyLinAlg

let private identity3 =
    { m00 = 1.0; m01 = 0.0; m02 = 0.0
      m10 = 0.0; m11 = 1.0; m12 = 0.0
      m20 = 0.0; m21 = 0.0; m22 = 1.0 }

let private geometry width height depth : StackAffineResampler.ImageGeom =
    { W = width
      H = height
      D = depth
      Origin = v3 0.0 0.0 0.0
      Spacing = v3 1.0 1.0 1.0
      Direction = identity3 }

let private writeTiffStack output (slices: seq<int * StackAffineResampler.Image<float32>>) =
    Directory.CreateDirectory(output) |> ignore

    for index, image in slices do
        let fileName = Path.Combine(output, sprintf "image_%03d.tiff" index)
        image.toFile(fileName)

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/resampleAffineTrilinearSlices"

    let chunks = "../tmp/resampleAffineTrilinearSlicesChunks"
    deleteIfExists chunks

    src
    |> normalNoise<float32> 80u 64u 32u 128.0 25.0
    >=> writeChunks chunks ".tiff" 16u 16u 8u
    >=> ignoreSingles ()
    |> sink

    let inputGeometry = geometry 80 64 32
    let outputGeometry = geometry 56 56 28
    let affine =
        { A = identity3
          T = v3 0.0 0.0 0.0
          C = v3 32.0 32.0 16.0 }

    let lerp (a: float32) (b: float32) (t: float32) = a + (b - a) * t

    resampleAffineTrilinearSlices chunks ".tiff" lerp 16 inputGeometry outputGeometry affine 0.0f
    |> writeTiffStack output

    0
