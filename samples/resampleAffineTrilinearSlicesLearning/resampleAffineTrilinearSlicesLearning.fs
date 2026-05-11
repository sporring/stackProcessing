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
    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/resampleAffineTrilinearSlices"
        | _ -> "../data/volume", "../tmp/resampleAffineTrilinearSlices"

    let inputGeometry = geometry 256 256 256
    let outputGeometry = geometry 256 256 256
    let affine =
        { A = identity3
          T = v3 4.0 -2.0 0.0
          C = v3 128.0 128.0 128.0 }

    let lerp (a: float32) (b: float32) (t: float32) = a + (b - a) * t

    resampleAffineTrilinearSlices input ".tiff" lerp 8 inputGeometry outputGeometry affine 0.0f
    |> writeTiffStack output

    0
