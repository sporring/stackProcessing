// Resample one small chunked volume with affine trilinear interpolation.
open StackProcessing
open TinyLinAlg

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input =
        match args with
        | [| input |] -> input
        | _ -> "../data/volume"

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

    src
    |> readRange<float32> 0u 1 15u input ".tiff"
    >=> resampleAffine
            (fun a b t -> a + (b - a) * t)
            16
            inputGeometry
            outputGeometry
            affine
            0.0f
    >=> write "../tmp/resampleAffineTrilinearSlices" ".tiff"
    |> sink

    0
