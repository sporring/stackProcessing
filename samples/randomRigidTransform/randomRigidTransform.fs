// Resample one stack with a seeded random rigid affine transform.
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

    let geometry : StackAffineResampler.ImageGeom =
        { W = 64; H = 64; D = 16
          Origin = v3 0.0 0.0 0.0
          Spacing = v3 1.0 1.0 1.0
          Direction = identity3 }

    let affine =
        randomRigidTransform 7 64u 64u 16u 0.0

    src
    |> read<float32> input ".tiff"
    >=> resampleAffine
            (fun a b t -> a + (b - a) * t)
            16
            geometry
            geometry
            affine
            0.0f
    >=> write "../tmp/randomRigidTransform" ".tiff"
    |> sink

    0
