// Detect 3D DoG keypoints and estimate an affine transform between two point sets.
//
// This compact version uses the same keypoints as fixed and moving points, so the
// written matrix should be close to identity. Replace one branch with another
// stack or point-set source to turn it into a real registration example.
open SlimPipeline
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/affineKeypointRegistration"
        | _ -> "../data/volume", "../tmp/affineKeypointRegistration"

    let keypoints =
        src
        |> read<float32> input ".tiff"
        >=> dogKeypoints<float32> 0.8 1.25 4u 0.0005 4u

    zip keypoints keypoints
    >=> affineRegistrationMatrices defaultAffineRegistrationOptions
    >=> writeMatrix output ".csv"
    |> sink

    0
