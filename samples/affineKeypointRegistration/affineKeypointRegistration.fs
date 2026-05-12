// Estimate an affine transform between two small 3D point sets.
//
// The compact sample writes a tiny point-set CSV and feeds it through the public
// readPointSet box twice, so the estimated matrix should be close to identity.
open SlimPipeline
open System.IO
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/affineKeypointRegistration"

    let pointFile = output + "-points.csv"
    let directory = Path.GetDirectoryName(pointFile)
    if not (System.String.IsNullOrWhiteSpace directory) then
        Directory.CreateDirectory(directory) |> ignore

    File.WriteAllLines(
        pointFile,
        [| "x,y,z,scale,response"
           "4,4,2,1,1"
           "24,5,3,1,1"
           "6,25,10,1,1"
           "26,26,12,1,1" |])

    zip (readPointSet pointFile src) (readPointSet pointFile src)
    >=> affineRegistrationMatrices { defaultAffineRegistrationOptions with MaxIterations = 1 }
    >=> writeMatrix output ".csv"
    |> sink

    0
