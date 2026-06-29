// Fit an affine transform from streamed 3D point-set correspondences.
open System
open System.Globalization
open SlimPipeline
open StackProcessing
open TinyLinAlg

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/affineKeypointRegistration"

    let movingOutput = output + "-moving"
    let fixedOutput = output + "-fixed"
    let movingFile = movingOutput + ".csv"
    let fixedFile = fixedOutput + ".csv"

    // Choose some random 3d points across (virtual) slices
    let fixedPoints = [ 
        v3 0.0 0.0 0.0;
        v3 12.0 0.0 1.0;
        v3 1.0 10.0 2.0;
        v3 2.0 1.0 9.0;
        v3 14.0 11.0 4.0;
        v3 5.0 8.0 13.0]
    fixedPoints |> toPointSet |> writePointSetFile fixedFile

    // Pick some affine transformation and use TinyLinAlg to artificially produce a transformed data set
    let affineTransform =
        { A = m3 1.10 0.08 -0.04  -0.03 0.95 0.06  0.02 -0.05 1.07
          T = v3 3.0 -2.5 1.25 }
    let movingPoints =
        fixedPoints |> List.map (affinePoint affineTransform)
    movingPoints |> toPointSet |> writePointSetFile movingFile

    // Start 2 streaming of points and reduce to a fitted affine matrix.
    let fixedPlan = src |> readPointSet fixedFile
    let movingPlan = src |> readPointSet movingFile
    let fittedAffineTransform =
        zip fixedPlan movingPlan
        >=> affineRegistrationMatrix defaultAffineRegistrationOptions
        |> drain
        |> ofHomogeneousMatrix

    // The point-wise affine registration should be very similar to the original transform
    let error = m3AffineNormMax affineTransform fittedAffineTransform
    printfn $"Fixed points:  [{fixedPoints}]"
    printfn $"Moving points: [{movingPoints}]"
    printfn $"Known affine matrix:  [{affineTransform}]"
    printfn $"Fitted affine matrix: [{fittedAffineTransform}]"
    printfn $"Matrix max error: {error}"

    0
