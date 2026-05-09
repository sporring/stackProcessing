module StackRansac

open System
open System.Collections.Generic

type RansacResult<'Model, 'Item> =
    { Model: 'Model
      Inliers: 'Item list
      Error: float }

type PointMatch2D =
    { FixedX: float
      FixedY: float
      MovingX: float
      MovingY: float }

let private solveLinearSystem (a: float[,]) (b: float[]) =
    let n = b.Length
    let m = Array2D.copy a
    let rhs = Array.copy b

    for i in 0 .. n - 1 do
        m[i, i] <- m[i, i] + 1.0e-10

    for k in 0 .. n - 1 do
        let mutable pivot = k
        let mutable pivotValue = abs m[k, k]
        for row in k + 1 .. n - 1 do
            let value = abs m[row, k]
            if value > pivotValue then
                pivot <- row
                pivotValue <- value

        if pivotValue < 1.0e-18 then
            invalidOp "Linear system is singular."

        if pivot <> k then
            for col in k .. n - 1 do
                let tmp = m[k, col]
                m[k, col] <- m[pivot, col]
                m[pivot, col] <- tmp
            let tmp = rhs[k]
            rhs[k] <- rhs[pivot]
            rhs[pivot] <- tmp

        let diag = m[k, k]
        for col in k .. n - 1 do
            m[k, col] <- m[k, col] / diag
        rhs[k] <- rhs[k] / diag

        for row in 0 .. n - 1 do
            if row <> k then
                let factor = m[row, k]
                for col in k .. n - 1 do
                    m[row, col] <- m[row, col] - factor * m[k, col]
                rhs[row] <- rhs[row] - factor * rhs[k]

    rhs

let fit sampleSize maxIterations minInlierRatio randomSeed tryFit errorForItem items =
    let items = items |> Seq.toArray
    if sampleSize <= 0 then invalidArg "sampleSize" "RANSAC sampleSize must be positive."
    if maxIterations <= 0 then invalidArg "maxIterations" "RANSAC maxIterations must be positive."
    if minInlierRatio <= 0.0 || minInlierRatio > 1.0 then invalidArg "minInlierRatio" "RANSAC minInlierRatio must be in (0, 1]."
    if items.Length < sampleSize then
        None
    else
        let rng = Random(randomSeed)
        let mutable best = None

        for _iteration in 1 .. maxIterations do
            let indices = HashSet<int>()
            while indices.Count < sampleSize do
                indices.Add(rng.Next(items.Length)) |> ignore

            let sample =
                indices
                |> Seq.map (fun index -> items[index])
                |> Seq.toList

            match tryFit sample with
            | Some model ->
                let inliers =
                    items
                    |> Array.choose (fun item ->
                        errorForItem model item
                        |> Option.map (fun error -> item, error))
                    |> Array.toList

                let inlierRatio = float inliers.Length / float items.Length
                if inlierRatio >= minInlierRatio && inliers.Length >= sampleSize then
                    let error = inliers |> List.averageBy snd
                    let candidate =
                        { Model = model
                          Inliers = inliers |> List.map fst
                          Error = error }

                    match best with
                    | Some current when current.Inliers.Length > candidate.Inliers.Length -> ()
                    | Some current when current.Inliers.Length = candidate.Inliers.Length && current.Error <= candidate.Error -> ()
                    | _ -> best <- Some candidate
            | None -> ()

        best

let affine2DFromMatches (matches: PointMatch2D list) =
    if matches.Length < 3 then
        None
    else
        let normal = Array2D.zeroCreate<float> 6 6
        let right = Array.zeroCreate<float> 6

        let addEquation (values: float[]) target =
            for row in 0 .. 5 do
                right[row] <- right[row] + values[row] * target
                for col in 0 .. 5 do
                    normal[row, col] <- normal[row, col] + values[row] * values[col]

        for matchItem in matches do
            addEquation [| matchItem.MovingX; matchItem.MovingY; 1.0; 0.0; 0.0; 0.0 |] matchItem.FixedX
            addEquation [| 0.0; 0.0; 0.0; matchItem.MovingX; matchItem.MovingY; 1.0 |] matchItem.FixedY

        try
            let solution = solveLinearSystem normal right
            Some
                [ [ solution[0]; solution[1]; solution[2] ]
                  [ solution[3]; solution[4]; solution[5] ]
                  [ 0.0; 0.0; 1.0 ] ]
        with _ ->
            None

let transformPoint2D (matrix: float list list) x y =
    matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
    matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]

let affine2DRansac maxIterations maxAlignmentError minInlierRatio randomSeed matches =
    let errorForItem matrix matchItem =
        let x, y = transformPoint2D matrix matchItem.MovingX matchItem.MovingY
        let dx = x - matchItem.FixedX
        let dy = y - matchItem.FixedY
        let error = sqrt (dx * dx + dy * dy)
        if error <= maxAlignmentError then Some error else None

    fit 3 maxIterations minInlierRatio randomSeed affine2DFromMatches errorForItem matches
    |> Option.bind (fun result ->
        match affine2DFromMatches result.Inliers with
        | Some refined -> Some refined
        | None -> Some result.Model)
