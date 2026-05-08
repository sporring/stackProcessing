module StackRegistration

open System
open SlimPipeline
open StackCore
open StackPoints
open TinyLinAlg

type AffineRegistrationOptions =
    { MaxIterations: int
      InitialLinearStep: float
      InitialTranslationStep: float
      MinStep: float
      StepShrink: float }

type AffineRegistrationResult =
    { Transform: Affine
      InverseTransform: Affine
      Distance: float
      Iterations: int
      Converged: bool }

let defaultAffineRegistrationOptions =
    { MaxIterations = 200
      InitialLinearStep = 0.05
      InitialTranslationStep = 1.0
      MinStep = 1.0e-4
      StepShrink = 0.5 }

let private pointToV3 point =
    v3 point.X point.Y point.Z

let private v3ToPoint (templatePoint: CoordinatePoint) value =
    { templatePoint with
        X = value.x
        Y = value.y
        Z = value.z }

let private distance (a: CoordinatePoint) (b: CoordinatePoint) =
    let dx = a.X - b.X
    let dy = a.Y - b.Y
    let dz = a.Z - b.Z
    sqrt (dx * dx + dy * dy + dz * dz)

let private centroid (points: CoordinatePoint[]) =
    if points.Length = 0 then
        invalidArg "points" "Affine registration needs at least one point in each set."

    let mutable sx = 0.0
    let mutable sy = 0.0
    let mutable sz = 0.0

    for point in points do
        sx <- sx + point.X
        sy <- sy + point.Y
        sz <- sz + point.Z

    let n = float points.Length
    v3 (sx / n) (sy / n) (sz / n)

let private hungarianDistance (cost: float[][]) =
    let n = cost.Length

    if n = 0 then
        0.0
    else
        let u = Array.zeroCreate<float> (n + 1)
        let v = Array.zeroCreate<float> (n + 1)
        let p = Array.zeroCreate<int> (n + 1)
        let way = Array.zeroCreate<int> (n + 1)

        for i in 1 .. n do
            p[0] <- i
            let minv = Array.create (n + 1) Double.PositiveInfinity
            let used = Array.create (n + 1) false
            let mutable j0 = 0
            let mutable donePath = false

            while not donePath do
                used[j0] <- true
                let i0 = p[j0]
                let mutable delta = Double.PositiveInfinity
                let mutable j1 = 0

                for j in 1 .. n do
                    if not used[j] then
                        let cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j] then
                            minv[j] <- cur
                            way[j] <- j0
                        if minv[j] < delta then
                            delta <- minv[j]
                            j1 <- j

                for j in 0 .. n do
                    if used[j] then
                        u[p[j]] <- u[p[j]] + delta
                        v[j] <- v[j] - delta
                    else
                        minv[j] <- minv[j] - delta

                j0 <- j1
                donePath <- p[j0] = 0

            let mutable continuePath = true
            while continuePath do
                let j1 = way[j0]
                p[j0] <- p[j1]
                j0 <- j1
                continuePath <- j0 <> 0

        -v[0] / float n

let private greedyTransportDistance (fixedPoints: CoordinatePoint[]) (movingPoints: CoordinatePoint[]) =
    if fixedPoints.Length = 0 && movingPoints.Length = 0 then
        0.0
    elif fixedPoints.Length = 0 || movingPoints.Length = 0 then
        Double.PositiveInfinity
    else
        let fixedMass = Array.create fixedPoints.Length (1.0 / float fixedPoints.Length)
        let movingMass = Array.create movingPoints.Length (1.0 / float movingPoints.Length)

        let pairs =
            [| for i in 0 .. fixedPoints.Length - 1 do
                   for j in 0 .. movingPoints.Length - 1 do
                       distance fixedPoints[i] movingPoints[j], i, j |]
            |> Array.sortBy (fun (d, i, j) -> d, i, j)

        let mutable cost = 0.0

        for d, i, j in pairs do
            let flow = min fixedMass[i] movingMass[j]

            if flow > 0.0 then
                cost <- cost + flow * d
                fixedMass[i] <- fixedMass[i] - flow
                movingMass[j] <- movingMass[j] - flow

        cost

let earthMoversDistance (fixedPoints: CoordinatePoint seq) (movingPoints: CoordinatePoint seq) =
    let fixedPoints = fixedPoints |> Seq.toArray
    let movingPoints = movingPoints |> Seq.toArray

    if fixedPoints.Length = movingPoints.Length then
        let cost =
            fixedPoints
            |> Array.map (fun fixedPoint ->
                movingPoints
                |> Array.map (fun movingPoint -> distance fixedPoint movingPoint))

        hungarianDistance cost
    else
        greedyTransportDistance fixedPoints movingPoints

let transformPointSet (transform: Affine) (points: PointSet) =
    { Points =
        points.Points
        |> List.map (fun point ->
            point
            |> pointToV3
            |> affinePoint transform
            |> v3ToPoint point) }

let inverseAffine (transform: Affine) =
    let inverseA = inv3 transform.A
    { A = inverseA
      T = scale -1.0 (mulMV inverseA transform.T)
      C = transform.C }

let private affineFromParameters (center: V3) (parameters: float[]) =
    { A =
        { m00 = parameters[0]; m01 = parameters[1]; m02 = parameters[2]
          m10 = parameters[3]; m11 = parameters[4]; m12 = parameters[5]
          m20 = parameters[6]; m21 = parameters[7]; m22 = parameters[8] }
      T = v3 parameters[9] parameters[10] parameters[11]
      C = center }

let affineToMatrix (transform: Affine) =
    let c = transform.C
    let ac = mulMV transform.A c
    let offset = add transform.T (sub c ac)
    let matrix = Array2D.zeroCreate<float> 4 4
    matrix[0, 0] <- transform.A.m00
    matrix[0, 1] <- transform.A.m01
    matrix[0, 2] <- transform.A.m02
    matrix[0, 3] <- offset.x
    matrix[1, 0] <- transform.A.m10
    matrix[1, 1] <- transform.A.m11
    matrix[1, 2] <- transform.A.m12
    matrix[1, 3] <- offset.y
    matrix[2, 0] <- transform.A.m20
    matrix[2, 1] <- transform.A.m21
    matrix[2, 2] <- transform.A.m22
    matrix[2, 3] <- offset.z
    matrix[3, 3] <- 1.0
    vectorizeMatrix matrix

let matrixToAffine (matrix: VectorizedMatrix) =
    let matrix = unvectorizeMatrix matrix
    if matrix.GetLength(0) <> 4 || matrix.GetLength(1) <> 4 then
        invalidArg "matrix" "matrixToAffine expects a 4x4 homogeneous affine matrix."
    if abs (matrix[3, 0]) > 1.0e-12 || abs (matrix[3, 1]) > 1.0e-12 || abs (matrix[3, 2]) > 1.0e-12 || abs (matrix[3, 3] - 1.0) > 1.0e-12 then
        invalidArg "matrix" "matrixToAffine expects the last row to be [0, 0, 0, 1]."

    { A =
        { m00 = matrix[0, 0]; m01 = matrix[0, 1]; m02 = matrix[0, 2]
          m10 = matrix[1, 0]; m11 = matrix[1, 1]; m12 = matrix[1, 2]
          m20 = matrix[2, 0]; m21 = matrix[2, 1]; m22 = matrix[2, 2] }
      T = v3 matrix[0, 3] matrix[1, 3] matrix[2, 3]
      C = v3 0.0 0.0 0.0 }

let private transformPoints transform (points: CoordinatePoint[]) =
    points
    |> Array.map (fun point ->
        point
        |> pointToV3
        |> affinePoint transform
        |> v3ToPoint point)

let affineRegistration
    (options: AffineRegistrationOptions)
    (fixedPoints: CoordinatePoint seq)
    (movingPoints: CoordinatePoint seq)
    =

    if options.MaxIterations <= 0 then invalidArg "options" "MaxIterations must be positive."
    if options.InitialLinearStep <= 0.0 then invalidArg "options" "InitialLinearStep must be positive."
    if options.InitialTranslationStep <= 0.0 then invalidArg "options" "InitialTranslationStep must be positive."
    if options.MinStep <= 0.0 then invalidArg "options" "MinStep must be positive."
    if options.StepShrink <= 0.0 || options.StepShrink >= 1.0 then invalidArg "options" "StepShrink must be between 0 and 1."

    let fixedPoints = fixedPoints |> Seq.toArray
    let movingPoints = movingPoints |> Seq.toArray
    let fixedCenter = centroid fixedPoints
    let movingCenter = centroid movingPoints
    let center = movingCenter

    let parameters =
        [| 1.0; 0.0; 0.0
           0.0; 1.0; 0.0
           0.0; 0.0; 1.0
           fixedCenter.x - movingCenter.x
           fixedCenter.y - movingCenter.y
           fixedCenter.z - movingCenter.z |]

    let steps =
        [| yield! Array.create 9 options.InitialLinearStep
           yield! Array.create 3 options.InitialTranslationStep |]

    let objective parameters =
        let transform = affineFromParameters center parameters
        let transformedMoving = transformPoints transform movingPoints
        earthMoversDistance fixedPoints transformedMoving

    let mutable bestParameters = Array.copy parameters
    let mutable bestDistance = objective bestParameters
    let mutable iteration = 0

    while iteration < options.MaxIterations && (steps |> Array.max) > options.MinStep do
        let mutable improved = false

        for parameterIndex in 0 .. bestParameters.Length - 1 do
            for direction in [ 1.0; -1.0 ] do
                let candidate = Array.copy bestParameters
                candidate[parameterIndex] <- candidate[parameterIndex] + direction * steps[parameterIndex]
                let candidateDistance = objective candidate

                if candidateDistance < bestDistance then
                    bestParameters <- candidate
                    bestDistance <- candidateDistance
                    improved <- true

        if not improved then
            for index in 0 .. steps.Length - 1 do
                steps[index] <- steps[index] * options.StepShrink

        iteration <- iteration + 1

    let transform = affineFromParameters center bestParameters

    { Transform = transform
      InverseTransform = inverseAffine transform
      Distance = bestDistance
      Iterations = iteration
      Converged = (steps |> Array.max) <= options.MinStep }

let affineRegistrationMatrices options : Stage<PointSet * PointSet, VectorizedMatrix> =
    Stage.map
        "affineRegistration"
        (fun _ (fixedPoints, movingPoints) ->
            let result = affineRegistration options fixedPoints.Points movingPoints.Points
            [ affineToMatrix result.Transform
              affineToMatrix result.InverseTransform ])
        (fun _ -> 0UL)
        (fun pointSets -> pointSets * 2UL)
    --> StackCore.flattenList ()
