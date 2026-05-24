module TinyLinAlg
// -------------------------
// Small 3D linear algebra
// -------------------------
[<Struct>]
type V3 = { x: float; y: float; z: float }

[<Struct>]
type M3 =
    { m00: float; m01: float; m02: float
      m10: float; m11: float; m12: float
      m20: float; m21: float; m22: float }

let v3 x y z = { x = x; y = y; z = z }

let add (a:V3) (b:V3) = v3 (a.x+b.x) (a.y+b.y) (a.z+b.z)
let sub (a:V3) (b:V3) = v3 (a.x-b.x) (a.y-b.y) (a.z-b.z)
let scale s (a:V3) = v3 (s*a.x) (s*a.y) (s*a.z)
let dot (a: V3) (b: V3) = a.x*b.x + a.y*b.y + a.z*b.z
let cross (a: V3) (b: V3) =
    v3
        (a.y * b.z - a.z * b.y)
        (a.z * b.x - a.x * b.z)
        (a.x * b.y - a.y * b.x)
let norm (a: V3) = sqrt (dot a a)

let normalize (a: V3) =
    let n = norm a
    if n < 1e-18 then v3 0.0 0.0 0.0 else scale (1.0 / n) a

let mulMV (m:M3) (v:V3) =
    v3
      (m.m00*v.x + m.m01*v.y + m.m02*v.z)
      (m.m10*v.x + m.m11*v.y + m.m12*v.z)
      (m.m20*v.x + m.m21*v.y + m.m22*v.z)

let det3 (m:M3) =
    m.m00*(m.m11*m.m22 - m.m12*m.m21)
  - m.m01*(m.m10*m.m22 - m.m12*m.m20)
  + m.m02*(m.m10*m.m21 - m.m11*m.m20)

let inv3 (m:M3) =
    let d = det3 m
    if abs d < 1e-18 then failwith "Singular 3x3 matrix"
    let invDet = 1.0 / d
    { m00 =  (m.m11*m.m22 - m.m12*m.m21) * invDet
      m01 = -(m.m01*m.m22 - m.m02*m.m21) * invDet
      m02 =  (m.m01*m.m12 - m.m02*m.m11) * invDet
      m10 = -(m.m10*m.m22 - m.m12*m.m20) * invDet
      m11 =  (m.m00*m.m22 - m.m02*m.m20) * invDet
      m12 = -(m.m00*m.m12 - m.m02*m.m10) * invDet
      m20 =  (m.m10*m.m21 - m.m11*m.m20) * invDet
      m21 = -(m.m00*m.m21 - m.m01*m.m20) * invDet
      m22 =  (m.m00*m.m11 - m.m01*m.m10) * invDet }

let symmetricEigen (m: M3) : (float * V3) list =
    let a = Array2D.zeroCreate<float> 3 3
    let v = Array2D.zeroCreate<float> 3 3

    a[0,0] <- m.m00; a[0,1] <- m.m01; a[0,2] <- m.m02
    a[1,0] <- m.m01; a[1,1] <- m.m11; a[1,2] <- m.m12
    a[2,0] <- m.m02; a[2,1] <- m.m12; a[2,2] <- m.m22

    for i in 0 .. 2 do
        for j in 0 .. 2 do
            v[i,j] <- if i = j then 1.0 else 0.0

    let rotate p q =
        if abs a[p,q] > 1e-14 then
            let tau = (a[q,q] - a[p,p]) / (2.0 * a[p,q])
            let t =
                let sign = if tau >= 0.0 then 1.0 else -1.0
                sign / (abs tau + sqrt (1.0 + tau * tau))
            let c = 1.0 / sqrt (1.0 + t * t)
            let s = t * c
            let app = a[p,p]
            let aqq = a[q,q]
            let apq = a[p,q]

            a[p,p] <- app - t * apq
            a[q,q] <- aqq + t * apq
            a[p,q] <- 0.0
            a[q,p] <- 0.0

            for r in 0 .. 2 do
                if r <> p && r <> q then
                    let arp = a[r,p]
                    let arq = a[r,q]
                    a[r,p] <- c * arp - s * arq
                    a[p,r] <- a[r,p]
                    a[r,q] <- s * arp + c * arq
                    a[q,r] <- a[r,q]

            for r in 0 .. 2 do
                let vrp = v[r,p]
                let vrq = v[r,q]
                v[r,p] <- c * vrp - s * vrq
                v[r,q] <- s * vrp + c * vrq

    for _ in 1 .. 32 do
        rotate 0 1
        rotate 0 2
        rotate 1 2

    [ for i in 0 .. 2 ->
        let vector = normalize (v3 v[0,i] v[1,i] v[2,i])
        a[i,i], vector ]
    |> List.sortByDescending fst

let symmetricEigenN (matrix: float[,]) : (float * float list) list =
    let n = matrix.GetLength(0)
    if n <> matrix.GetLength(1) then invalidArg "matrix" "Symmetric eigen decomposition expects a square matrix."
    if n < 2 then invalidArg "matrix" "Symmetric eigen decomposition expects at least two components."

    let a = Array2D.copy matrix
    let v = Array2D.zeroCreate<float> n n
    for i in 0 .. n - 1 do
        v[i, i] <- 1.0

    let rotate p q =
        if System.Math.Abs a[p, q] > 1e-14 then
            let tau = (a[q, q] - a[p, p]) / (2.0 * a[p, q])
            let sign = if tau >= 0.0 then 1.0 else -1.0
            let t = sign / (System.Math.Abs tau + System.Math.Sqrt (1.0 + tau * tau))
            let c = 1.0 / System.Math.Sqrt (1.0 + t * t)
            let s = t * c
            let app = a[p, p]
            let aqq = a[q, q]
            let apq = a[p, q]

            a[p, p] <- app - t * apq
            a[q, q] <- aqq + t * apq
            a[p, q] <- 0.0
            a[q, p] <- 0.0

            for r in 0 .. n - 1 do
                if r <> p && r <> q then
                    let arp = a[r, p]
                    let arq = a[r, q]
                    a[r, p] <- c * arp - s * arq
                    a[p, r] <- a[r, p]
                    a[r, q] <- s * arp + c * arq
                    a[q, r] <- a[r, q]

            for r in 0 .. n - 1 do
                let vrp = v[r, p]
                let vrq = v[r, q]
                v[r, p] <- c * vrp - s * vrq
                v[r, q] <- s * vrp + c * vrq

    for _ in 1 .. System.Math.Max(32, 8 * n * n) do
        for p in 0 .. n - 2 do
            for q in p + 1 .. n - 1 do
                rotate p q

    [ for i in 0 .. n - 1 ->
        let vector = [ for r in 0 .. n - 1 -> v[r, i] ]
        let norm = vector |> List.sumBy (fun value -> value * value) |> System.Math.Sqrt
        let vector = if norm < 1e-18 then vector else vector |> List.map (fun value -> value / norm)
        a[i, i], vector ]
    |> List.sortByDescending fst

type PcaAccumulator =
    { Count: uint64
      Components: int
      Sums: float[]
      Products: float[,] }

let zeroPcaAccumulator components : PcaAccumulator =
    if components < 2 then invalidArg "components" "PCA needs at least two vector components."
    { Count = 0UL
      Components = components
      Sums = Array.zeroCreate components
      Products = Array2D.zeroCreate components components }

let addPcaVector (state: PcaAccumulator) (values: float list) : PcaAccumulator =
    let values = values |> List.toArray
    if values.Length <> state.Components then
        invalidArg "values" $"PCA: expected {state.Components}-component vectors, got {values.Length}."

    let sums = Array.copy state.Sums
    let products = Array2D.copy state.Products
    values |> Array.iteri (fun i value -> sums[i] <- sums[i] + value)
    for i in 0 .. state.Components - 1 do
        for j in i .. state.Components - 1 do
            let product = values[i] * values[j]
            products[i, j] <- products[i, j] + product
            if i <> j then products[j, i] <- products[j, i] + product

    { state with
        Count = state.Count + 1UL
        Sums = sums
        Products = products }

let pcaEigenSystem (state: PcaAccumulator) : (float * float list) list =
    if state.Count = 0UL then
        invalidOp "PCA cannot reduce an empty vector sequence."

    let n = float state.Count
    let means = state.Sums |> Array.map (fun sum -> sum / n)
    let covariance = Array2D.zeroCreate<float> state.Components state.Components
    for i in 0 .. state.Components - 1 do
        for j in 0 .. state.Components - 1 do
            covariance[i, j] <- state.Products[i, j] / n - means[i] * means[j]

    symmetricEigenN covariance

// -------------------------
// Affine transform (output -> input)
// SimpleITK's AffineTransform uses center: p' = A*(p - c) + t + c
// -------------------------
type Affine =
    { A: M3
      T: V3
      C: V3 } // center

let affinePoint (a:Affine) (p:V3) : V3 =
    add (add (mulMV a.A (sub p a.C)) a.T) a.C

let identity3 =
    { m00 = 1.0; m01 = 0.0; m02 = 0.0
      m10 = 0.0; m11 = 1.0; m12 = 0.0
      m20 = 0.0; m21 = 0.0; m22 = 1.0 }

let matrixFromColumnAxes (xAxis: V3) (yAxis: V3) (zAxis: V3) =
    { m00 = xAxis.x; m01 = yAxis.x; m02 = zAxis.x
      m10 = xAxis.y; m11 = yAxis.y; m12 = zAxis.y
      m20 = xAxis.z; m21 = yAxis.z; m22 = zAxis.z }

let imageCenter (width: uint) (height: uint) (depth: uint) =
    v3
        ((float width - 1.0) / 2.0)
        ((float height - 1.0) / 2.0)
        ((float depth - 1.0) / 2.0)

let randomUnitVectorOnSphere (rng: System.Random) =
    let z = 2.0 * rng.NextDouble() - 1.0
    let theta = 2.0 * System.Math.PI * rng.NextDouble()
    let r = sqrt (max 0.0 (1.0 - z * z))
    v3 (r * cos theta) (r * sin theta) z

let randomRigidTransformAround (seed: int) (center: V3) (maxTranslation: float) =
    if maxTranslation < 0.0 then invalidArg "maxTranslation" "Translation radius must be non-negative."

    let rng = System.Random(seed)
    let normal = randomUnitVectorOnSphere rng
    let originalX = v3 1.0 0.0 0.0
    let originalY = v3 0.0 1.0 0.0
    let reference = if abs (dot normal originalX) < 0.95 then originalX else originalY
    let firstAxis = sub reference (scale (dot reference normal) normal) |> normalize
    let secondAxis = cross normal firstAxis |> normalize
    let rotation = matrixFromColumnAxes firstAxis secondAxis normal

    let translation =
        if maxTranslation = 0.0 then
            v3 0.0 0.0 0.0
        else
            v3
                ((2.0 * rng.NextDouble() - 1.0) * maxTranslation)
                ((2.0 * rng.NextDouble() - 1.0) * maxTranslation)
                ((2.0 * rng.NextDouble() - 1.0) * maxTranslation)

    { A = rotation
      T = translation
      C = center }

let randomRigidTransform (seed: int) (width: uint) (height: uint) (depth: uint) (maxTranslation: float) =
    randomRigidTransformAround seed (imageCenter width height depth) maxTranslation

module Dense =
    let private checkMatrixVectorShape (a: float[,]) (b: float[]) =
        if a.GetLength(0) <> b.Length then
            invalidArg "b" $"Matrix row count {a.GetLength(0)} must match vector length {b.Length}."

    let solveLinearSystem (a: float[,]) (b: float[]) =
        let n = a.GetLength(0)
        if a.GetLength(1) <> n then
            invalidArg "a" "Linear solve expects a square matrix."
        checkMatrixVectorShape a b

        let m = Array2D.zeroCreate<float> n (n + 1)
        for row in 0 .. n - 1 do
            for col in 0 .. n - 1 do
                m[row, col] <- a[row, col]
            m[row, n] <- b[row]

        for col in 0 .. n - 1 do
            let mutable pivotRow = col
            let mutable pivotValue = abs m[col, col]

            for row in col + 1 .. n - 1 do
                let candidate = abs m[row, col]
                if candidate > pivotValue then
                    pivotRow <- row
                    pivotValue <- candidate

            if pivotValue < 1e-12 then
                failwith "Singular matrix"

            if pivotRow <> col then
                for k in col .. n do
                    let tmp = m[col, k]
                    m[col, k] <- m[pivotRow, k]
                    m[pivotRow, k] <- tmp

            let pivot = m[col, col]
            for k in col .. n do
                m[col, k] <- m[col, k] / pivot

            for row in 0 .. n - 1 do
                if row <> col then
                    let factor = m[row, col]
                    if factor <> 0.0 then
                        for k in col .. n do
                            m[row, k] <- m[row, k] - factor * m[col, k]

        Array.init n (fun row -> m[row, n])

    let leastSquares ridge (a: float[,]) (y: float[]) =
        checkMatrixVectorShape a y
        if ridge < 0.0 then
            invalidArg "ridge" "Ridge regularization must be non-negative."

        let rows = a.GetLength(0)
        let cols = a.GetLength(1)
        let ata = Array2D.zeroCreate<float> cols cols
        let aty = Array.zeroCreate<float> cols

        for row in 0 .. rows - 1 do
            for col in 0 .. cols - 1 do
                let value = a[row, col]
                aty[col] <- aty[col] + value * y[row]

                for other in 0 .. cols - 1 do
                    ata[col, other] <- ata[col, other] + value * a[row, other]

        if ridge > 0.0 then
            for col in 0 .. cols - 1 do
                ata[col, col] <- ata[col, col] + ridge

        solveLinearSystem ata aty

    let nonNegativeLeastSquares ridge maxIterations tolerance (a: float[,]) (y: float[]) =
        checkMatrixVectorShape a y
        if ridge < 0.0 then
            invalidArg "ridge" "Ridge regularization must be non-negative."
        if maxIterations <= 0 then
            invalidArg "maxIterations" "Iteration count must be positive."
        if tolerance < 0.0 then
            invalidArg "tolerance" "Tolerance must be non-negative."

        let rows = a.GetLength(0)
        let cols = a.GetLength(1)
        let x = Array.zeroCreate<float> cols
        let residual = Array.copy y
        let mutable largestColumnNormSquared = 0.0

        let columnNormSquared col =
            let mutable total = ridge
            for row in 0 .. rows - 1 do
                let value = a[row, col]
                total <- total + value * value
            total

        let columnNorms = Array.init cols columnNormSquared

        for norm in columnNorms do
            if norm > largestColumnNormSquared then
                largestColumnNormSquared <- norm

        let toleranceScaled = tolerance * max 1.0 largestColumnNormSquared
        let mutable iteration = 0
        let mutable keepGoing = true

        while keepGoing && iteration < maxIterations do
            keepGoing <- false
            iteration <- iteration + 1

            for col in 0 .. cols - 1 do
                let oldValue = x[col]
                let mutable gradientNumerator = 0.0

                for row in 0 .. rows - 1 do
                    gradientNumerator <- gradientNumerator + a[row, col] * residual[row]

                gradientNumerator <- gradientNumerator - ridge * oldValue

                if columnNorms[col] > 0.0 then
                    let newValue = max 0.0 (oldValue + gradientNumerator / columnNorms[col])
                    let delta = newValue - oldValue

                    if abs delta > toleranceScaled then
                        keepGoing <- true

                    if delta <> 0.0 then
                        x[col] <- newValue

                        for row in 0 .. rows - 1 do
                            residual[row] <- residual[row] - a[row, col] * delta

        x

    let predict (a: float[,]) (coefficients: float[]) =
        if a.GetLength(1) <> coefficients.Length then
            invalidArg "coefficients" $"Matrix column count {a.GetLength(1)} must match coefficient count {coefficients.Length}."

        Array.init (a.GetLength(0)) (fun row ->
            let mutable total = 0.0
            for col in 0 .. coefficients.Length - 1 do
                total <- total + a[row, col] * coefficients[col]
            total)
