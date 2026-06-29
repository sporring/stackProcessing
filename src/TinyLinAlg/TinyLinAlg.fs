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

[<Struct>]
type SymmetricEigen3 =
    { Value0: float
      Vector0: V3
      Value1: float
      Vector1: V3
      Value2: float
      Vector2: V3 }

let v3 x y z = { x = x; y = y; z = z }
let v3Add (a:V3) (b:V3) = v3 (a.x+b.x) (a.y+b.y) (a.z+b.z)
let v3Sub(a:V3) (b:V3) = v3 (a.x-b.x) (a.y-b.y) (a.z-b.z)
let v3Scale (s: float) (a:V3) = v3 (s*a.x) (s*a.y) (s*a.z)
let v3Dot (a: V3) (b: V3) = a.x*b.x + a.y*b.y + a.z*b.z
let v3Cross (a: V3) (b: V3) =
    v3
        (a.y * b.z - a.z * b.y)
        (a.z * b.x - a.x * b.z)
        (a.x * b.y - a.y * b.x)
let v3Norm2 (a: V3) = 
    sqrt (v3Dot a a)
let v3NormMax (a: V3) = 
    max (abs a.x) (max (abs a.y) (abs a.z))
let v3Normalize (a: V3) =
    let n = v3Norm2 a
    if n < 1e-18 then v3 0.0 0.0 0.0 else v3Scale (1.0 / n) a

let m3 m00 m01 m02 m10 m11 m12 m20 m21 m22 =
    { m00 = m00; m01 = m01; m02 = m02
      m10 = m10; m11 = m11; m12 = m12
      m20 = m20; m21 = m21; m22 = m22 }
let m3Add (a:M3) (b:M3) =
    m3
        (a.m00+b.m00) (a.m01+b.m01) (a.m02+b.m02)
        (a.m10+b.m10) (a.m11+b.m11) (a.m12+b.m12)
        (a.m20+b.m20) (a.m21+b.m21) (a.m22+b.m22)
let m3Sub (a:M3) (b:M3) =
    m3
        (a.m00-b.m00) (a.m01-b.m01) (a.m02-b.m02)
        (a.m10-b.m10) (a.m11-b.m11) (a.m12-b.m12)
        (a.m20-b.m20) (a.m21-b.m21) (a.m22-b.m22)
let m3Scale (a:M3) (s:float) =
    m3
        (s*a.m00) (s*a.m01) (s*a.m02)
        (s*a.m10) (s*a.m11) (s*a.m12)
        (s*a.m20) (s*a.m21) (s*a.m22)

let m3Norm2 (m: M3) =
    let sumSquares =
        m.m00*m.m00 + m.m01*m.m01 + m.m02*m.m02 +
        m.m10*m.m10 + m.m11*m.m11 + m.m12*m.m12 +
        m.m20*m.m20 + m.m21*m.m21 + m.m22*m.m22
    sqrt sumSquares
let m3NormMax (m: M3) =
    let maxRow0 = max (abs m.m00) (max (abs m.m01) (abs m.m02))
    let maxRow1 = max (abs m.m10) (max (abs m.m11) (abs m.m12))
    let maxRow2 = max (abs m.m20) (max (abs m.m21) (abs m.m22))
    max maxRow0 (max maxRow1 maxRow2)

let m3v3Mul (m:M3) (v:V3) =
    v3
      (m.m00*v.x + m.m01*v.y + m.m02*v.z)
      (m.m10*v.x + m.m11*v.y + m.m12*v.z)
      (m.m20*v.x + m.m21*v.y + m.m22*v.z)

let m3Det(m:M3) =
    m.m00*(m.m11*m.m22 - m.m12*m.m21)
  - m.m01*(m.m10*m.m22 - m.m12*m.m20)
  + m.m02*(m.m10*m.m21 - m.m11*m.m20)

let m3Inv(m:M3) =
    let d = m3Det m
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

let private symmetricEigenJacobi (m: M3) : (float * V3) list =
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
        let vector = v3Normalize (v3 v[0,i] v[1,i] v[2,i])
        a[i,i], vector ]
    |> List.sortByDescending fst

let private sortEigen3Struct e0 v0 e1 v1 e2 v2 =
    let mutable a0 = e0
    let mutable u0 = v0
    let mutable a1 = e1
    let mutable u1 = v1
    let mutable a2 = e2
    let mutable u2 = v2
    if a0 < a1 then
        let ta = a0
        let tu = u0
        a0 <- a1
        u0 <- u1
        a1 <- ta
        u1 <- tu
    if a1 < a2 then
        let ta = a1
        let tu = u1
        a1 <- a2
        u1 <- u2
        a2 <- ta
        u2 <- tu
    if a0 < a1 then
        let ta = a0
        let tu = u0
        a0 <- a1
        u0 <- u1
        a1 <- ta
        u1 <- tu
    { Value0 = a0
      Vector0 = v3Normalize u0
      Value1 = a1
      Vector1 = v3Normalize u1
      Value2 = a2
      Vector2 = v3Normalize u2 }

let private symmetricEigenJacobi3 (m: M3) : SymmetricEigen3 =
    let mutable a00 = m.m00
    let mutable a01 = m.m01
    let mutable a02 = m.m02
    let mutable a11 = m.m11
    let mutable a12 = m.m12
    let mutable a22 = m.m22

    let mutable v00 = 1.0
    let mutable v01 = 0.0
    let mutable v02 = 0.0
    let mutable v10 = 0.0
    let mutable v11 = 1.0
    let mutable v12 = 0.0
    let mutable v20 = 0.0
    let mutable v21 = 0.0
    let mutable v22 = 1.0

    let inline coeff app aqq apq =
        let tau = (aqq - app) / (2.0 * apq)
        let sign = if tau >= 0.0 then 1.0 else -1.0
        let t = sign / (abs tau + sqrt (1.0 + tau * tau))
        let c = 1.0 / sqrt (1.0 + t * t)
        let s = t * c
        struct (t, c, s)

    let inline rotate01 () =
        if abs a01 > 1e-14 then
            let app = a00
            let aqq = a11
            let apq = a01
            let struct (t, c, s) = coeff app aqq apq
            a00 <- app - t * apq
            a11 <- aqq + t * apq
            a01 <- 0.0
            let a02Old = a02
            let a12Old = a12
            a02 <- c * a02Old - s * a12Old
            a12 <- s * a02Old + c * a12Old

            let r00 = v00
            let r01 = v01
            v00 <- c * r00 - s * r01
            v01 <- s * r00 + c * r01
            let r10 = v10
            let r11 = v11
            v10 <- c * r10 - s * r11
            v11 <- s * r10 + c * r11
            let r20 = v20
            let r21 = v21
            v20 <- c * r20 - s * r21
            v21 <- s * r20 + c * r21

    let inline rotate02 () =
        if abs a02 > 1e-14 then
            let app = a00
            let aqq = a22
            let apq = a02
            let struct (t, c, s) = coeff app aqq apq
            a00 <- app - t * apq
            a22 <- aqq + t * apq
            a02 <- 0.0
            let a01Old = a01
            let a12Old = a12
            a01 <- c * a01Old - s * a12Old
            a12 <- s * a01Old + c * a12Old

            let r00 = v00
            let r02 = v02
            v00 <- c * r00 - s * r02
            v02 <- s * r00 + c * r02
            let r10 = v10
            let r12 = v12
            v10 <- c * r10 - s * r12
            v12 <- s * r10 + c * r12
            let r20 = v20
            let r22 = v22
            v20 <- c * r20 - s * r22
            v22 <- s * r20 + c * r22

    let inline rotate12 () =
        if abs a12 > 1e-14 then
            let app = a11
            let aqq = a22
            let apq = a12
            let struct (t, c, s) = coeff app aqq apq
            a11 <- app - t * apq
            a22 <- aqq + t * apq
            a12 <- 0.0
            let a01Old = a01
            let a02Old = a02
            a01 <- c * a01Old - s * a02Old
            a02 <- s * a01Old + c * a02Old

            let r01 = v01
            let r02 = v02
            v01 <- c * r01 - s * r02
            v02 <- s * r01 + c * r02
            let r11 = v11
            let r12 = v12
            v11 <- c * r11 - s * r12
            v12 <- s * r11 + c * r12
            let r21 = v21
            let r22 = v22
            v21 <- c * r21 - s * r22
            v22 <- s * r21 + c * r22

    for _ in 1 .. 32 do
        rotate01()
        rotate02()
        rotate12()

    sortEigen3Struct
        a00 (v3 v00 v10 v20)
        a11 (v3 v01 v11 v21)
        a22 (v3 v02 v12 v22)

let private sortEigenvaluesDescending a b c =
    let mutable x = a
    let mutable y = b
    let mutable z = c
    if x < y then
        let t = x
        x <- y
        y <- t
    if y < z then
        let t = y
        y <- z
        z <- t
    if x < y then
        let t = x
        x <- y
        y <- t
    struct (x, y, z)

let private symmetricEigenvalues3 (m: M3) =
    let p1 = m.m01 * m.m01 + m.m02 * m.m02 + m.m12 * m.m12
    if p1 = 0.0 then
        sortEigenvaluesDescending m.m00 m.m11 m.m22
    else
        let q = (m.m00 + m.m11 + m.m22) / 3.0
        let a00 = m.m00 - q
        let a11 = m.m11 - q
        let a22 = m.m22 - q
        let p2 = a00 * a00 + a11 * a11 + a22 * a22 + 2.0 * p1
        let p = sqrt (p2 / 6.0)
        let b00 = a00 / p
        let b01 = m.m01 / p
        let b02 = m.m02 / p
        let b11 = a11 / p
        let b12 = m.m12 / p
        let b22 = a22 / p
        let detB =
            b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02)
        let r = detB / 2.0
        let phi =
            if r <= -1.0 then System.Math.PI / 3.0
            elif r >= 1.0 then 0.0
            else System.Math.Acos(r) / 3.0
        let eig0 = q + 2.0 * p * System.Math.Cos(phi)
        let eig2 = q + 2.0 * p * System.Math.Cos(phi + 2.0 * System.Math.PI / 3.0)
        let eig1 = 3.0 * q - eig0 - eig2
        sortEigenvaluesDescending eig0 eig1 eig2

let private tryEigenvectorByv3CrossProduct (m: M3) eigenvalue tolerance (result: byref<V3>) =
    let r0 = v3 (m.m00 - eigenvalue) m.m01 m.m02
    let r1 = v3 m.m01 (m.m11 - eigenvalue) m.m12
    let r2 = v3 m.m02 m.m12 (m.m22 - eigenvalue)
    let c01 = v3Cross r0 r1
    let c02 = v3Cross r0 r2
    let c12 = v3Cross r1 r2
    let n01 = v3Dot c01 c01
    let n02 = v3Dot c02 c02
    let n12 = v3Dot c12 c12

    let struct (candidate, normSquared) =
        if n01 >= n02 && n01 >= n12 then struct (c01, n01)
        elif n02 >= n12 then struct (c02, n02)
        else struct (c12, n12)

    if normSquared <= tolerance * tolerance then
        false
    else
        let vector = v3Scale (1.0 / sqrt normSquared) candidate
        let vector =
            let maxAbs = v3NormMax vector
            let sign =
                if abs vector.x = maxAbs then vector.x
                elif abs vector.y = maxAbs then vector.y
                else vector.z
            if sign < 0.0 then v3Scale -1.0 vector else vector
        result <- vector
        true

let private eigenResidualNorm (m: M3) eigenvalue vector =
    let av = m3v3Mul m vector
    let lv = v3Scale eigenvalue vector
    v3Norm2 (v3Sub av lv)

let private orthonormalComplement (axis: V3) =
    let reference =
        let ax = abs axis.x
        let ay = abs axis.y
        let az = abs axis.z
        if ax <= ay && ax <= az then v3 1.0 0.0 0.0
        elif ay <= az then v3 0.0 1.0 0.0
        else v3 0.0 0.0 1.0
    let u = v3Normalize (v3Cross axis reference)
    let v = v3Normalize (v3Cross axis u)
    struct (u, v)

let private eigenListToStruct (eigen: (float * V3) list) =
    let e0, v0 = eigen[0]
    let e1, v1 = eigen[1]
    let e2, v2 = eigen[2]
    { Value0 = e0
      Vector0 = v0
      Value1 = e1
      Vector1 = v1
      Value2 = e2
      Vector2 = v2 }

let symmetricEigen3 (m: M3) : SymmetricEigen3 =
    let scale = max 1.0 (m3NormMax m)
    let struct (l0, l1, l2) = symmetricEigenvalues3 m
    let gapTolerance = 1e-10 * scale

    if abs (l0 - l1) <= gapTolerance && abs (l1 - l2) <= gapTolerance then
        { Value0 = l0
          Vector0 = v3 1.0 0.0 0.0
          Value1 = l1
          Vector1 = v3 0.0 1.0 0.0
          Value2 = l2
          Vector2 = v3 0.0 0.0 1.0 }
    elif abs (l0 - l1) <= gapTolerance then
        let mutable v2 = v3 0.0 0.0 0.0
        if tryEigenvectorByv3CrossProduct m l2 (1e-12 * scale) &v2 then
            let struct (v0, v1) = orthonormalComplement v2
            { Value0 = l0
              Vector0 = v0
              Value1 = l1
              Vector1 = v1
              Value2 = l2
              Vector2 = v2 }
        else
            symmetricEigenJacobi3 m
    elif abs (l1 - l2) <= gapTolerance then
        let mutable v0 = v3 0.0 0.0 0.0
        if tryEigenvectorByv3CrossProduct m l0 (1e-12 * scale) &v0 then
            let struct (v1, v2) = orthonormalComplement v0
            { Value0 = l0
              Vector0 = v0
              Value1 = l1
              Vector1 = v1
              Value2 = l2
              Vector2 = v2 }
        else
            symmetricEigenJacobi3 m
    else
        let vectorTolerance = 1e-12 * scale
        let mutable v0 = v3 0.0 0.0 0.0
        let mutable v1 = v3 0.0 0.0 0.0
        let mutable v2 = v3 0.0 0.0 0.0
        if tryEigenvectorByv3CrossProduct m l0 vectorTolerance &v0
           && tryEigenvectorByv3CrossProduct m l1 vectorTolerance &v1
           && tryEigenvectorByv3CrossProduct m l2 vectorTolerance &v2 then
            let residualTolerance = 1e-7 * scale
            if eigenResidualNorm m l0 v0 <= residualTolerance
               && eigenResidualNorm m l1 v1 <= residualTolerance
               && eigenResidualNorm m l2 v2 <= residualTolerance then
                { Value0 = l0
                  Vector0 = v0
                  Value1 = l1
                  Vector1 = v1
                  Value2 = l2
                  Vector2 = v2 }
            else
                symmetricEigenJacobi3 m
        else
            symmetricEigenJacobi3 m

let symmetricEigen (m: M3) : (float * V3) list =
    let eigen = symmetricEigen3 m
    [ eigen.Value0, eigen.Vector0
      eigen.Value1, eigen.Vector1
      eigen.Value2, eigen.Vector2 ]

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
// Affine transform: p' = A*p + T
// -------------------------
type Affine =
    { A: M3
      T: V3 }

let affinePoint (a:Affine) (p:V3) : V3 =
    v3Add (m3v3Mul a.A p) a.T

let identity3 =
    { m00 = 1.0; m01 = 0.0; m02 = 0.0
      m10 = 0.0; m11 = 1.0; m12 = 0.0
      m20 = 0.0; m21 = 0.0; m22 = 1.0 }

let m3AffineNormMax (a:Affine) (b:Affine) = 
    max (m3NormMax (m3Sub a.A b.A)) (v3NormMax (v3Sub a.T b.T))

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
    let reference = if abs (v3Dot normal originalX) < 0.95 then originalX else originalY
    let firstAxis = v3Sub reference (v3Scale (v3Dot reference normal) normal) |> v3Normalize
    let secondAxis = v3Cross normal firstAxis |> v3Normalize
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
      T = v3Add translation (v3Sub center (m3v3Mul rotation center)) }

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
