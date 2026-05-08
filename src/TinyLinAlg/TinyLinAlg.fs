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
