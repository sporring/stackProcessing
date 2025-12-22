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

let inline v3 x y z = { x = x; y = y; z = z }

let inline add (a:V3) (b:V3) = v3 (a.x+b.x) (a.y+b.y) (a.z+b.z)
let inline sub (a:V3) (b:V3) = v3 (a.x-b.x) (a.y-b.y) (a.z-b.z)
let inline scale s (a:V3) = v3 (s*a.x) (s*a.y) (s*a.z)

let inline mulMV (m:M3) (v:V3) =
    v3
      (m.m00*v.x + m.m01*v.y + m.m02*v.z)
      (m.m10*v.x + m.m11*v.y + m.m12*v.z)
      (m.m20*v.x + m.m21*v.y + m.m22*v.z)

let inline det3 (m:M3) =
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
