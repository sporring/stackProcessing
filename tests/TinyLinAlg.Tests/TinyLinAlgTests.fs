module Tests.TinyLinAlgTests

open Expecto
open TinyLinAlg

let private tolerance = 1e-12

let private expectFloat actual expected message =
    Expect.floatClose Accuracy.high actual expected message

let private expectV3 actual expected message =
    expectFloat actual.x expected.x $"{message} x"
    expectFloat actual.y expected.y $"{message} y"
    expectFloat actual.z expected.z $"{message} z"

let private expectM3 actual expected message =
    expectFloat actual.m00 expected.m00 $"{message} m00"
    expectFloat actual.m01 expected.m01 $"{message} m01"
    expectFloat actual.m02 expected.m02 $"{message} m02"
    expectFloat actual.m10 expected.m10 $"{message} m10"
    expectFloat actual.m11 expected.m11 $"{message} m11"
    expectFloat actual.m12 expected.m12 $"{message} m12"
    expectFloat actual.m20 expected.m20 $"{message} m20"
    expectFloat actual.m21 expected.m21 $"{message} m21"
    expectFloat actual.m22 expected.m22 $"{message} m22"

let private identity =
    { m00 = 1.0; m01 = 0.0; m02 = 0.0
      m10 = 0.0; m11 = 1.0; m12 = 0.0
      m20 = 0.0; m21 = 0.0; m22 = 1.0 }

let private mulMM a b =
    { m00 = a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20
      m01 = a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21
      m02 = a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22
      m10 = a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20
      m11 = a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21
      m12 = a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22
      m20 = a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20
      m21 = a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21
      m22 = a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22 }

[<Tests>]
let tinyLinAlgSuite =
    testList "TinyLinAlg" [
        testCase "V3 arithmetic is componentwise" <| fun _ ->
            let a = v3 1.0 -2.0 3.5
            let b = v3 4.0 5.0 -6.0

            expectV3 (add a b) (v3 5.0 3.0 -2.5) "add"
            expectV3 (sub a b) (v3 -3.0 -7.0 9.5) "sub"
            expectV3 (scale 2.0 a) (v3 2.0 -4.0 7.0) "scale"
            expectV3 (scale -0.5 (add a b)) (v3 -2.5 -1.5 1.25) "scale after add"
            Expect.equal (v3 1.0 -2.0 3.5) a "v3 should construct the public V3 record shape."

        testCase "matrix-vector multiplication uses row-major M3 fields" <| fun _ ->
            let m =
                { m00 = 1.0; m01 = 2.0; m02 = 3.0
                  m10 = 4.0; m11 = 5.0; m12 = 6.0
                  m20 = 7.0; m21 = 8.0; m22 = 9.0 }

            expectV3 (mulMV m (v3 1.0 2.0 3.0)) (v3 14.0 32.0 50.0) "mulMV"
            expectV3 (mulMV identity (v3 -4.0 0.5 9.0)) (v3 -4.0 0.5 9.0) "identity mulMV"

        testCase "det3 covers common matrix families" <| fun _ ->
            let diagonal =
                { m00 = 2.0; m01 = 0.0; m02 = 0.0
                  m10 = 0.0; m11 = -3.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 4.0 }

            let triangular =
                { m00 = 5.0; m01 = 2.0; m02 = -1.0
                  m10 = 0.0; m11 = 3.0; m12 = 7.0
                  m20 = 0.0; m21 = 0.0; m22 = -2.0 }

            let swapXY =
                { m00 = 0.0; m01 = 1.0; m02 = 0.0
                  m10 = 1.0; m11 = 0.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 1.0 }

            expectFloat (det3 identity) 1.0 "identity determinant"
            expectFloat (det3 diagonal) -24.0 "diagonal determinant"
            expectFloat (det3 triangular) -30.0 "triangular determinant"
            expectFloat (det3 swapXY) -1.0 "orientation reversing determinant"

        testCase "det3 and inv3 agree with matrix identity" <| fun _ ->
            let m =
                { m00 = 4.0; m01 = 7.0; m02 = 2.0
                  m10 = 3.0; m11 = 6.0; m12 = 1.0
                  m20 = 2.0; m21 = 5.0; m22 = 3.0 }

            expectFloat (det3 m) 9.0 "determinant"
            expectM3 (mulMM m (inv3 m)) identity "m * inv(m)"
            expectM3 (mulMM (inv3 m) m) identity "inv(m) * m"

        testCase "inv3 handles identity diagonal and orientation reversing matrices" <| fun _ ->
            let diagonal =
                { m00 = 2.0; m01 = 0.0; m02 = 0.0
                  m10 = 0.0; m11 = -4.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 0.5 }

            let swapXY =
                { m00 = 0.0; m01 = 1.0; m02 = 0.0
                  m10 = 1.0; m11 = 0.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 1.0 }

            expectM3 (inv3 identity) identity "identity inverse"
            expectM3
                (inv3 diagonal)
                { m00 = 0.5; m01 = 0.0; m02 = 0.0
                  m10 = 0.0; m11 = -0.25; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 2.0 }
                "diagonal inverse"
            expectM3 (inv3 swapXY) swapXY "axis swap inverse"

        testCase "inv3 rejects singular matrices" <| fun _ ->
            let singular =
                { m00 = 1.0; m01 = 2.0; m02 = 3.0
                  m10 = 2.0; m11 = 4.0; m12 = 6.0
                  m20 = 7.0; m21 = 8.0; m22 = 9.0 }

            Expect.throws (fun () -> inv3 singular |> ignore) "Singular matrices should not be inverted."
            Expect.throws (fun () -> inv3 { identity with m22 = 1e-20 } |> ignore) "Nearly singular matrices should respect the singularity threshold."

        testCase "affinePoint follows SimpleITK center convention" <| fun _ ->
            let rotateZ90 =
                { m00 = 0.0; m01 = -1.0; m02 = 0.0
                  m10 = 1.0; m11 = 0.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 1.0 }

            let affine =
                { A = rotateZ90
                  T = v3 10.0 0.5 -2.0
                  C = v3 1.0 1.0 0.0 }

            expectV3 (affinePoint affine (v3 2.0 1.0 3.0)) (v3 11.0 2.5 1.0) "affinePoint"

        testCase "affinePoint composes identity linear part with translation and center" <| fun _ ->
            let affine =
                { A = identity
                  T = v3 -2.0 5.0 0.25
                  C = v3 100.0 -50.0 7.0 }

            expectV3 (affinePoint affine (v3 4.0 -3.0 2.0)) (v3 2.0 2.0 2.25) "identity affine translates independent of center"

        testCase "affinePoint handles scaling around a nonzero center" <| fun _ ->
            let scaleAroundCenter =
                { m00 = 2.0; m01 = 0.0; m02 = 0.0
                  m10 = 0.0; m11 = 3.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = -1.0 }

            let affine =
                { A = scaleAroundCenter
                  T = v3 0.5 -1.0 2.0
                  C = v3 1.0 2.0 -3.0 }

            expectV3 (affinePoint affine (v3 2.0 4.0 -1.0)) (v3 3.5 7.0 -3.0) "scaled affine"
    ]
