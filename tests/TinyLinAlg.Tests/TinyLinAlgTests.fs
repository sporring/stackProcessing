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

            expectV3 (v3Add a b) (v3 5.0 3.0 -2.5) "v3Add"
            expectV3 (v3Sub a b) (v3 -3.0 -7.0 9.5) "v3Sub"
            expectV3 (v3Scale 2.0 a) (v3 2.0 -4.0 7.0) "v3Scale"
            expectV3 (v3Scale -0.5 (v3Add a b)) (v3 -2.5 -1.5 1.25) "v3Scale after v3Add"
            Expect.equal (v3 1.0 -2.0 3.5) a "v3 should construct the public V3 record shape."

        testCase "m3v3Mul uses row-major M3 fields" <| fun _ ->
            let m =
                { m00 = 1.0; m01 = 2.0; m02 = 3.0
                  m10 = 4.0; m11 = 5.0; m12 = 6.0
                  m20 = 7.0; m21 = 8.0; m22 = 9.0 }

            expectV3 (m3v3Mul m (v3 1.0 2.0 3.0)) (v3 14.0 32.0 50.0) "m3v3Mul"
            expectV3 (m3v3Mul identity (v3 -4.0 0.5 9.0)) (v3 -4.0 0.5 9.0) "identity m3v3Mul"

        testCase "m3Det covers common matrix families" <| fun _ ->
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

            expectFloat (m3Det identity) 1.0 "identity determinant"
            expectFloat (m3Det diagonal) -24.0 "diagonal determinant"
            expectFloat (m3Det triangular) -30.0 "triangular determinant"
            expectFloat (m3Det swapXY) -1.0 "orientation reversing determinant"

        testCase "m3Det and m3Inv agree with matrix identity" <| fun _ ->
            let m =
                { m00 = 4.0; m01 = 7.0; m02 = 2.0
                  m10 = 3.0; m11 = 6.0; m12 = 1.0
                  m20 = 2.0; m21 = 5.0; m22 = 3.0 }

            expectFloat (m3Det m) 9.0 "determinant"
            expectM3 (mulMM m (m3Inv m)) identity "m * inv(m)"
            expectM3 (mulMM (m3Inv m) m) identity "inv(m) * m"

        testCase "m3Inv handles identity diagonal and orientation reversing matrices" <| fun _ ->
            let diagonal =
                { m00 = 2.0; m01 = 0.0; m02 = 0.0
                  m10 = 0.0; m11 = -4.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 0.5 }

            let swapXY =
                { m00 = 0.0; m01 = 1.0; m02 = 0.0
                  m10 = 1.0; m11 = 0.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 1.0 }

            expectM3 (m3Inv identity) identity "identity inverse"
            expectM3
                (m3Inv diagonal)
                { m00 = 0.5; m01 = 0.0; m02 = 0.0
                  m10 = 0.0; m11 = -0.25; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 2.0 }
                "diagonal inverse"
            expectM3 (m3Inv swapXY) swapXY "axis swap inverse"

        testCase "m3Inv rejects singular matrices" <| fun _ ->
            let singular =
                { m00 = 1.0; m01 = 2.0; m02 = 3.0
                  m10 = 2.0; m11 = 4.0; m12 = 6.0
                  m20 = 7.0; m21 = 8.0; m22 = 9.0 }

            Expect.throws (fun () -> m3Inv singular |> ignore) "Singular matrices should not be inverted."
            Expect.throws (fun () -> m3Inv { identity with m22 = 1e-20 } |> ignore) "Nearly singular matrices should respect the singularity threshold."

        testCase "symmetricEigen sorts eigenpairs for symmetric matrices" <| fun _ ->
            let diagonal =
                { m00 = 9.0; m01 = 0.0; m02 = 0.0
                  m10 = 0.0; m11 = 4.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 1.0 }

            let eigen = symmetricEigen diagonal

            Expect.equal (eigen |> List.map fst) [ 9.0; 4.0; 1.0 ] "Eigenvalues should be sorted descending."
            expectV3 (eigen[0] |> snd) (v3 1.0 0.0 0.0) "largest eigenvector"
            expectV3 (eigen[1] |> snd) (v3 0.0 1.0 0.0) "middle eigenvector"
            expectV3 (eigen[2] |> snd) (v3 0.0 0.0 1.0) "smallest eigenvector"

            let rotated =
                { m00 = 5.0; m01 = 2.0; m02 = 0.0
                  m10 = 2.0; m11 = 5.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 1.0 }

            let rotatedEigen = symmetricEigen rotated
            Expect.floatClose Accuracy.high (rotatedEigen[0] |> fst) 7.0 "rotated largest eigenvalue"
            Expect.floatClose Accuracy.high (rotatedEigen[1] |> fst) 3.0 "rotated middle eigenvalue"
            Expect.floatClose Accuracy.high (abs ((rotatedEigen[0] |> snd).x)) (1.0 / sqrt 2.0) "rotated eigenvector x magnitude"
            Expect.floatClose Accuracy.high (abs ((rotatedEigen[0] |> snd).y)) (1.0 / sqrt 2.0) "rotated eigenvector y magnitude"

        testCase "symmetricEigenN sorts eigenpairs for dense symmetric matrices" <| fun _ ->
            let diagonal =
                array2D
                    [ [ 9.0; 0.0; 0.0; 0.0 ]
                      [ 0.0; 4.0; 0.0; 0.0 ]
                      [ 0.0; 0.0; 1.0; 0.0 ]
                      [ 0.0; 0.0; 0.0; 0.5 ] ]

            let eigen = symmetricEigenN diagonal

            Expect.equal (eigen |> List.map fst) [ 9.0; 4.0; 1.0; 0.5 ] "Eigenvalues should be sorted descending."
            Expect.equal (eigen[0] |> snd) [ 1.0; 0.0; 0.0; 0.0 ] "largest eigenvector"
            Expect.equal (eigen[3] |> snd) [ 0.0; 0.0; 0.0; 1.0 ] "smallest eigenvector"

            let rotated =
                array2D
                    [ [ 5.0; 2.0; 0.0 ]
                      [ 2.0; 5.0; 0.0 ]
                      [ 0.0; 0.0; 1.0 ] ]
            let rotatedEigen = symmetricEigenN rotated
            Expect.floatClose Accuracy.high (rotatedEigen[0] |> fst) 7.0 "rotated largest eigenvalue"
            Expect.floatClose Accuracy.high (rotatedEigen[1] |> fst) 3.0 "rotated middle eigenvalue"
            Expect.floatClose Accuracy.high (abs ((rotatedEigen[0] |> snd)[0])) (1.0 / sqrt 2.0) "rotated eigenvector x magnitude"
            Expect.floatClose Accuracy.high (abs ((rotatedEigen[0] |> snd)[1])) (1.0 / sqrt 2.0) "rotated eigenvector y magnitude"

        testCase "PCA accumulator builds covariance eigensystem" <| fun _ ->
            let state =
                [ [ -1.0; 0.0; 0.0 ]; [ 1.0; 0.0; 0.0 ] ]
                |> List.fold addPcaVector (zeroPcaAccumulator 3)

            let eigen = pcaEigenSystem state

            Expect.equal (eigen |> List.map fst) [ 1.0; 0.0; 0.0 ] "Variance should be isolated along x."
            Expect.floatClose Accuracy.high (abs ((eigen[0] |> snd)[0])) 1.0 "First PCA eigenvector should align with x."
            Expect.floatClose Accuracy.high (abs ((eigen[0] |> snd)[1])) 0.0 "First PCA eigenvector y"
            Expect.floatClose Accuracy.high (abs ((eigen[0] |> snd)[2])) 0.0 "First PCA eigenvector z"

        testCase "affinePoint applies linear part and translation directly" <| fun _ ->
            let rotateZ90 =
                { m00 = 0.0; m01 = -1.0; m02 = 0.0
                  m10 = 1.0; m11 = 0.0; m12 = 0.0
                  m20 = 0.0; m21 = 0.0; m22 = 1.0 }

            let affine =
                { A = rotateZ90
                  T = v3 10.0 0.5 -2.0 }

            expectV3 (affinePoint affine (v3 2.0 1.0 3.0)) (v3 9.0 2.5 1.0) "affinePoint"

        testCase "affinePoint composes identity linear part with translation" <| fun _ ->
            let affine =
                { A = identity
                  T = v3 -2.0 5.0 0.25 }

            expectV3 (affinePoint affine (v3 4.0 -3.0 2.0)) (v3 2.0 2.0 2.25) "identity affine translates"

        testCase "randomRigidTransformAround folds the center into translation" <| fun _ ->
            let center = v3 4.0 5.0 6.0
            let affine = randomRigidTransformAround 123 center 0.0

            expectV3 (affinePoint affine center) center "rotation around center should keep center fixed when translation radius is zero"

        testCase "dense least squares recovers coefficients for overdetermined systems" <| fun _ ->
            let a =
                array2D
                    [ [ 1.0; 0.0 ]
                      [ 1.0; 1.0 ]
                      [ 1.0; 2.0 ] ]

            let coefficients = TinyLinAlg.Dense.leastSquares 0.0 a [| 2.0; 5.0; 8.0 |]

            Expect.floatClose Accuracy.high coefficients[0] 2.0 "intercept"
            Expect.floatClose Accuracy.high coefficients[1] 3.0 "slope"

            let predicted = TinyLinAlg.Dense.predict a coefficients
            Expect.sequenceEqual predicted [| 2.0; 5.0; 8.0 |] "Predictions should match exactly."

        testCase "dense least squares supports ridge regularized underdetermined systems" <| fun _ ->
            let a =
                array2D
                    [ [ 1.0; 1.0; 0.0 ]
                      [ 1.0; 0.0; 1.0 ] ]

            let coefficients = TinyLinAlg.Dense.leastSquares 1e-6 a [| 3.0; 4.0 |]
            let predicted = TinyLinAlg.Dense.predict a coefficients

            Expect.floatClose Accuracy.medium predicted[0] 3.0 "first row"
            Expect.floatClose Accuracy.medium predicted[1] 4.0 "second row"

        testCase "dense non-negative least squares keeps coefficients above lower bound" <| fun _ ->
            let a =
                array2D
                    [ [ 1.0; 0.0 ]
                      [ 0.0; 1.0 ]
                      [ 1.0; 1.0 ] ]

            let coefficients = TinyLinAlg.Dense.nonNegativeLeastSquares 1e-8 10000 1e-12 a [| 1.0; -2.0; 1.0 |]

            Expect.isGreaterThanOrEqual coefficients[0] 0.0 "first coefficient should be non-negative"
            Expect.isGreaterThanOrEqual coefficients[1] 0.0 "second coefficient should be non-negative"
            Expect.floatClose Accuracy.medium coefficients[0] 1.0 "positive coefficient should remain close"
            Expect.floatClose Accuracy.medium coefficients[1] 0.0 "negative contribution should be clamped away"
    ]
