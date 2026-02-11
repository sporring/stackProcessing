module Tests.complex

open Expecto
open Image   // open the namespace that defines Image<'T>
open ImageFunctions
open System.Numerics
open itk.simple

let private c (re: float) (im: float) = Complex(re, im)

let private makeComplexArray2D () =
    Array3D.init 2 2 2 (fun x y k ->
        let baseVal = float (1 + x + 2*y)
        if k = 0 then baseVal else baseVal + 9.0)

[<Tests>]
let complexSuite =
  testList "Complex image support" [

(*    testCase "ofArray3DComplex roundtrip" <| fun _ ->
      let arr = makeComplexArray2D ()
      let img = Image<Complex>.ofArray3DComplex arr
      Expect.equal img.[0,0] (c 1.0 10.0) "Complex value at (0,0) mismatch"
      Expect.equal img.[1,1] (c 4.0 13.0) "Complex value at (1,1) mismatch"
      let arr2 = img.toArray3DComplex()
      Expect.equal arr2 arr "Roundtrip array mismatch"
*)

    testCase "ofImagePairToComplex" <| fun _ ->
      let real = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let imag = Image<float>.ofArray2D (array2D [ [0.5; 1.5]; [2.5; 3.5] ])
      let img = Image<float>.ofImagePairToComplex real imag
      Expect.equal img.[0,1] (c 2.0 1.5) "Complex value at (0,1) mismatch"
      Expect.equal img.[1,0] (c 3.0 2.5) "Complex value at (1,0) mismatch"

    testCase "toComplex from vector image" <| fun _ ->
      let arr = Array3D.init 2 2 2 (
        fun x y k ->
            let baseVal = float (1 + x + 2*y)
            if k = 0 then baseVal else baseVal + 9.0)
      let vecImg = Image<float list>.ofArray3DVector arr
      let cImg = vecImg.toComplex()
      Expect.equal cImg.[1,0] (c 3.0 12.0) "toComplex conversion mismatch"

    testCase "FFTXY returns Complex image with vector-2 storage and expected values" <| fun _ ->
      let src = Image<float>.ofArray2D (array2D [ [1.0; 0.0]; [0.0; 0.0] ])
      let out = FFTXY src

      Expect.equal (out.GetNumberOfComponentsPerPixel()) 2u "Complex output should have 2 components per pixel"
      Expect.equal (out.toSimpleITK().GetPixelID()) PixelIDValueEnum.sitkVectorFloat64 "Complex output should use VectorFloat64 backing"

      Expect.floatClose Accuracy.high out.[0,0].Real 1.0 "FFT(impulse) real at (0,0)"
      Expect.floatClose Accuracy.high out.[1,0].Real 1.0 "FFT(impulse) real at (1,0)"
      Expect.floatClose Accuracy.high out.[0,1].Real 1.0 "FFT(impulse) real at (0,1)"
      Expect.floatClose Accuracy.high out.[1,1].Real 1.0 "FFT(impulse) real at (1,1)"
      Expect.floatClose Accuracy.high out.[0,0].Imaginary 0.0 "FFT(impulse) imag at (0,0)"
      Expect.floatClose Accuracy.high out.[1,0].Imaginary 0.0 "FFT(impulse) imag at (1,0)"
      Expect.floatClose Accuracy.high out.[0,1].Imaginary 0.0 "FFT(impulse) imag at (0,1)"
      Expect.floatClose Accuracy.high out.[1,1].Imaginary 0.0 "FFT(impulse) imag at (1,1)"

    testCase "directionalFFT returns Complex output with expected 1D FFT values" <| fun _ ->
      let src = Image<float>.ofArray2D (array2D [ [1.0; 0.0]; [0.0; 0.0] ])
      let out = directionalFFT 0u src

      Expect.equal (out.GetNumberOfComponentsPerPixel()) 2u "Complex output should have 2 components per pixel"
      Expect.equal (out.toSimpleITK().GetPixelID()) PixelIDValueEnum.sitkVectorFloat64 "Complex output should use VectorFloat64 backing"

      Expect.floatClose Accuracy.high out.[0,0].Real 1.0 "dirFFT real at (0,0)"
      Expect.floatClose Accuracy.high out.[1,0].Real 1.0 "dirFFT real at (1,0)"
      Expect.floatClose Accuracy.high out.[0,1].Real 0.0 "dirFFT real at (0,1)"
      Expect.floatClose Accuracy.high out.[1,1].Real 0.0 "dirFFT real at (1,1)"
      Expect.floatClose Accuracy.high out.[0,0].Imaginary 0.0 "dirFFT imag at (0,0)"
      Expect.floatClose Accuracy.high out.[1,0].Imaginary 0.0 "dirFFT imag at (1,0)"
      Expect.floatClose Accuracy.high out.[0,1].Imaginary 0.0 "dirFFT imag at (0,1)"
      Expect.floatClose Accuracy.high out.[1,1].Imaginary 0.0 "dirFFT imag at (1,1)"
  ]
