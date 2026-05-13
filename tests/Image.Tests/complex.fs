module Tests.complex

open Expecto
open Image   // open the namespace that defines Image<'T>
open ImageFunctions
open System.Numerics
open itk.simple

let private c (re: float) (im: float) = Complex(re, im)

let private expectComplexClose (actual: Complex) (expected: Complex) message =
    Expect.floatClose Accuracy.high actual.Real expected.Real $"{message} real"
    Expect.floatClose Accuracy.high actual.Imaginary expected.Imaginary $"{message} imaginary"

let private makeComplexArray2D () =
    Array3D.init 2 2 2 (fun x y k ->
        let baseVal = float (1 + x + 2*y)
        if k = 0 then baseVal else baseVal + 9.0)

[<Tests>]
let complexSuite =
  testList "Complex image support" [

    testCase "Re Im toComplex and bulk 2D complex array accessors" <| fun _ ->
      let arr =
        array2D [
          [ c 1.0 -1.0; c 2.0 -2.0 ]
          [ c 3.0 -3.0; c 4.0 -4.0 ] ]

      let img = Image<Complex>.ofComplexArray2D arr
      let re = Re img
      let im = Im img
      let rebuilt = toComplex re im

      Expect.equal (re.GetSize()) [2u; 2u] "Re should preserve image size."
      Expect.equal (im.GetSize()) [2u; 2u] "Im should preserve image size."
      Expect.floatClose Accuracy.high re.[1,0] 3.0 "Re should expose real values."
      Expect.floatClose Accuracy.high im.[1,0] -3.0 "Im should expose imaginary values."
      Expect.equal (rebuilt.toSimpleITK().GetPixelID()) PixelIDValueEnum.sitkComplexFloat64 "toComplex should return a native complex image."
      Expect.equal (rebuilt.toComplexArray2D()) arr "toComplexArray2D should bulk round-trip complex values."
      Expect.throws (fun () -> img.toComplexArray3D() |> ignore) "toComplexArray3D should reject 2D images."

    testCase "bulk 3D complex array accessors" <| fun _ ->
      let arr = Array3D.init 2 2 2 (fun x y z -> c (float (x + 10*y + 100*z)) (-float (x + 10*y + 100*z)))
      let img = Image<Complex>.ofComplexArray3D arr
      let back = img.toComplexArray3D()

      Expect.equal (img.GetSize()) [2u; 2u; 2u] "ofComplexArray3D should preserve the three spatial dimensions."
      Expect.equal back arr "toComplexArray3D should bulk round-trip complex volumes."
      Expect.throws (fun () -> img.toComplexArray2D() |> ignore) "toComplexArray2D should reject 3D images."

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

    testCase "complex arithmetic modulus arg conjugate and polar conversion" <| fun _ ->
      let real = Image<float>.ofArray2D (array2D [ [ 3.0; 1.0 ]; [ -2.0; 0.0 ] ])
      let imag = Image<float>.ofArray2D (array2D [ [ 4.0; -1.0 ]; [ 2.0; -5.0 ] ])
      let otherReal = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 0.5; -3.0 ] ])
      let otherImag = Image<float>.ofArray2D (array2D [ [ -2.0; 0.5 ]; [ 1.0; 2.0 ] ])
      let z = toComplex real imag
      let w = toComplex otherReal otherImag
      let sum = z + w
      let difference = z - w
      let product = z * w
      let quotient = z / w
      let zModulus = modulus z
      let zArg = arg z
      let zConjugate = conjugate z
      let fromPolar = polarToComplex zModulus zArg

      Expect.equal sum.[0,0] (c 4.0 2.0) "Complex addition should preserve native complex arithmetic."
      Expect.equal difference.[0,0] (c 2.0 6.0) "Complex subtraction should preserve native complex arithmetic."
      Expect.equal product.[0,0] (c 11.0 -2.0) "Complex multiplication should preserve native complex arithmetic."
      Expect.floatClose Accuracy.high quotient.[0,0].Real -1.0 "Complex division real part should match."
      Expect.floatClose Accuracy.high quotient.[0,0].Imaginary 2.0 "Complex division imaginary part should match."
      Expect.floatClose Accuracy.high zModulus.[0,0] 5.0 "modulus should compute sqrt(re^2 + im^2)."
      Expect.floatClose Accuracy.high zArg.[0,0] (System.Math.Atan2(4.0, 3.0)) "arg should compute the complex phase angle."
      Expect.equal zConjugate.[0,0] (c 3.0 -4.0) "conjugate should keep Re and negate Im."
      Expect.floatClose Accuracy.high fromPolar.[0,0].Real 3.0 "polarToComplex should recover the real component."
      Expect.floatClose Accuracy.high fromPolar.[0,0].Imaginary 4.0 "polarToComplex should recover the imaginary component."

    testCase "toComplex from vector image is not supported" <| fun _ ->
      let arr = Array3D.init 2 2 2 (
        fun x y k ->
            let baseVal = float (1 + x + 2*y)
            if k = 0 then baseVal else baseVal + 9.0)
      let vecImg = Image<float list>.ofArray3DVector arr
      Expect.throws (fun () -> vecImg.toComplex() |> ignore) "Vector images are not implicitly converted to native complex images"

    testCase "FFTXY returns native Complex image with expected values" <| fun _ ->
      let src = Image<float>.ofArray2D (array2D [ [1.0; 0.0]; [0.0; 0.0] ])
      let out = FFTXY src

      Expect.equal (out.GetNumberOfComponentsPerPixel()) 1u "Complex output should have one native complex component per pixel"
      Expect.equal (out.toSimpleITK().GetPixelID()) PixelIDValueEnum.sitkComplexFloat64 "Complex output should use native ComplexFloat64 backing"

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

      Expect.equal (out.GetNumberOfComponentsPerPixel()) 1u "Complex output should have one native complex component per pixel"
      Expect.equal (out.toSimpleITK().GetPixelID()) PixelIDValueEnum.sitkComplexFloat64 "Complex output should use native ComplexFloat64 backing"

      Expect.floatClose Accuracy.high out.[0,0].Real 1.0 "dirFFT real at (0,0)"
      Expect.floatClose Accuracy.high out.[1,0].Real 1.0 "dirFFT real at (1,0)"
      Expect.floatClose Accuracy.high out.[0,1].Real 0.0 "dirFFT real at (0,1)"
      Expect.floatClose Accuracy.high out.[1,1].Real 0.0 "dirFFT real at (1,1)"
      Expect.floatClose Accuracy.high out.[0,0].Imaginary 0.0 "dirFFT imag at (0,0)"
      Expect.floatClose Accuracy.high out.[1,0].Imaginary 0.0 "dirFFT imag at (1,0)"
      Expect.floatClose Accuracy.high out.[0,1].Imaginary 0.0 "dirFFT imag at (0,1)"
      Expect.floatClose Accuracy.high out.[1,1].Imaginary 0.0 "dirFFT imag at (1,1)"

    testCase "complex directional FFT inverse XY FFT and shift helpers" <| fun _ ->
      let src = Image<float>.ofArray2D (array2D [ [ 1.0; 0.0 ]; [ 0.0; 0.0 ] ])
      let spectrum = FFTXY src
      let zVolume = ImageFunctions.stack [ spectrum; spectrum ]
      let zTransformed = directionalFFTComplex 2u false zVolume
      let zRecovered = directionalFFTComplex 2u true zTransformed
      let xyRecovered = inverseFFTXY spectrum
      let shifted = shiftFFT zTransformed

      try
        Expect.equal (zTransformed.GetSize()) [ 2u; 2u; 2u ] "directionalFFTComplex should preserve volume shape."
        Expect.equal zTransformed.[0,0,0] (c 2.0 0.0) "z FFT should add equal z samples at zero frequency."
        expectComplexClose zTransformed.[0,0,1] Complex.Zero "z FFT should cancel equal z samples at Nyquist frequency."
        Expect.floatClose Accuracy.high zRecovered.[1,1,0].Real spectrum.[1,1].Real "inverse directionalFFTComplex should recover real values."
        Expect.floatClose Accuracy.high zRecovered.[1,1,0].Imaginary spectrum.[1,1].Imaginary "inverse directionalFFTComplex should recover imaginary values."
        Expect.floatClose Accuracy.high xyRecovered.[0,0].Real 1.0 "inverseFFTXY should recover the original impulse real part."
        Expect.floatClose Accuracy.high xyRecovered.[0,0].Imaginary 0.0 "inverseFFTXY should recover the original impulse imaginary part."
        Expect.floatClose Accuracy.high xyRecovered.[1,0].Real 0.0 "inverseFFTXY should recover zero pixels."
        Expect.equal shifted.[1,1,1] zTransformed.[0,0,0] "shiftFFT should move zero frequency to the center for even-sized volumes."
      finally
        spectrum.decRefCount()
        zVolume.decRefCount()
        zTransformed.decRefCount()
        zRecovered.decRefCount()
        xyRecovered.decRefCount()
        shifted.decRefCount()
  ]
