module Tests.WrapperCoverage

open System
open System.IO
open System.Numerics
open Expecto
open Image
open Image.InternalHelpers
open itk.simple

let private expectClose actual expected message =
    Expect.floatClose Accuracy.high actual expected message

let private tinyFloatImage () =
    Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])

[<Tests>]
let wrapperCoverage =
  testList "Image public wrapper coverage" [

    testCase "4D array roundtrip and collection helpers" <| fun _ ->
      let source = Array4D.init 2 3 2 2 (fun x y z t -> x + 10 * y + 100 * z + 1000 * t)
      let img = Image<int>.ofArray4D (source, "fourD", 7)
      let roundtrip = img.toArray4D()
      let mapped = Image.map ((+) 1) img
      let indexed = Image.mapi (fun idx value -> value + int idx[3]) img
      let sum = Image.fold (+) 0 img
      let delta = Image.fold2 (fun acc before after -> acc + after - before) 0 img mapped
      let indexedSum = Image.foldi (fun idx acc value -> acc + int idx[0] + int idx[3] + value) 0 img

      Expect.equal (img.GetSize()) [ 2u; 3u; 2u; 2u ] "ofArray4D should preserve all four dimensions."
      Expect.equal roundtrip source "toArray4D should preserve 4D pixel order."
      Expect.equal mapped.[1,2,1,1] (source[1,2,1,1] + 1) "Image.map should use the 4D array path."
      Expect.equal indexed.[0,0,0,1] (source[0,0,0,1] + 1) "Image.mapi should include the fourth index."
      Expect.equal sum (source |> Seq.cast<int> |> Seq.sum) "Image.fold should cover all 4D values."
      Expect.equal delta source.Length "Image.fold2 should pair all 4D values."
      Expect.equal indexedSum (source |> Seq.cast<int> |> Seq.sumBy id |> fun total -> total + 24) "Image.foldi should include 4D indices."

    testCase "prod and simple shape manipulation wrappers" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [ [ 1uy; 2uy ]; [ 3uy; 4uy ] ])
      let padded = ImageFunctions.constantPad2D [ 1u; 2u ] [ 0u; 1u ] 9.0 img
      let cropped = ImageFunctions.crop2D [ 1u; 2u ] [ 0u; 1u ] padded
      let squeezed = Image<int>.ofArray3D (Array3D.init 2 1 3 (fun x _ z -> x + 10 * z)) |> ImageFunctions.squeeze

      Expect.equal (ImageFunctions.prod img) 24uy "prod should multiply all pixels."
      Expect.equal (padded.GetSize()) [ 3u; 5u ] "constantPad2D should expand both 2D axes."
      Expect.equal padded.[0,0] 9uy "The configured pad value should be written into the padded area."
      Expect.equal (cropped.GetSize()) [ 2u; 2u ] "crop2D should reverse the padding in this case."
      Expect.equal cropped.[1,1] 4uy "Cropping should preserve the original data region."
      Expect.equal (squeezed.GetSize()) [ 2u; 3u ] "squeeze should remove singleton dimensions."
      Expect.equal squeezed.[1,2] 21 "squeeze should preserve pixel order through the removed dimension."

      Expect.throws (fun () -> ImageFunctions.constantPad2D [ 1u ] [ 1u; 1u ] 0.0 img |> ignore) "constantPad2D should reject non-2D bounds."
      Expect.throws (fun () -> ImageFunctions.crop2D [ 1u; 1u ] [ 1u ] img |> ignore) "crop2D should reject non-2D bounds."

    testCase "intensity and local filter wrappers return expected values and shapes" <| fun _ ->
      let img = tinyFloatImage ()
      let clamped = ImageFunctions.clampImage 1.5 3.5 img
      let rescaled = ImageFunctions.rescaleIntensity 0.0 1.0 img
      let windowed = ImageFunctions.intensityWindow 1.0 4.0 0.0 30.0 img
      let normalized = ImageFunctions.normalizeImage img
      let shifted = ImageFunctions.shiftScale 1.0 2.0 img
      let inverted = ImageFunctions.invertIntensity 10.0 img
      let median = ImageFunctions.median 1u img
      let bilateral = ImageFunctions.bilateral 1.0 10.0 img
      let gradient = ImageFunctions.gradientMagnitude img
      let sobel = ImageFunctions.sobelEdge img
      let laplacian = ImageFunctions.laplacian img
      let normal = ImageFunctions.addNormalNoise 10.0 0.0 img
      let saltAndPepper = ImageFunctions.addSaltAndPepperNoise 0.0 img
      let shot = ImageFunctions.addShotNoise 0.0 img
      let speckle = ImageFunctions.addSpeckleNoise 0.0 img

      Expect.equal (clamped.GetSize()) [ 2u; 2u ] "clampImage should preserve shape."
      expectClose clamped.[0,0] 1.5 "clampImage should apply the lower bound."
      expectClose clamped.[1,1] 3.5 "clampImage should apply the upper bound."
      expectClose rescaled.[0,0] 0.0 "rescaleIntensity should map the minimum value."
      expectClose rescaled.[1,1] 1.0 "rescaleIntensity should map the maximum value."
      expectClose windowed.[0,0] 0.0 "intensityWindow should map the lower window edge."
      expectClose windowed.[1,1] 30.0 "intensityWindow should map the upper window edge."
      Expect.isFalse (Double.IsNaN normalized.[0,0]) "normalizeImage should produce finite values."
      expectClose shifted.[0,0] 4.0 "shiftScale should use SimpleITK's (value + shift) * scale convention."
      expectClose inverted.[1,1] 6.0 "invertIntensity should subtract values from the configured maximum."
      Expect.equal (median.GetSize()) [ 2u; 2u ] "median should preserve shape."
      Expect.equal (bilateral.GetSize()) [ 2u; 2u ] "bilateral should preserve shape."
      Expect.equal (gradient.GetSize()) [ 2u; 2u ] "gradientMagnitude should preserve shape."
      Expect.equal (sobel.GetSize()) [ 2u; 2u ] "sobelEdge should preserve shape."
      Expect.equal (laplacian.GetSize()) [ 2u; 2u ] "laplacian should preserve shape."
      Expect.equal (normal.GetSize()) [ 2u; 2u ] "addNormalNoise should preserve shape."
      Expect.equal (saltAndPepper.GetSize()) [ 2u; 2u ] "addSaltAndPepperNoise should preserve shape."
      Expect.equal (shot.GetSize()) [ 2u; 2u ] "addShotNoise should preserve shape."
      Expect.equal (speckle.GetSize()) [ 2u; 2u ] "addSpeckleNoise should preserve shape."
      expectClose normal.[0,0] img.[0,0] "Zero-std normal noise should bypass SimpleITK and preserve pixels."
      expectClose saltAndPepper.[0,1] img.[0,1] "Zero-probability salt-and-pepper noise should preserve pixels."
      expectClose shot.[1,0] img.[1,0] "Zero-scale shot noise should preserve pixels."
      expectClose speckle.[1,1] img.[1,1] "Zero-std speckle noise should preserve pixels."
      Expect.isTrue (Object.ReferenceEquals(img, normal)) "Zero-std normal noise should retain and return the input image."
      Expect.isGreaterThanOrEqual (img.getNReferences()) 2 "Zero-noise bypass should increment the image reference count."

    testCase "comparison logical mask and transform wrappers" <| fun _ ->
      let a = Image<uint8>.ofArray2D (array2D [ [ 0uy; 1uy ]; [ 2uy; 3uy ] ])
      let b = Image<uint8>.ofArray2D (array2D [ [ 0uy; 2uy ]; [ 1uy; 3uy ] ])
      let ones = Image<uint8>.ofArray2D (array2D [ [ 1uy; 0uy ]; [ 1uy; 0uy ] ])
      let other = Image<uint8>.ofArray2D (array2D [ [ 1uy; 1uy ]; [ 0uy; 0uy ] ])
      let eq = ImageFunctions.equalImage a b
      let neq = ImageFunctions.notEqualImage a b
      let gt = ImageFunctions.greaterImage a b
      let ge = ImageFunctions.greaterEqualImage a b
      let lt = ImageFunctions.lessImage a b
      let le = ImageFunctions.lessEqualImage a b
      let anded = ImageFunctions.andImage ones other
      let ored = ImageFunctions.orImage ones other
      let xored = ImageFunctions.xorImage ones other
      let inverted = ImageFunctions.notImage ones
      let masked = ImageFunctions.mask 9.0 a ones
      let transformed = ImageFunctions.euler2DTransform a (0.0, 0.0, 0.0) (0.0, 0.0)
      let rotated = ImageFunctions.euler2DRotate a (0.0, 0.0) 0.0
      let resampled = ImageFunctions.resample2D InterpolatorEnum.sitkNearestNeighbor 4u 3u 0.5 0.5 a

      Expect.equal eq.[0,0] 1uy "equalImage should mark equal pixels."
      Expect.equal neq.[0,1] 1uy "notEqualImage should mark unequal pixels."
      Expect.equal gt.[1,0] 1uy "greaterImage should mark greater pixels."
      Expect.equal ge.[1,1] 1uy "greaterEqualImage should include equality."
      Expect.equal lt.[0,1] 1uy "lessImage should mark smaller pixels."
      Expect.equal le.[0,0] 1uy "lessEqualImage should include equality."
      Expect.equal anded.[0,0] 1uy "andImage should combine non-zero masks."
      Expect.equal ored.[0,1] 1uy "orImage should combine non-zero masks."
      Expect.equal xored.[1,0] 1uy "xorImage should mark pixels set in exactly one mask."
      Expect.equal inverted.[0,1] 1uy "notImage should invert zero to one."
      Expect.equal masked.[0,1] 9uy "mask should use the outside value where the mask is zero."
      Expect.equal (transformed.GetSize()) [ 2u; 2u ] "euler2DTransform should preserve reference image size."
      Expect.equal (rotated.GetSize()) [ 2u; 2u ] "euler2DRotate should preserve reference image size."
      Expect.equal (resampled.GetSize()) [ 4u; 3u ] "resample2D should use the requested output size."
      Expect.throws (fun () ->
        Image<uint8>.ofArray3D (Array3D.zeroCreate 2 2 2)
        |> ImageFunctions.resample2D InterpolatorEnum.sitkNearestNeighbor 2u 2u 1.0 1.0
        |> ignore) "resample2D should reject non-2D images."

    testCase "byte facts cover vector complex label and fallback branches" <| fun _ ->
      Expect.equal (getBytesPerComponent typeof<uint64 list>) 8u "uint64 vector components should be eight bytes."
      Expect.equal (getBytesPerComponent typeof<int64 list>) 8u "int64 vector components should be eight bytes."
      Expect.equal (getBytesPerComponent typeof<float list>) 8u "float vector components should be eight bytes."
      Expect.equal (getBytesPerComponent typeof<Complex>) 16u "Complex components should include real and imaginary doubles."
      Expect.equal (getBytesPerComponent typeof<string>) 8u "Unknown component types should use the fallback estimate."
      Expect.equal (getBytesPerSItkComponent PixelIDValueEnum.sitkLabelUInt8) 1u "LabelUInt8 components should be one byte."
      Expect.equal (getBytesPerSItkComponent PixelIDValueEnum.sitkLabelUInt16) 2u "LabelUInt16 components should be two bytes."
      Expect.equal (getBytesPerSItkComponent PixelIDValueEnum.sitkLabelUInt32) 4u "LabelUInt32 components should be four bytes."
      Expect.equal (getBytesPerSItkComponent PixelIDValueEnum.sitkLabelUInt64) 8u "LabelUInt64 components should be eight bytes."
      Expect.equal (ImageFacts.memoryBytesForType<Complex> 6UL 1u) 96UL "Complex memory estimates should use sixteen bytes per pixel."

    testCase "complex array conversions" <| fun _ ->
      let complex2D = Array3D.init 2 2 2 (fun x y c -> if c = 0 then float (x + 10 * y) else float (100 + x + 10 * y))
      let complex2DImg = Image<Complex>.ofArray3DComplex complex2D

      Expect.equal complex2DImg.[1,1] (Complex(11.0, 111.0)) "ofArray3DComplex should map the final array axis to real/imaginary parts."
      Expect.throws (fun () -> Image<Complex>.ofArray3DComplex (Array3D.zeroCreate 2 2 3) |> ignore) "ofArray3DComplex should require a real/imaginary component axis."

    testCase "complex helpers and Fourier wrappers cover native complex paths" <| fun _ ->
      let src = Image<float>.ofArray2D (array2D [ [ 1.0; 0.0 ]; [ 0.0; 0.0 ] ])
      let fft = ImageFunctions.FFTXY src
      let directional = ImageFunctions.directionalFFT 1u src
      let scalar = Image<uint8>.ofArray2D (array2D [ [ 1uy ] ])
      let tmp = Path.Combine(Path.GetTempPath(), $"stackprocessing-complex-{Guid.NewGuid():N}.mha")

      try
        fft.toFileComplex tmp
        Expect.equal (toComplexFloat32 [ 1.0f; -2.0f ]) (Complex(1.0, -2.0)) "toComplexFloat32 should map two components to real/imaginary parts."
        Expect.equal (toComplexFloat64 [ 3.0 ]) (Complex(3.0, 0.0)) "toComplexFloat64 should default a missing imaginary component to zero."
        Expect.equal (toComplexFloat64 []) Complex.Zero "toComplexFloat64 should default an empty value to zero."
        Expect.throws (fun () -> toComplexFloat32 [ 1.0f; 2.0f; 3.0f ] |> ignore) "toComplexFloat32 should reject ambiguous component counts."
        Expect.equal (fft.GetNumberOfComponentsPerPixel()) 1u "FFTXY should return a native one-component complex image."
        Expect.equal (directional.GetNumberOfComponentsPerPixel()) 1u "directionalFFT should return a native one-component complex image."
        Expect.throws (fun () -> Image<Complex>.ofFileComplex tmp |> ignore) "ofFileComplex should reject formats read back as two-component vector images."
        Expect.throws (fun () -> scalar.toFileComplex tmp) "toFileComplex should reject scalar non-complex images."
        Expect.throws (fun () -> ImageFunctions.FFTXY (Image<float>.ofArray3D (Array3D.zeroCreate 2 2 2)) |> ignore) "FFTXY should reject non-2D images."
        Expect.throws (fun () -> ImageFunctions.directionalFFT 2u src |> ignore) "directionalFFT should reject directions outside the image dimensions."
      finally
        if File.Exists tmp then File.Delete tmp

    testCase "ofImageList covers multi-component composition plus invalid counts" <| fun _ ->
      let make value = Image<uint8>.ofArray2D (array2D [ [ value; value + 1uy ]; [ value + 2uy; value + 3uy ] ])
      let four = [ 0uy; 10uy; 20uy; 30uy ] |> List.map make
      let five = [ 0uy; 10uy; 20uy; 30uy; 40uy ] |> List.map make
      let six = [ 0uy; 10uy; 20uy; 30uy; 40uy; 50uy ] |> List.map make
      let composed4 = Image<uint8>.ofImageList four
      let composed5 = Image<uint8>.ofImageList five
      let composed6 = Image<uint8>.ofImageList six
      let vectorImage = ImageFunctions.toVectorImage four
      let split5 = composed5.toImageList()

      Expect.equal (composed4.GetNumberOfComponentsPerPixel()) 4u "Four scalar images should compose into a four-component vector image."
      Expect.equal composed4.[1,1] [ 3uy; 13uy; 23uy; 33uy ] "The four-component composed pixel should preserve source order."
      Expect.equal vectorImage.[1,1] [ 3uy; 13uy; 23uy; 33uy ] "toVectorImage should make a vector-valued image, not add a stack dimension."
      Expect.equal (vectorImage.GetSize()) [ 2u; 2u ] "toVectorImage should preserve the input image domain."
      Expect.equal (composed5.GetNumberOfComponentsPerPixel()) 5u "Five scalar images should compose into a five-component vector image."
      Expect.equal (composed6.GetNumberOfComponentsPerPixel()) 6u "Six scalar images should compose through VectorOfImage."
      Expect.equal composed6.[1,1] [ 3uy; 13uy; 23uy; 33uy; 43uy; 53uy ] "The six-component composed pixel should preserve source order."
      Expect.equal split5[4].[1,1] 43uy "toImageList should recover the fifth component."
      Expect.throws (fun () -> Image<uint8>.ofImageList [] |> ignore) "ofImageList should reject an empty input list."
      Expect.throws (fun () -> Image<uint8>.ofImageList [ make 1uy ] |> ignore) "ofImageList should reject one image."

    testCase "scalar and vector conversion methods cover public cast helpers" <| fun _ ->
      let scalar = Image<int>.ofArray2D (array2D [ [ 1; 2 ]; [ 3; 4 ] ])
      let vector = Image<uint8 list>.ofArray3DVector (Array3D.init 2 2 3 (fun x y c -> uint8 (x + 10 * y + 40 * c)))

      Expect.equal (scalar.toUInt8()).[0,0] 1uy "toUInt8 should cast scalar images."
      Expect.equal (scalar.toInt8()).[0,0] 1y "toInt8 should cast scalar images."
      Expect.equal (scalar.toUInt16()).[0,0] 1us "toUInt16 should cast scalar images."
      Expect.equal (scalar.toInt16()).[0,0] 1s "toInt16 should cast scalar images."
      Expect.equal (scalar.toUInt()).[0,0] 1u "toUInt should cast scalar images."
      Expect.equal (scalar.toInt()).[0,0] 1 "toInt should cast scalar images."
      Expect.equal (scalar.toUInt64()).[0,0] 1UL "toUInt64 should cast scalar images."
      Expect.equal (scalar.toInt64()).[0,0] 1L "toInt64 should cast scalar images."
      Expect.equal (scalar.toFloat32()).[0,0] 1.0f "toFloat32 should cast scalar images."
      Expect.equal (scalar.toFloat()).[0,0] 1.0 "toFloat should cast scalar images."
      Expect.equal (vector.toVectorUInt8()).[1,1] [ 11uy; 51uy; 91uy ] "toVectorUInt8 should keep vector component values."
      Expect.equal (vector.toVectorInt8()).[1,1] [ 11y; 51y; 91y ] "toVectorInt8 should cast vector component values."
      Expect.equal (vector.toVectorUInt16()).[1,1] [ 11us; 51us; 91us ] "toVectorUInt16 should cast vector component values."
      Expect.equal (vector.toVectorInt16()).[1,1] [ 11s; 51s; 91s ] "toVectorInt16 should cast vector component values."
      Expect.equal (vector.toVectorUInt32()).[1,1] [ 11u; 51u; 91u ] "toVectorUInt32 should cast vector component values."
      Expect.equal (vector.toVectorInt32()).[1,1] [ 11; 51; 91 ] "toVectorInt32 should cast vector component values."
      Expect.equal (vector.toVectorUInt64()).[1,1] [ 11UL; 51UL; 91UL ] "toVectorUInt64 should cast vector component values."
      Expect.equal (vector.toVectorInt64()).[1,1] [ 11L; 51L; 91L ] "toVectorInt64 should cast vector component values."
      Expect.equal (vector.toVectorFloat32()).[1,1] [ 11.0f; 51.0f; 91.0f ] "toVectorFloat32 should cast vector components."
      Expect.equal (vector.toVectorFloat64()).[1,1] [ 11.0; 51.0; 91.0 ] "toVectorFloat64 should cast vector components."

    testCase "vector image file wrappers round-trip vector pixels and reject scalar images" <| fun _ ->
      let vector = Image<uint8 list>.ofArray3DVector (Array3D.init 2 2 3 (fun x y c -> uint8 (x + 10 * y + 40 * c)))
      let scalar = Image<uint8>.ofArray2D (array2D [ [ 1uy; 2uy ] ])
      let tmp = Path.Combine(Path.GetTempPath(), $"stackprocessing-vector-{Guid.NewGuid():N}.mha")
      let scalarTmp = Path.Combine(Path.GetTempPath(), $"stackprocessing-scalar-{Guid.NewGuid():N}.mha")

      try
        vector.toFileVector tmp
        scalar.toFile scalarTmp
        let reread = Image<uint8>.ofFileVector tmp

        Expect.equal (reread.GetNumberOfComponentsPerPixel()) 3u "ofFileVector should preserve vector component count."
        Expect.equal reread.[1,1] [ 11uy; 51uy; 91uy ] "ofFileVector should recover vector component values."
        Expect.throws (fun () -> scalar.toFileVector tmp) "toFileVector should reject scalar images."
        Expect.throws (fun () -> Image<uint8>.ofFileVector scalarTmp |> ignore) "ofFileVector should reject scalar files."
      finally
        if File.Exists tmp then File.Delete tmp
        if File.Exists scalarTmp then File.Delete scalarTmp

    testCase "vector image operations map extract dot and cross pixels" <| fun _ ->
      let x = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
      let y = Image<float>.ofArray2D (array2D [ [ 10.0; 20.0 ]; [ 30.0; 40.0 ] ])
      let z = Image<float>.ofArray2D (array2D [ [ 100.0; 200.0 ]; [ 300.0; 400.0 ] ])
      let v2 = ImageFunctions.toVectorImage [ x; y ]
      let v3a = ImageFunctions.appendVectorElement v2 z
      let v3b = Image<float list>.ofArray3DVector (Array3D.init 2 2 3 (fun _ _ c -> if c = 0 then 0.0 elif c = 1 then 1.0 else 0.0))
      let mapped = ImageFunctions.mapVectorElements (fun value -> value + 1.0) v2
      let second = ImageFunctions.vectorElement 1u v2
      let dot = ImageFunctions.vectorDot v3a v3b
      let cross = ImageFunctions.vectorCross3D v3a v3b

      Expect.equal (v2.GetSize()) [ 2u; 2u ] "toVectorImage should preserve image size."
      Expect.equal (v2.GetNumberOfComponentsPerPixel()) 2u "toVectorImage should make one component per input image."
      Expect.equal v2.[1,1] [ 4.0; 40.0 ] "toVectorImage should preserve component order."
      Expect.equal v3a.[1,1] [ 4.0; 40.0; 400.0 ] "appendVectorElement should append the scalar image as the last vector component."
      Expect.equal mapped.[1,1] [ 5.0; 41.0 ] "mapVectorElements should transform each component."
      Expect.equal second.[1,1] 40.0 "vectorElement should extract the requested component."
      Expect.equal dot.[1,1] 40.0 "vectorDot should compute a per-pixel dot product."
      Expect.equal cross.[1,1] [ -400.0; 0.0; 4.0 ] "vectorCross3D should compute a per-pixel 3D cross product."
      Expect.floatClose Accuracy.high (ImageFunctions.vectorAngleTo [ 0.0; 1.0 ] v2).[1,1] (atan (0.1)) "vectorAngleTo should compute the angle to the reference vector."
      Expect.throws (fun () -> ImageFunctions.vectorElement 2u v2 |> ignore) "vectorElement should reject missing components."

    testCase "structure tensor helpers keep compact tensors and split eigensystem images" <| fun _ ->
      let gradient =
        Image<float list>.ofArray3DVector (
          Array3D.init 2 2 3 (fun _ _ elementIndex ->
            if elementIndex = 0 then 2.0 elif elementIndex = 1 then 0.0 else 0.0))
      let tensor = ImageFunctions.structureTensorOuterProduct gradient
      let eigenImages = ImageFunctions.structureTensorEigenImages tensor

      try
        Expect.equal (tensor.GetNumberOfComponentsPerPixel()) 6u "The compact symmetric tensor should store six components."
        Expect.equal tensor.[1,1] [ 4.0; 0.0; 0.0; 0.0; 0.0; 0.0 ] "Outer products should use [xx xy xz yy yz zz] order."
        Expect.equal eigenImages.Length 4 "The eigensystem should split into eigenvalues plus three eigenvector images."
        eigenImages |> List.iter (fun image -> Expect.equal (image.GetNumberOfComponentsPerPixel()) 3u "Every eigensystem output image should be a 3-vector.")
        Expect.equal eigenImages[0].[1,1] [ 4.0; 0.0; 0.0 ] "Eigenvalues should be stored in the first 3-vector image."
        Expect.equal eigenImages[1].[1,1] [ 1.0; 0.0; 0.0 ] "The first eigenvector should follow the x direction for a pure x-gradient."
      finally
        eigenImages |> List.iter (fun image -> image.decRefCount())
        tensor.decRefCount()
        gradient.decRefCount()

    testCase "gradientVector3D bundles finite differences into a 3-vector image" <| fun _ ->
      let volume =
        Array3D.init 5 5 5 (fun x y z -> float x + 10.0 * float y + 100.0 * float z)
        |> Image<float>.ofArray3D

      let gradient = ImageFunctions.gradientVector3D 1u volume
      let split = gradient.toImageList()
      let direct =
        [ 0u; 1u; 2u ]
        |> List.map (fun direction ->
          let kernel = ImageFunctions.finiteDiffFilter3D direction 1u
          try
            ImageFunctions.convolve None None volume kernel
          finally
            kernel.decRefCount())

      try
        Expect.equal (gradient.GetSize()) (volume.GetSize()) "gradientVector3D should preserve the 3D image domain."
        Expect.equal (gradient.GetNumberOfComponentsPerPixel()) 3u "gradientVector3D should emit dx dy dz components."

        List.zip3 split direct [ "dx"; "dy"; "dz" ]
        |> List.iter (fun (actualComponent, expected, label) ->
          expectClose actualComponent.[2,2,2] expected.[2,2,2] $"{label} should match the corresponding finite-difference convolution.")

        Expect.equal gradient.[2,2,2] [ split[0].[2,2,2]; split[1].[2,2,2]; split[2].[2,2,2] ] "The vector pixel should preserve component order."
        Expect.throws (fun () -> ImageFunctions.gradientVector3D 1u (tinyFloatImage ()) |> ignore) "gradientVector3D should reject non-3D images."
      finally
        gradient.decRefCount()
        split |> List.iter (fun image -> image.decRefCount())
        direct |> List.iter (fun image -> image.decRefCount())
        volume.decRefCount()
  ]
