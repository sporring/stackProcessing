module simpleITKWrappers

open CoreTypes
open Vector
open itk.simple

/// <summary>
/// Converts an array of <c>uint64</c> values to a <c>VectorUInt32</c>, which is the native index type expected by SimpleITK image accessors.
/// </summary>
/// <param name="arr">An array of <c>uint64</c> values representing an image index.</param>
/// <returns>
/// A <c>VectorUInt32</c> containing the same values as the input array, cast to <c>uint32</c>.
/// </returns>
/// <remarks>
/// This function is commonly used when generating pixel indices for use with <c>Image.GetPixelAsUInt8</c>
/// or other SimpleITK image accessor functions, which require a <c>VectorUInt32</c> rather than a .NET array.
/// </remarks>
let toVectorUInt32 (arr: 'a[]) =
    let v = new VectorUInt32()
    arr |> Array.iter (fun x -> v.Add(uint32 x))
    v

/// <summary>
/// Retrieves the intensity value of a voxel at the specified 3D coordinates from an image.
/// </summary>
/// <param name="image">The 3D image to access.</param>
/// <param name="x">The x-coordinate (column) of the voxel.</param>
/// <param name="y">The y-coordinate (row) of the voxel.</param>
/// <param name="z">The z-coordinate (slice index) of the voxel.</param>
/// <returns>The intensity value at the specified location as a byte.</returns>
let getPixel3D (image: Image) (x: uint) (y: uint) (z: uint) : byte =
    image.GetPixelAsUInt8(new VectorUInt32([| x; y; z |]))

/// <summary>
/// Extracts a 2D slice from a 3D image at the given z-index.
/// </summary>
/// <param name="image">The 3D image volume to slice.</param>
/// <param name="z">The slice index along the z-axis.</param>
/// <returns>A 2D image representing the extracted slice.</returns>
/// <remarks>
/// Uses <c>ExtractImageFilter</c> with a fixed size in the z-dimension of 1.
/// </remarks>
let extractSlice (image: Image) (z: uint) : Image =
    let width = image.GetWidth() |> uint
    let height = image.GetHeight() |> uint
    let extractor = new ExtractImageFilter()
    extractor.SetSize(new VectorUInt32([| width; height; 1u |]))
    extractor.SetIndex(new VectorInt32([| 0; 0; int z |]))
    extractor.Execute(image)

/// <summary>
/// Stacks a list of 2D image slices into a single 3D volume.
/// </summary>
/// <param name="slices">A non-empty list of <c>ImageSlice</c> objects to stack.</param>
/// <returns>
/// A new <c>ImageSlice</c> containing a 3D volume built from the input slices, using the index of the first slice.
/// </returns>
/// <exception cref="System.ArgumentException">
/// Thrown if the input list is empty.
/// </exception>
/// <remarks>
/// This function assumes all input images are aligned and have matching dimensions.
/// </remarks>
let stack (slices: ImageSlice list) : ImageSlice =
    let idx = slices[0].Index
    let images = List.map (fun s -> s.Image) slices
    let joiner = new JoinSeriesImageFilter()
    let volume = joiner.Execute(new VectorOfImage(images))
    { Index = idx; Image = volume }

/// <summary>
/// Multiplies the pixel values of two image slices element-wise.
/// </summary>
/// <param name="s1">The first image slice.</param>
/// <param name="s2">The second image slice.</param>
/// <returns>
/// A new <c>ImageSlice</c> containing the result of the element-wise multiplication.
/// </returns>
/// <remarks>
/// Assumes both input slices have the same dimensions.
/// </remarks>
let multiplySlices (s1:ImageSlice) (s2:ImageSlice) : ImageSlice =
    let filter = new MultiplyImageFilter()
    { s1 with Image = filter.Execute (s1.Image, s2.Image) }

/// <summary>
/// Adds the pixel values of two image slices element-wise.
/// </summary>
/// <param name="s1">The first image slice.</param>
/// <param name="s2">The second image slice.</param>
/// <returns>
/// A new <c>ImageSlice</c> containing the sum of the pixel values.
/// </returns>
/// <remarks>
/// Assumes both input slices are aligned and dimensionally compatible.
/// </remarks>
let addSlices (s1:ImageSlice) (s2:ImageSlice) : ImageSlice =
    let filter = new AddImageFilter()
    { s1 with Image = filter.Execute (s1.Image, s2.Image) }

/// <summary>
/// Adds a constant to each pixel in the image.
/// </summary>
/// <param name="value">The constant value to add.</param>
let addConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new AddImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Adds each pixel of the input image to the corresponding pixel in another image.
/// </summary>
/// <param name="other">The second image.</param>
let addImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new AddImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Subtracts a constant from each pixel in the image.
/// </summary>
/// <param name="value">The constant value to subtract.</param>
let subtractConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new SubtractImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Subtracts the corresponding pixel values in another image from the input image.
/// </summary>
/// <param name="other">The second image to subtract from the input.</param>
let subtractImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new SubtractImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Divides each pixel in the image by a constant value.
/// </summary>
/// <param name="value">The constant divisor.</param>
let divideConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new DivideImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Divides each pixel in the input image by the corresponding pixel in another image.
/// </summary>
/// <param name="other">The divisor image.</param>
let divideImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new DivideImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that multiplies each pixel in the image slice by a constant value.
/// </summary>
/// <param name="value">The constant multiplier.</param>
let multiplyConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new MultiplyImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Returns a function that multiplies each pixel in the input image slice by the corresponding pixel in another image.
/// </summary>
/// <param name="other">The second image to multiply with.</param>
let multiplyImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new MultiplyImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that computes the pixel-wise modulus (remainder) of the image with respect to a constant integer value.
/// </summary>
/// <param name="value">The constant modulus divisor (must be a non-zero unsigned integer).</param>
let modulusConst (value: uint) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new ModulusImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Returns a function that computes the pixel-wise modulus of the image with respect to another image.
/// </summary>
/// <param name="other">The divisor image (modulo base).</param>
let modulusImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new ModulusImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that raises each pixel to the given constant power.
/// </summary>
/// <param name="exponent">The exponent to raise each pixel to.</param>
let powConst (exponent: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new PowImageFilter()
        { slice with Image = filter.Execute(slice.Image, exponent) }

/// <summary>
/// Returns a function that raises each pixel in the image to the power of the corresponding pixel in another image.
/// </summary>
/// <param name="other">The exponent image.</param>
let powImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new PowImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that applies the GreaterEqualImageFilter with a constant threshold.
/// Pixels in the input image that are greater than or equal to the constant will be set to 1; others to 0.
/// </summary>
/// <param name="value">The constant value to compare against each pixel.</param>
let greaterEqualConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new GreaterEqualImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Returns a function that applies the GreaterEqualImageFilter with another image.
/// Pixels in the input image that are greater than or equal to the corresponding pixels in the second image will be set to 1; others to 0.
/// </summary>
/// <param name="other">The image to compare against.</param>
let greaterEqualImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new GreaterEqualImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that applies the GreaterImageFilter with a constant threshold.
/// Pixels in the input image that are strictly greater than the constant will be set to 1; others to 0.
/// </summary>
/// <param name="value">The constant value to compare against each pixel.</param>
let greaterConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new GreaterImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Returns a function that applies the GreaterImageFilter with another image.
/// Pixels in the input image that are strictly greater than the corresponding pixels in the second image will be set to 1; others to 0.
/// </summary>
/// <param name="other">The image to compare against.</param>
let greaterImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new GreaterImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that applies the EqualImageFilter with a constant.
/// Pixels in the input image that are exactly equal to the constant will be set to 1; others to 0.
/// </summary>
/// <param name="value">The constant value to compare against each pixel.</param>
let equalConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new EqualImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Returns a function that applies the EqualImageFilter with another image.
/// Pixels in the input image that are equal to the corresponding pixels in the second image will be set to 1; others to 0.
/// </summary>
/// <param name="other">The image to compare against.</param>
let equalImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new EqualImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that applies the NotEqualImageFilter with a constant.
/// Pixels in the input image that are not equal to the constant will be set to 1; others to 0.
/// </summary>
/// <param name="value">The constant value to compare against each pixel.</param>
let notEqualConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new NotEqualImageFilter()
        { slice with Image = filter.Execute(slice.Image,value) }

/// <summary>
/// Returns a function that applies the NotEqualImageFilter with another image.
/// Pixels in the input image that are not equal to the corresponding pixels in the second image will be set to 1; others to 0.
/// </summary>
/// <param name="other">The image to compare against.</param>
let notEqualImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new NotEqualImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that applies the LessImageFilter with a constant threshold.
/// Pixels in the input image that are strictly less than the constant will be set to 1; others to 0.
/// </summary>
/// <param name="value">The constant value to compare against each pixel.</param>
let lessConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new LessImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Returns a function that applies the LessImageFilter with another image.
/// Pixels in the input image that are strictly less than the corresponding pixels in the second image will be set to 1; others to 0.
/// </summary>
/// <param name="other">The image to compare against.</param>
let lessImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new LessImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Returns a function that applies the LessEqualImageFilter with a constant threshold.
/// Pixels in the input image that are less than or equal to the constant will be set to 1; others to 0.
/// </summary>
/// <param name="value">The constant value to compare against each pixel.</param>
let lessEqualConst (value: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new LessEqualImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Returns a function that applies the LessEqualImageFilter with another image.
/// Pixels in the input image that are less than or equal to the corresponding pixels in the second image will be set to 1; others to 0.
/// </summary>
/// <param name="other">The image to compare against.</param>
let lessEqualImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new LessEqualImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Performs a bitwise AND between each pixel in the image and a constant integer value.
/// </summary>
/// <param name="value">The constant value to AND with each pixel.</param>
/// <returns>An <c>ImageSlice</c> where each pixel is <c>pixel &amp; value</c>.</returns>
let bitwiseAndConst (value: int) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new AndImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Performs a pixel-wise bitwise AND between two images.
/// </summary>
/// <param name="other">The second image to AND with the input slice.</param>
/// <returns>An <c>ImageSlice</c> where each pixel is <c>pixel1 &amp; pixel2</c>.</returns>
let bitwiseAndImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new AndImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Performs a bitwise OR between each pixel in the image and a constant integer value.
/// </summary>
/// <param name="value">The constant value to OR with each pixel.</param>
/// <returns>An <c>ImageSlice</c> where each pixel is <c>pixel | value</c>.</returns>
let bitwiseOrConst (value: int) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new OrImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Performs a pixel-wise bitwise OR between two images.
/// </summary>
/// <param name="other">The second image to OR with the input slice.</param>
/// <returns>An <c>ImageSlice</c> where each pixel is <c>pixel1 | pixel2</c>.</returns>
let bitwiseOrImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new OrImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Performs a bitwise XOR between each pixel in the image and a constant integer value.
/// </summary>
/// <param name="value">The constant value to XOR with each pixel.</param>
/// <returns>An <c>ImageSlice</c> where each pixel is <c>pixel ^ value</c>.</returns>
let bitwiseXorConst (value: int) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new XorImageFilter()
        { slice with Image = filter.Execute(slice.Image, value) }

/// <summary>
/// Performs a pixel-wise bitwise XOR between two images.
/// </summary>
/// <param name="other">The second image to XOR with the input slice.</param>
/// <returns>An <c>ImageSlice</c> where each pixel is <c>pixel1 ^ pixel2</c>.</returns>
let bitwiseXorImage (other: Image) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new XorImageFilter()
        { slice with Image = filter.Execute(slice.Image, other) }

/// <summary>
/// Inverts the intensity of each pixel (bitwise NOT for integers).
/// </summary>
/// <param name="maximum">The maximum intensity value (e.g. 255 for 8-bit).</param>
let bitwiseNot (maximum: int) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new InvertIntensityImageFilter()
        filter.SetMaximum maximum
        { slice with Image = filter.Execute(slice.Image) }

/// <summary>
/// Performs a left bitwise shift on each pixel by multiplying by 2ⁿ.
/// </summary>
/// <param name="shiftBits">The number of bits to shift left.</param>
let bitwiseLeftShift (shiftBits: int) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new ShiftScaleImageFilter()
        filter.SetShift(0.0)
        filter.SetScale(2.0 ** float shiftBits)
        { slice with Image = filter.Execute(slice.Image) }

/// <summary>
/// Simulates right shift by dividing each pixel by 2ⁿ (approximation for unsigned integers).
/// </summary>
/// <param name="shiftBits">The number of bits to shift right.</param>
let bitwiseRightShift (shiftBits: int) : ImageSlice -> ImageSlice =
    fun slice ->
        let divisor = 2.0 ** float shiftBits
        let filter = new DivideImageFilter()
        { slice with Image = filter.Execute(slice.Image, divisor) }

/// <summary>
/// Returns a function that computes the absolute value of each pixel in the image slice.
/// </summary>
/// <returns>An <c>ImageSlice</c> with pixel values set to <c>abs(pixel)</c>.</returns>
let absImage : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new AbsImageFilter()
        { slice with Image = filter.Execute(slice.Image) }

/// <summary>
/// Returns a function that computes the square root of each pixel in the image slice.
/// </summary>
/// <returns>An <c>ImageSlice</c> with pixel values set to <c>sqrt(pixel)</c>.</returns>
/// <remarks>
/// Input values must be non-negative; negative values will produce NaN.
/// </remarks>
let sqrtImage : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new SqrtImageFilter()
        { slice with Image = filter.Execute(slice.Image) }

/// <summary>
/// Returns a function that computes the natural logarithm of each pixel in the image slice.
/// </summary>
/// <returns>An <c>ImageSlice</c> with pixel values set to <c>log(pixel)</c>.</returns>
/// <remarks>
/// Input values must be positive; zero or negative inputs will result in -inf or NaN.
/// </remarks>
let logImage : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new LogImageFilter()
        { slice with Image = filter.Execute(slice.Image) }

/// <summary>
/// Returns a function that computes the exponential (e<sup>x</sup>) of each pixel in the image slice.
/// </summary>
/// <returns>An <c>ImageSlice</c> with pixel values set to <c>exp(pixel)</c>.</returns>
/// <remarks>
/// May overflow for large input values depending on the pixel type.
/// </remarks>
let expImage : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new ExpImageFilter()
        { slice with Image = filter.Execute(slice.Image) }

/// <summary>
/// Applies a linear transformation to the pixel values of an image slice using shift and scale.
/// </summary>
/// <param name="delta">The value to shift (add to) each pixel.</param>
/// <param name="factor">The scale multiplier for each pixel.</param>
/// <param name="slice">The input image slice to transform.</param>
/// <returns>
/// A new <c>ImageSlice</c> with transformed pixel values.
/// </returns>
let shiftScaleSlice (delta: float) (factor: float) (slice:ImageSlice) : ImageSlice =
    let filter = new ShiftScaleImageFilter ()
    filter.SetScale factor
    filter.SetShift delta
    { slice with Image = filter.Execute slice.Image }

/// <summary>
/// Applies additive Gaussian noise to an image slice.
/// </summary>
/// <param name="mean">The mean of the Gaussian noise.</param>
/// <param name="stddev">The standard deviation of the Gaussian noise.</param>
/// <param name="slice">The input image slice to which noise will be added.</param>
/// <returns>
/// A new <c>ImageSlice</c> with Gaussian noise applied.
/// </returns>
let additiveGaussianNoiseSlice (mean: float) (stddev: float) (slice: ImageSlice) : ImageSlice =
    let filter = new AdditiveGaussianNoiseImageFilter()
    filter.SetMean(mean)
    filter.SetStandardDeviation(stddev)
    { slice with Image = filter.Execute(slice.Image) }

/// <summary>
/// Applies a binary threshold to an image slice, setting pixel values outside the range to zero.
/// </summary>
/// <param name="lower">The lower threshold value (inclusive).</param>
/// <param name="upper">The upper threshold value (inclusive).</param>
/// <returns>
/// A function that transforms an <c>ImageSlice</c> by applying the binary threshold.
/// </returns>
/// <remarks>
/// Pixels with values between <paramref name="lower"/> and <paramref name="upper"/> are preserved;
/// others are set to zero or the background.
/// </remarks>
let thresholdSlice (lower: float) (upper: float) : ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new BinaryThresholdImageFilter()
        filter.SetLowerThreshold lower
        filter.SetUpperThreshold upper
        { slice with Image = filter.Execute slice.Image }

/// <summary>
/// Applies a 3D discrete Gaussian filter to a volumetric slice and extracts a representative 2D image from the center slice.
/// </summary>
/// <param name="sigma">The standard deviation (blur amount) of the Gaussian kernel.</param>
/// <param name="depth">The depth of the slice stack to simulate a 3D convolution.</param>
/// <returns>
/// A function that transforms an <c>ImageSlice</c> by applying 3D Gaussian smoothing and extracting the central slice.
/// </returns>
/// <remarks>
/// The center slice is computed as <c>depth / 2</c>. This is a pseudo-3D convolution assuming a volume around the input slice.
/// </remarks>
let convolve3DGaussianSlice (sigma: float) (depth: uint): ImageSlice -> ImageSlice =
    fun slice ->
        let filter = new DiscreteGaussianImageFilter()
        filter.SetVariance (sigma * sigma)
        let volume = filter.Execute slice.Image
        let image = extractSlice volume (depth/2u)
        { slice with Image = image }

/// <summary>
/// Computes the intensity histogram of a single 2D or 3D byte image slice.
/// </summary>
/// <param name="slice">An <c>ImageSlice</c> containing a SimpleITK image of pixel type <c>UInt8</c>.</param>
/// <returns>
/// An array of 256 integers where each index corresponds to the number of pixels with that intensity value (0–255).
/// </returns>
/// <exception cref="System.Exception">
/// Thrown if the image is not of type <c>UInt8</c> or has unsupported dimensionality.
/// </exception>
/// <remarks>
/// This function iterates over all pixels in the slice and accumulates their intensity counts
/// without explicitly referencing image bounds in the loop logic.
/// </remarks>
let histogramSlice (slice: ImageSlice) : Vector<int> =
    let image = slice.Image
    if image.GetPixelID() <> PixelIDValueEnum.sitkUInt8 then
        failwithf "Expected UInt8 image, got %A" (image.GetPixelID())

    let size = image.GetSize()
    let dims = size.Count
    let hist = Array.zeroCreate 256

    let indices =
        match dims with
        | 2 ->
            let width = int size.[0]
            let height = int size.[1]
            seq {
                for y in Seq.init height uint64 do
                    for x in Seq.init width uint64 do
                        yield [| x; y |]
            }
        | 3 ->
            let width = int size.[0]
            let height = int size.[1]
            let depth = int size.[2]
            seq {
                for z in Seq.init depth uint64 do
                    for y in Seq.init height uint64 do
                        for x in Seq.init width uint64 do
                            yield [| x; y; z |]
            }
        | _ -> failwith "Unsupported dimensionality"

    for idx in indices do
        let value = idx |> toVectorUInt32 |> image.GetPixelAsUInt8
        hist.[int value] <- hist.[int value] + 1

    Vector hist
