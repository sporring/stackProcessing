module ImageClass.ImageOps
open ImageClass
open itk.simple

/// Module with inline operator overloads for Image
let toVectorUInt32 (arr: int list) =
    let v = new VectorUInt32()
    arr |> List.iter (uint32 >> v.Add)
    v

let toVectorDouble (arr: float list) =
    let v = new VectorDouble()
    arr |> List.iter v.Add
    v

// ----- Operator Overloads -----

// operator overloading in F# is not easy. 
//   (=) etc. must be inline and cannot be overloaded directly. 
//   op_Equality etc. can but must return bools so cannot return Image of bools and cannot be curried directly
//   when ^T : (static member op_Explicit : ^T -> double) can be used for type checking
// Thus we settle for boxing:
//   + Single function handles all needed combinations.
//   + Allows generic numeric types: int, float, byte, etc.
//   + Works naturally in F# functional code (List.map ((+) 5.0)).
//   + Avoids F#'s restriction against multiple function/operator overloads.
//   - Type resolution for Image vs scalar happens at runtime via box, not static dispatch.
//   - You lose type-checking for "is this a scalar?" - but retain type-checking for Image.
let inline makeBinaryImageOperator
    (filter: 'Filter)
    (invokeImageImage: 'Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image)
    (invokeImageScalar: 'Filter -> itk.simple.Image -> double -> itk.simple.Image)
    (invokeScalarImage: 'Filter -> double -> itk.simple.Image -> itk.simple.Image)
    : (^A -> ^B -> Raw) 
    when ^A: (static member op_Explicit: ^A -> double)
     and ^B: (static member op_Explicit: ^B -> double) =
    let inline apply (a: ^A) (b: ^B) =
        match box a, box b with
        | (:? Raw as ia), (:? Raw as ib) ->
            Raw(invokeImageImage filter ia.Image ib.Image)
        | (:? Raw as ia), _ ->
            Raw(invokeImageScalar filter ia.Image (double b))
        | _, (:? Raw as ib) ->
            Raw(invokeScalarImage filter (double a) ib.Image)
        | _ ->
            failwithf "Invalid operands to image operator: %A and %A" typeof<^A> typeof<^B>
    apply

let inline makeBinaryUIntImageOperator
    (filter: 'Filter)
    (invokeImageImage: 'Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image)
    (invokeImageScalar: 'Filter -> itk.simple.Image -> uint -> itk.simple.Image)
    (invokeScalarImage: 'Filter -> uint -> itk.simple.Image -> itk.simple.Image)
    : (^A -> ^B -> Raw) =
    let inline apply (a: ^A) (b: ^B) =
        match box a, box b with
        | (:? Raw as ia), (:? Raw as ib) ->
            Raw(invokeImageImage filter ia.Image ib.Image)
        | (:? Raw as ia), _ ->
            Raw(invokeImageScalar filter ia.Image (uint b))
        | _, (:? Raw as ib) ->
            Raw(invokeScalarImage filter (uint a) ib.Image)
        | _ ->
            failwithf "Invalid operands to image operator: %A and %A" typeof<^A> typeof<^B>
    apply

let inline makeBinaryIntImageOperator
    (filter: 'Filter)
    (invokeImageImage: 'Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image)
    (invokeImageScalar: 'Filter -> itk.simple.Image -> int -> itk.simple.Image)
    (invokeScalarImage: 'Filter -> int -> itk.simple.Image -> itk.simple.Image)
    : (^A -> ^B -> Raw) =
    let inline apply (a: ^A) (b: ^B) =
        match box a, box b with
        | (:? Raw as ia), (:? Raw as ib) ->
            Raw(invokeImageImage filter ia.Image ib.Image)
        | (:? Raw as ia), _ ->
            Raw(invokeImageScalar filter ia.Image (int b))
        | _, (:? Raw as ib) ->
            Raw(invokeScalarImage filter (int a) ib.Image)
        | _ ->
            failwithf "Invalid operands to image operator: %A and %A" typeof<^A> typeof<^B>
    apply

// for static type checking, Execute for each combination of Image and
// double needs triple mentions
let inline (+.) a b =
    makeBinaryImageOperator
        (new AddImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline (-.) a b =
    makeBinaryImageOperator
        (new SubtractImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline ( *.) a b =
    makeBinaryImageOperator
        (new MultiplyImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline (/.) a b =
    makeBinaryImageOperator
        (new DivideImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b
let inline (===) a b =
    makeBinaryImageOperator
        (new EqualImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b
let inline (=/=) a b =
    makeBinaryImageOperator
        (new NotEqualImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b
let inline (<.) a b =
    makeBinaryImageOperator
        (new LessImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b
let inline (<=.) a b =
    makeBinaryImageOperator
        (new LessImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b
let inline (>.) a b =
    makeBinaryImageOperator
        (new GreaterImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b
let inline (>=.) a b =
    makeBinaryImageOperator
        (new GreaterEqualImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline ( %. ) a b =
    makeBinaryUIntImageOperator
        (new ModulusImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline ( **. ) a b =
    makeBinaryImageOperator
        (new PowImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline ( &&&. ) a b =
    makeBinaryIntImageOperator
        (new AndImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline ( ^^^. ) a b =
    makeBinaryIntImageOperator
        (new XorImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline ( |||. ) a b =
    makeBinaryIntImageOperator
        (new OrImageFilter())
        (fun f a b -> f.Execute(a, b))
        (fun f a s -> f.Execute(a, s))
        (fun f s b -> f.Execute(s, b))
        a b

let inline ( ~~~ ) (img: Raw) =
    let filter = new InvertIntensityImageFilter()
    // Default maximum is 1.0, but you can expose it if needed
    Raw(filter.Execute(img.Image))

// ----- basic mathematical functions -----
let inline makeUnaryImageOperator
    (filter: 'Filter)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image)
    : Raw -> Raw =
    fun (img: Raw) ->
        Raw(invoke filter img.Image)

let inline abs (img: Raw)    = makeUnaryImageOperator (new AbsImageFilter())    (fun f x -> f.Execute(x)) img
let inline log (img: Raw)    = makeUnaryImageOperator (new LogImageFilter())    (fun f x -> f.Execute(x)) img
let inline log10 (img: Raw)  = makeUnaryImageOperator (new Log10ImageFilter())  (fun f x -> f.Execute(x)) img
let inline exp (img: Raw)    = makeUnaryImageOperator (new ExpImageFilter())    (fun f x -> f.Execute(x)) img
let inline sqrt (img: Raw)   = makeUnaryImageOperator (new SqrtImageFilter())   (fun f x -> f.Execute(x)) img
let inline square (img: Raw) = makeUnaryImageOperator (new SquareImageFilter()) (fun f x -> f.Execute(x)) img
let inline sin (img: Raw)    = makeUnaryImageOperator (new SinImageFilter())    (fun f x -> f.Execute(x)) img
let inline cos (img: Raw)    = makeUnaryImageOperator (new CosImageFilter())    (fun f x -> f.Execute(x)) img
let inline tan (img: Raw)    = makeUnaryImageOperator (new TanImageFilter())    (fun f x -> f.Execute(x)) img
let inline asin (img: Raw)   = makeUnaryImageOperator (new AsinImageFilter())   (fun f x -> f.Execute(x)) img
let inline acos (img: Raw)   = makeUnaryImageOperator (new AcosImageFilter())   (fun f x -> f.Execute(x)) img
let inline atan (img: Raw)   = makeUnaryImageOperator (new AtanImageFilter())   (fun f x -> f.Execute(x)) img

let inline round (img: Raw)  = makeUnaryImageOperator (new RoundImageFilter())  (fun f x -> f.Execute(x)) img

// ----- basic image analysis functions -----
let fft3D (img: Raw) : Raw =
    let filter = new ForwardFFTImageFilter()
    Raw(filter.Execute(img.Image))

let ifft3D (img: Raw) : Raw =
    let filter = new InverseFFTImageFilter()
    Raw(filter.Execute(img.Image))

let real (img: Raw) : Raw =
    let filter = new ComplexToRealImageFilter()
    Raw(filter.Execute(img.Image))

let imag (img: Raw) : Raw =
    let filter = new ComplexToImaginaryImageFilter()
    Raw(filter.Execute(img.Image))

let cabs (img: Raw) : Raw =
    let filter = new ComplexToModulusImageFilter()
    Raw(filter.Execute(img.Image))

let carg (img: Raw) : Raw =
    let filter = new ComplexToPhaseImageFilter()
    Raw(filter.Execute(img.Image))

let convolve (kernel: Raw) (input: Raw) : Raw =
    let filter = new ConvolutionImageFilter()
    Raw(filter.Execute(input.Image, kernel.Image))

/// Gaussian kernel convolution
/// Isotropic Discrete Gaussian blur
let discreteGaussian (input: Raw) (sigma: float) : Raw =
    let filter = new DiscreteGaussianImageFilter()
    filter.SetVariance(sigma * sigma)
    Raw(filter.Execute(input.Image))

/// Recursive Gaussian blur in a specific direction (0 = x, 1 = y, 2 = z)
let recursiveGaussian (input: Raw) (sigma: float) (direction: uint) : Raw =
    let filter = new RecursiveGaussianImageFilter()
    filter.SetSigma(sigma)
    filter.SetDirection(direction)
    Raw(filter.Execute(input.Image))

/// Laplacian of Gaussian convolution
let laplacianConvolve (input: Raw) (sigma: float) : Raw =
    let filter = new LaplacianRecursiveGaussianImageFilter()
    filter.SetSigma(sigma)
    Raw(filter.Execute(input.Image))

/// Gradient convolution using Derivative filter
let gradientXConvolve (input: Raw) (order: uint32) : Raw =
    let filter = new DerivativeImageFilter()
    filter.SetDirection(0u) // X axis
    filter.SetOrder(order)
    Raw(filter.Execute(input.Image))

let gradientYConvolve (input: Raw) (order: uint32) : Raw =
    let filter = new DerivativeImageFilter()
    filter.SetDirection(1u) // Y axis
    filter.SetOrder(order)
    Raw(filter.Execute(input.Image))

let gradientZConvolve (input: Raw) (order: uint32) : Raw =
    let filter = new DerivativeImageFilter()
    filter.SetDirection(2u) // Z axis
    filter.SetOrder(order)
    Raw(filter.Execute(input.Image))

/// Image sources
/// Create a grid pattern image
let gridImage (size: int list) (spacing: float list) (origin: float list) : Raw =
    let source = new GridImageSource()
    source.SetSize(new VectorUInt32(size))
    source.SetSpacing(new VectorDouble(spacing))
    source.SetOrigin(new VectorDouble(origin))
    Raw(source.Execute())

/// Create a Gabor pattern image
let gaborImage (size: int list) (sigma: float list) (frequency: float) : Raw =
    let source = new GaborImageSource()
    source.SetSize(new VectorUInt32(size))
    source.SetSigma(new VectorDouble(sigma))
    source.SetFrequency(frequency)
    Raw(source.Execute())

/// Create a Gaussian pattern image
let gaussianImage (size: int list) (sigma: float list) : Raw =
    let source = new GaussianImageSource()
    source.SetSize(new VectorUInt32(size))
    source.SetSigma(new VectorDouble(sigma))
    Raw(source.Execute())

let constantImage (size: int list) (value: float) : Raw =
    let img = Raw.FromSize(size) 
    img + value

/// Mathematical morphology
/// Binary erosion
let binaryErode (radius: uint) (foreground: float) (img: Raw) : Raw =
    let filter = new BinaryErodeImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Raw(filter.Execute(img.Image))

/// Binary dilation
let binaryDilate (radius: uint) (foreground: float) (img: Raw) : Raw =
    let filter = new BinaryDilateImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Raw(filter.Execute(img.Image))

/// Binary opening (erode then dilate)
let binaryOpening (radius: uint) (foreground: float) (img: Raw) : Raw =
    let filter = new BinaryMorphologicalOpeningImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Raw(filter.Execute(img.Image))

/// Binary closing (dilate then erode)
let binaryClosing (radius: uint) (foreground: float) (img: Raw) : Raw =
    let filter = new BinaryMorphologicalClosingImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Raw(filter.Execute(img.Image))

/// Fill holes in binary regions
let binaryFillHoles (foreground: float) (img: Raw) : Raw =
    let filter = new BinaryFillholeImageFilter()
    filter.SetForegroundValue(foreground)
    Raw(filter.Execute(img.Image))

/// Connected components labeling
let connectedComponents (img: Raw) : Raw =
    let filter = new ConnectedComponentImageFilter()
    Raw(filter.Execute(img.Image))

/// Relabel components by size, optionally remove small objects
let relabelComponents (minSize: uint) (img: Raw) : Raw =
    let filter = new RelabelComponentImageFilter()
    filter.SetMinimumObjectSize(uint64 minSize)
    Raw(filter.Execute(img.Image))

/// Compute label shape statistics and return a dictionary of results
let labelShapeStatistics (img: Raw) : LabelShapeStatisticsImageFilter =
    let stats = new LabelShapeStatisticsImageFilter()
    stats.Execute(img.Image)
    stats

/// Compute signed Maurer distance map (positive outside, negative inside)
let signedDistanceMap (insideIsPositive: bool) (squaredDistance: bool) (img: Raw) : Raw =
    let filter = new SignedMaurerDistanceMapImageFilter()
    filter.SetInsideIsPositive(insideIsPositive)
    filter.SetSquaredDistance(squaredDistance)
    Raw(filter.Execute(img.Image))

/// Morphological watershed (binary or grayscale)
let watershed (img: Raw) (level: float) (markWatershedLine: bool) : Raw =
    let filter = new MorphologicalWatershedImageFilter()
    filter.SetLevel(level)
    filter.SetMarkWatershedLine(markWatershedLine)
    Raw(filter.Execute(img.Image))

/// Histogram related functions
type ImageStats =
    { 
        Mean: float
        StdDev: float
        Minimum: float
        Maximum: float
        Sum: float
        Variance: float 
    }

let computeStats (img: Raw) : ImageStats =
    let stats = new StatisticsImageFilter()
    stats.Execute(img.Image)
    { 
        Mean = stats.GetMean()
        StdDev = stats.GetSigma()
        Minimum = stats.GetMinimum()
        Maximum = stats.GetMaximum()
        Sum = stats.GetSum()
        Variance = stats.GetVariance() 
    }

/// Otsu threshold
let otsuThreshold (img: Raw) : Raw =
    let filter = new OtsuThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Raw(filter.Execute(img.Image))

/// Otsu multiple thresholds (returns a label map)
let otsuMultiThreshold (img: Raw) (numThresholds: byte) : Raw =
    let filter = new OtsuMultipleThresholdsImageFilter()
    filter.SetNumberOfThresholds(numThresholds)
    Raw(filter.Execute(img.Image))

/// Entropy-based threshold
let RenyiEntropyThreshold (img: Raw) : Raw =
    let filter = new RenyiEntropyThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Raw(filter.Execute(img.Image))

/// Moments-based threshold
let momentsThreshold (img: Raw) : Raw =
    let filter = new MomentsThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Raw(filter.Execute(img.Image))
