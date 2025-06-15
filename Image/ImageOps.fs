module ImageClass.ImageOps
open ImageClass

/// Module with inline operator overloads for Image
let toVectorUInt32 (lst: int list) =
    let v = new itk.simple.VectorUInt32()
    lst |> List.iter (uint32 >> v.Add)
    v

let toVectorDouble (lst: float list) =
    let v = new itk.simple.VectorDouble()
    lst |> List.iter v.Add
    v

let fromVectorUInt32 (v: itk.simple.VectorUInt32) : uint list =
    v |> Seq.map uint |> Seq.toList

let fromVectorDouble (v: itk.simple.VectorDouble) : float list =
    v |> Seq.toList
    
// ----- basic mathematical functions -----
let inline makeUnaryImageOperator
    (filter: 'Filter)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image)
    : Image -> Image =
    fun (img: Image) ->
        Image(invoke filter img.Image)

let inline makeBinaryImageOperator
    (filter: 'Filter)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image)
    : Image -> Image -> Image =
    fun a b -> Image(invoke filter a.Image b.Image)

let inline abs (img: Image)    = makeUnaryImageOperator (new itk.simple.AbsImageFilter())    (fun f x -> f.Execute(x)) img
let inline log (img: Image)    = makeUnaryImageOperator (new itk.simple.LogImageFilter())    (fun f x -> f.Execute(x)) img
let inline log10 (img: Image)  = makeUnaryImageOperator (new itk.simple.Log10ImageFilter())  (fun f x -> f.Execute(x)) img
let inline exp (img: Image)    = makeUnaryImageOperator (new itk.simple.ExpImageFilter())    (fun f x -> f.Execute(x)) img
let inline sqrt (img: Image)   = makeUnaryImageOperator (new itk.simple.SqrtImageFilter())   (fun f x -> f.Execute(x)) img
let inline square (img: Image) = makeUnaryImageOperator (new itk.simple.SquareImageFilter()) (fun f x -> f.Execute(x)) img
let inline sin (img: Image)    = makeUnaryImageOperator (new itk.simple.SinImageFilter())    (fun f x -> f.Execute(x)) img
let inline cos (img: Image)    = makeUnaryImageOperator (new itk.simple.CosImageFilter())    (fun f x -> f.Execute(x)) img
let inline tan (img: Image)    = makeUnaryImageOperator (new itk.simple.TanImageFilter())    (fun f x -> f.Execute(x)) img
let inline asin (img: Image)   = makeUnaryImageOperator (new itk.simple.AsinImageFilter())   (fun f x -> f.Execute(x)) img
let inline acos (img: Image)   = makeUnaryImageOperator (new itk.simple.AcosImageFilter())   (fun f x -> f.Execute(x)) img
let inline atan (img: Image)   = makeUnaryImageOperator (new itk.simple.AtanImageFilter())   (fun f x -> f.Execute(x)) img
let inline round (img: Image)  = makeUnaryImageOperator (new itk.simple.RoundImageFilter())  (fun f x -> f.Execute(x)) img
// ----- basic image analysis functions -----
let fft3D (img: Image)         = makeUnaryImageOperator (new itk.simple.ForwardFFTImageFilter()) (fun f x -> f.Execute(x)) img
let ifft3D (img: Image)        = makeUnaryImageOperator (new itk.simple.InverseFFTImageFilter()) (fun f x -> f.Execute(x)) img
let real (img: Image)          = makeUnaryImageOperator (new itk.simple.ComplexToRealImageFilter()) (fun f x -> f.Execute(x)) img
let imag (img: Image)          = makeUnaryImageOperator (new itk.simple.ComplexToImaginaryImageFilter()) (fun f x -> f.Execute(x)) img
let cabs (img: Image)          = makeUnaryImageOperator (new itk.simple.ComplexToModulusImageFilter()) (fun f x -> f.Execute(x)) img
let carg (img: Image)          = makeUnaryImageOperator (new itk.simple.ComplexToPhaseImageFilter()) (fun f x -> f.Execute(x)) img
let convolve (kern: Image) (img: Image)
    = makeBinaryImageOperator (new itk.simple.ConvolutionImageFilter()) (fun f a b -> f.Execute(a, b)) kern img

/// Gaussian kernel convolution
/// Isotropic Discrete Gaussian blur
let discreteGaussian (input: Image) (sigma: float) : Image =
    let filter = new itk.simple.DiscreteGaussianImageFilter()
    filter.SetVariance(sigma * sigma)
    Image(filter.Execute(input.Image))

/// Recursive Gaussian blur in a specific direction (0 = x, 1 = y, 2 = z)
let recursiveGaussian (input: Image) (sigma: float) (direction: uint) : Image =
    let filter = new itk.simple.RecursiveGaussianImageFilter()
    filter.SetSigma(sigma)
    filter.SetDirection(direction)
    Image(filter.Execute(input.Image))

/// Laplacian of Gaussian convolution
let laplacianConvolve (input: Image) (sigma: float) : Image =
    let filter = new itk.simple.LaplacianRecursiveGaussianImageFilter()
    filter.SetSigma(sigma)
    Image(filter.Execute(input.Image))

/// Gradient convolution using Derivative filter
let gradientXConvolve (input: Image) (order: uint32) : Image =
    let filter = new itk.simple.DerivativeImageFilter()
    filter.SetDirection(0u) // X axis
    filter.SetOrder(order)
    Image(filter.Execute(input.Image))

let gradientYConvolve (input: Image) (order: uint32) : Image =
    let filter = new itk.simple.DerivativeImageFilter()
    filter.SetDirection(1u) // Y axis
    filter.SetOrder(order)
    Image(filter.Execute(input.Image))

let gradientZConvolve (input: Image) (order: uint32) : Image =
    let filter = new itk.simple.DerivativeImageFilter()
    filter.SetDirection(2u) // Z axis
    filter.SetOrder(order)
    Image(filter.Execute(input.Image))

/// Image sources
/// Create a grid pattern image
let gridImage (size: int list) (spacing: float list) (origin: float list) : Image =
    let source = new itk.simple.GridImageSource()
    source.SetSize(new itk.simple.VectorUInt32(size))
    source.SetSpacing(new itk.simple.VectorDouble(spacing))
    source.SetOrigin(new itk.simple.VectorDouble(origin))
    Image(source.Execute())

/// Create a Gabor pattern image
let gaborImage (size: int list) (sigma: float list) (frequency: float) : Image =
    let source = new itk.simple.GaborImageSource()
    source.SetSize(new itk.simple.VectorUInt32(size))
    source.SetSigma(new itk.simple.VectorDouble(sigma))
    source.SetFrequency(frequency)
    Image(source.Execute())

/// Create a Gaussian pattern image
let gaussianImage (size: int list) (sigma: float list) : Image =
    let source = new itk.simple.GaussianImageSource()
    source.SetSize(new itk.simple.VectorUInt32(size))
    source.SetSigma(new itk.simple.VectorDouble(sigma))
    Image(source.Execute())

let constantImage (size: int list) (value: float) : Image =
    let img = Image.FromSize(size) 
    img + value

/// Mathematical morphology
/// Binary erosion
let binaryErode (radius: uint) (foreground: float) (img: Image) : Image =
    let filter = new itk.simple.BinaryErodeImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Image(filter.Execute(img.Image))

/// Binary dilation
let binaryDilate (radius: uint) (foreground: float) (img: Image) : Image =
    let filter = new itk.simple.BinaryDilateImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Image(filter.Execute(img.Image))

/// Binary opening (erode then dilate)
let binaryOpening (radius: uint) (foreground: float) (img: Image) : Image =
    let filter = new itk.simple.BinaryMorphologicalOpeningImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Image(filter.Execute(img.Image))

/// Binary closing (dilate then erode)
let binaryClosing (radius: uint) (foreground: float) (img: Image) : Image =
    let filter = new itk.simple.BinaryMorphologicalClosingImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Image(filter.Execute(img.Image))

/// Fill holes in binary regions
let binaryFillHoles (foreground: float) (img: Image) : Image =
    let filter = new itk.simple.BinaryFillholeImageFilter()
    filter.SetForegroundValue(foreground)
    Image(filter.Execute(img.Image))

/// Connected components labeling
let connectedComponents (img: Image) : Image =
    let filter = new itk.simple.ConnectedComponentImageFilter()
    Image(filter.Execute(img.Image))

/// Relabel components by size, optionally remove small objects
let relabelComponents (minSize: uint) (img: Image) : Image =
    let filter = new itk.simple.RelabelComponentImageFilter()
    filter.SetMinimumObjectSize(uint64 minSize)
    Image(filter.Execute(img.Image))

type LabelShapeStatistics = {
    Label: int64
    PhysicalSize: float
    Centroid: float list
    BoundingBox: uint32 list
    Elongation: float
    Flatness: float
    FeretDiameter: float
    EquivalentEllipsoidDiameter: float list
    EquivalentSphericalPerimeter: float
    EquivalentSphericalRadius: float
    Indexes: uint32 list
    NumberOfPixels: uint64
    NumberOfPixelsOnBorder: uint64
    OrientedBoundingBoxDirection: float list
    OrientedBoundingBoxOrigin: float list
    OrientedBoundingBoxSize: float list
    OrientedBoundingBoxVertices: float list
    Perimeter: float
    PerimeterOnBorder: float
    PerimeterOnBorderRatio: float
    PrincipalAxes: float list
    PrincipalMoments: float list
    Region: uint32 list
    RLEIndexes: uint32 list
    Roundness: float
}

let shapeStatsMap (filter: itk.simple.LabelShapeStatisticsImageFilter) : Map<int64, LabelShapeStatistics> =
    filter.GetLabels()
    |> Seq.map (fun label ->
        let stats = {
            Label = label
            PhysicalSize = filter.GetPhysicalSize(label)
            Centroid = filter.GetCentroid(label) |> fromVectorDouble
            BoundingBox = filter.GetBoundingBox(label)|> fromVectorUInt32
            Elongation = filter.GetElongation(label)
            Flatness = filter.GetFlatness(label)
            FeretDiameter = filter.GetFeretDiameter(label)
            EquivalentEllipsoidDiameter = filter.GetEquivalentEllipsoidDiameter(label) |> fromVectorDouble
            EquivalentSphericalPerimeter = filter.GetEquivalentSphericalPerimeter(label)
            EquivalentSphericalRadius = filter.GetEquivalentSphericalRadius(label)
            Indexes = filter.GetIndexes(label) |> fromVectorUInt32
            NumberOfPixels = filter.GetNumberOfPixels(label)
            NumberOfPixelsOnBorder = filter.GetNumberOfPixelsOnBorder(label)
            OrientedBoundingBoxDirection = filter.GetOrientedBoundingBoxDirection(label) |> fromVectorDouble
            OrientedBoundingBoxOrigin = filter.GetOrientedBoundingBoxOrigin(label) |> fromVectorDouble
            OrientedBoundingBoxSize = filter.GetOrientedBoundingBoxSize(label) |> fromVectorDouble
            OrientedBoundingBoxVertices = filter.GetOrientedBoundingBoxVertices(label) |> fromVectorDouble
            Perimeter = filter.GetPerimeter(label)
            PerimeterOnBorder = filter.GetPerimeterOnBorder(label)
            PerimeterOnBorderRatio = filter.GetPerimeterOnBorderRatio(label)
            PrincipalAxes = filter.GetPrincipalAxes(label) |> fromVectorDouble
            PrincipalMoments = filter.GetPrincipalMoments(label) |> fromVectorDouble
            Region = filter.GetRegion(label) |> fromVectorUInt32
            RLEIndexes = filter.GetRLEIndexes(label) |> fromVectorUInt32
            Roundness = filter.GetRoundness(label)
        }
        label, stats
    )
    |> Map.ofSeq

/// Compute label shape statistics and return a dictionary of results
let labelShapeStatistics (img: Image) : Map<int64, LabelShapeStatistics> =
    let stats = new itk.simple.LabelShapeStatisticsImageFilter()
    stats.Execute(img.Image)
    shapeStatsMap(stats)

/// Compute signed Maurer distance map (positive outside, negative inside)
let signedDistanceMap (insideIsPositive: bool) (squaredDistance: bool) (img: Image) : Image =
    let filter = new itk.simple.SignedMaurerDistanceMapImageFilter()
    filter.SetInsideIsPositive(insideIsPositive)
    filter.SetSquaredDistance(squaredDistance)
    Image(filter.Execute(img.Image))

/// Morphological watershed (binary or grayscale)
let watershed (img: Image) (level: float) (markWatershedLine: bool) : Image =
    let filter = new itk.simple.MorphologicalWatershedImageFilter()
    filter.SetLevel(level)
    filter.SetMarkWatershedLine(markWatershedLine)
    Image(filter.Execute(img.Image))

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

let computeStats (img: Image) : ImageStats =
    let stats = new itk.simple.StatisticsImageFilter()
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
let otsuThreshold (img: Image) : Image =
    let filter = new itk.simple.OtsuThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image(filter.Execute(img.Image))

/// Otsu multiple thresholds (returns a label map)
let otsuMultiThreshold (img: Image) (numThresholds: byte) : Image =
    let filter = new itk.simple.OtsuMultipleThresholdsImageFilter()
    filter.SetNumberOfThresholds(numThresholds)
    Image(filter.Execute(img.Image))

/// Entropy-based threshold
let RenyiEntropyThreshold (img: Image) : Image =
    let filter = new itk.simple.RenyiEntropyThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image(filter.Execute(img.Image))

/// Moments-based threshold
let momentsThreshold (img: Image) : Image =
    let filter = new itk.simple.MomentsThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image(filter.Execute(img.Image))
