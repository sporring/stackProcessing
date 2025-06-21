module ImageFunctions
open ImageClass

// ----- basic mathematical functions -----
let inline makeUnaryImageOperatorWith
    (createFilter: unit -> 'Filter when 'Filter :> System.IDisposable)
    (setup: 'Filter -> unit)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image)
    : (Image<'T> -> Image<'T>) =
    fun (img: Image<'T>) ->
        use filter = createFilter()
        setup filter
        Image<'T>.ofSimpleITK(invoke filter img.Image)

let inline makeUnaryImageOperator createFilter invoke = makeUnaryImageOperatorWith createFilter (fun _ -> ()) invoke
(*
let inline makeUnaryImageOperator
    (filter: 'Filter)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image)
    : Image<'T> -> Image<'T> =
    fun (img: Image<'T>) ->
        Image<'T>(invoke filter img.Image)
*)

let inline makeImageSourceOperatorWith
    (createSource: unit -> ^Source when ^Source :> System.IDisposable)
    (setup: ^Source -> unit)
    (invoke: ^Source -> itk.simple.Image)
    : Image<'T> =
    use source = createSource()
    setup source
    Image<'T>.ofSimpleITK(invoke source)

let inline makeImageSourceOperator createSource invoke = makeImageSourceOperatorWith createSource (fun _ -> ()) invoke

let inline makeBinaryImageOperator
    (filter: 'Filter)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image)
    : Image<'T> -> Image<'T> -> Image<'T> =
    fun a b -> Image<'T>.ofSimpleITK(invoke filter a.Image b.Image)

let inline abs (img: Image<'T>)    = makeUnaryImageOperator (fun () -> new itk.simple.AbsImageFilter())    (fun f x -> f.Execute(x)) img
let inline log (img: Image<'T>)    = makeUnaryImageOperator (fun () -> new itk.simple.LogImageFilter())    (fun f x -> f.Execute(x)) img
let inline log10 (img: Image<'T>)  = makeUnaryImageOperator (fun () -> new itk.simple.Log10ImageFilter())  (fun f x -> f.Execute(x)) img
let inline exp (img: Image<'T>)    = makeUnaryImageOperator (fun () -> new itk.simple.ExpImageFilter())    (fun f x -> f.Execute(x)) img
let inline sqrt (img: Image<'T>)   = makeUnaryImageOperator (fun () -> new itk.simple.SqrtImageFilter())   (fun f x -> f.Execute(x)) img
let inline square (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.SquareImageFilter()) (fun f x -> f.Execute(x)) img
let inline sin (img: Image<'T>)    = makeUnaryImageOperator (fun () -> new itk.simple.SinImageFilter())    (fun f x -> f.Execute(x)) img
let inline cos (img: Image<'T>)    = makeUnaryImageOperator (fun () -> new itk.simple.CosImageFilter())    (fun f x -> f.Execute(x)) img
let inline tan (img: Image<'T>)    = makeUnaryImageOperator (fun () -> new itk.simple.TanImageFilter())    (fun f x -> f.Execute(x)) img
let inline asin (img: Image<'T>)   = makeUnaryImageOperator (fun () -> new itk.simple.AsinImageFilter())   (fun f x -> f.Execute(x)) img
let inline acos (img: Image<'T>)   = makeUnaryImageOperator (fun () -> new itk.simple.AcosImageFilter())   (fun f x -> f.Execute(x)) img
let inline atan (img: Image<'T>)   = makeUnaryImageOperator (fun () -> new itk.simple.AtanImageFilter())   (fun f x -> f.Execute(x)) img
let inline round (img: Image<'T>)  = makeUnaryImageOperator (fun () -> new itk.simple.RoundImageFilter())  (fun f x -> f.Execute(x)) img
// ----- basic image analysis functions -----
let fft (img: Image<'T>)         = makeUnaryImageOperator (fun () -> new itk.simple.ForwardFFTImageFilter()) (fun f x -> f.Execute(x)) img
let ifft (img: Image<'T>)        = makeUnaryImageOperator (fun () -> new itk.simple.InverseFFTImageFilter()) (fun f x -> f.Execute(x)) img
let real (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToRealImageFilter()) (fun f x -> f.Execute(x)) img
let imag (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToImaginaryImageFilter()) (fun f x -> f.Execute(x)) img
let cabs (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToModulusImageFilter()) (fun f x -> f.Execute(x)) img
let carg (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToPhaseImageFilter()) (fun f x -> f.Execute(x)) img
let convolve (kern: Image<'T>) (img: Image<'T>)
    = makeBinaryImageOperator (new itk.simple.ConvolutionImageFilter()) (fun f a b -> f.Execute(a, b)) kern img

// --- basic manipulations ---
let squeeze (img: Image<'T>) : Image<'T> =
    let filter =  new itk.simple.ExtractImageFilter()
    let size = img.GetSize()
    let squeezedSize = size |> List.map (fun dim -> if dim = 1u then 0u else dim)
    filter.SetSize(squeezedSize |> toVectorUInt32)
    Image<'T>.ofSimpleITK(filter.Execute(img.Image))

let concatAlong (dim: uint) (a: Image<'T>) (b: Image<'T>) : Image<'T> =
    if a.GetDimension() <> b.GetDimension() then
        failwith "Images must have the same dimensionality."
    if a.GetNumberOfComponentsPerPixel() <> b.GetNumberOfComponentsPerPixel() then
        failwith "Images must have the same number of components."

    let sizeA = a.GetSize()
    let sizeB = b.GetSize()
    let sizeZipped = List.concat [List.zip sizeA sizeB; List.replicate (max 0 ((int dim)-sizeA.Length+1)) (1u,1u)]
    sizeZipped
    |> List.iteri (fun i (da,db) -> 
        if i <> int dim && da <> db then
            failwithf "Image sizes differ at dimension %d: %d vs %d" i da db)
    let newSize = 
        sizeZipped |> List.mapi (fun i (a,b) -> if i <> int dim then a else a+b)

    // Create output image
    let pt = fromType<'T>
    let itkId = pt.ToSimpleITK()
    let outImg = new itk.simple.Image(newSize |> toVectorUInt32, itkId, a.GetNumberOfComponentsPerPixel())

    let paste = new itk.simple.PasteImageFilter()
    // Paste image A at origin
    paste.SetDestinationIndex(List.replicate newSize.Length 0 |> toVectorInt32)
    paste.SetSourceSize(a.GetSize() |> toVectorUInt32)
    let outWithA = paste.Execute(outImg, a.Image)

    // Paste image B at offset
    let offset = 
        sizeZipped |> List.mapi (fun i (a,b) -> if i <> int dim then 0 else int a)
    paste.SetDestinationIndex(offset |> toVectorInt32)
    paste.SetSourceSize(b.GetSize() |> toVectorUInt32)
    let outWithBoth = paste.Execute(outWithA, b.Image)

    Image<'T>.ofSimpleITK(outWithBoth)

/// Gaussian kernel convolution
/// Isotropic Discrete Gaussian blur
let discreteGaussian (sigma: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.DiscreteGaussianImageFilter())
        (fun filter -> filter.SetVariance(sigma * sigma))
        (fun filter input -> filter.Execute(input))

/// Recursive Gaussian blur in a specific direction (0 = x, 1 = y, 2 = z)
let recursiveGaussian (sigma: float) (direction: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.RecursiveGaussianImageFilter())
        (fun f ->
            f.SetSigma(sigma)
            f.SetDirection(direction))
        (fun f x -> f.Execute(x))


/// Laplacian of Gaussian convolution
let laplacianConvolve (sigma: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.LaplacianRecursiveGaussianImageFilter())
        (fun f -> f.SetSigma(sigma))
        (fun f x -> f.Execute(x))


/// Gradient convolution using Derivative filter
let gradientConvolve (direction: uint) (order: uint32) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.DerivativeImageFilter())
        (fun f ->
            f.SetDirection(direction)
            f.SetOrder(order))
        (fun f x -> f.Execute(x))

/// Image sources
/// Create a grid pattern image
let gridImage (size: uint list) (spacing: float list) (origin: float list) : Image<'T> =
    makeImageSourceOperatorWith
        (fun () -> new itk.simple.GridImageSource())
        (fun s ->
            s.SetSize(toVectorUInt32 size)
            s.SetSpacing( toVectorFloat64 spacing)
            s.SetOrigin( toVectorFloat64 origin))
        (fun s -> s.Execute())


/// Create a Gabor pattern image
let gaborImage (size: uint list) (sigma: float list) (frequency: float) : Image<'T> =
    makeImageSourceOperatorWith
        (fun () -> new itk.simple.GaborImageSource())
        (fun s ->
            s.SetSize(toVectorUInt32 size)
            s.SetSigma( toVectorFloat64 sigma)
            s.SetFrequency(frequency))
        (fun s -> s.Execute())

/// Create a Gaussian pattern image
let gaussianImage (size: uint list) (sigma: float list) : Image<'T> =
    makeImageSourceOperatorWith
        (fun () -> new itk.simple.GaussianImageSource())
        (fun s ->
            s.SetSize(toVectorUInt32 size)
            s.SetSigma( toVectorFloat64 sigma))
        (fun s -> s.Execute())

/// Mathematical morphology
/// Binary erosion
let binaryErode (radius: uint) (foreground: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.BinaryErodeImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetForegroundValue(foreground))
        (fun f x -> f.Execute(x))


/// Binary dilation
let binaryDilate (radius: uint) (foreground: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.BinaryDilateImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetForegroundValue(foreground))
        (fun f x -> f.Execute(x))


/// Binary opening (erode then dilate)
let binaryOpening (radius: uint) (foreground: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.BinaryMorphologicalOpeningImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetForegroundValue(foreground))
        (fun f x -> f.Execute(x))


/// Binary closing (dilate then erode)
let binaryClosing (radius: uint) (foreground: float) (img: Image<'T>) : Image<'T> =
    use filter = new itk.simple.BinaryMorphologicalClosingImageFilter()
    filter.SetKernelRadius(radius)
    filter.SetForegroundValue(foreground)
    Image<'T>.ofSimpleITK(filter.Execute(img.Image))

/// Fill holes in binary regions
let binaryFillHoles (foreground: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.BinaryFillholeImageFilter())
        (fun f -> f.SetForegroundValue(foreground))
        (fun f x -> f.Execute(x))

/// Connected components labeling
// Currying and generic arguments causes value restriction error
let connectedComponents (img : Image<'T>) : Image<'T> =
    use filter = new itk.simple.ConnectedComponentImageFilter()
    Image<'T>.ofSimpleITK(filter.Execute(img.Image))

/// Relabel components by size, optionally remove small objects
let relabelComponents (minSize: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.RelabelComponentImageFilter())
        (fun f -> f.SetMinimumObjectSize(uint64 minSize))
        (fun f x -> f.Execute(x))

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
            Centroid = filter.GetCentroid(label) |>  fromVectorFloat64
            BoundingBox = filter.GetBoundingBox(label)|> fromVectorUInt32
            Elongation = filter.GetElongation(label)
            Flatness = filter.GetFlatness(label)
            FeretDiameter = filter.GetFeretDiameter(label)
            EquivalentEllipsoidDiameter = filter.GetEquivalentEllipsoidDiameter(label) |>  fromVectorFloat64
            EquivalentSphericalPerimeter = filter.GetEquivalentSphericalPerimeter(label)
            EquivalentSphericalRadius = filter.GetEquivalentSphericalRadius(label)
            Indexes = filter.GetIndexes(label) |> fromVectorUInt32
            NumberOfPixels = filter.GetNumberOfPixels(label)
            NumberOfPixelsOnBorder = filter.GetNumberOfPixelsOnBorder(label)
            OrientedBoundingBoxDirection = filter.GetOrientedBoundingBoxDirection(label) |>  fromVectorFloat64
            OrientedBoundingBoxOrigin = filter.GetOrientedBoundingBoxOrigin(label) |>  fromVectorFloat64
            OrientedBoundingBoxSize = filter.GetOrientedBoundingBoxSize(label) |>  fromVectorFloat64
            OrientedBoundingBoxVertices = filter.GetOrientedBoundingBoxVertices(label) |>  fromVectorFloat64
            Perimeter = filter.GetPerimeter(label)
            PerimeterOnBorder = filter.GetPerimeterOnBorder(label)
            PerimeterOnBorderRatio = filter.GetPerimeterOnBorderRatio(label)
            PrincipalAxes = filter.GetPrincipalAxes(label) |>  fromVectorFloat64
            PrincipalMoments = filter.GetPrincipalMoments(label) |>  fromVectorFloat64
            Region = filter.GetRegion(label) |> fromVectorUInt32
            RLEIndexes = filter.GetRLEIndexes(label) |> fromVectorUInt32
            Roundness = filter.GetRoundness(label)
        }
        label, stats
    )
    |> Map.ofSeq

/// Compute label shape statistics and return a dictionary of results
let labelShapeStatistics (img: Image<'T>) : Map<int64, LabelShapeStatistics> =
    use stats = new itk.simple.LabelShapeStatisticsImageFilter()
    stats.Execute(img.Image)
    shapeStatsMap(stats)


/// Compute signed Maurer distance map (positive outside, negative inside)
let signedDistanceMap (insideIsPositive: bool) (squaredDistance: bool) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.SignedMaurerDistanceMapImageFilter())
        (fun f ->
            f.SetInsideIsPositive(insideIsPositive)
            f.SetSquaredDistance(squaredDistance))
        (fun f x -> f.Execute(x))


/// Morphological watershed (binary or grayscale)
let watershed (level: float) (markWatershedLine: bool) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.MorphologicalWatershedImageFilter())
        (fun f ->
            f.SetLevel(level)
            f.SetMarkWatershedLine(markWatershedLine))
        (fun f x -> f.Execute(x))


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

let computeStats (img: Image<'T>) : ImageStats =
    use stats = new itk.simple.StatisticsImageFilter()
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
// Currying and generic arguments causes value restriction error
let otsuThreshold (img: Image<'T>) : Image<'T> =
    use filter = new itk.simple.OtsuThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image<'T>.ofSimpleITK(filter.Execute(img.Image))


/// Otsu multiple thresholds (returns a label map)
let otsuMultiThreshold (numThresholds: byte) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.OtsuMultipleThresholdsImageFilter())
        (fun f -> f.SetNumberOfThresholds(numThresholds))
        (fun f x -> f.Execute(x))

/// Entropy-based threshold
let renyiEntropyThreshold (img: Image<'T>) : Image<'T> =
    use filter = new itk.simple.RenyiEntropyThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image<'T>.ofSimpleITK(filter.Execute(img.Image))


/// Moments-based threshold
let momentsThreshold (img: Image<'T>) : Image<'T> =
    use filter = new itk.simple.MomentsThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image<'T>.ofSimpleITK(filter.Execute(img.Image))

/// Coordinate fields
// Cannot get TransformToDisplacementFieldFilter to work, so making it by hand.
let generateCoordinateAxis (axis: int) (size: int list) : Image<uint32> =
    let dim = size.Length
    let image = new itk.simple.Image(toVectorUInt32 (size |> List.map uint), itk.simple.PixelIDValueEnum.sitkUInt32)

    // Recursive generator for all N-dimensional indices
    let rec generateIndices dims =
        match dims with
        | [] -> [ [] ]
        | d :: ds ->
            List.allPairs [0 .. d - 1] (generateIndices ds)
            |> List.map (fun (i, rest) -> i :: rest)

    // Write coordinate values along the selected axis
    generateIndices size
    |> List.iter (fun index ->
        let coord = uint32 index.[axis]
        let idxVec = toVectorUInt32 (index |> List.map uint)
        image.SetPixelAsUInt32(idxVec, coord))

    Image<uint32>.ofSimpleITK(image)
