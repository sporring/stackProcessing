module ImageFunctions
open Image
open Image.InternalHelpers

// --- basic manipulations ---
let squeeze (img: Image<'T>) : Image<'T> =
    let filter =  new itk.simple.ExtractImageFilter()
    let size = img.GetSize()
    let squeezedSize = size |> List.map (fun dim -> if dim = 1u then 0u else dim)
    filter.SetSize(squeezedSize |> toVectorUInt32)
    Image<'T>.ofSimpleITK(filter.Execute(img.Image))

let expand (dim: uint) (zero: 'S) (a: 'S list) = 
    List.concat [a; List.replicate (max 0 ((int dim)-a.Length)) zero]

let concatAlong (dim: uint) (a: Image<'T>) (b: Image<'T>) : Image<'T> =
    // perhaps use JoinSeriesImageFilter for speed.
    if a.GetDimension() <> b.GetDimension() then
        failwith "Images must have the same dimensionality."
    if a.GetNumberOfComponentsPerPixel() <> b.GetNumberOfComponentsPerPixel() then
        failwith "Images must have the same number of components."

    let sizeA = a.GetSize()
    let sizeB = b.GetSize()
    let sizeZipped = List.zip sizeA sizeB |> expand dim (1u,1u)

    sizeZipped
    |> List.iteri (fun i (da,db) -> 
        if i <> int dim && da <> db then
            failwithf "Image sizes differ at dimension %d: %d vs %d" i da db)
    let newSize = 
        sizeZipped |> List.mapi (fun i (a,b) -> if i <> int dim then a else a+b)

    // Create output image
    let itkId = fromType<'T>
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

// ----- basic mathematical helper functions -----
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

let inline makeBinaryImageOperatorWith
    (createFilter: unit -> 'Filter when 'Filter :> System.IDisposable)
    (setup: 'Filter -> unit)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image) 
    : (Image<'T> -> Image<'T> -> Image<'T>) =
    fun (a: Image<'T>) (b: Image<'T>) ->
        use filter = createFilter()
        setup filter
        Image<'T>.ofSimpleITK(invoke filter a.Image b.Image)

let inline makeBinaryImageOperator createFilter invoke = makeBinaryImageOperatorWith createFilter (fun _ -> ()) invoke

// Basic unary operators
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
(* // I'm waiting with proper support of complex values
let fft (img: Image<'T>)         = makeUnaryImageOperator (fun () -> new itk.simple.ForwardFFTImageFilter()) (fun f x -> f.Execute(x)) img
let ifft (img: Image<'T>)        = makeUnaryImageOperator (fun () -> new itk.simple.InverseFFTImageFilter()) (fun f x -> f.Execute(x)) img
let real (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToRealImageFilter()) (fun f x -> f.Execute(x)) img
let imag (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToImaginaryImageFilter()) (fun f x -> f.Execute(x)) img
let cabs (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToModulusImageFilter()) (fun f x -> f.Execute(x)) img
let carg (img: Image<'T>)          = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToPhaseImageFilter()) (fun f x -> f.Execute(x)) img
*)

type BoundaryCondition = ZERO_PAD | PERIODIC_PAD | ZERO_FLUX_NEUMANN_PAD
type OutputRegionMode = VALID | SAME

let convolve (outputRegion: OutputRegionMode) (boundaryCondition: BoundaryCondition) (normalize: bool) : Image<'T> -> Image<'T> -> Image<'T> =
    makeBinaryImageOperatorWith
        (fun () -> new itk.simple.ConvolutionImageFilter())
        (fun (f : itk.simple.ConvolutionImageFilter) ->
            f.SetOutputRegionMode (
                match outputRegion with
                    | VALID -> itk.simple.ConvolutionImageFilter.OutputRegionModeType.VALID
                    | _ -> itk.simple.ConvolutionImageFilter.OutputRegionModeType.SAME)
            f.SetBoundaryCondition (
                match boundaryCondition with
                    | ZERO_FLUX_NEUMANN_PAD -> itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_FLUX_NEUMANN_PAD
                    | PERIODIC_PAD -> itk.simple.ConvolutionImageFilter.BoundaryConditionType.PERIODIC_PAD 
                    | _ -> itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_PAD)
            f.SetNormalize (
                match normalize with
                    | true -> true
                    | _ -> false))
        (fun (f : itk.simple.ConvolutionImageFilter) img ker -> f.Execute(img, ker))

let conv (img: Image<'T>) (ker: Image<'T>) : Image<'T> = convolve SAME ZERO_PAD false img ker


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
let relabelComponents (minObjectSize: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.RelabelComponentImageFilter())
        (fun f -> f.SetMinimumObjectSize(uint64 minObjectSize))
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

/// Compute label shape statistics and return a dictionary of results
let labelShapeStatistics (img: Image<'T>) : Map<int64, LabelShapeStatistics> =
    use stats = new itk.simple.LabelShapeStatisticsImageFilter()
    stats.Execute(img.Image)
    stats.GetLabels()
    |> Seq.map (fun label ->
        let stats = {
            Label = label
            PhysicalSize = stats.GetPhysicalSize(label)
            Centroid = stats.GetCentroid(label) |>  fromVectorFloat64
            BoundingBox = stats.GetBoundingBox(label)|> fromVectorUInt32
            Elongation = stats.GetElongation(label)
            Flatness = stats.GetFlatness(label)
            FeretDiameter = stats.GetFeretDiameter(label)
            EquivalentEllipsoidDiameter = stats.GetEquivalentEllipsoidDiameter(label) |>  fromVectorFloat64
            EquivalentSphericalPerimeter = stats.GetEquivalentSphericalPerimeter(label)
            EquivalentSphericalRadius = stats.GetEquivalentSphericalRadius(label)
            Indexes = stats.GetIndexes(label) |> fromVectorUInt32
            NumberOfPixels = stats.GetNumberOfPixels(label)
            NumberOfPixelsOnBorder = stats.GetNumberOfPixelsOnBorder(label)
            OrientedBoundingBoxDirection = stats.GetOrientedBoundingBoxDirection(label) |>  fromVectorFloat64
            OrientedBoundingBoxOrigin = stats.GetOrientedBoundingBoxOrigin(label) |>  fromVectorFloat64
            OrientedBoundingBoxSize = stats.GetOrientedBoundingBoxSize(label) |>  fromVectorFloat64
            OrientedBoundingBoxVertices = stats.GetOrientedBoundingBoxVertices(label) |>  fromVectorFloat64
            Perimeter = stats.GetPerimeter(label)
            PerimeterOnBorder = stats.GetPerimeterOnBorder(label)
            PerimeterOnBorderRatio = stats.GetPerimeterOnBorderRatio(label)
            PrincipalAxes = stats.GetPrincipalAxes(label) |>  fromVectorFloat64
            PrincipalMoments = stats.GetPrincipalMoments(label) |>  fromVectorFloat64
            Region = stats.GetRegion(label) |> fromVectorUInt32
            RLEIndexes = stats.GetRLEIndexes(label) |> fromVectorUInt32
            Roundness = stats.GetRoundness(label)
        }
        label, stats
    )
    |> Map.ofSeq

/// Compute signed Maurer distance map (positive outside, negative inside)
// ApproximateSignedDistanceMapImageFilter has an error. These cast an exception: 
//   [[1uy;0uy;1uy;0uy;1uy;0uy]] and [[1uy;0uy;1uy;0uy;0uy;0uy]]
// but these don't
//   [[1uy;0uy;1uy;1uy;1uy;0uy]] and [[1uy;0uy;0uy;0uy;1uy;0uy]]
let signedDistanceMap (inside: uint8) (outside: uint8) (img: Image<uint8>) : Image<float> =
    let f = new itk.simple.ApproximateSignedDistanceMapImageFilter()
    f.SetInsideValue(float inside)
    f.SetOutsideValue(float outside)
    Image<float>.ofSimpleITK(f.Execute(img.Image))

/// Morphological watershed (binary or grayscale)
let watershed (level: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.MorphologicalWatershedImageFilter())
        (fun f ->
            f.SetLevel(level)
            f.SetMarkWatershedLine(false))
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

let unique (img: Image<'T>) : 'T list when 'T : comparison =
    img.toArray2D()            // 'T [,]
    |> Seq.cast<'T>            // flatten to a seq<'T>
    |> Set.ofSeq               // remove duplicates
    |> Set.toList              // back to an ordered list
    
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
        let coord = uint32 index[axis]
        let idxVec = toVectorUInt32 (index |> List.map uint)
        image.SetPixelAsUInt32(idxVec, coord))

    Image<uint32>.ofSimpleITK(image)

let histogram (image: Image<'T>) : Map<'T, uint64> =
    let size = image.GetSize() |> List.map int
    let dim = image.GetDimension()
    let flat = 
        match dim with
            | 1u ->
                seq { 
                    for i0 in [0..(size[0] - 1)] do 
                        yield image[i0] }
            | 2u ->
                seq { 
                    for i0 in [0..(size[0] - 1)] do 
                        for i1 in [0..(size[1] - 1)] do 
                            yield image[i0,i1] }
            | 3u ->
                seq { 
                    for i0 in [0..(size[0] - 1)] do 
                        for i1 in [0..(size[1] - 1)] do 
                            for i2 in [0..(size[2] - 1)] do 
                               yield image[i0, i1, i2] }
            | 4u ->
                seq { 
                    for i0 in [0..(size[0] - 1)] do 
                        for i1 in [0..(size[1] - 1)] do 
                            for i2 in [0..(size[2] - 1)] do 
                                for i3 in [0..(size[2] - 1)] do 
                                    yield image[i0, i1, i2, i3] }
            | _ -> failwith $"Unsupported dimensionality {dim}"

    flat 
    |> Seq.fold 
        (fun acc elm -> Map.change elm (fun vopt -> match vopt with Some v -> Some (v+1uL) | None -> Some (1uL)) acc)
        Map.empty<'T, uint64>


let addNormalNoise (mean: float) (stddev: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.AdditiveGaussianNoiseImageFilter())
        (fun f -> 
            f.SetMean(mean)
            f.SetStandardDeviation(stddev))
        (fun f x -> f.Execute(x))

let threshold (lower: float) (upper: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        (fun () -> new itk.simple.BinaryThresholdImageFilter())
        (fun f -> 
            f.SetLowerThreshold lower
            f.SetUpperThreshold upper)
        (fun f x -> f.Execute(x))

let stack (images: Image<'T> list) : Image<'T> =
    if images.Length = 0 then
        failwith "stack: Cannot stack an empty list of image"
    let dim = max 3u (List.fold (fun acc (img:Image<'T>) -> max 0u (img.GetDimension())) 0u images)
    if dim = 0u then
        failwith "stack: Cannot stack a list of empty image"
    let cmp = List.map (fun (i:Image<'T>) -> i.GetNumberOfComponentsPerPixel()) images
    if (List.distinct cmp).Length > 1 then
        failwith "Images must have the same number of components."

    let sizes = List.map (fun (img:Image<'T>) -> expand dim 1u (img.GetSize())) images
    let sz0 = sizes[0]
    List.iteri (
        fun i szi -> (List.iteri (
            fun j szij -> 
                if j <> 2 && szij <> sz0[j] then
                    failwith "All images must have same dimensions except along the 3rd axis")
            sizes[i]))
         sizes
    let newSize = sizes |> List.reduce (fun acc sz -> List.init (int dim) (fun i -> if i = 2 then acc[2]+sz[2] else acc[i]))
    let itkId = fromType<'T>

    let paste = new itk.simple.PasteImageFilter()
    let mutable sitkImg = new itk.simple.Image(newSize |> toVectorUInt32, itkId, cmp[0])
    let sitkImages = images |> List.map (fun (img: Image<'T>)->img.Image)   
    let mutable z = 0
    List.iter 
        (fun (img: itk.simple.Image) -> 
            let offset = List.init (int dim) (fun i -> if i = 2 then z else 0)
            let szi = img.GetSize() |> fromVectorUInt32 |> expand dim 1u |> toVectorUInt32
            paste.SetDestinationIndex(offset |> toVectorInt32)
            paste.SetSourceSize(szi)
            sitkImg <- paste.Execute(sitkImg, img)
            z <- z + (int szi[2]))
        sitkImages
    Image<'T>.ofSimpleITK(sitkImg)

let extractSub (topLeft : uint list) (bottomRight: uint list) (img: Image<'T>) : Image<'T> =
    if topLeft.Length <> bottomRight.Length then
        failwith $"extractSub: topLeft and bottomRight lists must have equal lengths ({topLeft} vs {bottomRight})"
    if img.GetDimension() <> uint topLeft.Length then
        failwith $"extractSub: indices and image size does not match"
    let sz = List.zip topLeft bottomRight |> List.map (fun (a,b) -> b-a + 1u)
    if List.exists (fun a -> a <  1u) sz then
        failwith $"extractSub: no index of bottomRight must be smaller than topLeft  ({topLeft} vs {bottomRight})"

    let extractor = new itk.simple.ExtractImageFilter()
    extractor.SetSize(sz |> toVectorUInt32)
    extractor.SetIndex( topLeft |> List.map int |> toVectorInt32)
    Image<'T>.ofSimpleITK(extractor.Execute(img.Image))

let extractSlice (z: uint) (img: Image<'T>) =
    if img.GetDimension() <> 3u then
        failwith $"extractSlice: image must be 3D"
    let sz = img.GetSize()
    extractSub [0u; 0u; z] [sz[0]-1u; sz[1]-1u; z] img |> squeeze

type FileInfo = { dimensions: uint; size: uint64 list; componentType: string; numberOfComponents: uint}
let getFileInfo (filename: string) : FileInfo =
    use reader = new itk.simple.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    {
        dimensions = reader.GetDimension(); 
        size = reader.GetSize() |> fromVectorUInt64; 
        componentType = reader.GetPixelID() |> pixelIdToString
        numberOfComponents = reader.GetNumberOfComponents()
    }
