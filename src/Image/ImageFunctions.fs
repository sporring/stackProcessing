module ImageFunctions
open Image
open Image.InternalHelpers

// Image constant arithmetic operations
let inline imageAddScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.AddImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(img.toSimpleITK(), float i),"imageAddScalar",img.index)
let inline scalarAddImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    imageAddScalar img i
let inline imageSubScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.SubtractImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(img.toSimpleITK(), float i),"imageSubScalar",img.index)
let inline scalarSubImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    use filter = new itk.simple.SubtractImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(float i, img.toSimpleITK()),"scalarSubImage",img.index)
let inline imageMulScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.MultiplyImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(img.toSimpleITK(), float i),"imageMulScalar",img.index)
let inline scalarMulImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    imageMulScalar img i
let inline imageDivScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.DivideImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(img.toSimpleITK(), float i),"imageDivScalar",img.index)
let inline scalarDivImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    use filter = new itk.simple.DivideImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(float i, img.toSimpleITK()),"scalarDivImage",img.index)

let inline imagePowScalar<'S when ^S : equality
                  and  ^S : (static member op_Explicit : ^S -> float)
                  > (img: Image<'S>, i: 'S) =
    use filter = new itk.simple.PowImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(img.toSimpleITK(), float i),"imagePowScalar",img.index)


let inline scalarPowImage<'S when ^S : equality
                  and  ^S : (static member op_Explicit : ^S -> float)
                  > (i: 'S, img: Image<'S>) =
    use filter = new itk.simple.PowImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(float i, img.toSimpleITK()),"scalarPowImage",img.index)

let inline sum (img: Image<'T>) : ^T
    when ^T : (static member ( + ) : ^T * ^T -> ^T)
    and  ^T : (static member Zero : ^T) =
    let zero  : ^T = LanguagePrimitives.GenericZero
    img |> Image.fold (fun acc elm -> acc+elm) zero

let inline prod (img: Image<'T>) : ^T
    when ^T : (static member ( * ) : ^T * ^T -> ^T)
    and  ^T : (static member One : ^T) =
    let one  : ^T = LanguagePrimitives.GenericOne
    img |> Image.fold (fun acc elm -> acc*elm) one

let dump (img: Image<'T>) : string =
    img |> Image.foldi (fun idxLst acc elm -> acc+(sprintf "%A -> %A; " idxLst elm)) ""



// --- basic manipulations ---
let squeeze (img: Image<'T>) : Image<'T> =
    use filter = new itk.simple.ExtractImageFilter()
    let size = img.GetSize()
    let squeezedSize = size |> List.map (fun dim -> if dim = 1u then 0u else dim)
    filter.SetSize(squeezedSize |> toVectorUInt32)
    Image<'T>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"squeeze",img.index)

let expand (dim: uint) (zero: 'S) (a: 'S list) = 
    List.concat [a; List.replicate (max 0 ((int dim)-a.Length)) zero]

let concatAlong (dim: uint) (a: Image<'T>) (b: Image<'T>) : Image<'T> =
    // perhaps use JoinSeriesImageFilter for speed.
    if a.GetDimensions() <> b.GetDimensions() then
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
    use outImg = new itk.simple.Image(newSize |> toVectorUInt32, itkId, a.GetNumberOfComponentsPerPixel())

    use paste = new itk.simple.PasteImageFilter()
    // Paste image A at origin
    paste.SetDestinationIndex(List.replicate newSize.Length 0 |> toVectorInt32)
    paste.SetSourceSize(a.GetSize() |> toVectorUInt32)
    use outWithA = paste.Execute(outImg, a.toSimpleITK())

    // Paste image B at offset
    let offset = 
        sizeZipped |> List.mapi (fun i (a,b) -> if i <> int dim then 0 else int a)
    paste.SetDestinationIndex(offset |> toVectorInt32)
    paste.SetSourceSize(b.GetSize() |> toVectorUInt32)
    let outWithBoth = paste.Execute(outWithA, b.toSimpleITK())

    Image<'T>.ofSimpleITK(outWithBoth,"concatAlong",a.index)

let constantPad2D<'T when 'T : equality> (padLower : uint list) (padUpper : uint list) (c : double) (img : Image<'T>) : Image<'T> =
    if padLower.Length <> 2 || padUpper.Length <> 2 then
        invalidArg "padLower/padUpper" "Both bounds must have length 2 for a 2‑D image."
    use filter = new itk.simple.ConstantPadImageFilter()
    filter.SetPadLowerBound(padLower |> toVectorUInt32)
    filter.SetPadUpperBound(padUpper |> toVectorUInt32)
    filter.SetConstant(c)
    let padded = filter.Execute(img.toSimpleITK())
    Image<'T>.ofSimpleITK(padded,"constantPad2D",img.index)


// ----- basic mathematical helper functions -----
let inline makeUnaryImageOperatorWith
    (name: string)
    (createFilter: unit -> 'Filter when 'Filter :> System.IDisposable)
    (setup: 'Filter -> unit)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image)
    : (Image<'T> -> Image<'S>) =
        fun (img: Image<'T>) ->
            use filter = createFilter()
            setup filter
            Image<'S>.ofSimpleITK(invoke filter (img.toSimpleITK()),name,img.index)

let inline makeUnaryImageOperator name createFilter invoke = makeUnaryImageOperatorWith name createFilter (fun _ -> ()) invoke

let inline makeBinaryImageOperatorWith
    (name: string)
    (createFilter: unit -> 'Filter when 'Filter :> System.IDisposable)
    (setup: 'Filter -> unit)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image) 
    : (Image<'T> -> Image<'T> -> Image<'T>) =
    fun (a: Image<'T>) (b: Image<'T>) ->
        use filter = createFilter()
        setup filter
        let img = Image<'T>.ofSimpleITK(invoke filter (a.toSimpleITK()) (b.toSimpleITK()),name,a.index)
        img

let makeBinaryImageOperator name createFilter invoke = makeBinaryImageOperatorWith name createFilter (fun _ -> ()) invoke

// Basic unary operators
let absImage (img: Image<'T>)    = makeUnaryImageOperator "absImage" (fun () -> new itk.simple.AbsImageFilter())    (fun f x -> f.Execute(x)) img
let logImage (img: Image<'T>)    = makeUnaryImageOperator "logImage" (fun () -> new itk.simple.LogImageFilter())    (fun f x -> f.Execute(x)) img
let log10Image (img: Image<'T>)  = makeUnaryImageOperator "log10Image" (fun () -> new itk.simple.Log10ImageFilter())  (fun f x -> f.Execute(x)) img
let expImage (img: Image<'T>)    = makeUnaryImageOperator "expImage" (fun () -> new itk.simple.ExpImageFilter())    (fun f x -> f.Execute(x)) img
let sqrtImage (img: Image<'T>)   = makeUnaryImageOperator "sqrtImage" (fun () -> new itk.simple.SqrtImageFilter())   (fun f x -> f.Execute(x)) img
let squareImage (img: Image<'T>) = makeUnaryImageOperator "squareImage" (fun () -> new itk.simple.SquareImageFilter()) (fun f x -> f.Execute(x)) img
let sinImage (img: Image<'T>)    = makeUnaryImageOperator "sinImage" (fun () -> new itk.simple.SinImageFilter())    (fun f x -> f.Execute(x)) img
let cosImage (img: Image<'T>)    = makeUnaryImageOperator "cosImage" (fun () -> new itk.simple.CosImageFilter())    (fun f x -> f.Execute(x)) img
let tanImage (img: Image<'T>)    = makeUnaryImageOperator "tanImage" (fun () -> new itk.simple.TanImageFilter())    (fun f x -> f.Execute(x)) img
let asinImage (img: Image<'T>)   = makeUnaryImageOperator "asinImage" (fun () -> new itk.simple.AsinImageFilter())   (fun f x -> f.Execute(x)) img
let acosImage (img: Image<'T>)   = makeUnaryImageOperator "acosImage" (fun () -> new itk.simple.AcosImageFilter())   (fun f x -> f.Execute(x)) img
let atanImage (img: Image<'T>)   = makeUnaryImageOperator "atanImage" (fun () -> new itk.simple.AtanImageFilter())   (fun f x -> f.Execute(x)) img
let roundImage (img: Image<'T>)  = makeUnaryImageOperator "roundImage" (fun () -> new itk.simple.RoundImageFilter())  (fun f x -> f.Execute(x)) img

let clampImage (lower: double) (upper: double) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "clampImage"
        (fun () -> new itk.simple.ClampImageFilter())
        (fun f ->
            f.SetLowerBound(lower)
            f.SetUpperBound(upper))
        (fun f x -> f.Execute(x))

let rescaleIntensity (outputMinimum: double) (outputMaximum: double) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "rescaleIntensity"
        (fun () -> new itk.simple.RescaleIntensityImageFilter())
        (fun f ->
            f.SetOutputMinimum(outputMinimum)
            f.SetOutputMaximum(outputMaximum))
        (fun f x -> f.Execute(x))

let intensityWindow (windowMinimum: double) (windowMaximum: double) (outputMinimum: double) (outputMaximum: double) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "intensityWindow"
        (fun () -> new itk.simple.IntensityWindowingImageFilter())
        (fun f ->
            f.SetWindowMinimum(windowMinimum)
            f.SetWindowMaximum(windowMaximum)
            f.SetOutputMinimum(outputMinimum)
            f.SetOutputMaximum(outputMaximum))
        (fun f x -> f.Execute(x))

let normalizeImage (img: Image<'T>) : Image<float> =
    use filter = new itk.simple.NormalizeImageFilter()
    Image<float>.ofSimpleITK(filter.Execute(img.toSimpleITK()), "normalizeImage", img.index)

let shiftScale (shift: double) (scale: double) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "shiftScale"
        (fun () -> new itk.simple.ShiftScaleImageFilter())
        (fun f ->
            f.SetShift(shift)
            f.SetScale(scale))
        (fun f x -> f.Execute(x))

let invertIntensity (maximum: double) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "invertIntensity"
        (fun () -> new itk.simple.InvertIntensityImageFilter())
        (fun f -> f.SetMaximum(maximum))
        (fun f x -> f.Execute(x))

let median (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "median"
        (fun () -> new itk.simple.MedianImageFilter())
        (fun f -> f.SetRadius(radius))
        (fun f x -> f.Execute(x))

let bilateral (domainSigma: double) (rangeSigma: double) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "bilateral"
        (fun () -> new itk.simple.BilateralImageFilter())
        (fun f ->
            f.SetDomainSigma(domainSigma)
            f.SetRangeSigma(rangeSigma))
        (fun f x -> f.Execute(x))

let gradientMagnitude (img: Image<'T>) : Image<'T> =
    makeUnaryImageOperator "gradientMagnitude" (fun () -> new itk.simple.GradientMagnitudeImageFilter()) (fun f x -> f.Execute(x)) img

let sobelEdge (img: Image<'T>) : Image<'T> =
    makeUnaryImageOperator "sobelEdge" (fun () -> new itk.simple.SobelEdgeDetectionImageFilter()) (fun f x -> f.Execute(x)) img

let laplacian (img: Image<'T>) : Image<'T> =
    makeUnaryImageOperator "laplacian" (fun () -> new itk.simple.LaplacianImageFilter()) (fun f x -> f.Execute(x)) img

let equalImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.EqualImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "equalImage", a.index)

let notEqualImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.NotEqualImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "notEqualImage", a.index)

let greaterImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.GreaterImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "greaterImage", a.index)

let greaterEqualImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.GreaterEqualImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "greaterEqualImage", a.index)

let lessImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.LessImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "lessImage", a.index)

let lessEqualImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.LessEqualImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "lessEqualImage", a.index)

let andImage (a: Image<uint8>) (b: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.AndImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "andImage", a.index)

let orImage (a: Image<uint8>) (b: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.OrImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "orImage", a.index)

let xorImage (a: Image<uint8>) (b: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.XorImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(a.toSimpleITK(), b.toSimpleITK()), "xorImage", a.index)

let notImage (img: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.NotImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(img.toSimpleITK()), "notImage", img.index)

let mask (outsideValue: double) (img: Image<'T>) (mask: Image<uint8>) : Image<'T> =
    use filter = new itk.simple.MaskImageFilter()
    filter.SetOutsideValue(outsideValue)
    Image<'T>.ofSimpleITK(filter.Execute(img.toSimpleITK(), mask.toSimpleITK()), "mask", img.index)

// ----- basic image analysis functions -----
(* // I'm waiting with proper support of complex values
let fft (img: Image<'T>)  = makeUnaryImageOperator (fun () -> new itk.simple.ForwardFFTImageFilter()) (fun f x -> f.Execute(x)) img
let ifft (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.InverseFFTImageFilter()) (fun f x -> f.Execute(x)) img
let real (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToRealImageFilter()) (fun f x -> f.Execute(x)) img
let imag (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToImaginaryImageFilter()) (fun f x -> f.Execute(x)) img
let cabs (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToModulusImageFilter()) (fun f x -> f.Execute(x)) img
let carg (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToPhaseImageFilter()) (fun f x -> f.Execute(x)) img
*)


// Basic transformations
let euler2DTransform (img: Image<'T>) (cx:float,cy:float,a:float) (dx:float,dy:float): Image<'T> =
    use t = new itk.simple.Euler2DTransform()
    t.SetAngle(a)
    t.SetCenter([ cx; cy ] |> toVectorFloat64)
    t.SetTranslation([ dx; dy ] |> toVectorFloat64)

    use f = new itk.simple.ResampleImageFilter()
    f.SetReferenceImage(img.toSimpleITK())
    f.SetInterpolator(itk.simple.InterpolatorEnum.sitkLinear) // sitkNearestNeighbor, sitkLinear, sitkBSpline
    f.SetDefaultPixelValue(0.0) // This will probably break for vector valued functions...
    f.SetTransform(t.GetInverse()) 
    Image<'T>.ofSimpleITK(f.Execute (img.toSimpleITK()),"euler2DTransform",img.index)

let euler2DRotate (img: Image<'T>) (cx:float,cy:float) (a:float): Image<'T> =
    use t = new itk.simple.Euler2DTransform()
    t.SetAngle(a)
    t.SetCenter([ cx; cy ] |> toVectorFloat64)

    use f = new itk.simple.ResampleImageFilter()
    f.SetReferenceImage(img.toSimpleITK())
    f.SetInterpolator(itk.simple.InterpolatorEnum.sitkLinear)
    f.SetDefaultPixelValue(0.0) // This will probably break for vector valued functions...
    f.SetTransform(t) 
    Image<'T>.ofSimpleITK(f.Execute (img.toSimpleITK()),"euler2DRotate",img.index)

let resample2D (interpolator: itk.simple.InterpolatorEnum) (outputWidth: uint) (outputHeight: uint) (outputSpacingX: float) (outputSpacingY: float) (img: Image<'T>) : Image<'T> =
    if img.GetDimensions() <> 2u then
        failwith $"resample2D expects a 2D image, but got {img.GetDimensions()}D."

    use transform = new itk.simple.Transform(2u, itk.simple.TransformEnum.sitkIdentity)
    use filter = new itk.simple.ResampleImageFilter()
    filter.SetSize([ outputWidth; outputHeight ] |> toVectorUInt32)
    filter.SetOutputOrigin([ 0.0; 0.0 ] |> toVectorFloat64)
    filter.SetOutputSpacing([ outputSpacingX; outputSpacingY ] |> toVectorFloat64)
    filter.SetOutputDirection([ 1.0; 0.0; 0.0; 1.0 ] |> toVectorFloat64)
    filter.SetInterpolator(interpolator)
    filter.SetDefaultPixelValue(0.0)
    filter.SetTransform(transform)
    Image<'T>.ofSimpleITK(filter.Execute(img.toSimpleITK()), "resample2D", img.index)

type BoundaryCondition = ZeroPad | PerodicPad | ZeroFluxNeumannPad
type OutputRegionMode = Valid | Same

let internal convolve3 (img:itk.simple.Image) (ker:itk.simple.Image) (outputRegionMode: OutputRegionMode option) =
    // Tailored for sliding-window stacks: default/Same means x/y Same and z Valid.
    // Let SimpleITK do the heavy SAME convolution, then discard the unusable halo.
    let szImg = img.GetSize() |> fromVectorUInt32 |> List.map int
    let szKer = ker.GetSize() |> fromVectorUInt32 |> List.map int

    use filter = new itk.simple.ConvolutionImageFilter()
    filter.SetOutputRegionMode(itk.simple.ConvolutionImageFilter.OutputRegionModeType.SAME)
    filter.SetBoundaryCondition(itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_PAD)
    use same = filter.Execute(img, ker)

    let start, size =
        match outputRegionMode with
        | None | Some Same ->
            [ 0; 0; szKer[2] / 2 ],
            [ szImg[0]; szImg[1]; szImg[2] - szKer[2] + 1 ]
        | Some Valid ->
            szKer |> List.map (fun n -> n / 2),
            List.map2 (fun a b -> a - b + 1) szImg szKer

    use extractor = new itk.simple.ExtractImageFilter()
    extractor.SetSize(size |> List.map uint |> toVectorUInt32)
    extractor.SetIndex(start |> toVectorInt32)
    extractor.Execute(same)

// Convolve sets size after the first argument, so convolve img kernel!
let convolve (outputRegionMode: OutputRegionMode option) (boundaryCondition: BoundaryCondition option) : Image<'T> -> Image<'T> -> Image<'T> =
    makeBinaryImageOperatorWith
        "convolve"
        (fun () -> new itk.simple.ConvolutionImageFilter())
        (fun f ->
            f.SetOutputRegionMode (
                match outputRegionMode with
                    | Some Valid -> itk.simple.ConvolutionImageFilter.OutputRegionModeType.VALID
                    | _ ->         itk.simple.ConvolutionImageFilter.OutputRegionModeType.SAME)
            f.SetBoundaryCondition (
                match boundaryCondition with
                    | Some ZeroFluxNeumannPad -> itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_FLUX_NEUMANN_PAD
                    | Some PerodicPad ->         itk.simple.ConvolutionImageFilter.BoundaryConditionType.PERIODIC_PAD 
                    | _ ->                       itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_PAD)) 
        (fun f img ker -> 
            // $"convolve called"
            if img.GetNumberOfComponentsPerPixel() > 1u then
                failwith "Can't convolve vector images"
            // Convolve does not like images with singular dimensions
            let idImg = img.GetPixelID()
            let idKer = ker.GetPixelID()
            if idImg <> idKer then 
                failwith "Image and kernels must be of the same type"
            let dimImg = img.GetDimension()
            let dimKer = ker.GetDimension()
            if dimImg <> dimKer then
                failwith "Image and kernels must have the same number of dimensions"
            let szImg = img.GetSize() |> fromVectorUInt32 |> List.map int
            let szKer = ker.GetSize() |> fromVectorUInt32 |> List.map int
            let szZip = List.zip szImg szKer
            if List.exists (fun (a,b) -> a < b) szZip then
                failwith "Image must not be smaller than kernel in any dimension"
            let res =
                if dimImg = 3u && outputRegionMode <> Some Valid then
                    convolve3 img ker outputRegionMode
                elif List.forall (fun (a,b) -> a >= b && b > 1) szZip then
                    f.Execute(img,ker)
                elif dimImg = 3u then
                    convolve3 img ker outputRegionMode
                else
                    f.Execute(img,ker)
            res
        )

let conv (img: Image<'T>) (ker: Image<'T>) : Image<'T> = convolve None None img ker

let defaultGaussWindowSize (sigma: float) : uint = 1u + 2u*2u * uint sigma
/// Gaussian kernel convolution

let gauss (dim: uint) (sigma: float) (kernelSize: uint option) : Image<'T> =
    use f = new itk.simple.GaussianImageSource()
    let sz = Option.defaultValue (defaultGaussWindowSize sigma) kernelSize
    f.SetSize(List.replicate (int dim) sz |> toVectorUInt32)
    f.SetSigma(sigma)
    // Image coords: [0 1] (mean = 0.5), [0 1 2] (mean = 1) => mean = (sz-1)/2
    f.SetMean(List.replicate (int dim) ((float (sz-1u)) / 2.0) |> toVectorFloat64)
    f.SetScale(1.0)
    f.NormalizedOn()
    Image<'T>.ofSimpleITK(f.Execute(),"gauss")

let private stensil order = 
    // https://en.wikipedia.org/wiki/Finite_difference_coefficient 
    if   order = 1u then [1.0/2.0; 0.0; -1.0/2.0]
    elif order = 2u then [1.0; -2.0; 1.0]
    elif order = 3u then [1.0/2.0; -1.0; 0.0; 1.0; -1.0/2.0]
    elif order = 4u then [1.0; -4.0; 6.0; -4.0; 1.0]
    elif order = 5u then [1.0/2.0; -2.0; 5.0/2.0; 0.0; -5.0/2.0; 2.0; -1.0/2.0]
    elif order = 6u then [1.0; -6.0; 15.0; -20.0; 15.0; -6.0; 1.0]
    else failwith "[finiteDiffFilter] only implemented derivative order 1 <= order <= 6"

let finiteDiffFilter2D (direction: uint) (order: uint) : Image<float> =
    let lst = stensil order
    let n = lst.Length
    let arr = 
        if direction = 0u then
            Array2D.init n 1 (fun i _ -> lst[i])
        else
            Array2D.init 1 n (fun _ i -> lst[i])
    Image<float>.ofArray2D(arr, "stensil2D") 

let finiteDiffFilter3D (sigma:float) (direction: uint) (order: uint) : Image<float> =
    let lst = stensil order
    let n = lst.Length
    let arr = 
        if direction = 0u then
            Array3D.init n 1 1 (fun i _ _ -> lst[i])
        elif direction = 1u then
            Array3D.init 1 n 1 (fun _ i _ -> lst[i])
        else 
            Array3D.init 1 1 n (fun _ _ i -> lst[i])
    Image<float>.ofArray3D(arr, "stensil3D")

let finiteDiffFilter4D (direction: uint) (order: uint) : Image<float> =
    let lst = stensil order
    let n = lst.Length
    let arr = 
        if direction = 0u then
            Array4D.init n 1 1 1 (fun i _ _ _ -> lst[i])
        elif direction = 1u then
            Array4D.init 1 n 1 1 (fun _ i _ _ -> lst[i])
        elif direction = 2u then
            Array4D.init 1 1 n 1 (fun _ _ i _ -> lst[i])
        else 
            Array4D.init 1 1 1 n (fun _ _ _ i -> lst[i])
    Image<float>.ofArray4D(arr, "stensil4D")

let discreteGaussian (dim: uint) (sigma: float) (kernelSize: uint option) (outputRegionMode: OutputRegionMode option) (boundaryCondition: BoundaryCondition option) : Image<'T> -> Image<'T> =
    fun (input: Image<'T>) -> 
        let kern = gauss dim sigma kernelSize
        let res = convolve outputRegionMode boundaryCondition input kern
        kern.decRefCount()
        res

/// Gradient convolution using Derivative filter
let gradientConvolve (direction: uint) (order: uint32) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "gradientConvolve"
        (fun () -> new itk.simple.DerivativeImageFilter())
        (fun f ->
            f.SetDirection(direction)
            f.SetOrder(order))
        (fun f x -> f.Execute(x))

/// Mathematical morphology
/// Binary erosion
let binaryErode (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryErode"
        (fun () -> new itk.simple.BinaryErodeImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

/// Binary dilation
let binaryDilate (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryDilate"
        (fun () -> new itk.simple.BinaryDilateImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

/// Binary opening (erode then dilate)
let binaryOpening (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryOpening"
        (fun () -> new itk.simple.BinaryMorphologicalOpeningImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

/// Binary closing (dilate then erode)
let binaryClosing (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryClosing"
        (fun () -> new itk.simple.BinaryMorphologicalClosingImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

let grayscaleErode (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "grayscaleErode"
        (fun () -> new itk.simple.GrayscaleErodeImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

let grayscaleDilate (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "grayscaleDilate"
        (fun () -> new itk.simple.GrayscaleDilateImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

let grayscaleOpening (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "grayscaleOpening"
        (fun () -> new itk.simple.GrayscaleMorphologicalOpeningImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

let grayscaleClosing (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "grayscaleClosing"
        (fun () -> new itk.simple.GrayscaleMorphologicalClosingImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

let whiteTopHat (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "whiteTopHat"
        (fun () -> new itk.simple.WhiteTopHatImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

let blackTopHat (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "blackTopHat"
        (fun () -> new itk.simple.BlackTopHatImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

let morphologicalGradient (radius: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "morphologicalGradient"
        (fun () -> new itk.simple.MorphologicalGradientImageFilter())
        (fun f -> f.SetKernelRadius(radius))
        (fun f x -> f.Execute(x))

/// Fill holes in binary regions
let binaryFillHoles (img : Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.BinaryFillholeImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"binaryFillHoles",img.index)

let binaryContour (fullyConnected: bool) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryContour"
        (fun () -> new itk.simple.BinaryContourImageFilter())
        (fun f -> f.SetFullyConnected(fullyConnected))
        (fun f x -> f.Execute(x))

let binaryThinning (img: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.BinaryThinningImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(img.toSimpleITK()), "binaryThinning", img.index)

let binaryMedian (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryMedian"
        (fun () -> new itk.simple.BinaryMedianImageFilter())
        (fun f -> f.SetRadius(radius))
        (fun f x -> f.Execute(x))

let binaryOpeningByReconstruction (radius: uint) (fullyConnected: bool) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryOpeningByReconstruction"
        (fun () -> new itk.simple.BinaryOpeningByReconstructionImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetFullyConnected(fullyConnected))
        (fun f x -> f.Execute(x))

let binaryClosingByReconstruction (radius: uint) (fullyConnected: bool) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryClosingByReconstruction"
        (fun () -> new itk.simple.BinaryClosingByReconstructionImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetFullyConnected(fullyConnected))
        (fun f x -> f.Execute(x))

let binaryReconstructionByDilation (fullyConnected: bool) (marker: Image<uint8>) (mask: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.BinaryReconstructionByDilationImageFilter()
    filter.SetFullyConnected(fullyConnected)
    Image<uint8>.ofSimpleITK(filter.Execute(marker.toSimpleITK(), mask.toSimpleITK()), "binaryReconstructionByDilation", marker.index)

let binaryReconstructionByErosion (fullyConnected: bool) (marker: Image<uint8>) (mask: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.BinaryReconstructionByErosionImageFilter()
    filter.SetFullyConnected(fullyConnected)
    Image<uint8>.ofSimpleITK(filter.Execute(marker.toSimpleITK(), mask.toSimpleITK()), "binaryReconstructionByErosion", marker.index)

let votingBinaryHoleFilling (radius: uint) (majorityThreshold: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "votingBinaryHoleFilling"
        (fun () -> new itk.simple.VotingBinaryHoleFillingImageFilter())
        (fun f ->
            f.SetRadius(radius)
            f.SetMajorityThreshold(majorityThreshold))
        (fun f x -> f.Execute(x))

type ConnectedComponentsResult =
    { Labels : Image<uint64>
      ObjectCount : uint64 }

/// Connected components labeling
// Currying and generic arguments causes value restriction error
let connectedComponents (img : Image<uint8>) : ConnectedComponentsResult =
    use filter = new itk.simple.ConnectedComponentImageFilter()
    { Labels = Image<uint64>.ofSimpleITK(filter.Execute(img.toSimpleITK()), "connectedComponents",img.index)
      ObjectCount = uint64 (filter.GetObjectCount()) }

/// Relabel components by size, optionally remove small objects
let relabelComponents (minObjectSize: uint) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "relabelComponents"
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
    //Indexes: uint32 list
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
    //RLEIndexes: uint32 list
    Roundness: float
}

/// Compute label shape statistics and return a dictionary of results
let labelShapeStatistics (img: Image<'T>) : Map<int64, LabelShapeStatistics> =
    use stats = new itk.simple.LabelShapeStatisticsImageFilter()
    stats.Execute(img.toSimpleITK())
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
            //Indexes = stats.GetIndexes(label) |> fromVectorUInt32
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
            //RLEIndexes = stats.GetRLEIndexes(label) |> fromVectorUInt32
            Roundness = stats.GetRoundness(label)
        }
        label, stats
    )
    |> Map.ofSeq

type LabelIntensityStatistics = {
    Label: int64
    NumberOfPixels: uint64
    PhysicalSize: float
    Mean: float
    Median: float
    Minimum: float
    Maximum: float
    Sum: float
    StandardDeviation: float
    Variance: float
    Skewness: float
    Kurtosis: float
    Centroid: float list
    CenterOfGravity: float list
    BoundingBox: uint list
}

let labelIntensityStatistics (labelImage: Image<'L>) (intensityImage: Image<'T>) : Map<int64, LabelIntensityStatistics> =
    use stats = new itk.simple.LabelIntensityStatisticsImageFilter()
    stats.Execute(labelImage.toSimpleITK(), intensityImage.toSimpleITK())
    stats.GetLabels()
    |> Seq.map (fun label ->
        label,
        { Label = label
          NumberOfPixels = stats.GetNumberOfPixels(label)
          PhysicalSize = stats.GetPhysicalSize(label)
          Mean = stats.GetMean(label)
          Median = stats.GetMedian(label)
          Minimum = stats.GetMinimum(label)
          Maximum = stats.GetMaximum(label)
          Sum = stats.GetSum(label)
          StandardDeviation = stats.GetStandardDeviation(label)
          Variance = stats.GetVariance(label)
          Skewness = stats.GetSkewness(label)
          Kurtosis = stats.GetKurtosis(label)
          Centroid = stats.GetCentroid(label) |> fromVectorFloat64
          CenterOfGravity = stats.GetCenterOfGravity(label) |> fromVectorFloat64
          BoundingBox = stats.GetBoundingBox(label) |> fromVectorUInt32 })
    |> Map.ofSeq

type LabelOverlapMeasures = {
    MeanOverlap: float
    UnionOverlap: float
    JaccardCoefficient: float
    DiceCoefficient: float
    VolumeSimilarity: float
    FalseNegativeError: float
    FalsePositiveError: float
    FalseDiscoveryRate: float
}

let labelOverlapMeasures (source: Image<'T>) (target: Image<'T>) : LabelOverlapMeasures =
    use stats = new itk.simple.LabelOverlapMeasuresImageFilter()
    stats.Execute(source.toSimpleITK(), target.toSimpleITK())
    { MeanOverlap = stats.GetMeanOverlap()
      UnionOverlap = stats.GetUnionOverlap()
      JaccardCoefficient = stats.GetJaccardCoefficient()
      DiceCoefficient = stats.GetDiceCoefficient()
      VolumeSimilarity = stats.GetVolumeSimilarity()
      FalseNegativeError = stats.GetFalseNegativeError()
      FalsePositiveError = stats.GetFalsePositiveError()
      FalseDiscoveryRate = stats.GetFalseDiscoveryRate() }

let labelContour (fullyConnected: bool) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "labelContour"
        (fun () -> new itk.simple.LabelContourImageFilter())
        (fun f -> f.SetFullyConnected(fullyConnected))
        (fun f x -> f.Execute(x))

let changeLabel (fromLabel: double) (toLabel: double) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "changeLabel"
        (fun () -> new itk.simple.ChangeLabelImageFilter())
        (fun f ->
            let map = new itk.simple.DoubleDoubleMap()
            map.Add(fromLabel, toLabel)
            f.SetChangeMap(map))
        (fun f x -> f.Execute(x))

/// Compute signed Maurer distance map (positive outside, negative inside)
// ApproximateSignedDistanceMapImageFilter has an error. These cast an exception: 
//   [[1uy;0uy;1uy;0uy;1uy;0uy]] and [[1uy;0uy;1uy;0uy;0uy;0uy]]
// but these don't
//   [[1uy;0uy;1uy;1uy;1uy;0uy]] and [[1uy;0uy;0uy;0uy;1uy;0uy]]
let signedDistanceMap (inside: uint8) (outside: uint8) (img: Image<uint8>) : Image<float> =
    use f = new itk.simple.ApproximateSignedDistanceMapImageFilter()
    f.SetInsideValue(float inside)
    f.SetOutsideValue(float outside)
    Image<float>.ofSimpleITK(f.Execute(img.toSimpleITK()),"signedDistanceMap",img.index)

/// Morphological watershed (binary or grayscale)
let watershed (level: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "watershed"
        (fun () -> new itk.simple.MorphologicalWatershedImageFilter())
        (fun f ->
            f.SetLevel(level)
            f.SetMarkWatershedLine(false))
        (fun f x -> f.Execute(x))

/// Histogram related functions
type ImageStats =
    { 
        NumPixels: uint
        Mean: float
        Std: float
        Min: float
        Max: float
        Sum: float
        Var: float 
    }

let computeStats (img: Image<'T>) : ImageStats =
    use stats = new itk.simple.StatisticsImageFilter()
    stats.Execute(img.toSimpleITK())
    { 
        NumPixels = img.GetSize() |> List.reduce (*)
        Mean = stats.GetMean()
        Std = stats.GetSigma()
        Min = stats.GetMinimum()
        Max = stats.GetMaximum()
        Sum = stats.GetSum()
        Var = stats.GetVariance() 
    }

//var = 1/N sum_i (x_i-m)^2 = 1/N sum_i x_i^2-m^2
let addComputeStats (s1: ImageStats) (s2: ImageStats): ImageStats =
    let n1, n2 = s1.NumPixels |> float, s2.NumPixels |> float
    let m1, m2 = s1.Mean, s2.Mean
    let v1, v2 = s1.Var, s2.Var
    let s = v1*(n1-1.0)+m1**2.0*n1+v2*(n2-1.0)+m2**2.0*n2
    let n = n1 + n2
    let m = (s1.Sum+s2.Sum)/n
    let v = (s-n*m**2.0)/(n-1.0)
    { 
        NumPixels = n |> uint
        Mean = m
        Sum = s1.Sum+s2.Sum
        Var = v
        Std = sqrt(v)
        Min = min s1.Min s2.Min
        Max = max s1.Max s2.Max
    }


let unique (img: Image<'T>) : 'T list when 'T : comparison =
    img.toArray2D()            // 'T [,]
    |> Seq.cast<'T>            // flatten to a seq<'T>
    |> Set.ofSeq               // remove duplicates
    |> Set.toList              // back to an ordered list
    
let otsuThresholdFromHistogram (bins: uint) (images: Image<'T> list) : float =
    if bins < 2u then
        invalidArg "bins" "Otsu threshold estimation requires at least two bins."
    if List.isEmpty images then
        invalidArg "images" "Otsu threshold estimation requires at least one image."

    let toFloat value = System.Convert.ToDouble(box value, System.Globalization.CultureInfo.InvariantCulture)
    let values =
        images
        |> List.collect (fun image ->
            image.GetSize()
            |> flatIndices
            |> Seq.map (image.Get >> toFloat)
            |> Seq.toList)

    match values with
    | [] -> invalidArg "images" "Otsu threshold estimation requires at least one pixel."
    | _ ->
        let minValue = values |> List.min
        let maxValue = values |> List.max
        if minValue = maxValue then
            minValue
        else
            let binCount = int bins
            let width = (maxValue - minValue) / float binCount
            let histogram = Array.zeroCreate<uint64> binCount
            values
            |> List.iter (fun value ->
                let bin =
                    if value >= maxValue then
                        binCount - 1
                    else
                        int ((value - minValue) / width)
                        |> max 0
                        |> min (binCount - 1)
                histogram[bin] <- histogram[bin] + 1UL)

            let totalCount = histogram |> Array.sumBy float
            let totalMean =
                histogram
                |> Array.mapi (fun i count -> float i * float count)
                |> Array.sum

            let mutable bestThresholdBin = 1
            let mutable bestVariance = System.Double.NegativeInfinity
            let mutable backgroundWeight = 0.0
            let mutable backgroundSum = 0.0

            for thresholdBin in 1 .. binCount - 1 do
                let previousBin = thresholdBin - 1
                backgroundWeight <- backgroundWeight + float histogram[previousBin]
                backgroundSum <- backgroundSum + float previousBin * float histogram[previousBin]
                let foregroundWeight = totalCount - backgroundWeight
                if backgroundWeight > 0.0 && foregroundWeight > 0.0 then
                    let backgroundMean = backgroundSum / backgroundWeight
                    let foregroundMean = (totalMean - backgroundSum) / foregroundWeight
                    let variance = backgroundWeight * foregroundWeight * pown (backgroundMean - foregroundMean) 2
                    if variance > bestVariance then
                        bestVariance <- variance
                        bestThresholdBin <- thresholdBin

            minValue + width * (float bestThresholdBin + 0.5)

/// Otsu threshold estimated from a binned histogram of the image values.
let otsuThreshold (img: Image<'T>) : Image<uint8> =
    let thresholdValue = otsuThresholdFromHistogram 256u [ img ]
    use filter = new itk.simple.BinaryThresholdImageFilter()
    filter.SetLowerThreshold(thresholdValue)
    filter.SetUpperThreshold(System.Double.PositiveInfinity)
    filter.SetInsideValue(1uy)
    filter.SetOutsideValue(0uy)
    Image<uint8>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"otsuThreshold",img.index)

/// Otsu multiple thresholds (returns a label map)
let otsuMultiThreshold (numThresholds: byte) (img: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.OtsuMultipleThresholdsImageFilter()
    filter.SetNumberOfThresholds(numThresholds)
    Image<uint8>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"otsuMultiThreshold",img.index)

/// Moments-based threshold
let momentsThreshold (img: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.MomentsThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image<uint8>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"momentsThreshold",img.index)

/// Coordinate fields
// Cannot get TransformToDisplacementFieldFilter to work, so making it by hand.
let generateCoordinateAxis (axis: int) (size: int list) : Image<uint32> =
    let img = new itk.simple.Image(toVectorUInt32 (size |> List.map uint), itk.simple.PixelIDValueEnum.sitkUInt32)

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
        img.SetPixelAsUInt32(idxVec, coord))

    Image<uint32>.ofSimpleITK(img,"generateCoordinateAxis")

let histogram (img: Image<'T>) : Map<'T, uint64> =
    img.GetSize()
    |> flatIndices
    |> Seq.fold 
        (fun acc idx ->
            let elm = img.Get idx 
            Map.change elm (fun vopt -> 
                match vopt with 
                    Some v -> Some (v+1uL) 
                    | None -> Some (1uL)) 
                acc)
        Map.empty<'T, uint64>

let addHistogram (h1: Map<'T, uint64>) (h2: Map<'T, uint64>) : Map<'T, uint64> =
    Map.fold 
        (fun acc k2 v2 -> 
            Map.change 
                k2 
                (fun v1opt -> 
                    match v1opt with 
                        Some v1 -> Some (v1+v2) | None -> Some v2) 
                acc)
        h1
        h2

let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
    map |> Map.toList

let inline pairs2floats<^T, ^S when ^T : (static member op_Explicit : ^T -> float)
                                 and ^S : (static member op_Explicit : ^S -> float)>
                                 (pairs: (^T * ^S) list) : (float * float) list =
    // must be inline for not reducing 'T and 'S to ints
    pairs |> List.map (fun (k, v) -> (float k, float v)) 

let inline pairs2ints<^T, ^S when ^T : (static member op_Explicit : ^T -> int)
                                 and ^S : (static member op_Explicit : ^S -> int)>
                                 (pairs: (^T * ^S) list) : (int * int) list =
    // must be inline for not reducing 'T and 'S to ints
    pairs |> List.map (fun (k, v) -> (int k, int v)) 

let addNormalNoise (mean: float) (stddev: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "addNormalNoise"
        (fun () -> new itk.simple.AdditiveGaussianNoiseImageFilter())
        (fun f -> 
            f.SetMean(mean)
            f.SetStandardDeviation(stddev))
        (fun f x -> f.Execute(x))

let threshold (lower: float) (upper: float) (img: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.BinaryThresholdImageFilter()
    filter.SetLowerThreshold lower
    filter.SetUpperThreshold upper
    let res = filter.Execute(img.toSimpleITK()); 
    Image<uint8>.ofSimpleITK(res,"threshold",0)

let toVectorOfImage images =
    let v = new itk.simple.VectorOfImage()
    for img in images do
        v.Add(img)
    v

let stack (images: Image<'T> list) : Image<'T> =
    match images with
    | [] -> invalidArg "images" "Cannot stack an empty image list."
    | first :: _ ->
        let dim = first.GetDimensions()
        if images |> List.exists (fun image -> image.GetDimensions() <> dim) then
            failwith "Images must have the same dimensionality."
        if dim = 2u then
            let expectedSize = first.GetSize()
            images
            |> List.iter (fun image ->
                if image.GetSize() <> expectedSize then
                    failwithf "Image sizes differ: %A vs %A" (image.GetSize()) expectedSize)
            use filter = new itk.simple.JoinSeriesImageFilter ()
            filter.SetOrigin(0.0) |> ignore
            filter.SetSpacing(1.0) |> ignore
            use v = new itk.simple.VectorOfImage()
            images |> List.iter (fun (I:Image<'T>) -> v.Add (I.toSimpleITK()))
            v |> filter.Execute |> (fun sitk -> Image<'T>.ofSimpleITK(sitk,"stack",first.index) )
        else
            let stackDim = dim - 1u
            images |> List.reduce (concatAlong stackDim)

let extractSub (topLeft : uint list) (bottomRight: uint list) (img: Image<'T>) : Image<'T> =
    if topLeft.Length <> bottomRight.Length then
        failwith $"extractSub: topLeft and bottomRight lists must have equal lengths ({topLeft} vs {bottomRight})"
    if img.GetDimensions() <> uint topLeft.Length then
        failwith $"extractSub: indices and image size does not match"
    let sz = List.zip topLeft bottomRight |> List.map (fun (a,b) -> b-a + 1u)
    if List.exists (fun a -> a <  1u) sz then
        failwith $"extractSub: no index of bottomRight must be smaller than topLeft  ({topLeft} vs {bottomRight})"

    use extractor = new itk.simple.ExtractImageFilter()
    extractor.SetSize(sz |> toVectorUInt32)
    extractor.SetIndex( topLeft |> List.map int |> toVectorInt32)
    let res = Image<'T>.ofSimpleITK(extractor.Execute(img.toSimpleITK()),"extractSub",img.index)
    res

let extractSlice (dir: uint) (i: int) (img: Image<'T>) =
    if img.GetDimensions() <> 3u then
        failwith $"extractSlice: image must be 3D"
    let size = img.GetSize()
    use extractor = new itk.simple.ExtractImageFilter()
    let sz, idx =
        if dir = 0u then   [0u; size[1]; size[2]], [i; 0; 0] // Has extractSlice confused x-y and i-j?
        elif dir = 1u then [size[0]; 0u; size[2]], [0; i; 0] 
        else               [size[0]; size[1]; 0u], [0; 0; i]
    extractor.SetSize( sz |> toVectorUInt32)
    extractor.SetIndex( idx |> toVectorInt32)
    extractor.SetDirectionCollapseToStrategy(itk.simple.ExtractImageFilter.DirectionCollapseToStrategyType.DIRECTIONCOLLAPSETOIDENTITY)
    let res = Image<'T>.ofSimpleITK(extractor.Execute(img.toSimpleITK()),"extractSlice",i)
    res

let unstack (dir: uint) (vol: Image<'T>): Image<'T> list =
    let sz = vol.GetSize()
    let depth = sz[int dir] |> int
    let res = List.init depth (fun i -> extractSlice dir i vol)
    res

let unstackSkipNTakeM (N:uint) (mWish:uint) (vol: Image<'T>): Image<'T> list =
    let dim = vol.GetDimensions()
    if dim < 3u then
        failwith $"Cannot unstack a {dim}-dimensional image along the 3rd axis"
    let depth = vol.GetDepth() |> int
    let M = min (uint depth - N) mWish
    if (N+M > uint depth) then
        failwith $"Cannot unstack from z={N} to z={N+M-1u} of a stack of depth {depth}"
    let res = List.init (int M) (fun i -> extractSlice 2u (i+int N) vol)
    res

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

let toSeqSeq (I: Image<'T>): seq<seq<float>> =
    let toFloat (value: obj) =
        match value with
        | :? float   as f -> f
        | :? float32 as f -> float f
        | :? int     as i -> float i
        | :? byte    as b -> float b
        | :? int64   as l -> float l
        | _ -> failwithf "Cannot convert value of type %s to float" (value.GetType().FullName)
    let width = I.GetWidth() |> int
    let height = I.GetHeight() |> int
    Seq.init height (fun y ->
        Seq.init width (fun x ->
            I[x,y] |> box |> toFloat))

let permuteAxes (order: uint list) (img: Image<'T>) = 
    use filter = new itk.simple.PermuteAxesImageFilter()
    filter.SetOrder(order|>toVectorUInt32)
    Image<'S>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"permuteAxes",img.index)

// Fourier transform a 2d image
let FFTXY (image: Image<'T>) : Image<System.Numerics.Complex> =
    if image.GetDimensions() <> 2u then
        failwith $"FFTXY: image must be 2D, got {image.GetDimensions()}D"
    let input = ofCastItk<float> (image.toSimpleITK())
    use fft = new itk.simple.ForwardFFTImageFilter()
    let complexImg = fft.Execute(input)
    Image<System.Numerics.Complex>.ofSimpleITK(complexImg, "FFTXY", image.index)

// Fourier transform a 3d image along a specified axis direction
let directionalFFT (dir: uint) (image: Image<'T>) : Image<System.Numerics.Complex> =
    let dims = image.GetDimensions()
    if dir >= dims then
        failwith $"directionalFFT: dir={dir} is out of range for {dims}D image"
    let size = image.GetSize()
    let input = ofCastItk<float> (image.toSimpleITK())
    let inputImage = Image<float>.ofSimpleITK(input, "directionalFFTInput", image.index)
    let outputReal = new Image<float>(size, 1u, "directionalFFTReal", image.index, true)
    let outputImag = new Image<float>(size, 1u, "directionalFFTImag", image.index, true)

    let dimInt = int dims
    let rec baseCoords i acc =
        if i = dimInt then
            [List.rev acc]
        else
            if uint i = dir then
                baseCoords (i + 1) (0u :: acc)
            else
                [0u .. size[i] - 1u]
                |> List.collect (fun v -> baseCoords (i + 1) (v :: acc))

    let lineLength = size[int dir] |> int
    let lineLengthFloat = float lineLength
    for baseCoord in baseCoords 0 [] do
        let line =
            [| for n in 0 .. lineLength - 1 ->
                let coord = baseCoord |> List.mapi (fun i v -> if uint i = dir then uint n else v)
                inputImage.Get coord |]

        for k in 0 .. lineLength - 1 do
            let mutable re = 0.0
            let mutable im = 0.0
            for n in 0 .. lineLength - 1 do
                let theta = -2.0 * System.Math.PI * float (k * n) / lineLengthFloat
                re <- re + line[n] * cos theta
                im <- im + line[n] * sin theta
            let coord = baseCoord |> List.mapi (fun i v -> if uint i = dir then uint k else v)
            outputReal.Set coord re
            outputImag.Set coord im

    Image<float>.ofImagePairToComplex outputReal outputImag
