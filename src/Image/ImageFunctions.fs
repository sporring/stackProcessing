module ImageFunctions
open System
open System.Collections.Generic
open Image
open Image.InternalHelpers
open TinyLinAlg

let imageFromTemporarySimpleITK<'T when 'T: equality> name index (itkImage: itk.simple.Image) : Image<'T> =
    Image<'T>.ofSimpleITKNDispose(itkImage, name, index)

// Image constant arithmetic operations
let inline imageAddScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.AddImageFilter()
    imageFromTemporarySimpleITK<'S> "imageAddScalar" img.index (filter.Execute(img.toSimpleITK(), float i))
let inline scalarAddImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    imageAddScalar img i
let inline imageSubScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.SubtractImageFilter()
    imageFromTemporarySimpleITK<'S> "imageSubScalar" img.index (filter.Execute(img.toSimpleITK(), float i))
let inline scalarSubImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    use filter = new itk.simple.SubtractImageFilter()
    imageFromTemporarySimpleITK<'S> "scalarSubImage" img.index (filter.Execute(float i, img.toSimpleITK()))
let inline imageMulScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.MultiplyImageFilter()
    imageFromTemporarySimpleITK<'S> "imageMulScalar" img.index (filter.Execute(img.toSimpleITK(), float i))
let inline scalarMulImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    imageMulScalar img i
let inline imageDivScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (img: Image<'S>) (i: ^S) =
    use filter = new itk.simple.DivideImageFilter()
    imageFromTemporarySimpleITK<'S> "imageDivScalar" img.index (filter.Execute(img.toSimpleITK(), float i))
let inline scalarDivImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (img: Image<'S>) =
    use filter = new itk.simple.DivideImageFilter()
    imageFromTemporarySimpleITK<'S> "scalarDivImage" img.index (filter.Execute(float i, img.toSimpleITK()))

let inline imagePowScalar<'S when ^S : equality
                  and  ^S : (static member op_Explicit : ^S -> float)
                  > (img: Image<'S>, i: 'S) =
    use filter = new itk.simple.PowImageFilter()
    imageFromTemporarySimpleITK<'S> "imagePowScalar" img.index (filter.Execute(img.toSimpleITK(), float i))


let inline scalarPowImage<'S when ^S : equality
                  and  ^S : (static member op_Explicit : ^S -> float)
                  > (i: 'S, img: Image<'S>) =
    use filter = new itk.simple.PowImageFilter()
    imageFromTemporarySimpleITK<'S> "scalarPowImage" img.index (filter.Execute(float i, img.toSimpleITK()))

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
    imageFromTemporarySimpleITK "squeeze" img.index (filter.Execute(img.toSimpleITK()))

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
    imageFromTemporarySimpleITK "concatAlong" a.index outWithBoth

let constantPad2D<'T when 'T : equality> (padLower : uint list) (padUpper : uint list) (c : double) (img : Image<'T>) : Image<'T> =
    if padLower.Length <> 2 || padUpper.Length <> 2 then
        invalidArg "padLower/padUpper" "Both bounds must have length 2 for a 2‑D image."
    use filter = new itk.simple.ConstantPadImageFilter()
    filter.SetPadLowerBound(padLower |> toVectorUInt32)
    filter.SetPadUpperBound(padUpper |> toVectorUInt32)
    filter.SetConstant(c)
    let padded = filter.Execute(img.toSimpleITK())
    imageFromTemporarySimpleITK "constantPad2D" img.index padded

let crop2D<'T when 'T : equality> (cropLower : uint list) (cropUpper : uint list) (img : Image<'T>) : Image<'T> =
    if cropLower.Length <> 2 || cropUpper.Length <> 2 then
        invalidArg "cropLower/cropUpper" "Both bounds must have length 2 for a 2-D image."
    use filter = new itk.simple.CropImageFilter()
    filter.SetLowerBoundaryCropSize(cropLower |> toVectorUInt32)
    filter.SetUpperBoundaryCropSize(cropUpper |> toVectorUInt32)
    let cropped = filter.Execute(img.toSimpleITK())
    imageFromTemporarySimpleITK "crop2D" img.index cropped


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
            imageFromTemporarySimpleITK<'S> name img.index (invoke filter (img.toSimpleITK()))

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
        let img = imageFromTemporarySimpleITK<'T> name a.index (invoke filter (a.toSimpleITK()) (b.toSimpleITK()))
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
    imageFromTemporarySimpleITK "normalizeImage" img.index (filter.Execute(img.toSimpleITK()))

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
    imageFromTemporarySimpleITK "equalImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let notEqualImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.NotEqualImageFilter()
    imageFromTemporarySimpleITK "notEqualImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let greaterImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.GreaterImageFilter()
    imageFromTemporarySimpleITK "greaterImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let greaterEqualImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.GreaterEqualImageFilter()
    imageFromTemporarySimpleITK "greaterEqualImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let lessImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.LessImageFilter()
    imageFromTemporarySimpleITK "lessImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let lessEqualImage (a: Image<'T>) (b: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.LessEqualImageFilter()
    imageFromTemporarySimpleITK "lessEqualImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let andImage (a: Image<uint8>) (b: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.AndImageFilter()
    imageFromTemporarySimpleITK "andImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let orImage (a: Image<uint8>) (b: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.OrImageFilter()
    imageFromTemporarySimpleITK "orImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let xorImage (a: Image<uint8>) (b: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.XorImageFilter()
    imageFromTemporarySimpleITK "xorImage" a.index (filter.Execute(a.toSimpleITK(), b.toSimpleITK()))

let notImage (img: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.NotImageFilter()
    imageFromTemporarySimpleITK "notImage" img.index (filter.Execute(img.toSimpleITK()))

let mask (outsideValue: double) (img: Image<'T>) (mask: Image<uint8>) : Image<'T> =
    use filter = new itk.simple.MaskImageFilter()
    filter.SetOutsideValue(outsideValue)
    imageFromTemporarySimpleITK "mask" img.index (filter.Execute(img.toSimpleITK(), mask.toSimpleITK()))

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
type ResampleInterpolation =
    | NearestNeighbor
    | Linear

module ResampleInterpolation =
    let parse (value: string) =
        match value.Trim().ToLowerInvariant().Replace("_", "").Replace("-", "").Replace(" ", "") with
        | "nearest"
        | "nearestneighbor"
        | "nn" -> NearestNeighbor
        | "linear" -> Linear
        | _ -> failwith $"Unknown resampling interpolation '{value}'. Use NearestNeighbor or Linear."

    let internal toItk = function
        | NearestNeighbor -> itk.simple.InterpolatorEnum.sitkNearestNeighbor
        | Linear -> itk.simple.InterpolatorEnum.sitkLinear

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
    imageFromTemporarySimpleITK "euler2DTransform" img.index (f.Execute (img.toSimpleITK()))

let euler2DRotate (img: Image<'T>) (cx:float,cy:float) (a:float): Image<'T> =
    use t = new itk.simple.Euler2DTransform()
    t.SetAngle(a)
    t.SetCenter([ cx; cy ] |> toVectorFloat64)

    use f = new itk.simple.ResampleImageFilter()
    f.SetReferenceImage(img.toSimpleITK())
    f.SetInterpolator(itk.simple.InterpolatorEnum.sitkLinear)
    f.SetDefaultPixelValue(0.0) // This will probably break for vector valued functions...
    f.SetTransform(t) 
    imageFromTemporarySimpleITK "euler2DRotate" img.index (f.Execute (img.toSimpleITK()))

let resample2D (interpolation: ResampleInterpolation) (outputWidth: uint) (outputHeight: uint) (outputSpacingX: float) (outputSpacingY: float) (img: Image<'T>) : Image<'T> =
    if img.GetDimensions() <> 2u then
        failwith $"resample2D expects a 2D image, but got {img.GetDimensions()}D."

    use transform = new itk.simple.Transform(2u, itk.simple.TransformEnum.sitkIdentity)
    use filter = new itk.simple.ResampleImageFilter()
    filter.SetSize([ outputWidth; outputHeight ] |> toVectorUInt32)
    filter.SetOutputOrigin([ 0.0; 0.0 ] |> toVectorFloat64)
    filter.SetOutputSpacing([ outputSpacingX; outputSpacingY ] |> toVectorFloat64)
    filter.SetOutputDirection([ 1.0; 0.0; 0.0; 1.0 ] |> toVectorFloat64)
    filter.SetInterpolator(ResampleInterpolation.toItk interpolation)
    filter.SetDefaultPixelValue(0.0)
    filter.SetTransform(transform)
    imageFromTemporarySimpleITK "resample2D" img.index (filter.Execute(img.toSimpleITK()))

type BoundaryCondition = ZeroPad | PerodicPad | ZeroFluxNeumannPad
type OutputRegionMode = Valid | Same

let private convolutionBoundaryConditionType (boundaryCondition: BoundaryCondition option) =
    match boundaryCondition with
    | Some ZeroFluxNeumannPad -> itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_FLUX_NEUMANN_PAD
    | Some PerodicPad ->         itk.simple.ConvolutionImageFilter.BoundaryConditionType.PERIODIC_PAD 
    | _ ->                       itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_PAD

let internal convolve3 (img:itk.simple.Image) (ker:itk.simple.Image) (outputRegionMode: OutputRegionMode option) (boundaryCondition: BoundaryCondition option) =
    // Let SimpleITK do the heavy SAME convolution, then discard halos only for Valid output.
    let szImg = img.GetSize() |> fromVectorUInt32 |> List.map int
    let szKer = ker.GetSize() |> fromVectorUInt32 |> List.map int

    use filter = new itk.simple.ConvolutionImageFilter()
    filter.SetOutputRegionMode(itk.simple.ConvolutionImageFilter.OutputRegionModeType.SAME)
    filter.SetBoundaryCondition(convolutionBoundaryConditionType boundaryCondition)
    use same = filter.Execute(img, ker)

    let start, size =
        match outputRegionMode with
        | None | Some Same ->
            [ 0; 0; 0 ],
            szImg
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
            f.SetBoundaryCondition(convolutionBoundaryConditionType boundaryCondition)) 
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
                    convolve3 img ker outputRegionMode boundaryCondition
                elif List.forall (fun (a,b) -> a >= b && b > 1) szZip then
                    f.Execute(img,ker)
                elif dimImg = 3u then
                    convolve3 img ker outputRegionMode boundaryCondition
                else
                    f.Execute(img,ker)
            res
        )

let conv (img: Image<'T>) (ker: Image<'T>) : Image<'T> = convolve None None img ker

let defaultGaussWindowSize (sigma: float) : uint =
    let radius = System.Math.Ceiling(2.0 * sigma) |> uint
    2u * radius + 1u
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
    let raw = imageFromTemporarySimpleITK "gauss.raw" 0 (f.Execute())
    try
        let discreteSum = sum raw
        if discreteSum = 0.0 then
            invalidOp "Gaussian kernel has zero discrete sum."

        let normalized = imageDivScalar raw discreteSum
        if typeof<'T> = typeof<float> then
            unbox<Image<'T>> (box normalized)
        else
            try
                Image<'T>.ofSimpleITK(normalized.toSimpleITK(),"gauss")
            finally
                normalized.decRefCount()
    finally
        raw.decRefCount()

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

let finiteDiffFilter3D (direction: uint) (order: uint) : Image<float> =
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

let gradientVector3D (order: uint) (img: Image<float>) : Image<float list> =
    if img.GetDimensions() <> 3u then
        invalidArg "img" $"gradientVector3D: expected a 3D image, got {img.GetDimensions()}D."

    let derivative direction =
        let kernel = finiteDiffFilter3D direction order
        try
            convolve None None img kernel
        finally
            kernel.decRefCount()

    let derivatives = [ 0u; 1u; 2u ] |> List.map derivative
    try
        Image<float>.ofImageList derivatives
    finally
        derivatives |> List.iter (fun derivativeImage -> derivativeImage.decRefCount())

let discreteGaussian (dim: uint) (sigma: float) (kernelSize: uint option) (outputRegionMode: OutputRegionMode option) (boundaryCondition: BoundaryCondition option) : Image<'T> -> Image<'T> =
    fun (input: Image<'T>) ->
        let kern = gauss dim sigma kernelSize
        try
            convolve outputRegionMode boundaryCondition input kern
        finally
            kern.decRefCount()

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
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetKernelType(itk.simple.KernelEnum.sitkBall)
            f.SetForegroundValue(1.0)
            f.SetBackgroundValue(0.0))
        (fun f x -> f.Execute(x))

/// Binary dilation
let binaryDilate (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryDilate"
        (fun () -> new itk.simple.BinaryDilateImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetKernelType(itk.simple.KernelEnum.sitkBall)
            f.SetForegroundValue(1.0)
            f.SetBackgroundValue(0.0))
        (fun f x -> f.Execute(x))

let private sphericalOffsets dimensions (radius: uint) =
    if dimensions <> 2u && dimensions <> 3u then
        invalidArg "dimensions" $"Spherical binary dilation supports 2D and 3D images, got {dimensions}D."

    let r = int radius
    let radiusLimit = r * r + r
    let offsets = ResizeArray<int * int * int>()
    let zMin, zMax = if dimensions = 2u then 0, 0 else -r, r

    for dz in zMin .. zMax do
        for dy in -r .. r do
            for dx in -r .. r do
                let d2 = dx * dx + dy * dy + dz * dz
                if d2 <= radiusLimit then
                    offsets.Add(dx, dy, dz)

    offsets
    |> Seq.sortBy (fun (dx, dy, dz) -> dx * dx + dy * dy + dz * dz)
    |> Seq.toArray

/// CPU implementation of binary dilation with a digital spherical structuring element.
///
/// This is intended as an experimental native baseline beside the SimpleITK implementation.
/// It uses the same binary convention as <c>binaryDilate</c>: foreground pixels have value 1
/// and the output contains only 0/1 values. The footprint follows SimpleITK's <c>sitkBall</c>
/// convention so results can be compared directly.
let binaryDilateSphericalNative (radius: uint) (img: Image<uint8>) : Image<uint8> =
    let dimensions = img.GetDimensions()
    if dimensions <> 2u && dimensions <> 3u then
        invalidArg "img" $"binaryDilateSphericalNative supports 2D and 3D images, got {dimensions}D."

    let offsets = sphericalOffsets dimensions radius

    if dimensions = 2u then
        let width = int (img.GetWidth())
        let height = int (img.GetHeight())
        let input = copyScalarPixels<uint8> img.Image (width * height)
        let output = Array.zeroCreate<uint8> (width * height)

        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                let mutable found = false
                let mutable i = 0
                while not found && i < offsets.Length do
                    let dx, dy, _dz = offsets[i]
                    let xx = x + dx
                    let yy = y + dy
                    if xx >= 0 && xx < width && yy >= 0 && yy < height && input[yy * width + xx] = 1uy then
                        found <- true
                    else
                        i <- i + 1

                if found then
                    output[y * width + x] <- 1uy

        Image<uint8>.ofSimpleITKNDispose(importScalarImage (img.GetSize()) output, "binaryDilateSphericalNative", img.index)
    else
        let width = int (img.GetWidth())
        let height = int (img.GetHeight())
        let depth = int (img.GetDepth())
        let input = copyScalarPixels<uint8> img.Image (width * height * depth)
        let output = Array.zeroCreate<uint8> (width * height * depth)
        let plane = width * height

        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    let mutable found = false
                    let mutable i = 0
                    while not found && i < offsets.Length do
                        let dx, dy, dz = offsets[i]
                        let xx = x + dx
                        let yy = y + dy
                        let zz = z + dz
                        if xx >= 0 && xx < width && yy >= 0 && yy < height && zz >= 0 && zz < depth && input[zz * plane + yy * width + xx] = 1uy then
                            found <- true
                        else
                            i <- i + 1

                    if found then
                        output[z * plane + y * width + x] <- 1uy

        Image<uint8>.ofSimpleITKNDispose(importScalarImage (img.GetSize()) output, "binaryDilateSphericalNative", img.index)

let private zonohedralBestCoefficients =
    // Best-approximation coefficients from Gorpho/pygorpho's implementation of
    // Jensen et al., "Zonohedral Approximation of Spherical Structuring Element
    // for Volumetric Morphology" (SCIA 2019).
    [| 2, 0, 0
       4, 0, 0
       3, 0, 2
       4, 0, 2
       3, 2, 2
       4, 2, 2
       3, 2, 3
       4, 3, 2
       3, 3, 3
       4, 4, 2
       6, 3, 3
       4, 4, 3
       6, 4, 3
       8, 3, 4
       6, 4, 4
       7, 5, 3
       6, 5, 4
       7, 6, 3
       9, 5, 4
       7, 6, 4
       9, 5, 5
       7, 7, 4
       9, 6, 5
       11, 5, 6
       9, 7, 5
       11, 6, 6
       12, 7, 5
       10, 8, 5
       9, 8, 6
       10, 9, 5
       12, 8, 6
       14, 7, 7 |]

let private zonohedralLineSteps =
    [| ( 1,  0,  0)
       ( 0, -1,  0)
       ( 0,  0,  1)
       ( 1,  1,  0)
       (-1,  1,  0)
       (-1,  0, -1)
       ( 1,  0, -1)
       ( 0,  1,  1)
       ( 0, -1,  1)
       (-1, -1, -1)
       ( 1,  1, -1)
       ( 1, -1,  1)
       (-1,  1,  1) |]

let zonohedralBestLines radius =
    if radius = 0u then
        [||]
    elif int radius <= zonohedralBestCoefficients.Length then
        let a1, a2, a3 = zonohedralBestCoefficients[int radius - 1]
        let lengths =
            [| a1; a1; a1
               a2; a2; a2; a2; a2; a2
               a3; a3; a3; a3 |]

        Array.zip zonohedralLineSteps lengths
        |> Array.choose (fun ((dx, dy, dz), length) ->
            if length > 1 then Some(dx, dy, dz, length) else None)
    else
        invalidArg "radius" $"binaryDilateZonohedralNative currently has best-approximation coefficients through radius {zonohedralBestCoefficients.Length}; got radius {radius}."

let private vhgwDilateLine (length: int) (count: int) (line: uint8[]) (prefix: uint8[]) (suffix: uint8[]) (lineOutput: uint8[]) =
    if count > 0 then
        let left = length - length / 2 - 1
        let right = length / 2

        for i in 0 .. count - 1 do
            if i % length = 0 then
                prefix[i] <- line[i]
            else
                prefix[i] <- max prefix[i - 1] line[i]

        for i in count - 1 .. -1 .. 0 do
            if i = count - 1 || i % length = length - 1 then
                suffix[i] <- line[i]
            else
                suffix[i] <- max suffix[i + 1] line[i]

        for i in 0 .. count - 1 do
            let lo = max 0 (i - left)
            let hi = min (count - 1) (i + right)
            lineOutput[i] <- max suffix[lo] prefix[hi]

let private lineStarts3D width height depth dx dy dz =
    let seen = HashSet<int>()
    let starts = ResizeArray<int * int * int>()
    let add x y z =
        if x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth then
            let key = z * width * height + y * width + x
            if seen.Add key then
                starts.Add(x, y, z)

    if dx > 0 then
        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                add 0 y z
    elif dx < 0 then
        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                add (width - 1) y z

    if dy > 0 then
        for z in 0 .. depth - 1 do
            for x in 0 .. width - 1 do
                add x 0 z
    elif dy < 0 then
        for z in 0 .. depth - 1 do
            for x in 0 .. width - 1 do
                add x (height - 1) z

    if dz > 0 then
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                add x y 0
    elif dz < 0 then
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                add x y (depth - 1)

    starts
    |> Seq.toArray

let private lineDilate3D (width: int) (height: int) (depth: int) (input: uint8[]) (dx: int, dy: int, dz: int, length: int) =
    let output = Array.zeroCreate<uint8> input.Length
    let plane = width * height
    let maxLine = max width (max height depth)
    let indices = Array.zeroCreate<int> maxLine
    let line = Array.zeroCreate<uint8> maxLine
    let prefix = Array.zeroCreate<uint8> maxLine
    let suffix = Array.zeroCreate<uint8> maxLine
    let lineOutput = Array.zeroCreate<uint8> maxLine

    for x0, y0, z0 in lineStarts3D width height depth dx dy dz do
        let mutable x = x0
        let mutable y = y0
        let mutable z = z0
        let mutable count = 0

        while x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth do
            let index = z * plane + y * width + x
            indices[count] <- index
            line[count] <- input[index]
            count <- count + 1
            x <- x + dx
            y <- y + dy
            z <- z + dz

        vhgwDilateLine length count line prefix suffix lineOutput

        for i in 0 .. count - 1 do
            output[indices[i]] <- lineOutput[i]

    output

let private lineErode3D (width: int) (height: int) (depth: int) (input: uint8[]) (dx: int, dy: int, dz: int, length: int) =
    let output = Array.zeroCreate<uint8> input.Length
    let plane = width * height
    let left = length - length / 2 - 1
    let right = length / 2

    for z in 0 .. depth - 1 do
        for y in 0 .. height - 1 do
            let row = z * plane + y * width
            for x in 0 .. width - 1 do
                let mutable inside = true
                let mutable t = -left
                while inside && t <= right do
                    let xx = x + t * dx
                    let yy = y + t * dy
                    let zz = z + t * dz
                    if xx < 0 || xx >= width || yy < 0 || yy >= height || zz < 0 || zz >= depth || input[zz * plane + yy * width + xx] <> 1uy then
                        inside <- false
                    t <- t + 1
                if inside then
                    output[row + x] <- 1uy

    output

let private lineDilate3DRange
    (width: int)
    (height: int)
    (depth: int)
    (inputValidLow: int)
    (inputValidHigh: int)
    (outputLow: int)
    (outputHigh: int)
    (input: uint8[])
    (dx: int, dy: int, dz: int, length: int) =
    let output = Array.zeroCreate<uint8> input.Length
    let plane = width * height
    let maxLine = max width (max height depth)
    let indices = Array.zeroCreate<int> maxLine
    let zValues = Array.zeroCreate<int> maxLine
    let line = Array.zeroCreate<uint8> maxLine
    let prefix = Array.zeroCreate<uint8> maxLine
    let suffix = Array.zeroCreate<uint8> maxLine
    let lineOutput = Array.zeroCreate<uint8> maxLine

    for x0, y0, z0 in lineStarts3D width height depth dx dy dz do
        let mutable x = x0
        let mutable y = y0
        let mutable z = z0
        let mutable count = 0

        while x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth do
            if z >= inputValidLow && z <= inputValidHigh then
                let index = z * plane + y * width + x
                indices[count] <- index
                zValues[count] <- z
                line[count] <- input[index]
                count <- count + 1
            x <- x + dx
            y <- y + dy
            z <- z + dz

        vhgwDilateLine length count line prefix suffix lineOutput

        for i in 0 .. count - 1 do
            let z = zValues[i]
            if z >= outputLow && z <= outputHigh then
                output[indices[i]] <- lineOutput[i]

    output

let private expandZRangeForLine depth outputLow outputHigh (_dx: int, _dy: int, dz: int, length: int) =
    let left = length - length / 2 - 1
    let right = length / 2
    let zA = -left * dz
    let zB = right * dz
    max 0 (outputLow + min zA zB), min (depth - 1) (outputHigh + max zA zB)

let zonohedralZHalo radius =
    zonohedralBestLines radius
    |> Array.sumBy (fun (_dx, _dy, dz, length) ->
        let left = length - length / 2 - 1
        let right = length / 2
        max (abs (left * dz)) (abs (right * dz)))

let binaryDilateZonohedralValidSlicesNative (radius: uint) (outputStart: uint) (outputCount: uint) (images: Image<uint8> list) : Image<uint8> list =
    match images with
    | [] -> []
    | first :: _ ->
        let width = int (first.GetWidth())
        let height = int (first.GetHeight())
        let depth = images.Length
        let plane = width * height
        let requestedLow = int outputStart
        let requestedHigh = min (depth - 1) (requestedLow + int outputCount - 1)

        if requestedLow > requestedHigh then
            []
        else
            let lines = zonohedralBestLines radius
            let neededAfter = Array.zeroCreate<int * int> lines.Length
            let mutable neededLow = requestedLow
            let mutable neededHigh = requestedHigh

            for i in lines.Length - 1 .. -1 .. 0 do
                neededAfter[i] <- neededLow, neededHigh
                let lo, hi = expandZRangeForLine depth neededLow neededHigh lines[i]
                neededLow <- lo
                neededHigh <- hi

            let mutable current = Array.zeroCreate<uint8> (plane * depth)
            images
            |> List.iteri (fun z image ->
                if image.GetWidth() <> first.GetWidth() || image.GetHeight() <> first.GetHeight() then
                    invalidArg "images" "All slices in a zonohedral dilation window must have the same width and height."
                let slice = copyScalarPixels<uint8> image.Image plane
                Array.Copy(slice, 0, current, z * plane, plane))

            let mutable validLow = 0
            let mutable validHigh = depth - 1

            for i in 0 .. lines.Length - 1 do
                let outLow, outHigh = neededAfter[i]
                current <- lineDilate3DRange width height depth validLow validHigh outLow outHigh current lines[i]
                validLow <- outLow
                validHigh <- outHigh

            [ for z in requestedLow .. requestedHigh do
                let slice = Array.zeroCreate<uint8> plane
                Array.Copy(current, z * plane, slice, 0, plane)
                let index =
                    if z < images.Length then images[z].index
                    else first.index + z
                yield Image<uint8>.ofSimpleITKNDispose(importScalarImage [ uint width; uint height ] slice, "binaryDilateZonohedralValidSlicesNative", index) ]

/// Experimental binary dilation using Jensen et al.'s zonohedral best approximation of a spherical structuring element.
///
/// The approximation is represented as a composition of line dilations in the 13 directions used by
/// Gorpho/pygorpho. It is only a 3D experimental baseline for now.
let binaryDilateZonohedralNative (radius: uint) (img: Image<uint8>) : Image<uint8> =
    if img.GetDimensions() <> 3u then
        invalidArg "img" $"binaryDilateZonohedralNative supports 3D images, got {img.GetDimensions()}D."

    let width = int (img.GetWidth())
    let height = int (img.GetHeight())
    let depth = int (img.GetDepth())
    let lines = zonohedralBestLines radius

    let mutable current = copyScalarPixels<uint8> img.Image (width * height * depth)
    for line in lines do
        current <- lineDilate3D width height depth current line

    Image<uint8>.ofSimpleITKNDispose(importScalarImage (img.GetSize()) current, "binaryDilateZonohedralNative", img.index)

/// Experimental binary erosion using the same zonohedral approximation as
/// <c>binaryDilateZonohedralNative</c>.
let binaryErodeZonohedralNative (radius: uint) (img: Image<uint8>) : Image<uint8> =
    if img.GetDimensions() <> 3u then
        invalidArg "img" $"binaryErodeZonohedralNative supports 3D images, got {img.GetDimensions()}D."

    let width = int (img.GetWidth())
    let height = int (img.GetHeight())
    let depth = int (img.GetDepth())
    let lines = zonohedralBestLines radius

    let mutable current = copyScalarPixels<uint8> img.Image (width * height * depth)
    for line in lines do
        current <- lineErode3D width height depth current line

    Image<uint8>.ofSimpleITKNDispose(importScalarImage (img.GetSize()) current, "binaryErodeZonohedralNative", img.index)

/// Binary opening (erode then dilate)
let binaryOpening (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryOpening"
        (fun () -> new itk.simple.BinaryMorphologicalOpeningImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetKernelType(itk.simple.KernelEnum.sitkBall)
            f.SetForegroundValue(1.0)
            f.SetBackgroundValue(0.0))
        (fun f x -> f.Execute(x))

/// Binary closing (dilate then erode)
let binaryClosing (radius: uint) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryClosing"
        (fun () -> new itk.simple.BinaryMorphologicalClosingImageFilter())
        (fun f ->
            f.SetKernelRadius(radius)
            f.SetKernelType(itk.simple.KernelEnum.sitkBall)
            // SimpleITK's closing filter exposes foreground but not background configuration.
            f.SetForegroundValue(1.0))
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
    imageFromTemporarySimpleITK "binaryFillHoles" img.index (filter.Execute(img.toSimpleITK()))

let binaryContour (fullyConnected: bool) : Image<uint8> -> Image<uint8> =
    makeUnaryImageOperatorWith
        "binaryContour"
        (fun () -> new itk.simple.BinaryContourImageFilter())
        (fun f -> f.SetFullyConnected(fullyConnected))
        (fun f x -> f.Execute(x))

let binaryThinning (img: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.BinaryThinningImageFilter()
    imageFromTemporarySimpleITK "binaryThinning" img.index (filter.Execute(img.toSimpleITK()))

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
    imageFromTemporarySimpleITK "binaryReconstructionByDilation" marker.index (filter.Execute(marker.toSimpleITK(), mask.toSimpleITK()))

let binaryReconstructionByErosion (fullyConnected: bool) (marker: Image<uint8>) (mask: Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.BinaryReconstructionByErosionImageFilter()
    filter.SetFullyConnected(fullyConnected)
    imageFromTemporarySimpleITK "binaryReconstructionByErosion" marker.index (filter.Execute(marker.toSimpleITK(), mask.toSimpleITK()))

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
    { Labels = imageFromTemporarySimpleITK "connectedComponents" img.index (filter.Execute(img.toSimpleITK()))
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
let signedDistanceMap (inside: uint8) (outside: uint8) (img: Image<uint8>) : Image<float> =
    ignore inside
    use f = new itk.simple.SignedMaurerDistanceMapImageFilter()
    f.SetBackgroundValue(float outside)
    f.SetSquaredDistance(false)
    f.SetInsideIsPositive(false)
    imageFromTemporarySimpleITK "signedDistanceMap" img.index (f.Execute(img.toSimpleITK()))

let bandSignedDistanceMap (bandRadius: uint) (img: Image<uint8>) : Image<float> =
    if bandRadius = 0u then
        invalidArg "bandRadius" "Band signed distance requires a positive band radius."

    let binary =
        Image.map (fun value -> if value = 0uy then 0uy else 1uy) img

    let distance =
        try
            signedDistanceMap 1uy 0uy binary
        finally
            binary.decRefCount()

    try
        let limit = float bandRadius
        Image.map (fun value -> if abs value < limit then value else System.Double.NaN) distance
    finally
        distance.decRefCount()

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

let private imageValues (img: Image<'T>) : 'T seq =
    match img.GetDimensions() with
    | 2u -> img.toArray2D() |> Seq.cast<'T>
    | 3u -> img.toArray3D() |> Seq.cast<'T>
    | 4u -> img.toArray4D() |> Seq.cast<'T>
    | _ -> Image.fold (fun acc value -> value :: acc) [] img |> List.rev :> seq<'T>
    
let private valuesFromImages bins (images: Image<'T> list) operation =
    if bins < 2u then
        invalidArg "bins" $"{operation} threshold estimation requires at least two bins."
    if List.isEmpty images then
        invalidArg "images" $"{operation} threshold estimation requires at least one image."

    let toFloat value = System.Convert.ToDouble(box value, System.Globalization.CultureInfo.InvariantCulture)
    let values =
        images
        |> List.collect (fun image ->
            imageValues image
            |> Seq.map toFloat
            |> Seq.toList)

    match values with
    | [] -> invalidArg "images" $"{operation} threshold estimation requires at least one pixel."
    | _ -> values

let private binnedHistogram bins values =
    let minValue = values |> List.min
    let maxValue = values |> List.max
    if minValue = maxValue then
        minValue, maxValue, 1.0, [| uint64 values.Length |]
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
        minValue, maxValue, width, histogram

let private orderedHistogramValues (histogram: Map<'T, uint64>) operation =
    if Map.isEmpty histogram then
        invalidArg "histogram" $"{operation} threshold estimation requires a non-empty histogram."

    let ordered =
        histogram
        |> Map.toList
        |> List.choose (fun (value, count) ->
            if count = 0UL then
                None
            else
                Some(System.Convert.ToDouble(box value, System.Globalization.CultureInfo.InvariantCulture), count))
        |> List.sortBy fst

    if List.isEmpty ordered then
        invalidArg "histogram" $"{operation} threshold estimation requires a histogram with non-zero counts."

    ordered

let otsuThresholdFromHistogram (histogram: Map<'T, uint64>) : float =
    let ordered = orderedHistogramValues histogram "Otsu"
    match ordered with
    | [ value, _ ] -> value
    | _ ->
        let totalCount = ordered |> List.sumBy (snd >> float)
        let totalMean = ordered |> List.sumBy (fun (value, count) -> value * float count)
        let mutable bestThreshold = fst ordered[0]
        let mutable bestVariance = System.Double.NegativeInfinity
        let mutable backgroundWeight = 0.0
        let mutable backgroundSum = 0.0

        for index in 0 .. ordered.Length - 2 do
            let value, count = ordered[index]
            backgroundWeight <- backgroundWeight + float count
            backgroundSum <- backgroundSum + value * float count
            let foregroundWeight = totalCount - backgroundWeight
            if backgroundWeight > 0.0 && foregroundWeight > 0.0 then
                let backgroundMean = backgroundSum / backgroundWeight
                let foregroundMean = (totalMean - backgroundSum) / foregroundWeight
                let variance = backgroundWeight * foregroundWeight * pown (backgroundMean - foregroundMean) 2
                if variance > bestVariance then
                    bestVariance <- variance
                    let nextValue = fst ordered[index + 1]
                    bestThreshold <- 0.5 * (value + nextValue)

        bestThreshold

let private otsuThresholdFromImages bins images =
    let values = valuesFromImages bins images "Otsu"
    let minValue, maxValue, width, histogram = binnedHistogram bins values
    if minValue = maxValue then
        minValue
    else
            let binCount = histogram.Length
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
    let thresholdValue = otsuThresholdFromImages 256u [ img ]
    use filter = new itk.simple.BinaryThresholdImageFilter()
    filter.SetLowerThreshold(thresholdValue)
    filter.SetUpperThreshold(System.Double.PositiveInfinity)
    filter.SetInsideValue(1uy)
    filter.SetOutsideValue(0uy)
    imageFromTemporarySimpleITK "otsuThreshold" img.index (filter.Execute(img.toSimpleITK()))

/// Otsu multiple thresholds (returns a label map)
let otsuMultiThreshold (numThresholds: byte) (img: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.OtsuMultipleThresholdsImageFilter()
    filter.SetNumberOfThresholds(numThresholds)
    imageFromTemporarySimpleITK "otsuMultiThreshold" img.index (filter.Execute(img.toSimpleITK()))

/// Moments-based threshold
let momentsThresholdFromHistogram (histogram: Map<'T, uint64>) : float =
    let ordered = orderedHistogramValues histogram "Moments"
    match ordered with
    | [ value, _ ] -> value
    | _ ->
        let totalCount = ordered |> List.sumBy (snd >> float)
        let moment power =
            ordered
            |> List.sumBy (fun (value, count) -> (value ** power) * float count)
            |> fun value -> value / totalCount

        let m0 = 1.0
        let m1 = moment 1.0
        let m2 = moment 2.0
        let m3 = moment 3.0
        let cd = m0 * m2 - m1 * m1

        if abs cd < 1e-12 then
            m1
        else
            let c0 = (-m2 * m2 + m1 * m3) / cd
            let c1 = (-m3 + m2 * m1) / cd
            let discriminant = c1 * c1 - 4.0 * c0

            if discriminant < 0.0 then
                m1
            else
                let root = sqrt discriminant
                let z0 = 0.5 * (-c1 - root)
                let z1 = 0.5 * (-c1 + root)
                let denominator = z1 - z0

                if abs denominator < 1e-12 then
                    m1
                else
                    let p0 = (z1 - m1) / denominator |> max 0.0 |> min 1.0
                    let target = p0 * totalCount
                    let mutable cumulative = 0.0
                    let mutable threshold = fst (List.last ordered)
                    let mutable found = false

                    for index in 0 .. ordered.Length - 1 do
                        let value, count = ordered[index]
                        if not found then
                            cumulative <- cumulative + float count
                            if cumulative >= target then
                                threshold <-
                                    if index < ordered.Length - 1 then
                                        0.5 * (value + fst ordered[index + 1])
                                    else
                                        value
                                found <- true

                    threshold

let private momentsThresholdFromImages bins images =
    let values = valuesFromImages bins images "Moments"
    let minValue, maxValue, width, histogram = binnedHistogram bins values
    if minValue = maxValue then
        minValue
    else
            let binCount = histogram.Length
            let totalCount = histogram |> Array.sumBy float
            let moment power =
                histogram
                |> Array.mapi (fun i count -> (float i ** power) * float count)
                |> Array.sum
                |> fun value -> value / totalCount

            let m0 = 1.0
            let m1 = moment 1.0
            let m2 = moment 2.0
            let m3 = moment 3.0
            let cd = m0 * m2 - m1 * m1

            if abs cd < 1e-12 then
                minValue + width * (m1 + 0.5)
            else
                let c0 = (-m2 * m2 + m1 * m3) / cd
                let c1 = (-m3 + m2 * m1) / cd
                let discriminant = c1 * c1 - 4.0 * c0

                if discriminant < 0.0 then
                    minValue + width * (m1 + 0.5)
                else
                    let root = sqrt discriminant
                    let z0 = 0.5 * (-c1 - root)
                    let z1 = 0.5 * (-c1 + root)
                    let denominator = z1 - z0

                    if abs denominator < 1e-12 then
                        minValue + width * (m1 + 0.5)
                    else
                        let p0 = (z1 - m1) / denominator |> max 0.0 |> min 1.0
                        let target = p0 * totalCount
                        let mutable cumulative = 0.0
                        let mutable thresholdBin = binCount - 1
                        let mutable found = false

                        for i in 0 .. binCount - 1 do
                            if not found then
                                cumulative <- cumulative + float histogram[i]
                                if cumulative >= target then
                                    thresholdBin <- i
                                    found <- true

                        minValue + width * (float thresholdBin + 0.5)

let momentsThreshold (img: Image<'T>) : Image<uint8> =
    let thresholdValue = momentsThresholdFromImages 256u [ img ]
    use filter = new itk.simple.BinaryThresholdImageFilter()
    filter.SetLowerThreshold(thresholdValue)
    filter.SetUpperThreshold(System.Double.PositiveInfinity)
    filter.SetInsideValue(1uy)
    filter.SetOutsideValue(0uy)
    imageFromTemporarySimpleITK "momentsThreshold" img.index (filter.Execute(img.toSimpleITK()))

/// Coordinate fields
// Cannot get TransformToDisplacementFieldFilter to work, so making it by hand.
let generateCoordinateAxis (axis: int) (size: int list) : Image<uint32> =
    match size with
    | [ width; height ] ->
        Array2D.init width height (fun x y ->
            if axis = 0 then uint32 x
            elif axis = 1 then uint32 y
            else invalidArg "axis" $"Axis {axis} is outside a 2D image.")
        |> fun values -> Image<uint32>.ofArray2D(values, "generateCoordinateAxis")
    | [ width; height; depth ] ->
        Array3D.init width height depth (fun x y z ->
            if axis = 0 then uint32 x
            elif axis = 1 then uint32 y
            elif axis = 2 then uint32 z
            else invalidArg "axis" $"Axis {axis} is outside a 3D image.")
        |> fun values -> Image<uint32>.ofArray3D(values, "generateCoordinateAxis")
    | _ ->
        failwith $"Unsupported dimensionality {size.Length}"

let histogram (img: Image<'T>) : Map<'T, uint64> =
    let addValue acc elm =
        Map.change elm (fun vopt -> 
            match vopt with 
                Some v -> Some (v+1uL) 
                | None -> Some (1uL)) 
            acc

    imageValues img |> Seq.fold addValue Map.empty<'T, uint64>

let histogramFixedBins firstLeftEdge lastLeftEdge bins (img: Image<'T>) : Map<float, uint64> =
    if bins = 0u then
        invalidArg (nameof bins) "Histogram bin count must be positive."

    let binWidth =
        if bins = 1u then
            1.0
        else
            let width = (lastLeftEdge - firstLeftEdge) / float (bins - 1u)
            if width <= 0.0 then
                invalidArg (nameof lastLeftEdge) "Last left edge must be greater than first left edge when using more than one histogram bin."
            width

    let leftEdge index = firstLeftEdge + float index * binWidth

    let initial =
        [ 0 .. int bins - 1 ]
        |> List.map (fun index -> leftEdge index, 0UL)
        |> Map.ofList

    let addValue acc (pixel: 'T) =
        let value = Convert.ToDouble(box pixel)
        if Double.IsNaN value || Double.IsInfinity value then
            acc
        else
            let binIndex = int (Math.Floor((value - firstLeftEdge) / binWidth))
            if binIndex < 0 || binIndex >= int bins then
                acc
            else
                let edge = leftEdge binIndex
                Map.change edge (function Some count -> Some(count + 1UL) | None -> Some 1UL) acc

    imageValues img |> Seq.fold addValue initial

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

let quantilesFromHistogram (quantiles: float list) (histogram: Map<'T, uint64>) : float list =
    if Map.isEmpty histogram then
        invalidArg "histogram" "Cannot estimate quantiles from an empty histogram."

    let total = histogram |> Map.fold (fun acc _ count -> acc + count) 0UL
    if total = 0UL then
        invalidArg "histogram" "Cannot estimate quantiles from a histogram with zero total count."

    let ordered =
        histogram
        |> Map.toList
        |> List.map (fun (value, count) -> System.Convert.ToDouble(box value, System.Globalization.CultureInfo.InvariantCulture), count)
        |> List.sortBy fst

    quantiles
    |> List.map (fun quantile ->
        if quantile < 0.0 || quantile > 1.0 || System.Double.IsNaN quantile then
            invalidArg "quantiles" "Quantiles must be finite numbers between 0 and 1."

        let target = uint64 (ceil (quantile * float total))
        let target = max 1UL target
        let mutable cumulative = 0UL

        ordered
        |> List.pick (fun (value, count) ->
            cumulative <- cumulative + count
            if cumulative >= target then Some value else None))

let private retainNoNoise (img: Image<'T>) =
    img.incRefCount()
    img

let addNormalNoise (mean: float) (stddev: float) : Image<'T> -> Image<'T> =
    if stddev <= 0.0 then
        retainNoNoise
    else
        makeUnaryImageOperatorWith
            "addNormalNoise"
            (fun () -> new itk.simple.AdditiveGaussianNoiseImageFilter())
            (fun f -> 
                f.SetMean(mean)
                f.SetStandardDeviation(stddev))
            (fun f x -> f.Execute(x))

let addSaltAndPepperNoise (probability: float) : Image<'T> -> Image<'T> =
    if probability <= 0.0 then
        retainNoNoise
    else
        makeUnaryImageOperatorWith
            "addSaltAndPepperNoise"
            (fun () -> new itk.simple.SaltAndPepperNoiseImageFilter())
            (fun f -> f.SetProbability(probability))
            (fun f x -> f.Execute(x))

let addShotNoise (scale: float) : Image<'T> -> Image<'T> =
    if scale <= 0.0 then
        retainNoNoise
    else
        makeUnaryImageOperatorWith
            "addShotNoise"
            (fun () -> new itk.simple.ShotNoiseImageFilter())
            (fun f -> f.SetScale(scale))
            (fun f x -> f.Execute(x))

let addSpeckleNoise (stddev: float) : Image<'T> -> Image<'T> =
    if stddev <= 0.0 then
        retainNoNoise
    else
        makeUnaryImageOperatorWith
            "addSpeckleNoise"
            (fun () -> new itk.simple.SpeckleNoiseImageFilter())
            (fun f -> f.SetStandardDeviation(stddev))
            (fun f x -> f.Execute(x))

let threshold (lower: float) (upper: float) (img: Image<'T>) : Image<uint8> =
    use filter = new itk.simple.BinaryThresholdImageFilter()
    filter.SetLowerThreshold lower
    filter.SetUpperThreshold upper
    let res = filter.Execute(img.toSimpleITK()); 
    imageFromTemporarySimpleITK "threshold" img.index res

let toVectorImage (images: Image<'T> list) : Image<'T list> =
    Image<'T>.ofImageList images

let private composeVectorAndRelease<'T when 'T : equality> (components: Image<'T> list) =
    try
        Image<'T>.ofImageList components
    finally
        components |> List.iter (fun comp -> comp.decRefCount())

let private mapScalarComponentsAndCompose (f: Image<float> -> Image<float>) (img: Image<float list>) =
    let components = img.toImageList()
    let mapped = components |> List.map f
    components |> List.iter (fun comp -> comp.decRefCount())
    composeVectorAndRelease mapped

let private scalarMapArray name (f: float -> float) (img: Image<float>) =
    match img.GetDimensions() with
    | 2u ->
        img.toArray2D()
        |> Array2D.map f
        |> fun output -> Image<float>.ofArray2D(output, name, img.index)
    | 3u ->
        img.toArray3D()
        |> Array3D.map f
        |> fun output -> Image<float>.ofArray3D(output, name, img.index)
    | dims ->
        failwith $"{name}: only 2D and 3D images are supported, got {dims}D"

let vectorElement<'T when 'T : equality> (componentId: uint) (img: Image<'T list>) : Image<'T> =
    let componentIndex = int componentId
    let componentCount = int (img.GetNumberOfComponentsPerPixel())
    if componentIndex < 0 || componentIndex >= componentCount then
        invalidArg "componentId" $"vectorElement: component {componentId} is outside the available component range 0..{componentCount - 1}."

    use filter = new itk.simple.VectorIndexSelectionCastImageFilter()
    filter.SetIndex(componentId)
    imageFromTemporarySimpleITK "vectorElement" img.index (filter.Execute(img.toSimpleITK()))

let vectorRange<'T when 'T : equality> (firstComponent: uint) (componentCount: uint) (img: Image<'T list>) : Image<'T list> =
    let first = int firstComponent
    let count = int componentCount
    let available = int (img.GetNumberOfComponentsPerPixel())
    if count <= 0 then
        invalidArg "componentCount" "vectorRange: componentCount must be positive."
    if first < 0 || first + count > available then
        invalidArg "firstComponent" $"vectorRange: requested components {first}..{first + count - 1}, but available range is 0..{available - 1}."

    let components = img.toImageList()
    let selected = components |> List.skip first |> List.take count
    try
        let result = Image<'T>.ofImageList selected
        result.index <- img.index
        result
    finally
        components |> List.iter (fun comp -> comp.decRefCount())

let private requireThreeComponents name (img: Image<'T list>) =
    if img.GetNumberOfComponentsPerPixel() <> 3u then
        invalidArg "img" $"{name}: expected a 3-component image, got {img.GetNumberOfComponentsPerPixel()} components."

let private clampByte value =
    if value <= 0.0 then 0uy
    elif value >= 255.0 then 255uy
    else byte (Math.Round value)

let vector3ToColor (inputMinimum: float) (inputMaximum: float) (img: Image<float list>) : Image<uint8 list> =
    requireThreeComponents "vector3ToColor" img
    if inputMaximum <= inputMinimum then
        invalidArg "inputMaximum" "vector3ToColor: inputMaximum must be larger than inputMinimum."

    let scale = 255.0 / (inputMaximum - inputMinimum)
    let components = img.toImageList()
    let mapped =
        components
        |> List.map (fun comp ->
            match comp.GetDimensions() with
            | 2u ->
                comp.toArray2D()
                |> Array2D.map (fun value -> (value - inputMinimum) * scale |> clampByte)
                |> fun output -> Image<uint8>.ofArray2D(output, "vector3ToColor", comp.index)
            | 3u ->
                comp.toArray3D()
                |> Array3D.map (fun value -> (value - inputMinimum) * scale |> clampByte)
                |> fun output -> Image<uint8>.ofArray3D(output, "vector3ToColor", comp.index)
            | dims ->
                failwith $"vector3ToColor: only 2D and 3D images are supported, got {dims}D")
    components |> List.iter (fun comp -> comp.decRefCount())
    composeVectorAndRelease mapped

let colorToVector3 (outputMinimum: float) (outputMaximum: float) (img: Image<uint8 list>) : Image<float list> =
    requireThreeComponents "colorToVector3" img
    if outputMaximum <= outputMinimum then
        invalidArg "outputMaximum" "colorToVector3: outputMaximum must be larger than outputMinimum."

    let scale = (outputMaximum - outputMinimum) / 255.0
    let components = img.toImageList()
    let mapped =
        components
        |> List.map (fun comp ->
            match comp.GetDimensions() with
            | 2u ->
                comp.toArray2D()
                |> Array2D.map (fun value -> outputMinimum + float value * scale)
                |> fun output -> Image<float>.ofArray2D(output, "colorToVector3", comp.index)
            | 3u ->
                comp.toArray3D()
                |> Array3D.map (fun value -> outputMinimum + float value * scale)
                |> fun output -> Image<float>.ofArray3D(output, "colorToVector3", comp.index)
            | dims ->
                failwith $"colorToVector3: only 2D and 3D images are supported, got {dims}D")
    components |> List.iter (fun comp -> comp.decRefCount())
    composeVectorAndRelease mapped

let appendVectorElement (vector: Image<float list>) (element: Image<float>) : Image<float list> =
    if vector.GetSize() <> element.GetSize() then
        invalidArg "element" $"appendVectorElement: image sizes differ: {vector.GetSize()} vs {element.GetSize()}."

    let components = vector.toImageList()
    try
        Image<float>.ofImageList (components @ [ element ])
    finally
        components |> List.iter (fun comp -> comp.decRefCount())

let mapVectorElements (f: float -> float) (img: Image<float list>) : Image<float list> =
    mapScalarComponentsAndCompose (scalarMapArray "mapVectorElements" f) img

let private ensureMatchingVectorImages name (a: Image<float list>) (b: Image<float list>) =
    if a.GetSize() <> b.GetSize() then
        invalidArg "b" $"{name}: image sizes differ: {a.GetSize()} vs {b.GetSize()}."
    if a.GetNumberOfComponentsPerPixel() <> b.GetNumberOfComponentsPerPixel() then
        invalidArg "b" $"{name}: component counts differ: {a.GetNumberOfComponentsPerPixel()} vs {b.GetNumberOfComponentsPerPixel()}."

let vectorDot (a: Image<float list>) (b: Image<float list>) : Image<float> =
    ensureMatchingVectorImages "vectorDot" a b
    let aComponents = a.toImageList()
    let bComponents = b.toImageList()
    try
        match a.GetDimensions() with
        | 2u ->
            let av = aComponents |> List.map (fun comp -> comp.toArray2D())
            let bv = bComponents |> List.map (fun comp -> comp.toArray2D())
            Array2D.init (av.Head.GetLength 0) (av.Head.GetLength 1) (fun x y ->
                List.zip av bv |> List.sumBy (fun (aValues, bValues) -> aValues[x, y] * bValues[x, y]))
            |> fun output -> Image<float>.ofArray2D(output, "vectorDot", a.index)
        | 3u ->
            let av = aComponents |> List.map (fun comp -> comp.toArray3D())
            let bv = bComponents |> List.map (fun comp -> comp.toArray3D())
            Array3D.init (av.Head.GetLength 0) (av.Head.GetLength 1) (av.Head.GetLength 2) (fun x y z ->
                List.zip av bv |> List.sumBy (fun (aValues, bValues) -> aValues[x, y, z] * bValues[x, y, z]))
            |> fun output -> Image<float>.ofArray3D(output, "vectorDot", a.index)
        | dims ->
            failwith $"vectorDot: only 2D and 3D images are supported, got {dims}D"
    finally
        aComponents |> List.iter (fun comp -> comp.decRefCount())
        bComponents |> List.iter (fun comp -> comp.decRefCount())

let vectorCross3D (a: Image<float list>) (b: Image<float list>) : Image<float list> =
    ensureMatchingVectorImages "vectorCross3D" a b
    if a.GetNumberOfComponentsPerPixel() <> 3u then
        invalidArg "a" $"vectorCross3D: expected 3-component vector images, got {a.GetNumberOfComponentsPerPixel()} components."

    let aComponents = a.toImageList()
    let bComponents = b.toImageList()
    let make2D (ax: float[,]) (ay: float[,]) (az: float[,]) (bx: float[,]) (by: float[,]) (bz: float[,]) =
        [ Array2D.init (ax.GetLength 0) (ax.GetLength 1) (fun x y -> ay[x, y] * bz[x, y] - az[x, y] * by[x, y]) |> fun values -> Image<float>.ofArray2D(values, "vectorCross3D", a.index)
          Array2D.init (ax.GetLength 0) (ax.GetLength 1) (fun x y -> az[x, y] * bx[x, y] - ax[x, y] * bz[x, y]) |> fun values -> Image<float>.ofArray2D(values, "vectorCross3D", a.index)
          Array2D.init (ax.GetLength 0) (ax.GetLength 1) (fun x y -> ax[x, y] * by[x, y] - ay[x, y] * bx[x, y]) |> fun values -> Image<float>.ofArray2D(values, "vectorCross3D", a.index) ]
    let make3D (ax: float[,,]) (ay: float[,,]) (az: float[,,]) (bx: float[,,]) (by: float[,,]) (bz: float[,,]) =
        [ Array3D.init (ax.GetLength 0) (ax.GetLength 1) (ax.GetLength 2) (fun x y z -> ay[x, y, z] * bz[x, y, z] - az[x, y, z] * by[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "vectorCross3D", a.index)
          Array3D.init (ax.GetLength 0) (ax.GetLength 1) (ax.GetLength 2) (fun x y z -> az[x, y, z] * bx[x, y, z] - ax[x, y, z] * bz[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "vectorCross3D", a.index)
          Array3D.init (ax.GetLength 0) (ax.GetLength 1) (ax.GetLength 2) (fun x y z -> ax[x, y, z] * by[x, y, z] - ay[x, y, z] * bx[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "vectorCross3D", a.index) ]
    try
        let resultComponents =
            match aComponents, bComponents, a.GetDimensions() with
            | [ ax; ay; az ], [ bx; by; bz ], 2u ->
                make2D (ax.toArray2D()) (ay.toArray2D()) (az.toArray2D()) (bx.toArray2D()) (by.toArray2D()) (bz.toArray2D())
            | [ ax; ay; az ], [ bx; by; bz ], 3u ->
                make3D (ax.toArray3D()) (ay.toArray3D()) (az.toArray3D()) (bx.toArray3D()) (by.toArray3D()) (bz.toArray3D())
            | _, _, dims ->
                failwith $"vectorCross3D: only 2D and 3D images are supported, got {dims}D"
        composeVectorAndRelease resultComponents
    finally
        aComponents |> List.iter (fun comp -> comp.decRefCount())
        bComponents |> List.iter (fun comp -> comp.decRefCount())

let vectorAngleTo (reference: float list) (img: Image<float list>) : Image<float> =
    if reference.Length <> int (img.GetNumberOfComponentsPerPixel()) then
        invalidArg "reference" $"vectorAngleTo: reference vector has {reference.Length} components, image has {img.GetNumberOfComponentsPerPixel()}."

    let referenceNorm = reference |> List.sumBy (fun value -> value * value) |> sqrt
    if referenceNorm < 1e-18 then
        invalidArg "reference" "vectorAngleTo: reference vector must be non-zero."

    let angle values =
        let valueNorm = values |> Seq.sumBy (fun value -> value * value) |> sqrt
        if valueNorm < 1e-18 then
            System.Double.NaN
        else
            let cosTheta =
                Seq.zip values reference
                |> Seq.sumBy (fun (value, refValue) -> value * refValue)
                |> fun dot -> dot / (valueNorm * referenceNorm)
                |> max -1.0
                |> min 1.0
            acos cosTheta

    let components = img.toImageList()
    try
        match img.GetDimensions() with
        | 2u ->
            let values = components |> List.map (fun comp -> comp.toArray2D())
            Array2D.init (values.Head.GetLength 0) (values.Head.GetLength 1) (fun x y ->
                values |> Seq.map (fun comp -> comp[x, y]) |> angle)
            |> fun output -> Image<float>.ofArray2D(output, "vectorAngleTo", img.index)
        | 3u ->
            let values = components |> List.map (fun comp -> comp.toArray3D())
            Array3D.init (values.Head.GetLength 0) (values.Head.GetLength 1) (values.Head.GetLength 2) (fun x y z ->
                values |> Seq.map (fun comp -> comp[x, y, z]) |> angle)
            |> fun output -> Image<float>.ofArray3D(output, "vectorAngleTo", img.index)
        | dims ->
            failwith $"vectorAngleTo: only 2D and 3D images are supported, got {dims}D"
    finally
        components |> List.iter (fun comp -> comp.decRefCount())

let structureTensorOuterProduct (gradient: Image<float list>) : Image<float list> =
    if gradient.GetNumberOfComponentsPerPixel() <> 3u then
        invalidArg "gradient" $"structureTensorOuterProduct: expected a 3-component gradient image, got {gradient.GetNumberOfComponentsPerPixel()} components."

    let components = gradient.toImageList()
    let make2D (gx: float[,]) (gy: float[,]) (gz: float[,]) =
        [ Array2D.init (gx.GetLength 0) (gx.GetLength 1) (fun x y -> gx[x, y] * gx[x, y]) |> fun values -> Image<float>.ofArray2D(values, "structureTensorOuterProduct", gradient.index)
          Array2D.init (gx.GetLength 0) (gx.GetLength 1) (fun x y -> gx[x, y] * gy[x, y]) |> fun values -> Image<float>.ofArray2D(values, "structureTensorOuterProduct", gradient.index)
          Array2D.init (gx.GetLength 0) (gx.GetLength 1) (fun x y -> gx[x, y] * gz[x, y]) |> fun values -> Image<float>.ofArray2D(values, "structureTensorOuterProduct", gradient.index)
          Array2D.init (gx.GetLength 0) (gx.GetLength 1) (fun x y -> gy[x, y] * gy[x, y]) |> fun values -> Image<float>.ofArray2D(values, "structureTensorOuterProduct", gradient.index)
          Array2D.init (gx.GetLength 0) (gx.GetLength 1) (fun x y -> gy[x, y] * gz[x, y]) |> fun values -> Image<float>.ofArray2D(values, "structureTensorOuterProduct", gradient.index)
          Array2D.init (gx.GetLength 0) (gx.GetLength 1) (fun x y -> gz[x, y] * gz[x, y]) |> fun values -> Image<float>.ofArray2D(values, "structureTensorOuterProduct", gradient.index) ]
    let make3D (gx: float[,,]) (gy: float[,,]) (gz: float[,,]) =
        [ Array3D.init (gx.GetLength 0) (gx.GetLength 1) (gx.GetLength 2) (fun x y z -> gx[x, y, z] * gx[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "structureTensorOuterProduct", gradient.index)
          Array3D.init (gx.GetLength 0) (gx.GetLength 1) (gx.GetLength 2) (fun x y z -> gx[x, y, z] * gy[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "structureTensorOuterProduct", gradient.index)
          Array3D.init (gx.GetLength 0) (gx.GetLength 1) (gx.GetLength 2) (fun x y z -> gx[x, y, z] * gz[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "structureTensorOuterProduct", gradient.index)
          Array3D.init (gx.GetLength 0) (gx.GetLength 1) (gx.GetLength 2) (fun x y z -> gy[x, y, z] * gy[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "structureTensorOuterProduct", gradient.index)
          Array3D.init (gx.GetLength 0) (gx.GetLength 1) (gx.GetLength 2) (fun x y z -> gy[x, y, z] * gz[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "structureTensorOuterProduct", gradient.index)
          Array3D.init (gx.GetLength 0) (gx.GetLength 1) (gx.GetLength 2) (fun x y z -> gz[x, y, z] * gz[x, y, z]) |> fun values -> Image<float>.ofArray3D(values, "structureTensorOuterProduct", gradient.index) ]
    try
        let resultComponents =
            match components, gradient.GetDimensions() with
            | [ gx; gy; gz ], 2u -> make2D (gx.toArray2D()) (gy.toArray2D()) (gz.toArray2D())
            | [ gx; gy; gz ], 3u -> make3D (gx.toArray3D()) (gy.toArray3D()) (gz.toArray3D())
            | _, dims -> failwith $"structureTensorOuterProduct: only 2D and 3D images are supported, got {dims}D"
        composeVectorAndRelease resultComponents
    finally
        components |> List.iter (fun comp -> comp.decRefCount())

let smoothVectorElements3D (sigma: float) (img: Image<float list>) : Image<float list> =
    if img.GetDimensions() <> 3u then
        invalidArg "img" $"smoothVectorElements3D: expected a 3D vector image, got {img.GetDimensions()}D."
    if sigma <= 0.0 then
        mapVectorElements id img
    else
        let roundFloatToUint v = uint (v + 0.5)
        let kernelSize = max 1u (4.0 * sigma + 1.0 |> roundFloatToUint)
        let components = img.toImageList()
        // Reuse the same Gaussian kernel for every vector component; constructing it per component is avoidable overhead.
        let kern = gauss 3u sigma (Some kernelSize)
        let smoothed =
            components
            |> List.map (fun elementImage ->
                convolve None None elementImage kern)
        try
            Image<float>.ofImageList smoothed
        finally
            kern.decRefCount()
            components |> List.iter (fun elementImage -> elementImage.decRefCount())
            smoothed |> List.iter (fun elementImage -> elementImage.decRefCount())

let structureTensorEigenImages (tensor: Image<float list>) : Image<float list> list =
    if tensor.GetNumberOfComponentsPerPixel() <> 6u then
        invalidArg "tensor" $"structureTensorEigenImages: expected a 6-component symmetric tensor image, got {tensor.GetNumberOfComponentsPerPixel()} components."

    if tensor.GetDimensions() = 2u then
        let input = Image<float>.toArray3DVector tensor
        let width = input.GetLength 0
        let height = input.GetLength 1
        let eigenvalues = Array3D.zeroCreate<float> width height 3
        let eigenvector0 = Array3D.zeroCreate<float> width height 3
        let eigenvector1 = Array3D.zeroCreate<float> width height 3
        let eigenvector2 = Array3D.zeroCreate<float> width height 3

        for x in 0 .. width - 1 do
            for y in 0 .. height - 1 do
                let xx = input[x, y, 0]
                let xy = input[x, y, 1]
                let xz = input[x, y, 2]
                let yy = input[x, y, 3]
                let yz = input[x, y, 4]
                let zz = input[x, y, 5]
                let matrix =
                    { m00 = xx; m01 = xy; m02 = xz
                      m10 = xy; m11 = yy; m12 = yz
                      m20 = xz; m21 = yz; m22 = zz }
                let eigen = symmetricEigen matrix
                let values = eigen |> List.map fst
                for k in 0 .. 2 do
                    eigenvalues[x, y, k] <- values[k]
                eigen
                |> List.map snd
                |> function
                    | [ v0; v1; v2 ] ->
                        eigenvector0[x, y, 0] <- v0.x
                        eigenvector0[x, y, 1] <- v0.y
                        eigenvector0[x, y, 2] <- v0.z
                        eigenvector1[x, y, 0] <- v1.x
                        eigenvector1[x, y, 1] <- v1.y
                        eigenvector1[x, y, 2] <- v1.z
                        eigenvector2[x, y, 0] <- v2.x
                        eigenvector2[x, y, 1] <- v2.y
                        eigenvector2[x, y, 2] <- v2.z
                    | _ -> failwith "structureTensorEigenImages: expected three eigenvectors."

        [ Image<float>.ofArray3DVector(eigenvalues, "structureTensorEigenValues", tensor.index)
          Image<float>.ofArray3DVector(eigenvector0, "structureTensorEigenvector0", tensor.index)
          Image<float>.ofArray3DVector(eigenvector1, "structureTensorEigenvector1", tensor.index)
          Image<float>.ofArray3DVector(eigenvector2, "structureTensorEigenvector2", tensor.index) ]
    elif tensor.GetDimensions() = 3u then
        let components = tensor.toImageList()
        try
            match components with
            | [ xxImg; xyImg; xzImg; yyImg; yzImg; zzImg ] ->
                let xxValues = xxImg.toArray3D()
                let xyValues = xyImg.toArray3D()
                let xzValues = xzImg.toArray3D()
                let yyValues = yyImg.toArray3D()
                let yzValues = yzImg.toArray3D()
                let zzValues = zzImg.toArray3D()
                let width = xxValues.GetLength 0
                let height = xxValues.GetLength 1
                let depth = xxValues.GetLength 2
                let eigenvalues = [ for _ in 1 .. 3 -> Array3D.zeroCreate<float> width height depth ]
                let eigenvector0 = [ for _ in 1 .. 3 -> Array3D.zeroCreate<float> width height depth ]
                let eigenvector1 = [ for _ in 1 .. 3 -> Array3D.zeroCreate<float> width height depth ]
                let eigenvector2 = [ for _ in 1 .. 3 -> Array3D.zeroCreate<float> width height depth ]

                for x in 0 .. width - 1 do
                    for y in 0 .. height - 1 do
                        for z in 0 .. depth - 1 do
                            let matrix =
                                { m00 = xxValues[x, y, z]; m01 = xyValues[x, y, z]; m02 = xzValues[x, y, z]
                                  m10 = xyValues[x, y, z]; m11 = yyValues[x, y, z]; m12 = yzValues[x, y, z]
                                  m20 = xzValues[x, y, z]; m21 = yzValues[x, y, z]; m22 = zzValues[x, y, z] }
                            let eigen = symmetricEigen matrix
                            let values = eigen |> List.map fst
                            for k in 0 .. 2 do
                                eigenvalues[k][x, y, z] <- values[k]
                            eigen
                            |> List.map snd
                            |> function
                                | [ v0; v1; v2 ] ->
                                    eigenvector0[0][x, y, z] <- v0.x
                                    eigenvector0[1][x, y, z] <- v0.y
                                    eigenvector0[2][x, y, z] <- v0.z
                                    eigenvector1[0][x, y, z] <- v1.x
                                    eigenvector1[1][x, y, z] <- v1.y
                                    eigenvector1[2][x, y, z] <- v1.z
                                    eigenvector2[0][x, y, z] <- v2.x
                                    eigenvector2[1][x, y, z] <- v2.y
                                    eigenvector2[2][x, y, z] <- v2.z
                                | _ -> failwith "structureTensorEigenImages: expected three eigenvectors."

                let compose name arrays =
                    arrays
                    |> List.map (fun values -> Image<float>.ofArray3D(values, name, tensor.index))
                    |> composeVectorAndRelease

                [ compose "structureTensorEigenValues" eigenvalues
                  compose "structureTensorEigenvector0" eigenvector0
                  compose "structureTensorEigenvector1" eigenvector1
                  compose "structureTensorEigenvector2" eigenvector2 ]
            | values ->
                failwith $"structureTensorEigenImages: expected 6 components, got {values.Length}."
        finally
            components |> List.iter (fun comp -> comp.decRefCount())
    else
        failwith $"structureTensorEigenImages: only 2D and 3D images are supported, got {tensor.GetDimensions()}D"

let private tensorEigenMatrixValues xx xy xz yy yz zz =
    let matrix =
        { m00 = xx; m01 = xy; m02 = xz
          m10 = xy; m11 = yy; m12 = yz
          m20 = xz; m21 = yz; m22 = zz }
    let eigen = symmetricEigen matrix
    let values = eigen |> List.map fst
    let vectors =
        eigen
        |> List.collect (fun (_, v) -> [ v.x; v.y; v.z ])
    values @ vectors

let structureTensorEigenMatrix (tensor: Image<float list>) : Image<float list> =
    if tensor.GetNumberOfComponentsPerPixel() <> 6u then
        invalidArg "tensor" $"structureTensorEigenMatrix: expected a 6-component symmetric tensor image, got {tensor.GetNumberOfComponentsPerPixel()} components."

    if tensor.GetDimensions() = 2u then
        let input = Image<float>.toArray3DVector tensor
        let output = Array3D.zeroCreate<float> (input.GetLength 0) (input.GetLength 1) 12
        for x in 0 .. input.GetLength 0 - 1 do
            for y in 0 .. input.GetLength 1 - 1 do
                let values =
                    tensorEigenMatrixValues
                        input[x, y, 0]
                        input[x, y, 1]
                        input[x, y, 2]
                        input[x, y, 3]
                        input[x, y, 4]
                        input[x, y, 5]
                for k in 0 .. 11 do
                    output[x, y, k] <- values[k]
        Image<float>.ofArray3DVector(output, "structureTensorEigenMatrix", tensor.index)
    elif tensor.GetDimensions() = 3u then
        let components = tensor.toImageList()
        try
            match components with
            | [ xxImg; xyImg; xzImg; yyImg; yzImg; zzImg ] ->
                let xxValues = xxImg.toArray3D()
                let xyValues = xyImg.toArray3D()
                let xzValues = xzImg.toArray3D()
                let yyValues = yyImg.toArray3D()
                let yzValues = yzImg.toArray3D()
                let zzValues = zzImg.toArray3D()
                let width = xxValues.GetLength 0
                let height = xxValues.GetLength 1
                let depth = xxValues.GetLength 2
                let output = [ for _ in 1 .. 12 -> Array3D.zeroCreate<float> width height depth ]
                for x in 0 .. width - 1 do
                    for y in 0 .. height - 1 do
                        for z in 0 .. depth - 1 do
                            let values =
                                tensorEigenMatrixValues
                                    xxValues[x, y, z]
                                    xyValues[x, y, z]
                                    xzValues[x, y, z]
                                    yyValues[x, y, z]
                                    yzValues[x, y, z]
                                    zzValues[x, y, z]
                            for k in 0 .. 11 do
                                output[k][x, y, z] <- values[k]
                output
                |> List.map (fun values -> Image<float>.ofArray3D(values, "structureTensorEigenMatrix", tensor.index))
                |> composeVectorAndRelease
            | values ->
                failwith $"structureTensorEigenMatrix: expected 6 components, got {values.Length}."
        finally
            components |> List.iter (fun comp -> comp.decRefCount())
    else
        failwith $"structureTensorEigenMatrix: only 2D and 3D images are supported, got {tensor.GetDimensions()}D"

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
            let reference = first.toSimpleITK()
            let referenceSpacing = reference.GetSpacing()
            let referenceOrigin = reference.GetOrigin()
            let referenceDirection = reference.GetDirection()
            use filter = new itk.simple.JoinSeriesImageFilter ()
            filter.SetOrigin(0.0) |> ignore
            filter.SetSpacing(1.0) |> ignore
            use v = new itk.simple.VectorOfImage()
            let normalizedImages = ResizeArray<itk.simple.Image>()
            try
                images
                |> List.iter (fun (image: Image<'T>) ->
                    let sitk = image.toSimpleITK()
                    let normalized = new itk.simple.Image(sitk)
                    normalized.SetSpacing(referenceSpacing)
                    normalized.SetOrigin(referenceOrigin)
                    normalized.SetDirection(referenceDirection)
                    normalizedImages.Add normalized
                    v.Add normalized)
                v |> filter.Execute |> imageFromTemporarySimpleITK "stack" first.index
            finally
                normalizedImages |> Seq.iter (fun image -> image.Dispose())
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
    imageFromTemporarySimpleITK "extractSub" img.index (extractor.Execute(img.toSimpleITK()))

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
    imageFromTemporarySimpleITK "extractSlice" i (extractor.Execute(img.toSimpleITK()))

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
    imageFromTemporarySimpleITK "permuteAxes" img.index (filter.Execute(img.toSimpleITK()))

// Fourier transform a 2d image
let FFTXY (image: Image<'T>) : Image<System.Numerics.Complex> =
    if image.GetDimensions() <> 2u then
        failwith $"FFTXY: image must be 2D, got {image.GetDimensions()}D"
    use input = ofCastITK<float> (image.toSimpleITK())
    use fft = new itk.simple.ForwardFFTImageFilter()
    let complexImg = fft.Execute(input)
    imageFromTemporarySimpleITK "FFTXY" image.index complexImg

let FFTXYFloat32 (image: Image<'T>) : Image<ComplexFloat32> =
    if image.GetDimensions() <> 2u then
        failwith $"FFTXYFloat32: image must be 2D, got {image.GetDimensions()}D"
    use input = ofCastITK<float32> (image.toSimpleITK())
    use fft = new itk.simple.ForwardFFTImageFilter()
    let complexImg = fft.Execute(input)
    imageFromTemporarySimpleITK "FFTXYFloat32" image.index complexImg

let private dftLine inverse (line: System.Numerics.Complex[]) =
    let lineLength = line.Length
    let lineLengthFloat = float lineLength
    let sign = if inverse then 1.0 else -1.0
    let scale = if inverse then 1.0 / lineLengthFloat else 1.0
    let twiddles =
        Array2D.init lineLength lineLength (fun k n ->
            let theta = sign * 2.0 * System.Math.PI * float (k * n) / lineLengthFloat
            System.Numerics.Complex(System.Math.Cos theta, System.Math.Sin theta))

    Array.init lineLength (fun k ->
        let mutable value = System.Numerics.Complex.Zero
        for n in 0 .. lineLength - 1 do
            value <- value + line[n] * twiddles[k, n]
        value * scale)

let private directionalDftComplex2D dir inverse (input: System.Numerics.Complex[,]) =
    let width = input.GetLength 0
    let height = input.GetLength 1
    let output = Array2D.zeroCreate<System.Numerics.Complex> width height

    match dir with
    | 0u ->
        for y in 0 .. height - 1 do
            let transformed = dftLine inverse [| for x in 0 .. width - 1 -> input[x, y] |]
            for x in 0 .. width - 1 do
                output[x, y] <- transformed[x]
    | 1u ->
        for x in 0 .. width - 1 do
            let transformed = dftLine inverse [| for y in 0 .. height - 1 -> input[x, y] |]
            for y in 0 .. height - 1 do
                output[x, y] <- transformed[y]
    | _ ->
        failwith $"directionalDftComplex2D: dir={dir} is out of range for 2D image"

    output

let private directionalDftComplex3D dir inverse (input: System.Numerics.Complex[,,]) =
    let width = input.GetLength 0
    let height = input.GetLength 1
    let depth = input.GetLength 2
    let output = Array3D.zeroCreate<System.Numerics.Complex> width height depth

    match dir with
    | 0u ->
        for y in 0 .. height - 1 do
            for z in 0 .. depth - 1 do
                let transformed = dftLine inverse [| for x in 0 .. width - 1 -> input[x, y, z] |]
                for x in 0 .. width - 1 do
                    output[x, y, z] <- transformed[x]
    | 1u ->
        for x in 0 .. width - 1 do
            for z in 0 .. depth - 1 do
                let transformed = dftLine inverse [| for y in 0 .. height - 1 -> input[x, y, z] |]
                for y in 0 .. height - 1 do
                    output[x, y, z] <- transformed[y]
    | 2u ->
        for x in 0 .. width - 1 do
            for y in 0 .. height - 1 do
                let transformed = dftLine inverse [| for z in 0 .. depth - 1 -> input[x, y, z] |]
                for z in 0 .. depth - 1 do
                    output[x, y, z] <- transformed[z]
    | _ ->
        failwith $"directionalDftComplex3D: dir={dir} is out of range for 3D image"

    output

let private toComplex64 (value: ComplexFloat32) =
    System.Numerics.Complex(float value.Real, float value.Imaginary)

let private toComplex32 (value: System.Numerics.Complex) =
    ComplexFloat32(float32 value.Real, float32 value.Imaginary)

let private directionalDftComplexFloat322D dir inverse (input: ComplexFloat32[,]) =
    input
    |> Array2D.map toComplex64
    |> directionalDftComplex2D dir inverse
    |> Array2D.map toComplex32

let private directionalDftComplexFloat323D dir inverse (input: ComplexFloat32[,,]) =
    input
    |> Array3D.map toComplex64
    |> directionalDftComplex3D dir inverse
    |> Array3D.map toComplex32

// Fourier transform a 3d image along a specified axis direction
let directionalFFT (dir: uint) (image: Image<'T>) : Image<System.Numerics.Complex> =
    let dims = image.GetDimensions()
    if dir >= dims then
        failwith $"directionalFFT: dir={dir} is out of range for {dims}D image"
    use input = ofCastITK<float> (image.toSimpleITK())
    let inputImage = Image<float>.ofSimpleITKAlias(input, "directionalFFTInput", image.index)
    try
        match dims with
        | 2u ->
            let input = inputImage.toArray2D() |> Array2D.map (fun value -> System.Numerics.Complex(value, 0.0))
            let output = directionalDftComplex2D dir false input
            Image<System.Numerics.Complex>.ofComplexArray2D(output, "directionalFFT", image.index)
        | 3u ->
            let input = inputImage.toArray3D() |> Array3D.map (fun value -> System.Numerics.Complex(value, 0.0))
            let output = directionalDftComplex3D dir false input
            Image<System.Numerics.Complex>.ofComplexArray3D(output, "directionalFFT", image.index)
        | _ ->
            failwith $"directionalFFT: only 2D and 3D images are supported, got {dims}D"
    finally
        inputImage.decRefCount()

let directionalFFTFloat32 (dir: uint) (image: Image<'T>) : Image<ComplexFloat32> =
    let dims = image.GetDimensions()
    if dir >= dims then
        failwith $"directionalFFTFloat32: dir={dir} is out of range for {dims}D image"
    use input = ofCastITK<float32> (image.toSimpleITK())
    let inputImage = Image<float32>.ofSimpleITKAlias(input, "directionalFFTFloat32Input", image.index)

    try
        match dims with
        | 2u ->
            let input = inputImage.toArray2D() |> Array2D.map (fun value -> ComplexFloat32(value, 0.0f))
            let output = directionalDftComplexFloat322D dir false input
            Image<ComplexFloat32>.ofComplexFloat32Array2D(output, "directionalFFTFloat32", image.index)
        | 3u ->
            let input = inputImage.toArray3D() |> Array3D.map (fun value -> ComplexFloat32(value, 0.0f))
            let output = directionalDftComplexFloat323D dir false input
            Image<ComplexFloat32>.ofComplexFloat32Array3D(output, "directionalFFTFloat32", image.index)
        | _ ->
            failwith $"directionalFFTFloat32: only 2D and 3D images are supported, got {dims}D"
    finally
        inputImage.decRefCount()

let directionalFFTComplex (dir: uint) (inverse: bool) (image: Image<System.Numerics.Complex>) : Image<System.Numerics.Complex> =
    let dims = image.GetDimensions()
    if dir >= dims then
        failwith $"directionalFFTComplex: dir={dir} is out of range for {dims}D image"

    match dims with
    | 2u ->
        let output = image.toComplexArray2D() |> directionalDftComplex2D dir inverse
        Image<System.Numerics.Complex>.ofComplexArray2D(output, "directionalFFTComplex", image.index)
    | 3u ->
        let output = image.toComplexArray3D() |> directionalDftComplex3D dir inverse
        Image<System.Numerics.Complex>.ofComplexArray3D(output, "directionalFFTComplex", image.index)
    | _ ->
        failwith $"directionalFFTComplex: only 2D and 3D images are supported, got {dims}D"

let directionalFFTComplexFloat32 (dir: uint) (inverse: bool) (image: Image<ComplexFloat32>) : Image<ComplexFloat32> =
    let dims = image.GetDimensions()
    if dir >= dims then
        failwith $"directionalFFTComplexFloat32: dir={dir} is out of range for {dims}D image"

    match dims with
    | 2u ->
        let output = image.toComplexFloat32Array2D() |> directionalDftComplexFloat322D dir inverse
        Image<ComplexFloat32>.ofComplexFloat32Array2D(output, "directionalFFTComplexFloat32", image.index)
    | 3u ->
        let output = image.toComplexFloat32Array3D() |> directionalDftComplexFloat323D dir inverse
        Image<ComplexFloat32>.ofComplexFloat32Array3D(output, "directionalFFTComplexFloat32", image.index)
    | _ ->
        failwith $"directionalFFTComplexFloat32: only 2D and 3D images are supported, got {dims}D"

let inverseFFTXY (image: Image<System.Numerics.Complex>) : Image<System.Numerics.Complex> =
    if image.GetDimensions() <> 2u then
        failwith $"inverseFFTXY: image must be 2D, got {image.GetDimensions()}D"

    image.toComplexArray2D()
    |> directionalDftComplex2D 0u true
    |> directionalDftComplex2D 1u true
    |> fun recovered -> Image<System.Numerics.Complex>.ofComplexArray2D(recovered, "inverseFFTXY", image.index)

let inverseFFTXYFloat32 (image: Image<ComplexFloat32>) : Image<ComplexFloat32> =
    if image.GetDimensions() <> 2u then
        failwith $"inverseFFTXYFloat32: image must be 2D, got {image.GetDimensions()}D"

    image.toComplexFloat32Array2D()
    |> directionalDftComplexFloat322D 0u true
    |> directionalDftComplexFloat322D 1u true
    |> fun recovered -> Image<ComplexFloat32>.ofComplexFloat32Array2D(recovered, "inverseFFTXYFloat32", image.index)

let realPart (image: Image<System.Numerics.Complex>) : Image<float> =
    match image.GetDimensions() with
    | 2u ->
        let values = image.toComplexArray2D()
        Array2D.init (values.GetLength 0) (values.GetLength 1) (fun x y -> values[x, y].Real)
        |> fun real -> Image<float>.ofArray2D(real, "realPart", image.index)
    | 3u ->
        let values = image.toComplexArray3D()
        Array3D.init (values.GetLength 0) (values.GetLength 1) (values.GetLength 2) (fun x y z -> values[x, y, z].Real)
        |> fun real -> Image<float>.ofArray3D(real, "realPart", image.index)
    | dims ->
        failwith $"realPart: only 2D and 3D images are supported, got {dims}D"

let realPartFloat32 (image: Image<ComplexFloat32>) : Image<float32> =
    match image.GetDimensions() with
    | 2u ->
        let values = image.toComplexFloat32Array2D()
        Array2D.init (values.GetLength 0) (values.GetLength 1) (fun x y -> values[x, y].Real)
        |> fun real -> Image<float32>.ofArray2D(real, "realPartFloat32", image.index)
    | 3u ->
        let values = image.toComplexFloat32Array3D()
        Array3D.init (values.GetLength 0) (values.GetLength 1) (values.GetLength 2) (fun x y z -> values[x, y, z].Real)
        |> fun real -> Image<float32>.ofArray3D(real, "realPartFloat32", image.index)
    | dims ->
        failwith $"realPartFloat32: only 2D and 3D images are supported, got {dims}D"

let inverseFFTXYReal (image: Image<System.Numerics.Complex>) : Image<float> =
    if image.GetDimensions() <> 2u then
        failwith $"inverseFFTXYReal: image must be 2D, got {image.GetDimensions()}D"

    let complex = inverseFFTXY image
    let real = realPart complex
    complex.decRefCount()
    real

let shiftFFT (image: Image<System.Numerics.Complex>) : Image<System.Numerics.Complex> =
    match image.GetDimensions() with
    | 2u ->
        let input = image.toComplexArray2D()
        let width = input.GetLength 0
        let height = input.GetLength 1
        Array2D.init width height (fun x y ->
            input[(x + width - width / 2) % width, (y + height - height / 2) % height])
        |> fun output -> Image<System.Numerics.Complex>.ofComplexArray2D(output, "shiftFFT", image.index)
    | 3u ->
        let input = image.toComplexArray3D()
        let width = input.GetLength 0
        let height = input.GetLength 1
        let depth = input.GetLength 2
        Array3D.init width height depth (fun x y z ->
            input[(x + width - width / 2) % width,
                  (y + height - height / 2) % height,
                  (z + depth - depth / 2) % depth])
        |> fun output -> Image<System.Numerics.Complex>.ofComplexArray3D(output, "shiftFFT", image.index)
    | dims ->
        failwith $"shiftFFT: only 2D and 3D images are supported, got {dims}D"

let shiftFFTFloat32 (image: Image<ComplexFloat32>) : Image<ComplexFloat32> =
    match image.GetDimensions() with
    | 2u ->
        let input = image.toComplexFloat32Array2D()
        let width = input.GetLength 0
        let height = input.GetLength 1
        Array2D.init width height (fun x y ->
            input[(x + width - width / 2) % width, (y + height - height / 2) % height])
        |> fun output -> Image<ComplexFloat32>.ofComplexFloat32Array2D(output, "shiftFFTFloat32", image.index)
    | 3u ->
        let input = image.toComplexFloat32Array3D()
        let width = input.GetLength 0
        let height = input.GetLength 1
        let depth = input.GetLength 2
        Array3D.init width height depth (fun x y z ->
            input[(x + width - width / 2) % width,
                  (y + height - height / 2) % height,
                  (z + depth - depth / 2) % depth])
        |> fun output -> Image<ComplexFloat32>.ofComplexFloat32Array3D(output, "shiftFFTFloat32", image.index)
    | dims ->
        failwith $"shiftFFTFloat32: only 2D and 3D images are supported, got {dims}D"
