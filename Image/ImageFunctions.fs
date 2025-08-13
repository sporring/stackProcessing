module ImageFunctions
open Image
open Image.InternalHelpers

// Image constant arithmetic operations
let inline imageAddScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (f1: Image<'S>) (i: ^S) =
    let filter = new itk.simple.AddImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), float i),"imageAddScalar")
let inline scalarAddImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (f1: Image<'S>) =
    imageAddScalar f1 i
let inline imageSubScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (f1: Image<'S>) (i: ^S) =
    let filter = new itk.simple.SubtractImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), float i),"imageSubScalar")
let inline scalarSubImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (f1: Image<'S>) =
    let filter = new itk.simple.SubtractImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(float i, f1.toSimpleITK()),"scalarSubImage")
let inline imageMulScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (f1: Image<'S>) (i: ^S) =
    let filter = new itk.simple.MultiplyImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), float i),"imageMulScalar")
let inline scalarMulImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (f1: Image<'S>) =
    imageMulScalar f1 i
let inline imageDivScalar<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (f1: Image<'S>) (i: ^S) =
    let filter = new itk.simple.DivideImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), float i),"imageDivScalar")
let inline scalarDivImage<'S when ^S : equality
                             and  ^S : (static member op_Explicit : ^S -> float)
                             > (i: ^S) (f1: Image<'S>) =
    let filter = new itk.simple.DivideImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(float i, f1.toSimpleITK()),"scalarDivImage")

let inline imagePowScalar<'S when ^S : equality
                  and  ^S : (static member op_Explicit : ^S -> float)
                  > (f1: Image<'S>, i: 'S) =
    let filter = new itk.simple.PowImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), float i),"imagePowScalar")


let inline scalarPowImage<'S when ^S : equality
                  and  ^S : (static member op_Explicit : ^S -> float)
                  > (i: 'S, f1: Image<'S>) =
    let filter = new itk.simple.PowImageFilter()
    Image<'S>.ofSimpleITK(filter.Execute(float i, f1.toSimpleITK()),"scalarPowImage")

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
    let filter =  new itk.simple.ExtractImageFilter()
    let size = img.GetSize()
    let squeezedSize = size |> List.map (fun dim -> if dim = 1u then 0u else dim)
    filter.SetSize(squeezedSize |> toVectorUInt32)
    Image<'T>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"squeeze")

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
    let outImg = new itk.simple.Image(newSize |> toVectorUInt32, itkId, a.GetNumberOfComponentsPerPixel())

    let paste = new itk.simple.PasteImageFilter()
    // Paste image A at origin
    paste.SetDestinationIndex(List.replicate newSize.Length 0 |> toVectorInt32)
    paste.SetSourceSize(a.GetSize() |> toVectorUInt32)
    let outWithA = paste.Execute(outImg, a.toSimpleITK())

    // Paste image B at offset
    let offset = 
        sizeZipped |> List.mapi (fun i (a,b) -> if i <> int dim then 0 else int a)
    paste.SetDestinationIndex(offset |> toVectorInt32)
    paste.SetSourceSize(b.GetSize() |> toVectorUInt32)
    let outWithBoth = paste.Execute(outWithA, b.toSimpleITK())

    Image<'T>.ofSimpleITK(outWithBoth,"concatAlong")

let constantPad2D<'T when 'T : equality> (padLower : uint list) (padUpper : uint list) (c : double) (img : Image<'T>) : Image<'T> =
    if padLower.Length <> 2 || padUpper.Length <> 2 then
        invalidArg "padLower/padUpper" "Both bounds must have length 2 for a 2‑D image."
    let filter = new itk.simple.ConstantPadImageFilter()
    filter.SetPadLowerBound(padLower |> toVectorUInt32)
    filter.SetPadUpperBound(padUpper |> toVectorUInt32)
    filter.SetConstant(c)
    let padded = filter.Execute(img.toSimpleITK())
    Image<'T>.ofSimpleITK(padded,"constantPad2D")


// ----- basic mathematical helper functions -----
let inline makeUnaryImageOperatorWith
    (name: string)
    (createFilter: unit -> 'Filter when 'Filter :> System.IDisposable)
    (setup: 'Filter -> unit)
    (invoke: 'Filter -> itk.simple.Image -> itk.simple.Image)
    : (Image<'T> -> Image<'T>) =
        fun (img: Image<'T>) ->
            use filter = createFilter()
            setup filter
            Image<'T>.ofSimpleITK(invoke filter (img.toSimpleITK()),name,0)

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
        let img = Image<'T>.ofSimpleITK(invoke filter (a.toSimpleITK()) (b.toSimpleITK()),name)
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

// ----- basic image analysis functions -----
(* // I'm waiting with proper support of complex values
let fft (img: Image<'T>)  = makeUnaryImageOperator (fun () -> new itk.simple.ForwardFFTImageFilter()) (fun f x -> f.Execute(x)) img
let ifft (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.InverseFFTImageFilter()) (fun f x -> f.Execute(x)) img
let real (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToRealImageFilter()) (fun f x -> f.Execute(x)) img
let imag (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToImaginaryImageFilter()) (fun f x -> f.Execute(x)) img
let cabs (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToModulusImageFilter()) (fun f x -> f.Execute(x)) img
let carg (img: Image<'T>) = makeUnaryImageOperator (fun () -> new itk.simple.ComplexToPhaseImageFilter()) (fun f x -> f.Execute(x)) img
*)


type BoundaryCondition = ZeroPad | PerodicPad | ZeroFluxNeumannPad
type OutputRegionMode = Valid | Same

let internal convolve3 (img:itk.simple.Image) (ker:itk.simple.Image) (outputRegionMode: OutputRegionMode option) =
    // taylored for sliding window convolution of image stacks, so z outputRegionMode is always valid
    let idImg = img.GetPixelID()
    let idKer = ker.GetPixelID()
    let szImg = img.GetSize() |> fromVectorUInt32 |> List.map int
    let szKer = ker.GetSize() |> fromVectorUInt32 |> List.map int

    let szRes, offset = 
        match outputRegionMode with
        | None | Some Same -> 
            szImg, 
            List.map (fun a -> a/2) szKer // floor
        | Some Valid ->
            List.map2 (fun a b -> a - b + 1) szImg szKer,
            szKer

    let res = new itk.simple.Image(szRes |> List.map uint |> toVectorUInt32, idImg)
    for i0 in [0..szRes[0]-1] do // Coordinates in the result
        for i1 in [0..szRes[1]-1] do
            for i2 in [0..szRes[2]-1] do
                let i0i1i2 = [i0; i1; i2]
                let mutable s = getBoxedZero idImg None
                for k0 in [0..szKer[0]-1] do // Coordinates of the kernel
                    for k1 in [0..szKer[1]-1] do
                        for k2 in [0..szKer[2]-1] do
                            let k0k1k2 = [k0; k1; k2]
                            let kerVal = getBoxedPixel ker idKer (k0k1k2|>List.map uint|>toVectorUInt32)
                            let j0j1j2 = List.zip3 i0i1i2 k0k1k2 offset |> List.map (fun (i, k, o) -> o + i - k) // Coordinates in the image
                            let imgVal =
                                if List.forall2 (fun a b -> a>=0 && a < b) j0j1j2 szImg then
                                    getBoxedPixel img idImg (j0j1j2|>List.map uint|>toVectorUInt32)
                                else
                                    getBoxedZero idImg None
                            s <- mulAdd idImg s kerVal imgVal
                setBoxedPixel res idImg (i0i1i2|>List.map uint|>toVectorUInt32) s
    res

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
            if dimImg <> 3u then
                failwith "Image must be 3-dimensional"
            let szImg = img.GetSize() |> fromVectorUInt32 |> List.map int
            let szKer = ker.GetSize() |> fromVectorUInt32 |> List.map int
            let szZip = List.zip szImg szKer
            let res =
                if List.forall (fun (a,b) -> a >= b && b > 1) szZip then
                    //printfn $"Using simple itk to convolve {img.GetSize()|>fromVectorUInt32} {ker.GetSize()|>fromVectorUInt32}"
                    f.Execute(img,ker)
                else
                    //printfn $"Using convolve3 to convolve {img.GetSize()} {ker.GetSize()}"
                    convolve3 img ker outputRegionMode
            //printfn "convolve done"
            res
        )

(*
let convolve (xyOutputRegionMode: OutputRegionMode option) (boundaryCondition: BoundaryCondition option) : Image<'T> -> Image<'T> -> Image<'T> =
    // For sliding filter convolution, OutputRegionMode is the x-y coordinates, since z-coordinate output 
    // region is handled outside this function, and therefore, the result is always Valid in the z-direction.
    // The OutputRegionMode is relative to the first argument.
    fun (image: Image<'T>) (kernel: Image<'T>) ->
        let img = image.toSimpleITK()
        let ker = kernel.toSimpleITK()
        if img.GetNumberOfComponentsPerPixel() > 1u then
            failwith "Can't convolve vector images"
        let idImg = img.GetPixelID()
        let idKer = ker.GetPixelID()
        if idImg <> idKer then 
            failwith $"image ({idImg}) and kernel ({idKer}) must be of the same type"
        let dimImg = img.GetDimension()
        let dimKer = img.GetDimension()
        if dimImg <> dimKer then 
            failwith $"image ({dimImg})and kernel({dimKer}) must have the same dimensions"
        if dimImg < 2u || dimImg > 3u then
            failwith "convolve is only implemented for 2D and 3D images"
        let szImg = img.GetSize() |> fromVectorUInt32 |> List.map int
        let szKer = ker.GetSize() |> fromVectorUInt32 |> List.map int
        let pairs = List.zip szImg szKer
        if List.exists (fun (a,b) -> a < b) pairs then
            failwith $"all image axes ({szImg}) must be longer than the kernel axes ({szKer})"
        let szRes =
            match xyOutputRegionMode with
                Some Valid -> pairs |> List.map (fun (a,b) -> int (a-b+1))
                | _ -> szImg 
        let res = 
            if dimImg = 2u then
                use f = new itk.simple.ConvolutionImageFilter()
                f.SetOutputRegionMode (
                    match xyOutputRegionMode with
                        | Some Valid -> itk.simple.ConvolutionImageFilter.OutputRegionModeType.VALID
                        | _ ->         itk.simple.ConvolutionImageFilter.OutputRegionModeType.SAME)
                f.SetBoundaryCondition (
                    match boundaryCondition with
                        | Some ZeroFluxNeumannPad -> itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_FLUX_NEUMANN_PAD
                        | Some PerodicPad ->         itk.simple.ConvolutionImageFilter.BoundaryConditionType.PERIODIC_PAD 
                        | _ ->                       itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_PAD)
                if List.forall (fun (a,b) -> a > b && b > 1) pairs then
                    f.Execute(img,ker) // Simple itk relies on the first argument being the bigger
                else
                    printfn "Convolving 2D images %A * %A, output Region %A" szImg szKer xyOutputRegionMode
                    let res = new itk.simple.Image(szRes |> List.map uint |> toVectorUInt32, idImg)

                    for m0 in [0..szRes[0]-1] do
                        for m1 in [0..szRes[1]-1] do
                            let m0m1 = [uint m0; uint m1]
                            let mutable s = getBoxedZero idImg None
                            for k0 in [0..szKer[0]-1] do
                                for k1 in [0..szKer[1]-1] do
                                    let k0k1 = [uint k0; uint k1]
                                    let kerVal = getBoxedPixel ker idKer (k0k1|>toVectorUInt32)
                                    let i0i1k0k1 = [uint (szKer[0]-1+m0-k0); uint (szKer[1]-1+m1-k1)] 
                                    let imgVal = 
                                        let pairs = List.zip i0i1k0k1 (szImg|>List.map uint)
                                        if List.forall (fun (a,b) -> a>=0u && a < b) pairs then
                                            getBoxedPixel img idImg (i0i1k0k1|>toVectorUInt32)
                                        else
                                            getBoxedZero idImg None
                                    s <- mulAdd idImg s kerVal imgVal
                            printfn "%A: %A" szRes m0m1
                            setBoxedPixel res idImg (m0m1|>toVectorUInt32) s
                    res
            else // dime = 3u
                // taylored for sliding window convolution of image stacks, so z outputRegionMode is always valid
                let szRes = [szRes[0];szRes[1];szImg[2]-szKer[2]+1] 
                let res = new itk.simple.Image(szRes |> List.map uint |> toVectorUInt32, idImg)

                for m0 in [0..szRes[0]-1] do
                    for m1 in [0..szRes[1]-1] do
                        for m2 in [0..szRes[2]-1] do
                            let m0m1m2 = [uint m0; uint m1; uint m2]
                            let mutable s = getBoxedZero idImg None
                            for k0 in [0..szKer[0]-1] do
                                for k1 in [0..szKer[1]-1] do
                                    for k2 in [0..szKer[2]-1] do
                                        let k0k1k2 = [uint k0; uint k1; uint k2]
                                        let i0i1i2k0k1k2 = [uint (szKer[0]-1+m0-k0); uint (szKer[1]-1+m1-k1); uint (szKer[2]-1+m2-k2)] 
                                        let kerVal = getBoxedPixel ker idKer (k0k1k2|>toVectorUInt32)
                                        let imgVal = 
                                            let pairs = List.zip i0i1i2k0k1k2 (szImg|>List.map uint)
                                            if List.forall (fun (a,b) -> a>=0u && a < b) pairs then
                                                getBoxedPixel img idImg (i0i1i2k0k1k2|>toVectorUInt32)
                                            else
                                                getBoxedZero idImg None
                                        s <- mulAdd idImg s kerVal imgVal
                            setBoxedPixel res idImg (m0m1m2|>toVectorUInt32) s
                res
        Image<'T>.ofSimpleITK(res)
*)

let conv (img: Image<'T>) (ker: Image<'T>) : Image<'T> = convolve None None img ker

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
    arr |> Image<float>.ofArray2D

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
    arr |> Image<float>.ofArray3D

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
    arr |> Image<float>.ofArray4D

/// Gaussian kernel convolution
let gauss (dim: uint) (sigma: float) (kernelSize: uint option) : Image<'T> =
    let f = new itk.simple.GaussianImageSource()
    let sz = Option.defaultValue (1u + 2u*2u * uint sigma) kernelSize
    f.SetSize(List.replicate (int dim) sz |> toVectorUInt32)
    f.SetSigma(sigma)
    // Image coords: [0 1] (mean = 0.5), [0 1 2] (mean = 1) => mean = (sz-1)/2
    f.SetMean(List.replicate (int dim) ((float (sz-1u)) / 2.0) |> toVectorFloat64)
    f.SetScale(1.0)
    f.NormalizedOn()
    Image<'T>.ofSimpleITK(f.Execute(),"gauss")

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

/// Fill holes in binary regions
let binaryFillHoles (img : Image<uint8>) : Image<uint8> =
    use filter = new itk.simple.BinaryFillholeImageFilter()
    Image<uint8>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"binaryFillHoles")

/// Connected components labeling
// Currying and generic arguments causes value restriction error
let connectedComponents (img : Image<uint8>) : Image<uint64> =
    use filter = new itk.simple.ConnectedComponentImageFilter()
    Image<uint64>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"connectedComponents")

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
    Image<float>.ofSimpleITK(f.Execute(img.toSimpleITK()),"signedDistanceMap")

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
    
/// Otsu threshold
// Currying and generic arguments causes value restriction error
let otsuThreshold (img: Image<'T>) : Image<'T> =
    use filter = new itk.simple.OtsuThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image<'T>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"otsuThreshold")


/// Otsu multiple thresholds (returns a label map)
let otsuMultiThreshold (numThresholds: byte) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "otsuMultiThreshold"
        (fun () -> new itk.simple.OtsuMultipleThresholdsImageFilter())
        (fun f -> f.SetNumberOfThresholds(numThresholds))
        (fun f x -> f.Execute(x))

/// Moments-based threshold
let momentsThreshold (img: Image<'T>) : Image<'T> =
    use filter = new itk.simple.MomentsThresholdImageFilter()
    filter.SetInsideValue(0uy)
    filter.SetOutsideValue(1uy)
    Image<'T>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"momentsThreshold")

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

    Image<uint32>.ofSimpleITK(image,"generateCoordinateAxis")

let histogram (image: Image<'T>) : Map<'T, uint64> =
    image.GetSize()
    |> flatIndices
    |> Seq.fold 
        (fun acc idx ->
            let elm = image.Get idx 
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

let threshold (lower: float) (upper: float) : Image<'T> -> Image<'T> =
    makeUnaryImageOperatorWith
        "threshold"
        (fun () -> new itk.simple.BinaryThresholdImageFilter())
        (fun f -> 
            f.SetLowerThreshold lower
            f.SetUpperThreshold upper)
        (fun f x -> f.Execute(x))


let toVectorOfImage images =
    let v = new itk.simple.VectorOfImage()
    for img in images do
        v.Add(img)
    v

let stack (images: Image<'T> list) : Image<'T> =
    let filter = new itk.simple.JoinSeriesImageFilter ()
    filter.SetOrigin(0.0) |> ignore
    filter.SetSpacing(1.0) |> ignore
    //printfn "Stacking"
    //images |> List.iter (fun (I:Image<'T>) -> printf $"{I.Name}, ")
    //printfn ""
    let v = new itk.simple.VectorOfImage()
    images |> List.iter (fun (I:Image<'T>) -> v.Add (I.toSimpleITK()))
    let res = v |> filter.Execute |> (fun sitk -> Image<'T>.ofSimpleITK(sitk,"stack",0) )
    //printfn $"Stack: input {images.Length} {res.GetDepth()}"
    res

let stackOld (images: Image<'T> list) : Image<'T> =
    if images.Length = 0 then
        failwith "stack: Cannot stack an empty list of image"
    let dim = max 3u (List.fold (fun acc (img:Image<'T>) -> max 0u (img.GetDimensions())) 0u images)
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
    let sitkImages = images |> List.map (fun (img: Image<'T>)->img.toSimpleITK())   
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
    if img.GetDimensions() <> uint topLeft.Length then
        failwith $"extractSub: indices and image size does not match"
    let sz = List.zip topLeft bottomRight |> List.map (fun (a,b) -> b-a + 1u)
    if List.exists (fun a -> a <  1u) sz then
        failwith $"extractSub: no index of bottomRight must be smaller than topLeft  ({topLeft} vs {bottomRight})"

    let extractor = new itk.simple.ExtractImageFilter()
    extractor.SetSize(sz |> toVectorUInt32)
    extractor.SetIndex( topLeft |> List.map int |> toVectorInt32)
    let res = Image<'T>.ofSimpleITK(extractor.Execute(img.toSimpleITK()),"extractSub",0)
    res

let extractSlice (z: int) (img: Image<'T>) =
    if img.GetDimensions() <> 3u then
        failwith $"extractSlice: image must be 3D"
    let sz = img.GetSize()
    if sz[2] = 1u then
        // Should make a function for this...
        let filter =  new itk.simple.ExtractImageFilter() 
        let remove3rdDim = (List.take 2 sz)@[0u]
        filter.SetSize(remove3rdDim |> toVectorUInt32)
        Image<'T>.ofSimpleITK(filter.Execute(img.toSimpleITK()),"extractSlice",0)
    else
        let extractor = new itk.simple.ExtractImageFilter()
        extractor.SetSize([sz[0];sz[1];0u] |> toVectorUInt32)
        extractor.SetIndex( [0;0;z] |> toVectorInt32)
        let res = Image<'T>.ofSimpleITK(extractor.Execute(img.toSimpleITK()),"extractSlice",z)
        res

let unstack (vol: Image<'T>): Image<'T> list =
    let dim = vol.GetDimensions()
    if dim < 3u then
        failwith $"Cannot unstack a {dim}-dimensional image along the 3rd axis"
    let depth = vol.GetDepth() |> int
    let res = List.init depth (fun i -> extractSlice i vol)
    //printfn $"unstack: input {vol.GetSize()} {res.Length}"
    res

let unstackSkipNTakeM (N:uint) (M:uint) (vol: Image<'T>): Image<'T> list =
    let dim = vol.GetDimensions()
    if dim < 3u then
        failwith $"Cannot unstack a {dim}-dimensional image along the 3rd axis"
    let depth = vol.GetDepth() |> int
    if (N+M > uint depth) then
        failwith $"Cannot unstack from z={N} to z={N+M-1u} of a stack of depth {depth}"
    let res = List.init (int M) (fun i -> extractSlice (i+int N) vol)
    //printfn $"unstack: input {vol.GetSize()} {res.Length}"
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
