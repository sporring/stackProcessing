module ImageClass
open FSharp.Collections

type PixelType =
    | UInt8
    | Int8
    | UInt16
    | Int16
    | UInt32
    | Int32
    | UInt64
    | Int64
    | Float32
    | Float64
    | ComplexFloat32
    | ComplexFloat64
    | VectorUInt8
    | VectorInt8
    | VectorUInt16
    | VectorInt16
    | VectorUInt32
    | VectorInt32
    | VectorUInt64
    | VectorInt64
    | VectorFloat32
    | VectorFloat64
    | LabelUInt8
    | LabelUInt16
    | LabelUInt32
    | LabelUInt64

    member this.ToSimpleITK() =
        match this with
        | UInt8            -> itk.simple.PixelIDValueEnum.sitkUInt8
        | Int8             -> itk.simple.PixelIDValueEnum.sitkInt8
        | UInt16           -> itk.simple.PixelIDValueEnum.sitkUInt16
        | Int16            -> itk.simple.PixelIDValueEnum.sitkInt16
        | UInt32           -> itk.simple.PixelIDValueEnum.sitkUInt32
        | Int32            -> itk.simple.PixelIDValueEnum.sitkInt32
        | UInt64           -> itk.simple.PixelIDValueEnum.sitkUInt64
        | Int64            -> itk.simple.PixelIDValueEnum.sitkInt64
        | Float32          -> itk.simple.PixelIDValueEnum.sitkFloat32
        | Float64          -> itk.simple.PixelIDValueEnum.sitkFloat64
        | ComplexFloat32   -> itk.simple.PixelIDValueEnum.sitkComplexFloat32
        | ComplexFloat64   -> itk.simple.PixelIDValueEnum.sitkComplexFloat64
        | VectorUInt8      -> itk.simple.PixelIDValueEnum.sitkVectorUInt8
        | VectorInt8       -> itk.simple.PixelIDValueEnum.sitkVectorInt8
        | VectorUInt16     -> itk.simple.PixelIDValueEnum.sitkVectorUInt16
        | VectorInt16      -> itk.simple.PixelIDValueEnum.sitkVectorInt16
        | VectorUInt32     -> itk.simple.PixelIDValueEnum.sitkVectorUInt32
        | VectorInt32      -> itk.simple.PixelIDValueEnum.sitkVectorInt32
        | VectorUInt64     -> itk.simple.PixelIDValueEnum.sitkVectorUInt64
        | VectorInt64      -> itk.simple.PixelIDValueEnum.sitkVectorInt64
        | VectorFloat32    -> itk.simple.PixelIDValueEnum.sitkVectorFloat32
        | VectorFloat64    -> itk.simple.PixelIDValueEnum.sitkVectorFloat64
        | LabelUInt8       -> itk.simple.PixelIDValueEnum.sitkLabelUInt8
        | LabelUInt16      -> itk.simple.PixelIDValueEnum.sitkLabelUInt16
        | LabelUInt32      -> itk.simple.PixelIDValueEnum.sitkLabelUInt32
        | LabelUInt64      -> itk.simple.PixelIDValueEnum.sitkLabelUInt64

    member this.Zero() =
        match this with
        | UInt8 | VectorUInt8 | LabelUInt8         -> box 0uy
        | Int8  | VectorInt8                        -> box 0y
        | UInt16 | VectorUInt16 | LabelUInt16      -> box 0us
        | Int16 | VectorInt16                       -> box 0s
        | UInt32 | VectorUInt32 | LabelUInt32      -> box 0u
        | Int32 | VectorInt32                       -> box 0
        | UInt64 | VectorUInt64 | LabelUInt64      -> box 0UL
        | Int64 | VectorInt64                       -> box 0L
        | Float32 | VectorFloat32                  -> box 0.0f
        | Float64 | VectorFloat64                  -> box 0.0
        | ComplexFloat32                           -> box (System.Numerics.Complex(0.0, 0.0))
        | ComplexFloat64                           -> box (System.Numerics.Complex(0.0, 0.0))

    member this.One() =
        match this with
        | UInt8 | VectorUInt8 | LabelUInt8         -> box 1uy
        | Int8  | VectorInt8                        -> box 1y
        | UInt16 | VectorUInt16 | LabelUInt16      -> box 1us
        | Int16 | VectorInt16                       -> box 1s
        | UInt32 | VectorUInt32 | LabelUInt32      -> box 1u
        | Int32 | VectorInt32                       -> box 1
        | UInt64 | VectorUInt64 | LabelUInt64      -> box 1UL
        | Int64 | VectorInt64                       -> box 1L
        | Float32 | VectorFloat32                  -> box 1.0f
        | Float64 | VectorFloat64                  -> box 1.0
        | ComplexFloat32                           -> box (System.Numerics.Complex(1.0, 0.0))
        | ComplexFloat64                           -> box (System.Numerics.Complex(1.0, 0.0))

/// Module with inline operator overloads for Image
let toVectorUInt32 (lst: uint list) =
    let v = new itk.simple.VectorUInt32()
    lst |> List.iter v.Add
    v

let toVectorInt32 (lst: int list) =
    let v = new itk.simple.VectorInt32()
    lst |> List.iter v.Add
    v

let toVectorDouble (lst: float list) =
    let v = new itk.simple.VectorDouble()
    lst |> List.iter v.Add
    v

let fromVectorUInt32 (v: itk.simple.VectorUInt32) : uint list =
    v |> Seq.map uint |> Seq.toList

let fromVectorInt32 (v: itk.simple.VectorInt32) : int list =
    v |> Seq.map int |> Seq.toList

let fromVectorDouble (v: itk.simple.VectorDouble) : float list =
    v |> Seq.toList

type Image (img: itk.simple.Image) =
    static let extractSlice (axis: int) (index: int) (img: Image) : Image =
        let roi = new itk.simple.RegionOfInterestImageFilter()
        let size = (img.Image :> itk.simple.Image).GetSize()
        let start = List.init 3 (fun i -> if i = axis then index else 0)
        let extent = List.init 3 (fun i -> if i = axis then 1u else 0u)

        roi.SetSize(extent |> toVectorUInt32)
        roi.SetIndex(start |> toVectorInt32)
        Image(roi.Execute(img.Image))

    static let concatAlongAxis (axis: int) (images: Image list) : Image =
        if List.isEmpty images then failwith "Empty image list"

        // Helper to convert offset array to vector
        let offsetToVector offset = offset |> List.ofArray |> toVectorInt32

        // Start with the first image and zero offset
        let initialImage = new itk.simple.Image(images.Head.Image)
        let initialOffset = [| 0; 0; 0 |]

        // Fold over the tail, carrying (result image, offset) as state
        let finalImage, _ =
            images.Tail
            |> List.fold (fun (result, offset) img ->
                let size = img.Image.GetSize()

                let paste = new itk.simple.PasteImageFilter()
                paste.SetDestinationIndex(offsetToVector offset)
                paste.SetSourceSize(size)
                paste.SetSourceIndex(offsetToVector [| 0; 0; 0 |])

                let newResult = paste.Execute(result, img.Image)

                // Compute new offset immutably
                let newOffset =
                    offset
                    |> Array.mapi (fun i v -> if i = axis then v + (int size.[i]) else v)

                (newResult, newOffset)
            ) (initialImage, initialOffset)

        Image(finalImage)


    static let concatZ (images: Image list) : Image =
        if List.isEmpty images then failwith "Empty image list"

        let join = new itk.simple.JoinSeriesImageFilter()
        join.SetOrigin(0.0)

        let vector = new itk.simple.VectorOfImage()
        images |> List.iter (fun img -> vector.Add(img.Image))

        Image(join.Execute(vector))

    /// Underlying itk.simple image
    member this.Image = img

    /// String representation
    override this.ToString() = img.ToString()

    /// generate images
    static member FromSimpleITK(img: itk.simple.Image) : Image =
        Image(img)

    static member Create(size: uint list, ?pixelType: PixelType, ?value: obj) =
        let pt = defaultArg pixelType PixelType.UInt8
        let itkId = pt.ToSimpleITK()
        let img = new itk.simple.Image(size |> toVectorUInt32, itkId)
        let imgFilled =
            match value with
            | Some v ->
                let scalarFilter = new itk.simple.ShiftScaleImageFilter()
                scalarFilter.SetShift(v |> unbox |> float)
                scalarFilter.Execute(img)
            | None ->
                img
        Image(imgFilled)

    static member Create(size: uint list, value: uint8)  = Image.Create(size, PixelType.UInt8,  box value)
    static member Create(size: uint list, value: int8)   = Image.Create(size, PixelType.Int8,   box value)
    static member Create(size: uint list, value: uint16) = Image.Create(size, PixelType.UInt16, box value)
    static member Create(size: uint list, value: int16)  = Image.Create(size, PixelType.Int16,  box value)
    static member Create(size: uint list, value: uint32) = Image.Create(size, PixelType.UInt32, box value)
    static member Create(size: uint list, value: int32)  = Image.Create(size, PixelType.Int32,  box value)
    static member Create(size: uint list, value: uint64) = Image.Create(size, PixelType.UInt64, box value)
    static member Create(size: uint list, value: int64)  = Image.Create(size, PixelType.Int64,  box value)
    static member Create(size: uint list, value: float32)= Image.Create(size, PixelType.Float32, box value)
    static member Create(size: uint list, value: float)  = Image.Create(size, PixelType.Float64, box value)
    static member Create(size: uint list, value: System.Numerics.Complex) = Image.Create(size, PixelType.ComplexFloat64, box value)

    static member FromFile(filename: string) : Image =
        let reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        Image(reader.Execute())

    static member (+) (f1 : Image, f2 : Image) =
        let filter = new itk.simple.AddImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member (+) (f1: Image, i : float) =
        let filter = new itk.simple.AddImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member (+) (i : float, f2: Image) =
        let filter = new itk.simple.AddImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member (-) (f1: Image, f2: Image) =
        let filter = new itk.simple.SubtractImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member (-) (f1: Image, i: float) =
        let filter = new itk.simple.SubtractImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member (-) (i: float, f2: Image) =
        let filter = new itk.simple.SubtractImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member (*) (f1: Image, f2: Image) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member (*) (f1: Image, i: float) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member (*) (i: float, f2: Image) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member (/) (f1: Image, f2: Image) =
        let filter = new itk.simple.DivideImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member (/) (f1: Image, i: float) =
        let filter = new itk.simple.DivideImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member (/) (i: float, f2: Image) =
        let filter = new itk.simple.DivideImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member op_Equality (f1: Image, f2: Image) =
        let filter = new itk.simple.EqualImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_Equality (f1: Image, i: float) =
        let filter = new itk.simple.EqualImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_Equality (i: float, f2: Image) =
        let filter = new itk.simple.EqualImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member op_Inequality (f1: Image, f2: Image) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_Inequality (f1: Image, i: float) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_Inequality (i: float, f2: Image) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member op_LessThan (f1: Image, f2: Image) =
        let filter = new itk.simple.LessImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_LessThan (f1: Image, i: float) =
        let filter = new itk.simple.LessImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_LessThan (i: float, f2: Image) =
        let filter = new itk.simple.LessImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member op_LessThanOrEqual (f1: Image, f2: Image) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_LessThanOrEqual (f1: Image, i: float) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_LessThanOrEqual (i: float, f2: Image) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member op_GreaterThan (f1: Image, f2: Image) =
        let filter = new itk.simple.GreaterImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_GreaterThan (f1: Image, i: float) =
        let filter = new itk.simple.GreaterImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_GreaterThan (i: float, f2: Image) =
        let filter = new itk.simple.GreaterImageFilter()
        Image(filter.Execute(i, f2.Image))

    static member op_GreaterThanOrEqual (f1: Image, f2: Image) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_GreaterThanOrEqual (f1: Image, i: float) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_GreaterThanOrEqual (i: float, f2: Image) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image(filter.Execute(i, f2.Image))

    // Modulus ( % )
    static member op_Modulus (f1: Image, f2: Image) =
        let filter = new itk.simple.ModulusImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_Modulus (f1: Image, i: uint) =
        let filter = new itk.simple.ModulusImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_Modulus (i: uint, f2: Image) =
        let filter = new itk.simple.ModulusImageFilter()
        Image(filter.Execute(i, f2.Image))

    // Power (no direct operator for ** in .NET) - provide a named method instead
    static member Pow (f1: Image, f2: Image) =
        let filter = new itk.simple.PowImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member Pow (f1: Image, i: float) =
        let filter = new itk.simple.PowImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member Pow (i: float, f2: Image) =
        let filter = new itk.simple.PowImageFilter()
        Image(filter.Execute(i, f2.Image))

    // Bitwise AND ( &&& )
    static member op_BitwiseAnd (f1: Image, f2: Image) =
        let filter = new itk.simple.AndImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_BitwiseAnd (f1: Image, i: int) =
        let filter = new itk.simple.AndImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_BitwiseAnd (i: int, f2: Image) =
        let filter = new itk.simple.AndImageFilter()
        Image(filter.Execute(i, f2.Image))

    // Bitwise XOR ( ^^^ )
    static member op_ExclusiveOr (f1: Image, f2: Image) =
        let filter = new itk.simple.XorImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_ExclusiveOr (f1: Image, i: int) =
        let filter = new itk.simple.XorImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_ExclusiveOr (i: int, f2: Image) =
        let filter = new itk.simple.XorImageFilter()
        Image(filter.Execute(i, f2.Image))

    // Bitwise OR ( ||| )
    static member op_BitwiseOr (f1: Image, f2: Image) =
        let filter = new itk.simple.OrImageFilter()
        Image(filter.Execute(f1.Image, f2.Image))

    static member op_BitwiseOr (f1: Image, i: int) =
        let filter = new itk.simple.OrImageFilter()
        Image(filter.Execute(f1.Image, i))

    static member op_BitwiseOr (i: int, f2: Image) =
        let filter = new itk.simple.OrImageFilter()
        Image(filter.Execute(i, f2.Image))

    // Unary bitwise NOT ( ~~~ )
    static member op_LogicalNot (f: Image) =
        let filter = new itk.simple.InvertIntensityImageFilter()
        Image(filter.Execute(f.Image))

    /// Indexing
    member this.Item
        with get (x: int, y: int) =
            this.Image.GetPixelAsFloat([uint x; uint y] |> toVectorUInt32)

    member this.Item
        with get (x: int, y: int, z: int) =
            this.Image.GetPixelAsFloat([uint x; uint y; uint z] |> toVectorUInt32)

    member this.Slice(axis: int, index: int) = extractSlice axis index this

    // Concatenation
    static member ConcatX(images: Image list) = concatAlongAxis 0 images
    static member ConcatY(images: Image list) = concatAlongAxis 1 images
    static member ConcatZ(images: Image list) = concatZ images
