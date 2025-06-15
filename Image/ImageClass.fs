module ImageClass

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
    | VectorUInt8
    | VectorFloat32
    | LabelUInt8
    | LabelUInt16
    | LabelUInt32
    | LabelUInt64

    member this.ToSimpleITK() =
        match this with
        | UInt8         -> itk.simple.PixelIDValueEnum.sitkUInt8
        | Int8          -> itk.simple.PixelIDValueEnum.sitkInt8
        | UInt16        -> itk.simple.PixelIDValueEnum.sitkUInt16
        | Int16         -> itk.simple.PixelIDValueEnum.sitkInt16
        | UInt32        -> itk.simple.PixelIDValueEnum.sitkUInt32
        | Int32         -> itk.simple.PixelIDValueEnum.sitkInt32
        | UInt64        -> itk.simple.PixelIDValueEnum.sitkUInt64
        | Int64         -> itk.simple.PixelIDValueEnum.sitkInt64
        | Float32       -> itk.simple.PixelIDValueEnum.sitkFloat32
        | Float64       -> itk.simple.PixelIDValueEnum.sitkFloat64
        | VectorUInt8   -> itk.simple.PixelIDValueEnum.sitkVectorUInt8
        | VectorFloat32 -> itk.simple.PixelIDValueEnum.sitkVectorFloat32
        | LabelUInt8    -> itk.simple.PixelIDValueEnum.sitkLabelUInt8
        | LabelUInt16   -> itk.simple.PixelIDValueEnum.sitkLabelUInt16
        | LabelUInt32   -> itk.simple.PixelIDValueEnum.sitkLabelUInt32
        | LabelUInt64   -> itk.simple.PixelIDValueEnum.sitkLabelUInt64

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

type Image(img: itk.simple.Image) =

    static let extractSlice (axis: int) (index: int) (img: Image) : Image =
        let roi = new itk.simple.RegionOfInterestImageFilter()
        let size = (img.Image :> itk.simple.Image).GetSize()
        let start = [|0; 0; 0|]
        let extent = [|0u; 0u; 0u|]
        start.[axis] <- index
        extent.[axis] <- 1u

        roi.SetSize(extent |> List.ofArray |> toVectorUInt32)
        roi.SetIndex(start |> List.ofArray |> toVectorInt32)
        Image(roi.Execute(img.Image))

    static let concatAlongAxis (axis: int) (images: Image list) : Image =
        if List.isEmpty images then failwith "Empty image list"

        // Create a copy of the first image to paste into
        let mutable result = new itk.simple.Image(images.Head.Image)
        let mutable offset = [| 0; 0; 0 |]

        for img in images.Tail do
            let size = img.Image.GetSize()
            let paste = new itk.simple.PasteImageFilter()
            paste.SetDestinationIndex(offset |> List.ofArray |> toVectorInt32)
            paste.SetSourceSize(size)
            paste.SetSourceIndex([| 0; 0; 0 |] |> List.ofArray |> toVectorInt32)
            result <- paste.Execute(result, img.Image)

            // Update offset on the selected axis
            offset.[axis] <- offset.[axis] + (int size.[axis])

        Image(result)

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
    static member FromSize(size: int list) : Image =
        let vec = new itk.simple.VectorUInt32()
        size |> List.iter (uint32 >> vec.Add)
        let img = new itk.simple.Image(vec, PixelType.Float32.ToSimpleITK())
        Image(img)

    static member FromSizeAndType(size: int list, pixelType: PixelType) : Image =
        let vec = new itk.simple.VectorUInt32()
        size |> List.iter (uint32 >> vec.Add)
        let img = new itk.simple.Image(vec, pixelType.ToSimpleITK())
        Image(img)

    static member FromConstant(size: int list, pixelType: PixelType, value: float) : Image =
        let vec = new itk.simple.VectorUInt32()
        size |> List.iter (uint32 >> vec.Add)
        let img = new itk.simple.Image(vec, pixelType.ToSimpleITK())
        let scalarFilter = new itk.simple.ShiftScaleImageFilter()
        scalarFilter.SetShift(value)
        Image(scalarFilter.Execute(img))

    static member FromSimpleITK(img: itk.simple.Image) : Image =
        Image(img)

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
            this.Image.GetPixelAsFloat([|uint x; uint y|] |>List.ofArray |> toVectorUInt32)

    member this.Item
        with get (x: int, y: int, z: int) =
            this.Image.GetPixelAsFloat([|uint x; uint y; uint z|] |> List.ofArray |> toVectorUInt32)

    member this.Slice(axis: int, index: int) = extractSlice axis index this

    // Concatenation
    static member ConcatX(images: Image list) = concatAlongAxis 0 images
    static member ConcatY(images: Image list) = concatAlongAxis 1 images
    static member ConcatZ(images: Image list) = concatZ images
