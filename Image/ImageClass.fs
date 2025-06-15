namespace ImageClass

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

type Image(img: itk.simple.Image) =

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
