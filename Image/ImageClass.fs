namespace ImageClass
open itk.simple

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
        | UInt8         -> PixelIDValueEnum.sitkUInt8
        | Int8          -> PixelIDValueEnum.sitkInt8
        | UInt16        -> PixelIDValueEnum.sitkUInt16
        | Int16         -> PixelIDValueEnum.sitkInt16
        | UInt32        -> PixelIDValueEnum.sitkUInt32
        | Int32         -> PixelIDValueEnum.sitkInt32
        | UInt64        -> PixelIDValueEnum.sitkUInt64
        | Int64         -> PixelIDValueEnum.sitkInt64
        | Float32       -> PixelIDValueEnum.sitkFloat32
        | Float64       -> PixelIDValueEnum.sitkFloat64
        | VectorUInt8   -> PixelIDValueEnum.sitkVectorUInt8
        | VectorFloat32 -> PixelIDValueEnum.sitkVectorFloat32
        | LabelUInt8    -> PixelIDValueEnum.sitkLabelUInt8
        | LabelUInt16   -> PixelIDValueEnum.sitkLabelUInt16
        | LabelUInt32   -> PixelIDValueEnum.sitkLabelUInt32
        | LabelUInt64   -> PixelIDValueEnum.sitkLabelUInt64

type Raw(img: itk.simple.Image) =

    /// Underlying itk.simple image
    member this.Image = img

    /// String representation
    override this.ToString() = img.ToString()

    /// generate images
    static member FromSize(size: int list) : Raw =
        let vec = new VectorUInt32()
        size |> List.iter (uint32 >> vec.Add)
        let img = new itk.simple.Image(vec, PixelType.Float32.ToSimpleITK())
        new Raw(img)

    static member FromSizeAndType(size: int list, pixelType: PixelType) : Raw =
        let vec = new VectorUInt32()
        size |> List.iter (uint32 >> vec.Add)
        let img = new itk.simple.Image(vec, pixelType.ToSimpleITK())
        new Raw(img)

    static member FromConstant(size: int list, pixelType: PixelType, value: float) : Raw =
        let vec = new VectorUInt32()
        size |> List.iter (uint32 >> vec.Add)
        let img = new itk.simple.Image(vec, pixelType.ToSimpleITK())
        let scalarFilter = new ShiftScaleImageFilter()
        scalarFilter.SetShift(value)
        new Raw(scalarFilter.Execute(img))

    static member FromSimpleITK(img: itk.simple.Image) : Raw =
        new Raw(img)

    static member FromFile(filename: string) : Raw =
        let reader = new ImageFileReader()
        reader.SetFileName(filename)
        new Raw(reader.Execute())

    static member (+) (f1 : Raw, f2 : Raw) =
        let filter = new AddImageFilter()
        new Raw(filter.Execute(f1.Image, f2.Image))

    static member (+) (f1: Raw, i : float) =
        let filter = new AddImageFilter()
        new Raw(filter.Execute(f1.Image, i))

    static member (+) (i : float, f2: Raw) =
        let filter = new AddImageFilter()
        new Raw(filter.Execute(i, f2.Image))

