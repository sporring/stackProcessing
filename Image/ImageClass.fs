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
        | UInt8          -> itk.simple.PixelIDValueEnum.sitkUInt8
        | Int8           -> itk.simple.PixelIDValueEnum.sitkInt8
        | UInt16         -> itk.simple.PixelIDValueEnum.sitkUInt16
        | Int16          -> itk.simple.PixelIDValueEnum.sitkInt16
        | UInt32         -> itk.simple.PixelIDValueEnum.sitkUInt32
        | Int32          -> itk.simple.PixelIDValueEnum.sitkInt32
        | UInt64         -> itk.simple.PixelIDValueEnum.sitkUInt64
        | Int64          -> itk.simple.PixelIDValueEnum.sitkInt64
        | Float32        -> itk.simple.PixelIDValueEnum.sitkFloat32
        | Float64        -> itk.simple.PixelIDValueEnum.sitkFloat64
        | ComplexFloat32 -> itk.simple.PixelIDValueEnum.sitkComplexFloat32
        | ComplexFloat64 -> itk.simple.PixelIDValueEnum.sitkComplexFloat64
        | VectorUInt8    -> itk.simple.PixelIDValueEnum.sitkVectorUInt8
        | VectorInt8     -> itk.simple.PixelIDValueEnum.sitkVectorInt8
        | VectorUInt16   -> itk.simple.PixelIDValueEnum.sitkVectorUInt16
        | VectorInt16    -> itk.simple.PixelIDValueEnum.sitkVectorInt16
        | VectorUInt32   -> itk.simple.PixelIDValueEnum.sitkVectorUInt32
        | VectorInt32    -> itk.simple.PixelIDValueEnum.sitkVectorInt32
        | VectorUInt64   -> itk.simple.PixelIDValueEnum.sitkVectorUInt64
        | VectorInt64    -> itk.simple.PixelIDValueEnum.sitkVectorInt64
        | VectorFloat32  -> itk.simple.PixelIDValueEnum.sitkVectorFloat32
        | VectorFloat64  -> itk.simple.PixelIDValueEnum.sitkVectorFloat64
        | LabelUInt8     -> itk.simple.PixelIDValueEnum.sitkLabelUInt8
        | LabelUInt16    -> itk.simple.PixelIDValueEnum.sitkLabelUInt16
        | LabelUInt32    -> itk.simple.PixelIDValueEnum.sitkLabelUInt32
        | LabelUInt64    -> itk.simple.PixelIDValueEnum.sitkLabelUInt64

    member this.Zero() =
        match this with
        | UInt8 | VectorUInt8 | LabelUInt8    -> box 0uy
        | Int8  | VectorInt8                  -> box 0y
        | UInt16 | VectorUInt16 | LabelUInt16 -> box 0us
        | Int16 | VectorInt16                 -> box 0s
        | UInt32 | VectorUInt32 | LabelUInt32 -> box 0u
        | Int32 | VectorInt32                 -> box 0
        | UInt64 | VectorUInt64 | LabelUInt64 -> box 0UL
        | Int64 | VectorInt64                 -> box 0L
        | Float32 | VectorFloat32             -> box 0.0f
        | Float64 | VectorFloat64             -> box 0.0
        | ComplexFloat32                      -> box (System.Numerics.Complex(0.0, 0.0))
        | ComplexFloat64                      -> box (System.Numerics.Complex(0.0, 0.0))

    member this.One() =
        match this with
        | UInt8 | VectorUInt8 | LabelUInt8    -> box 1uy
        | Int8  | VectorInt8                  -> box 1y
        | UInt16 | VectorUInt16 | LabelUInt16 -> box 1us
        | Int16 | VectorInt16                 -> box 1s
        | UInt32 | VectorUInt32 | LabelUInt32 -> box 1u
        | Int32 | VectorInt32                 -> box 1
        | UInt64 | VectorUInt64 | LabelUInt64 -> box 1UL
        | Int64 | VectorInt64                 -> box 1L
        | Float32 | VectorFloat32             -> box 1.0f
        | Float64 | VectorFloat64             -> box 1.0
        | ComplexFloat32                      -> box (System.Numerics.Complex(1.0, 0.0))
        | ComplexFloat64                      -> box (System.Numerics.Complex(1.0, 0.0))

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

let fromType<'T> : PixelType =
    let t = typeof<'T>
    if t = typeof<uint8> then UInt8
    elif t = typeof<int8> then Int8
    elif t = typeof<uint16> then UInt16
    elif t = typeof<int16> then Int16
    elif t = typeof<uint32> then UInt32
    elif t = typeof<int32> then Int32
    elif t = typeof<uint64> then UInt64
    elif t = typeof<int64> then Int64
    elif t = typeof<float32> then Float32
    elif t = typeof<float> then Float64
    elif t = typeof<System.Numerics.Complex> then ComplexFloat64
    elif t = typeof<int8[]> then VectorInt8
    elif t = typeof<uint16[]> then VectorUInt16
    elif t = typeof<int16[]> then VectorInt16
    elif t = typeof<uint32[]> then VectorUInt32
    elif t = typeof<int32[]> then VectorInt32
    elif t = typeof<uint64[]> then VectorUInt64
    elif t = typeof<int64[]> then VectorInt64
    elif t = typeof<float32[]> then VectorFloat32
    elif t = typeof<float[]> then VectorFloat64
    else failwithf "Unsupported pixel type: %O" t

type Image<'T> (img: itk.simple.Image) =

    /// Underlying itk.simple image
    member this.Image = img

    /// String representation
    override this.ToString() = img.ToString()

    static member Create(size: uint list, ?value: obj) =
        let pt =
            try fromType<'T>
            with _ -> failwithf "Unsupported pixel type: %O" typeof<'T>
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
        Image<'T>(imgFilled)

    static member FromFile(filename: string) =
        let reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        Image<'T>(reader.Execute())

    static member (+) (f1 : Image<'T>, f2 : Image<'T>) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member (+) (f1: Image<'T>, i : float) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member (+) (i : float, f2: Image<'T>) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member (-) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member (-) (f1: Image<'T>, i: float) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member (-) (i: float, f2: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member (*) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member (*) (f1: Image<'T>, i: float) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member (*) (i: float, f2: Image<'T>) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member (/) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member (/) (f1: Image<'T>, i: float) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member (/) (i: float, f2: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member op_Equality (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.EqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_Equality (f1: Image<'T>, i: float) =
        let filter = new itk.simple.EqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_Equality (i: float, f2: Image<'T>) =
        let filter = new itk.simple.EqualImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member op_Inequality (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_Inequality (f1: Image<'T>, i: float) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_Inequality (i: float, f2: Image<'T>) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member op_LessThan (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.LessImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_LessThan (f1: Image<'T>, i: float) =
        let filter = new itk.simple.LessImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_LessThan (i: float, f2: Image<'T>) =
        let filter = new itk.simple.LessImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member op_LessThanOrEqual (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_LessThanOrEqual (f1: Image<'T>, i: float) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_LessThanOrEqual (i: float, f2: Image<'T>) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member op_GreaterThan (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.GreaterImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_GreaterThan (f1: Image<'T>, i: float) =
        let filter = new itk.simple.GreaterImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_GreaterThan (i: float, f2: Image<'T>) =
        let filter = new itk.simple.GreaterImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    static member op_GreaterThanOrEqual (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_GreaterThanOrEqual (f1: Image<'T>, i: float) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_GreaterThanOrEqual (i: float, f2: Image<'T>) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    // Modulus ( % )
    static member op_Modulus (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.ModulusImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_Modulus (f1: Image<'T>, i: uint) =
        let filter = new itk.simple.ModulusImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_Modulus (i: uint, f2: Image<'T>) =
        let filter = new itk.simple.ModulusImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    // Power (no direct operator for ** in .NET) - provide a named method instead
    static member Pow (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.PowImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member Pow (f1: Image<'T>, i: float) =
        let filter = new itk.simple.PowImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member Pow (i: float, f2: Image<'T>) =
        let filter = new itk.simple.PowImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    // Bitwise AND ( &&& )
    static member op_BitwiseAnd (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.AndImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_BitwiseAnd (f1: Image<'T>, i: int) =
        let filter = new itk.simple.AndImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_BitwiseAnd (i: int, f2: Image<'T>) =
        let filter = new itk.simple.AndImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    // Bitwise XOR ( ^^^ )
    static member op_ExclusiveOr (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.XorImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_ExclusiveOr (f1: Image<'T>, i: int) =
        let filter = new itk.simple.XorImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_ExclusiveOr (i: int, f2: Image<'T>) =
        let filter = new itk.simple.XorImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    // Bitwise OR ( ||| )
    static member op_BitwiseOr (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.OrImageFilter()
        Image<'T>(filter.Execute(f1.Image, f2.Image))

    static member op_BitwiseOr (f1: Image<'T>, i: int) =
        let filter = new itk.simple.OrImageFilter()
        Image<'T>(filter.Execute(f1.Image, i))

    static member op_BitwiseOr (i: int, f2: Image<'T>) =
        let filter = new itk.simple.OrImageFilter()
        Image<'T>(filter.Execute(i, f2.Image))

    // Unary bitwise NOT ( ~~~ )
    static member op_LogicalNot (f: Image<'T>) =
        let filter = new itk.simple.InvertIntensityImageFilter()
        Image<'T>(filter.Execute(f.Image))

    member this.Get (coords: uint list) : 'T =
        let u = coords |> toVectorUInt32
        let t = typeof<'T>
        let raw =
            if   t = typeof<uint8>                   then box (img.GetPixelAsUInt8(u))
            elif t = typeof<int8>                    then box (img.GetPixelAsInt8(u))
            elif t = typeof<uint16>                  then box (img.GetPixelAsUInt16(u))
            elif t = typeof<int16>                   then box (img.GetPixelAsInt16(u))
            elif t = typeof<uint32>                  then box (img.GetPixelAsUInt32(u))
            elif t = typeof<int32>                   then box (img.GetPixelAsInt32(u))
            elif t = typeof<uint64>                  then box (img.GetPixelAsUInt64(u))
            elif t = typeof<int64>                   then box (img.GetPixelAsInt64(u))
            elif t = typeof<float32>                 then box (img.GetPixelAsFloat(u))
            elif t = typeof<float>                   then box (img.GetPixelAsFloat(u))
            elif t = typeof<System.Numerics.Complex> then box (img.GetPixelAsComplexFloat64(u))
            elif t = typeof<int8 list>               then box (img.GetPixelAsVectorInt8(u))
            elif t = typeof<uint16 list>             then box (img.GetPixelAsVectorUInt16(u))
            elif t = typeof<int16 list>              then box (img.GetPixelAsVectorInt16(u))
            elif t = typeof<uint32 list>             then box (img.GetPixelAsVectorUInt32(u))
            elif t = typeof<int32 list>              then box (img.GetPixelAsVectorInt32(u))
            elif t = typeof<uint64 list>             then box (img.GetPixelAsVectorUInt64(u))
            elif t = typeof<int64 list>              then box (img.GetPixelAsVectorInt64(u))
            elif t = typeof<float32 list>            then box (img.GetPixelAsVectorFloat32(u))
            elif t = typeof<float list>              then box (img.GetPixelAsVectorFloat64(u))
            else failwithf "Unsupported pixel type: %O" t
        raw :?> 'T

    member this.Item
        with get(x: int, y: int) : 'T =
            this.Get([x;y]|>List.map uint)

    member this.Item
        with get(x: int, y: int, z: int) : 'T =
            this.Get([x;y;z]|>List.map uint)
