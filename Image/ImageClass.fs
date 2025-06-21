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
        | ComplexFloat32 -> itk.simple.PixelIDValueEnum.sitkVectorFloat32 // simple itk does not expose the complex type returned by sitkComplex*
        | ComplexFloat64 -> itk.simple.PixelIDValueEnum.sitkVectorFloat64
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
let inline toItkVector (createFilter: unit -> ^Filter when ^Filter :> System.IDisposable) =
    fun lst ->
        let v = createFilter()
        lst |> List.iter (fun x -> (^Filter : (member Add : _ -> unit) (v, x)))
        v

let inline toVectorUInt8 lst = toItkVector (fun () -> new itk.simple.VectorUInt8()) lst
let inline toVectorInt8 lst = toItkVector (fun () -> new itk.simple.VectorInt8()) lst
let inline toVectorUInt16 lst = toItkVector (fun () -> new itk.simple.VectorUInt16()) lst
let inline toVectorInt16 lst = toItkVector (fun () -> new itk.simple.VectorInt16()) lst
let inline toVectorUInt32 lst = toItkVector (fun () -> new itk.simple.VectorUInt32()) lst
let inline toVectorInt32 lst = toItkVector (fun () -> new itk.simple.VectorInt32()) lst
let inline toVectorUInt64 lst = toItkVector (fun () -> new itk.simple.VectorUInt64()) lst
let inline toVectorInt64 lst = toItkVector (fun () -> new itk.simple.VectorInt64()) lst
let inline toVectorFloat32 lst = toItkVector (fun () -> new itk.simple.VectorFloat()) lst
let inline toVectorFloat64 lst = toItkVector (fun () -> new itk.simple.VectorDouble()) lst

let inline fromItkVector f v = 
    v |> Seq.map f |> Seq.toList

let inline fromVectorUInt8 (v: itk.simple.VectorUInt8) : uint8 list = fromItkVector uint8 v
let inline fromVectorInt8 (v: itk.simple.VectorInt8) : int8 list = fromItkVector int8 v
let inline fromVectorUInt16 (v: itk.simple.VectorUInt16) : uint16 list = fromItkVector uint16 v
let inline fromVectorInt16 (v: itk.simple.VectorInt16) : int16 list = fromItkVector int16 v
let inline fromVectorUInt32 (v: itk.simple.VectorUInt32) : uint list = fromItkVector uint v
let inline fromVectorInt32 (v: itk.simple.VectorInt32) : int list = fromItkVector int v
let inline fromVectorUInt64 (v: itk.simple.VectorUInt64) : uint64 list = fromItkVector uint64 v
let inline fromVectorInt64 (v: itk.simple.VectorInt64) : int64 list = fromItkVector int64 v
let inline fromVectorFloat32 (v: itk.simple.VectorFloat) : float32 list = fromItkVector float32 v
let inline fromVectorFloat64 (v: itk.simple.VectorDouble) : float list = fromItkVector float v

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
    elif t = typeof<uint8 list> then VectorUInt8
    elif t = typeof<int8 list> then VectorInt8
    elif t = typeof<uint16 list> then VectorUInt16
    elif t = typeof<int16 list> then VectorInt16
    elif t = typeof<uint32 list> then VectorUInt32
    elif t = typeof<int32 list> then VectorInt32
    elif t = typeof<uint64 list> then VectorUInt64
    elif t = typeof<int64 list> then VectorInt64
    elif t = typeof<float32 list> then VectorFloat32
    elif t = typeof<float list> then VectorFloat64
    else failwithf "Unsupported pixel type: %O" t

let forcePixelType<'T> (img: itk.simple.Image) : itk.simple.Image =
    let expectedId = (fromType<'T>).ToSimpleITK()
    if img.GetPixelID() = expectedId then
        img // No casting needed
    else
        let cast = new itk.simple.CastImageFilter()
        cast.SetOutputPixelType(expectedId)
        cast.Execute(img)

[<StructuredFormatDisplay("{Display}")>] // Prevent fsi printing information about its members such as img
type Image<'T>(sz: uint list, ?numberComp: uint) =
    let pt =
        try fromType<'T>
        with _ -> failwithf "Unsupported pixel type: %O" typeof<'T>
    let itkId = pt.ToSimpleITK()

    // Is 'T a list type? (e.g., float list)
    let isListType = typeof<'T>.IsGenericType && typeof<'T>.GetGenericTypeDefinition() = typedefof<list<_>>

    let mutable img = 
        match numberComp with 
        | Some v when isListType && v < 2u ->
            new itk.simple.Image(sz |> toVectorUInt32, itkId, 2u)
        | Some v when not isListType && v > 1u ->
            new itk.simple.Image(sz |> toVectorUInt32, itkId)
        | Some v -> 
            new itk.simple.Image(sz |> toVectorUInt32, itkId, v)
        | None -> 
            new itk.simple.Image(sz |> toVectorUInt32, itkId)

    member this.Image = img
    member private this.SetImg (itkImg: itk.simple.Image) : unit =
        img <- itkImg
    member this.GetSize () = img.GetSize() |> fromVectorUInt32
    member this.GetDepth() = img.GetDepth()
    member this.GetDimension() = img.GetDimension()
    member this.GetHeight() = img.GetHeight()
    member this.GetWidth() = img.GetWidth()
    member this.GetNumberOfComponentsPerPixel() = img.GetNumberOfComponentsPerPixel()

    override this.ToString() = 
        let sz = this.GetSize()
        let szStr = List.fold (fun acc elm -> acc + $"x{elm}") (sz |> List.head |> string) (List.tail sz)
        let comp = this.GetNumberOfComponentsPerPixel()
        let vecStr = if comp = 1u then "Scalar" else sprintf $"{comp}-Vector "
        sprintf "%s %s" szStr vecStr
    member this.Display = this.ToString() // related to [<StructuredFormatDisplay>]

    static member ofSimpleITK (itkImg: itk.simple.Image) : Image<'T> =
        let itkImgCast = forcePixelType<'T> itkImg
        let img = Image<'T>([0u;0u])
        img.SetImg itkImgCast
        img

    static member ofArray2D (arr: 'T[,]) : Image<'T> =
        let pt =
            try fromType<'T>
            with _ -> failwithf "Unsupported pixel type: %O" typeof<'T>
        let itkId = pt.ToSimpleITK()
        let sz = [arr.GetLength(0); arr.GetLength(1)] |> List.map uint
        let img = new itk.simple.Image(sz |> toVectorUInt32, itkId)

        let t = typeof<'T>
        arr
        |> Array2D.mapi (fun x y value ->
            let u = [ uint x; uint y ] |> toVectorUInt32
            if      t = typeof<uint8>                   then img.SetPixelAsUInt8(u, unbox value)
            elif    t = typeof<int8>                    then img.SetPixelAsInt8(u, unbox value)
            elif    t = typeof<uint16>                  then img.SetPixelAsUInt16(u, unbox value)
            elif    t = typeof<int16>                   then img.SetPixelAsInt16(u, unbox value)
            elif    t = typeof<uint32>                  then img.SetPixelAsUInt32(u, unbox value)
            elif    t = typeof<int32>                   then img.SetPixelAsInt32(u, unbox value)
            elif    t = typeof<uint64>                  then img.SetPixelAsUInt64(u, unbox value)
            elif    t = typeof<int64>                   then img.SetPixelAsInt64(u, unbox value)
            elif    t = typeof<float32>                 then img.SetPixelAsFloat(u, unbox value)
            elif    t = typeof<float>                   then img.SetPixelAsDouble(u, unbox value)
            elif    t = typeof<System.Numerics.Complex> then
                let c = unbox<System.Numerics.Complex> value
                let v = toVectorFloat64 [ c.Real; c.Imaginary ]
                img.SetPixelAsVectorFloat64(u, v)
            elif    t = typeof<uint8 list>              then img.SetPixelAsVectorUInt8(u, toVectorUInt8 (unbox value))
            elif    t = typeof<int8 list>               then img.SetPixelAsVectorInt8(u, toVectorInt8 (unbox value))
            elif    t = typeof<uint16 list>             then img.SetPixelAsVectorUInt16(u, toVectorUInt16 (unbox value))
            elif    t = typeof<int16 list>              then img.SetPixelAsVectorInt16(u, toVectorInt16 (unbox value))
            elif    t = typeof<uint32 list>             then img.SetPixelAsVectorUInt32(u, toVectorUInt32 (unbox value))
            elif    t = typeof<int32 list>              then img.SetPixelAsVectorInt32(u, toVectorInt32 (unbox value))
            elif    t = typeof<uint64 list>             then img.SetPixelAsVectorUInt64(u, toVectorUInt64 (unbox value))
            elif    t = typeof<int64 list>              then img.SetPixelAsVectorInt64(u, toVectorInt64 (unbox value))
            elif    t = typeof<float32 list>            then img.SetPixelAsVectorFloat32(u, toVectorFloat32 (unbox value))
            elif    t = typeof<float list>              then img.SetPixelAsVectorFloat64(u, toVectorFloat64 (unbox value))
            else failwithf "Unsupported pixel type: %O" t
        )
        |> ignore
        let wrapped = Image<'T>([0u;0u])
        wrapped.SetImg img
        wrapped

    member this.toArray2D: 'T[,] =
        let img = this.Image
        let size = img.GetSize()
        if size.Count <> 2 then
            failwithf "toArray2D requires a 2D image, but got %dD" size.Count

        let width, height = int size.[0], int size.[1]
        let arr = Array2D.zeroCreate<'T> width height

        let t = typeof<'T>
        Array2D.mapi (fun x y _ ->
            let idx = toVectorUInt32 [ uint x; uint y ]
            let value =
                if      t = typeof<uint8>                   then box (img.GetPixelAsUInt8(idx))
                elif    t = typeof<int8>                    then box (img.GetPixelAsInt8(idx))
                elif    t = typeof<uint16>                  then box (img.GetPixelAsUInt16(idx))
                elif    t = typeof<int16>                   then box (img.GetPixelAsInt16(idx))
                elif    t = typeof<uint32>                  then box (img.GetPixelAsUInt32(idx))
                elif    t = typeof<int32>                   then box (img.GetPixelAsInt32(idx))
                elif    t = typeof<uint64>                  then box (img.GetPixelAsUInt64(idx))
                elif    t = typeof<int64>                   then box (img.GetPixelAsInt64(idx))
                elif    t = typeof<float32>                 then box (img.GetPixelAsFloat(idx))
                elif    t = typeof<float>                   then box (img.GetPixelAsDouble(idx))
                elif    t = typeof<System.Numerics.Complex> then
                    let v = img.GetPixelAsVectorFloat64(idx)
                    box (System.Numerics.Complex(v.[0], v.[1]))
                elif    t = typeof<uint8 list>              then box (img.GetPixelAsVectorUInt8(idx) |> fromVectorUInt8)
                elif    t = typeof<int8 list>               then box (img.GetPixelAsVectorInt8(idx) |> fromVectorInt8)
                elif    t = typeof<uint16 list>             then box (img.GetPixelAsVectorUInt16(idx) |> fromVectorUInt16)
                elif    t = typeof<int16 list>              then box (img.GetPixelAsVectorInt16(idx) |> fromVectorInt16)
                elif    t = typeof<uint32 list>             then box (img.GetPixelAsVectorUInt32(idx) |> fromVectorUInt32)
                elif    t = typeof<int32 list>              then box (img.GetPixelAsVectorInt32(idx) |> fromVectorInt32)
                elif    t = typeof<uint64 list>             then box (img.GetPixelAsVectorUInt64(idx) |> fromVectorUInt64)
                elif    t = typeof<int64 list>              then box (img.GetPixelAsVectorInt64(idx) |> fromVectorInt64)
                elif    t = typeof<float32 list>            then box (img.GetPixelAsVectorFloat32(idx) |> fromVectorFloat32)
                elif    t = typeof<float list>              then box (img.GetPixelAsVectorFloat64(idx) |> fromVectorFloat64)
                else failwithf "Unsupported pixel type: %O" t
            arr.[x, y] <- unbox value
            arr.[x, y]  // for mapi result (unused)
        ) |> ignore

        arr

    static member FromFile(filename: string) : Image<'T> =
        let reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        let itkImg = reader.Execute()
        let numComp = itkImg.GetNumberOfComponentsPerPixel()
        let tType = typeof<'T>
        let isVectorType =
            (tType.IsGenericType && tType.GetGenericTypeDefinition() = typedefof<list<_>>)
            || tType.IsArray

        // Validate number of components matches expectations
        match isVectorType, numComp with
        | true, n when n < 2u ->
            failwithf "Pixel type '%O' expects a vector (>=2 components), but image has %d component(s)." tType n
        | false, n when n > 1u ->
            failwithf "Pixel type '%O' expects a scalar (1 component), but image has %d component(s)." tType n
        | _ ->
            Image<'T>.ofSimpleITK(itkImg)

    member this.ToFile(filename: string, ?format: string) =
        let writer = new itk.simple.ImageFileWriter()
        writer.SetFileName(filename)
        match format with
        | Some fmt -> writer.SetImageIO(fmt)
        | None -> ()

        writer.Execute(this.Image)

    // Float images
    static member inline (+) (f1: Image<float>, i: int) =
        let filter = new itk.simple.AddImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, float i))

    static member (+) (i: int, f1: Image<float>) = f1 + i

    static member (+) (f1: Image<'T>, f2: Image<float>) =
        let filter = new itk.simple.AddImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member (+) (f1: Image<float>, f2: Image<'T>) = f2 + f1

    // Int images
    static member (+) (f1: Image<int>, i: float) =
        let filter = new itk.simple.AddImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (+) (i: float, f1: Image<int>) = f1 + i

    static member (+) (f1: Image<int>, i: int) =
        let filter = new itk.simple.AddImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (+) (i: int, f1: Image<int>) = f1 + i

    // Generic same-type images
    static member (+) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))


    // Float image - int scalar
    static member inline (-) (f1: Image<float>, i: int) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, float i))

    static member (-) (i: int, f1: Image<float>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(float i, f1.Image))

    // Image<'T> - Image<float>
    static member (-) (f1: Image<'T>, f2: Image<float>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    // Image<float> - Image<'T>
    static member (-) (f1: Image<float>, f2: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    // Image<int> - float
    static member (-) (f1: Image<int>, i: float) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (-) (i: float, f1: Image<int>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(i, f1.Image))

    // Image<int> - int
    static member (-) (f1: Image<int>, i: int) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (-) (i: int, f1: Image<int>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(i, f1.Image))

    // Same-type subtraction
    static member (-) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    // Float image * int scalar
    static member inline (*) (f1: Image<float>, i: int) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, float i))

    static member (*) (i: int, f1: Image<float>) = (*) f1 i

    // Image<'T> * Image<float>
    static member (*) (f1: Image<'T>, f2: Image<float>) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    // Image<float> * Image<'T>
    static member (*) (f1: Image<float>, f2: Image<'T>) = (*) f2 f1

    // Image<int> * float
    static member (*) (f1: Image<int>, i: float) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (*) (i: float, f1: Image<int>) = (*) f1 i

    // Image<int> * int
    static member (*) (f1: Image<int>, i: int) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (*) (i: int, f1: Image<int>) = (*) f1 i

    // Same-type image * image
    static member (*) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    // Float image / int scalar
    static member inline (/) (f1: Image<float>, i: int) =
        let filter = new itk.simple.DivideImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, float i))

    static member (/) (i: int, f1: Image<float>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(float i, f1.Image))

    // Image<'T> / Image<float>
    static member (/) (f1: Image<'T>, f2: Image<float>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    // Image<float> / Image<'T>
    static member (/) (f1: Image<float>, f2: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    // Image<int> / float
    static member (/) (f1: Image<int>, i: float) =
        let filter = new itk.simple.DivideImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (/) (i: float, f1: Image<int>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(i, f1.Image))

    // Image<int> / int
    static member (/) (f1: Image<int>, i: int) =
        let filter = new itk.simple.DivideImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member (/) (i: int, f1: Image<int>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(i, f1.Image))

    // Same-type image / image
    static member (/) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_Equality (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.EqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_Equality (f1: Image<'T>, i: float) =
        let filter = new itk.simple.EqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_Equality (i: float, f2: Image<'T>) =
        let filter = new itk.simple.EqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    static member op_Inequality (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_Inequality (f1: Image<'T>, i: float) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_Inequality (i: float, f2: Image<'T>) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    static member op_LessThan (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.LessImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_LessThan (f1: Image<'T>, i: float) =
        let filter = new itk.simple.LessImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_LessThan (i: float, f2: Image<'T>) =
        let filter = new itk.simple.LessImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    static member op_LessThanOrEqual (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_LessThanOrEqual (f1: Image<'T>, i: float) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_LessThanOrEqual (i: float, f2: Image<'T>) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    static member op_GreaterThan (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.GreaterImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_GreaterThan (f1: Image<'T>, i: float) =
        let filter = new itk.simple.GreaterImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_GreaterThan (i: float, f2: Image<'T>) =
        let filter = new itk.simple.GreaterImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    static member op_GreaterThanOrEqual (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_GreaterThanOrEqual (f1: Image<'T>, i: float) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_GreaterThanOrEqual (i: float, f2: Image<'T>) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    // Modulus ( % )
    static member op_Modulus (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.ModulusImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_Modulus (f1: Image<'T>, i: uint) =
        let filter = new itk.simple.ModulusImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_Modulus (i: uint, f2: Image<'T>) =
        let filter = new itk.simple.ModulusImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    // Power (no direct operator for ** in .NET) - provide a named method instead
    static member Pow (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.PowImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member Pow (f1: Image<'T>, i: float) =
        let filter = new itk.simple.PowImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member Pow (f1: Image<'T>, i: int) =
        let filter = new itk.simple.PowImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member Pow (i: float, f2: Image<'T>) =
        let filter = new itk.simple.PowImageFilter()
        Image<float>.ofSimpleITK(filter.Execute(i, f2.Image))

    static member Pow (i: int, f2: Image<'T>) =
        let filter = new itk.simple.PowImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(i, f2.Image))

    // Bitwise AND ( &&& )
    static member op_BitwiseAnd (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.AndImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_BitwiseAnd (f1: Image<int>, i: int) =
        let filter = new itk.simple.AndImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_BitwiseAnd (i: int, f2: Image<int>) =
        let filter = new itk.simple.AndImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(i, f2.Image))

    // Bitwise XOR ( ^^^ )
    static member op_ExclusiveOr (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.XorImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_ExclusiveOr (f1: Image<int>, i: int) =
        let filter = new itk.simple.XorImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_ExclusiveOr (i: int, f2: Image<int>) =
        let filter = new itk.simple.XorImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(i, f2.Image))

    // Bitwise OR ( ||| )
    static member op_BitwiseOr (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.OrImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member op_BitwiseOr (f1: Image<int>, i: int) =
        let filter = new itk.simple.OrImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(f1.Image, i))

    static member op_BitwiseOr (i: int, f2: Image<int>) =
        let filter = new itk.simple.OrImageFilter()
        Image<int>.ofSimpleITK(filter.Execute(i, f2.Image))

    // Unary bitwise NOT ( ~~~ )
    static member op_LogicalNot (f: Image<'T>) =
        let filter = new itk.simple.InvertIntensityImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f.Image))

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
            elif t = typeof<float>                   then box (img.GetPixelAsDouble(u))
            elif t = typeof<System.Numerics.Complex> then box (img.GetPixelAsVectorFloat64(u) |> fromVectorFloat64)
            elif t = typeof<int8 list>               then box (img.GetPixelAsVectorInt8(u) |> fromVectorInt8)
            elif t = typeof<uint16 list>             then box (img.GetPixelAsVectorUInt16(u) |> fromVectorUInt16)
            elif t = typeof<int16 list>              then box (img.GetPixelAsVectorInt16(u) |> fromVectorInt16)
            elif t = typeof<uint32 list>             then box (img.GetPixelAsVectorUInt32(u) |> fromVectorUInt32)
            elif t = typeof<int32 list>              then box (img.GetPixelAsVectorInt32(u) |> fromVectorInt32)
            elif t = typeof<uint64 list>             then box (img.GetPixelAsVectorUInt64(u) |> fromVectorUInt64)
            elif t = typeof<int64 list>              then box (img.GetPixelAsVectorInt64(u) |> fromVectorInt64)
            elif t = typeof<float32 list>            then box (img.GetPixelAsVectorFloat32(u) |> fromVectorFloat32)
            elif t = typeof<float list>              then box (img.GetPixelAsVectorFloat64(u) |> fromVectorFloat64)
            else failwithf "Unsupported pixel type: %O" t
        raw :?> 'T

    member this.Set (coords: uint list, value: 'T) : unit =
        let u = toVectorUInt32 coords
        let t = typeof<'T>
        if      t = typeof<uint8>                   then this.Image.SetPixelAsUInt8(u, unbox value)
        elif    t = typeof<int8>                    then this.Image.SetPixelAsInt8(u, unbox value)
        elif    t = typeof<uint16>                  then this.Image.SetPixelAsUInt16(u, unbox value)
        elif    t = typeof<int16>                   then this.Image.SetPixelAsInt16(u, unbox value)
        elif    t = typeof<uint32>                  then this.Image.SetPixelAsUInt32(u, unbox value)
        elif    t = typeof<int32>                   then this.Image.SetPixelAsInt32(u, unbox value)
        elif    t = typeof<uint64>                  then this.Image.SetPixelAsUInt64(u, unbox value)
        elif    t = typeof<int64>                   then this.Image.SetPixelAsInt64(u, unbox value)
        elif    t = typeof<float32>                 then this.Image.SetPixelAsFloat(u, unbox value)
        elif    t = typeof<float>                   then this.Image.SetPixelAsDouble(u, unbox value)
        elif    t = typeof<System.Numerics.Complex> then
            let c = unbox<System.Numerics.Complex> value
            let v = toVectorFloat64 [ c.Real; c.Imaginary ]
            this.Image.SetPixelAsVectorFloat64(u, v)
        elif    t = typeof<uint8 list>              then this.Image.SetPixelAsVectorUInt8(u, toVectorUInt8 (unbox value))
        elif    t = typeof<int8 list>               then this.Image.SetPixelAsVectorInt8(u, toVectorInt8 (unbox value))
        elif    t = typeof<uint16 list>             then this.Image.SetPixelAsVectorUInt16(u, toVectorUInt16 (unbox value))
        elif    t = typeof<int16 list>              then this.Image.SetPixelAsVectorInt16(u, toVectorInt16 (unbox value))
        elif    t = typeof<uint32 list>             then this.Image.SetPixelAsVectorUInt32(u, toVectorUInt32 (unbox value))
        elif    t = typeof<int32 list>              then this.Image.SetPixelAsVectorInt32(u, toVectorInt32 (unbox value))
        elif    t = typeof<uint64 list>             then this.Image.SetPixelAsVectorUInt64(u, toVectorUInt64 (unbox value))
        elif    t = typeof<int64 list>              then this.Image.SetPixelAsVectorInt64(u, toVectorInt64 (unbox value))
        elif    t = typeof<float32 list>            then this.Image.SetPixelAsVectorFloat32(u, toVectorFloat32 (unbox value))
        elif    t = typeof<float list>              then this.Image.SetPixelAsVectorFloat64(u, toVectorFloat64 (unbox value))
        else failwithf "Unsupported pixel type: %O" t

    member this.Item
        with get(x: int, y: int) : 'T =
            this.Get([ uint x; uint y ])
        and set(x: int, y: int) (value: 'T) : unit =
            this.Set([ uint x; uint y ], value)

    member this.Item
        with get(x: int, y: int, z: int) : 'T =
            this.Get([ uint x; uint y; uint z ])
        and set(x: int, y: int, z: int) (value: 'T) : unit =
            this.Set([ uint x; uint y; uint z ], value)
