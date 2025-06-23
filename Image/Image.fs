module Image
open FSharp.Collections

module internal InternalHelpers =
    let toVectorUInt8 (lst: uint8 list)     = new itk.simple.VectorUInt8(lst)
    let toVectorInt8 (lst: int8 list)       = new itk.simple.VectorInt8(lst)
    let toVectorUInt16 (lst: uint16 list)   = new itk.simple.VectorUInt16(lst)
    let toVectorInt16 (lst: int16 list)     = new itk.simple.VectorInt16(lst)
    let toVectorUInt32 (lst: uint32 list)   = new itk.simple.VectorUInt32(lst)
    let toVectorInt32 (lst: int32 list)     = new itk.simple.VectorInt32(lst)
    let toVectorUInt64 (lst: uint64 list)   = new itk.simple.VectorUInt64(lst)
    let toVectorInt64 (lst: int64 list)     = new itk.simple.VectorInt64(lst)
    let toVectorFloat32 (lst: float32 list) = new itk.simple.VectorFloat(lst)
    let toVectorFloat64 (lst: float list) = new itk.simple.VectorDouble(lst)

    let fromItkVector f v = 
        v |> Seq.map f |> Seq.toList

    let fromVectorUInt8 (v: itk.simple.VectorUInt8) : uint8 list = fromItkVector uint8 v
    let fromVectorInt8 (v: itk.simple.VectorInt8) : int8 list = fromItkVector int8 v
    let fromVectorUInt16 (v: itk.simple.VectorUInt16) : uint16 list = fromItkVector uint16 v
    let fromVectorInt16 (v: itk.simple.VectorInt16) : int16 list = fromItkVector int16 v
    let fromVectorUInt32 (v: itk.simple.VectorUInt32) : uint list = fromItkVector uint v
    let fromVectorInt32 (v: itk.simple.VectorInt32) : int list = fromItkVector int v
    let fromVectorUInt64 (v: itk.simple.VectorUInt64) : uint64 list = fromItkVector uint64 v
    let fromVectorInt64 (v: itk.simple.VectorInt64) : int64 list = fromItkVector int64 v
    let fromVectorFloat32 (v: itk.simple.VectorFloat) : float32 list = fromItkVector float32 v
    let fromVectorFloat64 (v: itk.simple.VectorDouble) : float list = fromItkVector float v

    let fromType<'T> =
        let t = typeof<'T>
        if t = typeof<uint8> then itk.simple.PixelIDValueEnum.sitkUInt8
        elif t = typeof<int8> then itk.simple.PixelIDValueEnum.sitkInt8
        elif t = typeof<uint16> then itk.simple.PixelIDValueEnum.sitkUInt16
        elif t = typeof<int16> then itk.simple.PixelIDValueEnum.sitkInt16
        elif t = typeof<uint32> then itk.simple.PixelIDValueEnum.sitkUInt32
        elif t = typeof<int32> then itk.simple.PixelIDValueEnum.sitkInt32
        elif t = typeof<uint64> then itk.simple.PixelIDValueEnum.sitkUInt64
        elif t = typeof<int64> then itk.simple.PixelIDValueEnum.sitkInt64
        elif t = typeof<float32> then itk.simple.PixelIDValueEnum.sitkFloat32
        elif t = typeof<float> then itk.simple.PixelIDValueEnum.sitkFloat64
        elif t = typeof<System.Numerics.Complex> then itk.simple.PixelIDValueEnum.sitkVectorFloat64
        elif t = typeof<uint8 list> then itk.simple.PixelIDValueEnum.sitkVectorUInt8
        elif t = typeof<int8 list> then itk.simple.PixelIDValueEnum.sitkVectorInt8
        elif t = typeof<uint16 list> then itk.simple.PixelIDValueEnum.sitkVectorUInt16
        elif t = typeof<int16 list> then itk.simple.PixelIDValueEnum.sitkVectorInt16
        elif t = typeof<uint32 list> then itk.simple.PixelIDValueEnum.sitkVectorUInt32
        elif t = typeof<int32 list> then itk.simple.PixelIDValueEnum.sitkVectorInt32
        elif t = typeof<uint64 list> then itk.simple.PixelIDValueEnum.sitkVectorUInt64
        elif t = typeof<int64 list> then itk.simple.PixelIDValueEnum.sitkVectorInt64
        elif t = typeof<float32 list> then itk.simple.PixelIDValueEnum.sitkVectorFloat32
        elif t = typeof<float list> then itk.simple.PixelIDValueEnum.sitkVectorFloat64
        else failwithf "Unsupported pixel type: %O" t

    let ofCastItk<'T> (img: itk.simple.Image) : itk.simple.Image =
        let expectedId = fromType<'T>
        if img.GetPixelID() = expectedId then
            img // No casting needed
        else
            let cast = new itk.simple.CastImageFilter()
            cast.SetOutputPixelType(expectedId)
            cast.Execute(img)

    let GetArrayFromImage (img: itk.simple.Image): 'T[,] =
        let size = img.GetSize()
        if size.Count <> 2 then
            failwithf "GetArrayFromImage requires a 2D image, but got %dD" size.Count

        let width, height = int size.[0], int size.[1]
        let t = typeof<'T>
        Array2D.init width height (fun x y ->
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
            unbox value
        )

open InternalHelpers

[<StructuredFormatDisplay("{Display}")>] // Prevent fsi printing information about its members such as img
type Image<'T>(sz: uint list, ?numberComp: uint) =
    let itkId = fromType<'T>
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
    member this.GetDepth() = max 1u (img.GetDepth()) // Non-vector images returns 0
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
        let itkImgCast = ofCastItk<'T> itkImg
        let img = Image<'T>([0u;0u])
        img.SetImg itkImgCast
        img

    member this.toSimpleITK () : itk.simple.Image =
        img

    member this.castTo<'S> () : Image<'S> =
        Image<'S>.ofSimpleITK img

    static member ofArray2D (arr: 'T[,]) : Image<'T> =
        let itkId = fromType<'T>
        let sz = [arr.GetLength(0); arr.GetLength(1)] |> List.map uint
        let img = new itk.simple.Image(sz |> toVectorUInt32, itkId)

        let t = typeof<'T>
        arr
        |> Array2D.iteri (fun x y value ->
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
        let wrapped = Image<'T>([0u;0u])
        wrapped.SetImg img
        wrapped

    member this.toArray2D (): 'T[,] =
        GetArrayFromImage this.Image

    static member ofImageList (images: Image<'S> list) : Image<'S list> =
        let itkImages = images |> List.map (fun img -> img.Image)
        use filter = new itk.simple.ComposeImageFilter()
        match itkImages with // seems no other way than unrolling them manually
        | [i1; i2] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2))
        | [i1; i2; i3] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2, i3))
        | [i1; i2; i3; i4] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2, i3, i4))
        | [i1; i2; i3; i4; i5] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2, i3, i4, i5))
        | [] ->
            invalidArg "images" "At least two images are required for ComposeImageFilter."
        | _ ->
            invalidArg "images" "ComposeImageFilter supports up to 10 images."

    member this.toImageList () : Image<'S> list =
        let filter = new itk.simple.VectorIndexSelectionCastImageFilter()
        let n = this.Image.GetNumberOfComponentsPerPixel() |> int
        List.init n (fun i ->
            filter.SetIndex(uint i)
            let scalarItk = filter.Execute(this.Image)
            Image<'S>.ofSimpleITK scalarItk
        )

    static member ofFile(filename: string) : Image<'T> =
        use reader = new itk.simple.ImageFileReader()
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

    member this.toFile(filename: string, ?format: string) =
        use writer = new itk.simple.ImageFileWriter()
        writer.SetFileName(filename)
        match format with
        | Some fmt -> writer.SetImageIO(fmt)
        | None -> ()
        writer.Execute(this.Image)

    // Addition
    static member (+) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member (+) (f1: Image<'T>, i: int8) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: int8, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: uint8) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: uint8, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: int16) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: int16, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: uint16) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: uint16, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: int32) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: int32, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: uint32) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: uint32, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: int64) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: int64, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: uint64) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: uint64, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: float32) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: float32, f1: Image<'T>) = f1 + i

    static member (+) (f1: Image<'T>, i: float) =
        let filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (+) (i: float, f1: Image<'T>) = f1 + i

    // Subtraction
    static member (-) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member (-) (f1: Image<'T>, i: int8) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: int8, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (-) (f1: Image<'T>, i: uint8) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: uint8, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (-) (f1: Image<'T>, i: int16) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: int16, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (-) (f1: Image<'T>, i: uint16) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: uint16, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (-) (f1: Image<'T>, i: int32) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: int32, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (-) (f1: Image<'T>, i: uint32) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: uint32, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (-) (f1: Image<'T>, i: int64) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: int64, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (-) (f1: Image<'T>, i: uint64) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: uint64, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (-) (f1: Image<'T>, i: float32) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: float32, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (-) (f1: Image<'T>, i: float) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (-) (i: float, f1: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    // Multiplication
    static member (*) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member (*) (f1: Image<'T>, i: int8) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: int8, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: uint8) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: uint8, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: int16) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: int16, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: uint16) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: uint16, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: int32) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: int32, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: uint32) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: uint32, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: int64) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: int64, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: uint64) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: uint64, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: float32) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: float32, f1: Image<'T>) = f1 * i

    static member (*) (f1: Image<'T>, i: float) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (*) (i: float, f1: Image<'T>) = f1 * i

    // Division
    static member (/) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, f2.Image))

    static member (/) (f1: Image<'T>, i: int8) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: int8, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (/) (f1: Image<'T>, i: uint8) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: uint8, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (/) (f1: Image<'T>, i: int16) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: int16, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (/) (f1: Image<'T>, i: uint16) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: uint16, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (/) (f1: Image<'T>, i: int32) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: int32, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (/) (f1: Image<'T>, i: uint32) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: uint32, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (/) (f1: Image<'T>, i: int64) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: int64, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (/) (f1: Image<'T>, i: uint64) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: uint64, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    static member (/) (f1: Image<'T>, i: float32) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: float32, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i,f1.Image))

    static member (/) (f1: Image<'T>, i: float) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.Image, float i))
    static member (/) (i: float, f1: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(float i, f1.Image))

    // Comparision
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
