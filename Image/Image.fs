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
//        elif t = typeof<System.Numerics.Complex> then itk.simple.PixelIDValueEnum.sitkVectorFloat64
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

    let ofCastItk<'T> (itkImg: itk.simple.Image) : itk.simple.Image =
        let expectedId = fromType<'T>
        if itkImg.GetPixelID() = expectedId then
            itkImg // No casting needed
        else
            let cast = new itk.simple.CastImageFilter()
            cast.SetOutputPixelType(expectedId)
            cast.Execute(itkImg)

    let array2dZip (a: 'T[,]) (b: 'U[,]) : ('T * 'U)[,] =
        let wA, hA = a.GetLength(0), a.GetLength(1)
        let wB, hB = b.GetLength(0), b.GetLength(1)
        if wA <> wB || hA <> hB then
            invalidArg "b" $"Array dimensions must match: {wA}x{hA} vs {wB}x{hB}"
        Array2D.init wA hA (fun x y -> a[x, y], b[x, y])

    let pixelIdToString (id: itk.simple.PixelIDValueEnum) : string =
        (id.ToString()).Substring(4)

    let flatIndices (size: uint list): seq<uint list> = 
        match size.Length with
            | 2 ->
                seq { 
                    for i0 in [0..(int size[0] - 1)] do 
                        for i1 in [0..(int size[1] - 1)] do 
                            yield [uint i0; uint i1] }
            | 3 ->
                seq { 
                    for i0 in [0..(int size[0] - 1)] do 
                        for i1 in [0..(int size[1] - 1)] do 
                            for i2 in [0..(int size[2] - 1)] do 
                               yield [uint i0; uint i1; uint i2] }
            | 4 ->
                seq { 
                    for i0 in [0..(int size[0] - 1)] do 
                        for i1 in [0..(int size[1] - 1)] do 
                            for i2 in [0..(int size[2] - 1)] do 
                                for i3 in [0..(int size[3] - 1)] do 
                                    yield [uint i0; uint i1; uint i2; uint i3] }
            | _ -> failwith $"Unsupported dimensionality {size.Length}"

    let setBoxedPixel (sitkImg: itk.simple.Image) (t: itk.simple.PixelIDValueEnum) (u: itk.simple.VectorUInt32) (value: obj) : unit =
        if      t = fromType<uint8>                   then sitkImg.SetPixelAsUInt8(u, unbox value)
        elif    t = fromType<int8>                    then sitkImg.SetPixelAsInt8(u, unbox value)
        elif    t = fromType<uint16>                  then sitkImg.SetPixelAsUInt16(u, unbox value)
        elif    t = fromType<int16>                   then sitkImg.SetPixelAsInt16(u, unbox value)
        elif    t = fromType<uint32>                  then sitkImg.SetPixelAsUInt32(u, unbox value)
        elif    t = fromType<int32>                  then sitkImg.SetPixelAsInt32(u, unbox value)
        elif    t = fromType<uint64>                  then sitkImg.SetPixelAsUInt64(u, unbox value)
        elif    t = fromType<int64>                  then sitkImg.SetPixelAsInt64(u, unbox value)
        elif    t = fromType<float32>                 then sitkImg.SetPixelAsFloat(u, unbox value)
        elif    t = fromType<float>                   then sitkImg.SetPixelAsDouble(u, unbox value)
//        elif    t = fromType<System.Numerics.Complex> then
//            let c = unbox<System.Numerics.Complex> value
//            let v = toVectorFloat64 [ c.Real; c.Imaginary ]
//            sitkImg.SetPixelAsVectorFloat64(u, v)
        elif    t = fromType<uint8 list>              then sitkImg.SetPixelAsVectorUInt8(u, toVectorUInt8 (unbox value))
        elif    t = fromType<int8 list>               then sitkImg.SetPixelAsVectorInt8(u, toVectorInt8 (unbox value))
        elif    t = fromType<uint16 list>             then sitkImg.SetPixelAsVectorUInt16(u, toVectorUInt16 (unbox value))
        elif    t = fromType<int16 list>              then sitkImg.SetPixelAsVectorInt16(u, toVectorInt16 (unbox value))
        elif    t = fromType<uint32 list>             then sitkImg.SetPixelAsVectorUInt32(u, toVectorUInt32 (unbox value))
        elif    t = fromType<int32 list>              then sitkImg.SetPixelAsVectorInt32(u, toVectorInt32 (unbox value))
        elif    t = fromType<uint64 list>             then sitkImg.SetPixelAsVectorUInt64(u, toVectorUInt64 (unbox value))
        elif    t = fromType<int64 list>              then sitkImg.SetPixelAsVectorInt64(u, toVectorInt64 (unbox value))
        elif    t = fromType<float32 list>            then sitkImg.SetPixelAsVectorFloat32(u, toVectorFloat32 (unbox value))
        elif    t = fromType<float list>              then sitkImg.SetPixelAsVectorFloat64(u, toVectorFloat64 (unbox value))
        else failwithf "Unsupported pixel type: %O" t

    let getBoxedPixel (img : itk.simple.Image) (t   : itk.simple.PixelIDValueEnum) (u   : itk.simple.VectorUInt32) : obj =

        if      t = fromType<uint8>                   then box (img.GetPixelAsUInt8   u)
        elif    t = fromType<int8>                    then box (img.GetPixelAsInt8    u)
        elif    t = fromType<uint16>                  then box (img.GetPixelAsUInt16  u)
        elif    t = fromType<int16>                   then box (img.GetPixelAsInt16   u)
        elif    t = fromType<uint32>                  then box (img.GetPixelAsUInt32  u)
        elif    t = fromType<int32>                   then box (img.GetPixelAsInt32   u)
        elif    t = fromType<uint64>                  then box (img.GetPixelAsUInt64  u)
        elif    t = fromType<int64>                   then box (img.GetPixelAsInt64   u)
        elif    t = fromType<float32>                 then box (img.GetPixelAsFloat   u)
        elif    t = fromType<float>                   then box (img.GetPixelAsDouble  u)
//        elif    t = fromType<System.Numerics.Complex>                 then
//            img.GetPixelAsVectorFloat64 u
//            |> fromVectorFloat64                    // float64 list → Complex
//            |> box
        elif    t = fromType<uint8   list>            then box (img.GetPixelAsVectorUInt8   u |> fromVectorUInt8)
        elif    t = fromType<int8    list>            then box (img.GetPixelAsVectorInt8    u |> fromVectorInt8)
        elif    t = fromType<uint16  list>            then box (img.GetPixelAsVectorUInt16  u |> fromVectorUInt16)
        elif    t = fromType<int16   list>            then box (img.GetPixelAsVectorInt16   u |> fromVectorInt16)
        elif    t = fromType<uint32  list>            then box (img.GetPixelAsVectorUInt32  u |> fromVectorUInt32)
        elif    t = fromType<int32   list>            then box (img.GetPixelAsVectorInt32   u |> fromVectorInt32)
        elif    t = fromType<uint64  list>            then box (img.GetPixelAsVectorUInt64  u |> fromVectorUInt64)
        elif    t = fromType<int64   list>            then box (img.GetPixelAsVectorInt64   u |> fromVectorInt64)
        elif    t = fromType<float32 list>            then box (img.GetPixelAsVectorFloat32 u |> fromVectorFloat32)
        elif    t = fromType<float   list>            then box (img.GetPixelAsVectorFloat64 u |> fromVectorFloat64)
        else
            failwithf "Unsupported pixel type: %O" t

    let getBoxedZero (t : itk.simple.PixelIDValueEnum) (vSize: uint option) : obj =

        let ncomp = match vSize with Some v -> int v | None -> 1
        if      t = fromType<uint8>                   then box 0uy
        elif    t = fromType<int8>                    then box 0y
        elif    t = fromType<uint16>                  then box 0us
        elif    t = fromType<int16>                   then box 0s
        elif    t = fromType<uint32>                  then box 0u
        elif    t = fromType<int32>                   then box 0
        elif    t = fromType<uint64>                  then box 0uL
        elif    t = fromType<int64>                   then box 0L
        elif    t = fromType<float32>                 then box 0.0f
        elif    t = fromType<float>                   then box 0.0
//        elif    t = fromType<System.Numerics.Complex>                 then
//            img.GetPixelAsVectorFloat64 u
//            |> fromVectorFloat64                    // float64 list → Complex
//            |> box
        elif    t = fromType<uint8   list>            then box (List.replicate ncomp 0uy)
        elif    t = fromType<int8    list>            then box (List.replicate ncomp 0u)
        elif    t = fromType<uint16  list>            then box (List.replicate ncomp 0us)
        elif    t = fromType<int16   list>            then box (List.replicate ncomp 0s)
        elif    t = fromType<uint32  list>            then box (List.replicate ncomp 0u)
        elif    t = fromType<int32   list>            then box (List.replicate ncomp 0)
        elif    t = fromType<uint64  list>            then box (List.replicate ncomp 0uL)
        elif    t = fromType<int64   list>            then box (List.replicate ncomp 0L)
        elif    t = fromType<float32 list>            then box (List.replicate ncomp 0.0f)
        elif    t = fromType<float   list>            then box (List.replicate ncomp 0.0)
        else
            failwithf "Unsupported pixel type: %O" t

    let inline mulAdd (t : itk.simple.PixelIDValueEnum) (acc : obj) (k : obj) (p : obj) : obj =
          //if      t = fromType<uint8>                   then box 0uy
        //elif    t = fromType<int8>                    then box 0y
        //elif    t = fromType<uint16>                  then box 0us
        //elif    t = fromType<int16>                   then box 0s
        //elif    t = fromType<uint32>                  then box 0u
        if      t = fromType<int32>                   then box ((unbox acc : int)     + (unbox k : int)     * (unbox p : int))
        //elif    t = fromType<uint64>                  then box 0uL
        elif    t = fromType<int64>                   then box ((unbox acc : int64)   + (unbox k : int64)   * (unbox p : int64))
        elif    t = fromType<float32>                 then box ((unbox acc : float32) + (unbox k : float32) * (unbox p : float32))
        elif    t = fromType<float>                   then box ((unbox acc : float)   + (unbox k : float)   * (unbox p : float))
        else failwithf "mulAdd: unsupported pixel type %A" id

open InternalHelpers
let getBytesPerComponent t =
    if t = typeof<uint8> then 1u
    elif t = typeof<int8> then 1u
    elif t = typeof<uint8 list> then 1u
    elif t = typeof<int8 list> then 1u
    elif t = typeof<uint16> then 2u
    elif t = typeof<int16> then 2u
    elif t = typeof<uint16 list> then 2u
    elif t = typeof<int16 list> then 2u
    elif t = typeof<uint32> then 4u
    elif t = typeof<int32> then 4u
    elif t = typeof<float32> then 4u
    elif t = typeof<uint32 list> then 4u
    elif t = typeof<int32 list> then 4u
    elif t = typeof<float32 list> then 4u
    elif t = typeof<uint64> then 8u
    elif t = typeof<int64> then 8u
    elif t = typeof<float> then 8u
    elif t = typeof<uint64 list> then 8u
    elif t = typeof<int64 list> then 8u
    elif t = typeof<float list> then 8u
    elif t = typeof<System.Numerics.Complex> then 16u
    else 8u // guessing here

let equalOne (v : 'T) : bool =
    match box v with
    | :? uint8   as b -> b = 1uy
    | :? int8    as b -> b = 1y
    | :? uint16  as b -> b = 1us
    | :? int16   as b -> b = 1s
    | :? uint32  as b -> b = 1u
    | :? int32   as b -> b = 1
    | :? uint64  as b -> b = 1uL
    | :? int64   as b -> b = 1L
    | :? float32 as b -> b = 1.0f
    | :? float   as b -> b = 1.0
    | _ -> failwithf "Don't know the value of 1 for %A" (typeof<'T>)

[<StructuredFormatDisplay("{Display}")>] // Prevent fsi printing information about its members such as img
type Image<'T when 'T : equality>(sz: uint list, ?numberComp: uint, ?name: string, ?index: uint) =
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

    static let mutable totalImages = 0
    do totalImages <- totalImages+1
    do printfn "Image count: %d" totalImages

    // 🧹 Decrement on disposal
    interface System.IDisposable with
        member _.Dispose() =
            totalImages <- totalImages - 1

    member this.Image = img
    member val Name = (Option.defaultValue "" name) with get
    member val index = (Option.defaultValue 0u index) with get, set
    member private this.SetImg (itkImg: itk.simple.Image) : unit =
        img <- itkImg
    member this.GetSize () = img.GetSize() |> fromVectorUInt32
    member this.GetDepth() = max 1u (img.GetDepth()) // Non-vector images returns 0
    member this.GetDimensions() = img.GetDimension()
    member this.GetHeight() = img.GetHeight()
    member this.GetWidth() = img.GetWidth()
    member this.GetNumberOfComponentsPerPixel() = img.GetNumberOfComponentsPerPixel()

    override this.ToString() = 
        let sz = this.GetSize()
        let szStr = List.fold (fun acc elm -> acc + $"x{elm}") (sz |> List.head |> string) (List.tail sz)
        let comp = this.GetNumberOfComponentsPerPixel()
        let vecStr = if comp = 1u then "Scalar" else sprintf $"{comp}-Vector "
        sprintf "%s %s<%s,%A>" szStr vecStr (typeof<'T>.Name) (img.GetPixelID())
    member this.Display = this.ToString() // related to [<StructuredFormatDisplay>]

    interface System.IEquatable<Image<'T>> with
        member this.Equals(other: Image<'T>) = Image<'T>.eq(this, other)

    interface System.IComparable with
        member this.CompareTo(obj: obj) =
            match obj with
            | :? Image<'T> as other -> this.CompareTo(other)
            | _ -> invalidArg "obj" "Expected Image<'T>"

    member this.memoryEstimate(): uint = // Intended to be mostly immutable, but better safe than sorry.
        let t = typeof<'T>
        let bytesPerComponent = getBytesPerComponent t
        bytesPerComponent * this.GetNumberOfComponentsPerPixel() * (this.GetSize() |> List.reduce (*));

    static member ofSimpleITK (itkImg: itk.simple.Image, ?name: string, ?index: uint) : Image<'T> =
        let itkImgCast = ofCastItk<'T> itkImg
        let img = new Image<'T>([0u;0u],itkImgCast.GetNumberOfComponentsPerPixel(),Option.defaultValue "" name,Option.defaultValue 0u index)
        img.SetImg itkImgCast
        img

    member this.toSimpleITK () : itk.simple.Image =
        img

    member this.castTo<'S when 'S: equality> () : Image<'S> = Image<'S>.ofSimpleITK img
    member this.toUInt8 ()   : Image<uint8>   = Image<uint8>.ofSimpleITK img
    member this.toInt8 ()    : Image<int8>    = Image<int8>.ofSimpleITK img
    member this.toUInt16 ()  : Image<uint16>  = Image<uint16>.ofSimpleITK img
    member this.toInt16 ()   : Image<int16>   = Image<int16>.ofSimpleITK img
    member this.toUInt ()    : Image<uint>    = Image<uint>.ofSimpleITK img
    member this.toInt ()     : Image<int>     = Image<int>.ofSimpleITK img
    member this.toUInt64 ()  : Image<uint64>  = Image<uint64>.ofSimpleITK img
    member this.toInt64 ()   : Image<int64>   = Image<int64>.ofSimpleITK img
    member this.toFloat32 () : Image<float32> = Image<float32>.ofSimpleITK img
    member this.toFloat ()   : Image<float>   = Image<float>.ofSimpleITK img

    static member ofArray2D (arr: 'T[,]) : Image<'T> =
        let sz = [arr.GetLength(0); arr.GetLength(1)] |> List.map uint
        let img = new Image<'T>(sz,1u)
        img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1]])
        img

    static member ofArray3D (arr: 'T[,,]) : Image<'T> =
        let sz = [arr.GetLength(0); arr.GetLength(1); arr.GetLength(2)] |> List.map uint
        let img = new Image<'T>(sz,1u)
        img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1],int idxLst[2]])
        img

    static member ofArray4D (arr: 'T[,,,]) : Image<'T> =
        let sz = [arr.GetLength(0); arr.GetLength(1); arr.GetLength(2); arr.GetLength(3)] |> List.map uint
        let img = new Image<'T>(sz,1u)
        img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1],int idxLst[2],int idxLst[3]])
        img

    member this.toArray2D (): 'T[,] =
        let sz = this.GetSize() |> List.map int
        Array2D.init sz[0] sz[1] (fun i0 i1 -> this.Get([uint i0; uint i1]))

    member this.toArray3D (): 'T[,,] =
        let sz = this.GetSize() |> List.map int
        Array3D.init sz[0] sz[1] sz[2] (fun i0 i1 i2 -> this.Get([uint i0; uint i1; uint i2]))

    member this.toArray4D (): 'T[,,,] =
        let sz = this.GetSize() |> List.map int
        Array4D.init sz[0] sz[1] sz[2] sz[3] (fun i0 i1 i2 i3 -> this.Get([uint i0; uint i1; uint i2; uint i3]))

    static member ofImageList (images: Image<'S> list) : Image<'S list> =
        let itkImages = images |> List.map (fun img -> img.toSimpleITK())
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

    static member ofFile(filename: string, ?name: string, ?index: uint) : Image<'T> =
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
            Image<'T>.ofSimpleITK(itkImg,Option.defaultValue "" name,Option.defaultValue 0u index)

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
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member (-) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member ( * ) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member (/) (f1: Image<'T>, f2: Image<'T>) =
        let filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))

    // Collection type
    static member map (f:'T->'T) (im1: Image<'T>) : Image<'T> =
        let sz = im1.GetSize()
        let comp = im1.GetNumberOfComponentsPerPixel()
        let im = new Image<'T>(sz,comp)
        sz
        |> flatIndices
        |> Seq.iter (fun idx -> im1.Get idx |> f |> (im.Set idx))
        im

    static member mapi (f:uint list->'T->'T) (im1: Image<'T>) : Image<'T> =
        let sz = im1.GetSize()
        let comp = im1.GetNumberOfComponentsPerPixel()
        let im = new Image<'T>(sz,comp)
        sz
        |> flatIndices
        |> Seq.iter (fun idx -> im1.Get idx |> f idx |> (im.Set idx))
        im

    static member iter (f:'T->unit) (im1: Image<'T>) : unit = 
        let sz = im1.GetSize()
        sz
        |> flatIndices
        |> Seq.iter (fun idx -> im1.Get idx |> f)

    static member iteri (f:uint list->'T->unit) (im1: Image<'T>) : unit = 
        let sz = im1.GetSize()
        sz
        |> flatIndices
        |> Seq.iter (fun idx -> im1.Get idx |> f idx)

    static member fold (f:'S->'T->'S) (acc0: 'S) (im1: Image<'T>) : 'S = 
        let sz = im1.GetSize()
        sz
        |> flatIndices
        |> Seq.fold (fun acc idx -> im1.Get idx |> f acc) acc0

    static member foldi (f:uint list->'S->'T->'S) (acc0: 'S) (im1: Image<'T>) : 'S =
        let sz = im1.GetSize()
        sz
        |> flatIndices
        |> Seq.fold (fun acc idx -> im1.Get idx |> f idx acc) acc0

    static member zip (imLst: Image<'T> list) : Image<'T list> =
        let nComp = imLst.Length
        if nComp < 2 then 
            failwith "can't zip list of less than 2 elements"
        else
            let sz = imLst[0].GetSize()
            let result = new Image<'T list>(sz,uint nComp)
            sz
            |> flatIndices
            |> Seq.iter (fun idxLst -> 
                List.map (fun (im: Image<'T>) -> im.Get idxLst) imLst
                |> result.Set idxLst)
            result

    static member unzip (im: Image<'T list>) : Image<'T> list =
        let sz = im.GetSize()
        let comp = im.GetNumberOfComponentsPerPixel()
        let imLst = List.init (int comp) (fun i -> new Image<'T>(sz,1u))
        im |> Image.iteri (fun idxLst vLst ->
            List.iteri (fun i v -> imLst[i].Set idxLst v) vLst )
        imLst

    member this.Get (coords: uint list) : 'T =
        let u = coords |> toVectorUInt32
        let t = fromType<'T>
        let raw = getBoxedPixel this.Image t u
        raw :?> 'T

    member this.Set (coords: uint list) (value: 'T) : unit =
        let u = toVectorUInt32 coords
        let t = fromType<'T>
        setBoxedPixel this.Image t u value

    member this.Item
        with get(i0: int, i1: int) : 'T =
            this.Get([ uint i0; uint i1 ])
        and set(i0: int, i1: int) (value: 'T) : unit =
            this.Set [ uint i0; uint i1 ] value
    member this.Item
        with get(i0: int, i1: int, i2: int) : 'T =
            this.Get([ uint i0; uint i1; uint i2 ])
        and set(i0: int, i1: int, i2: int) (value: 'T) : unit =
            this.Set [ uint i0; uint i1; uint i2 ] value
    member this.Item
        with get(i0: int, i1: int, i2: int, i3: int) : 'T =
            this.Get([ uint i0; uint i1; uint i2; uint i3 ])
        and set(i0: int, i1: int, i2: int, i3: int) (value: 'T) : unit =
            this.Set [ uint i0; uint i1; uint i2; uint i3 ] value

    // Slicing is available as https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/arrays


    member this.forAll (p: 'T -> bool) : bool =
        this |> Image.fold (fun acc elm -> acc && p elm) true

    override this.Equals(obj: obj) = // For some reason, it's not enough with this.Equals(other: Image<'T>)
        match obj with
        | :? Image<'T> as other -> (this :> System.IEquatable<_>).Equals(other)
        | _ -> false

    override this.GetHashCode() =
        let dim = this.GetDimensions()
        if dim = 2u then
            hash (this.toArray2D())
        elif dim = 3u then
            hash (this.toArray3D())
        elif dim = 4u then
            hash (this.toArray4D())
        else
            failwith "No hashcode defined for images with dimensions less than 2 or greater than 4"

    member this.CompareTo(other: Image<'T>) =
        let diff : Image<'T> = this - other
        let pixels = diff.toArray2D() |> Seq.cast<obj>

        let sum =
            pixels
            |> Seq.sumBy System.Convert.ToDouble

        if sum < 0.0 then -1
        elif sum > 0.0 then 1
        else 0

    /// Comparison operators
    static member isEqual (f1: Image<'S>, f2: Image<'S>) = // Curried form confuses fsharp
        let filter = new itk.simple.EqualImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member eq (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isEqual(f1, f2)).forAll equalOne

    static member isNotEqual (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.NotEqualImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member neq (f1: Image<'S>, f2: Image<'S>) =
        (Image<float>.isNotEqual(f1, f2)).forAll equalOne

    static member isLessThan (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.LessImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member lt (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isLessThan(f1, f2)).forAll equalOne

    static member isLessThanEqual (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.LessEqualImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member lte (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isLessThanEqual(f1, f2)).forAll equalOne

    static member isGreater (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.GreaterImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member gt (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isGreater(f1, f2)).forAll equalOne

    // greater than or equal
    static member isGreaterEqual (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.GreaterEqualImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))
    static member gte (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isGreaterEqual(f1, f2)).forAll equalOne

    // Power (no direct operator for ** in .NET) - provide a named method instead
    static member Pow (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.PowImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))

    // Bitwise AND ( &&& )
    static member op_BitwiseAnd (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.AndImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))

    // Bitwise XOR ( ^^^ )
    static member op_ExclusiveOr (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.XorImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))

    // Bitwise OR ( ||| )
    static member op_BitwiseOr (f1: Image<'S>, f2: Image<'S>) =
        let filter = new itk.simple.OrImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()))

    // Unary bitwise NOT ( ~~~ )
    static member op_LogicalNot (f: Image<'S>) =
        let filter = new itk.simple.InvertIntensityImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f.toSimpleITK()))
