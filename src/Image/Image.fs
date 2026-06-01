module Image
open FSharp.Collections
open System
open System.Buffers
open System.Runtime.InteropServices

[<Struct>]
type ComplexFloat32 =
    val Real: float32
    val Imaginary: float32
    new(real: float32, imaginary: float32) = { Real = real; Imaginary = imaginary }
    static member Zero = ComplexFloat32(0.0f, 0.0f)

module InternalHelpers = // internal
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
    let toComplexFloat32 (lst: float32 list) =
        match lst with
        | [re; im] -> ComplexFloat32(re, im)
        | [re] -> ComplexFloat32(re, 0.0f)
        | [] -> ComplexFloat32.Zero
        | _ -> invalidArg "lst" "Expected 0, 1, or 2 elements for complex value."
    let toComplexFloat64 (lst: float list) =
        match lst with
        | [re; im] -> System.Numerics.Complex(re, im)
        | [re] -> System.Numerics.Complex(re, 0.0)
        | [] -> System.Numerics.Complex(0.0, 0.0)
        | _ -> invalidArg "lst" "Expected 0, 1, or 2 elements for complex value."

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
        elif t = typeof<ComplexFloat32> then itk.simple.PixelIDValueEnum.sitkComplexFloat32
        elif t = typeof<System.Numerics.Complex> then itk.simple.PixelIDValueEnum.sitkComplexFloat64
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

    let isComplexPixelId (pid: itk.simple.PixelIDValueEnum) =
        pid = itk.simple.PixelIDValueEnum.sitkComplexFloat32
        || pid = itk.simple.PixelIDValueEnum.sitkComplexFloat64

    let isComplexCompatibleImage (itkImg: itk.simple.Image) =
        isComplexPixelId (itkImg.GetPixelID())

    let isScalarImportSupported<'T> =
        let t = typeof<'T>
        t = typeof<uint8>
        || t = typeof<int8>
        || t = typeof<uint16>
        || t = typeof<int16>
        || t = typeof<uint32>
        || t = typeof<int32>
        || t = typeof<uint64>
        || t = typeof<int64>
        || t = typeof<float32>
        || t = typeof<float>

    let setImportBuffer<'T> (importer: itk.simple.ImportImageFilter) (buffer: nativeint) =
        let t = typeof<'T>
        if t = typeof<uint8> then importer.SetBufferAsUInt8(buffer)
        elif t = typeof<int8> then importer.SetBufferAsInt8(buffer)
        elif t = typeof<uint16> then importer.SetBufferAsUInt16(buffer)
        elif t = typeof<int16> then importer.SetBufferAsInt16(buffer)
        elif t = typeof<uint32> then importer.SetBufferAsUInt32(buffer)
        elif t = typeof<int32> then importer.SetBufferAsInt32(buffer)
        elif t = typeof<uint64> then importer.SetBufferAsUInt64(buffer)
        elif t = typeof<int64> then importer.SetBufferAsInt64(buffer)
        elif t = typeof<float32> then importer.SetBufferAsFloat(buffer)
        elif t = typeof<float> then importer.SetBufferAsDouble(buffer)
        else invalidArg "T" $"Unsupported scalar import pixel type: {t.Name}"

    let scalarComponentByteSize<'T> =
        let t = typeof<'T>
        if t = typeof<uint8> || t = typeof<int8> then 1
        elif t = typeof<uint16> || t = typeof<int16> then 2
        elif t = typeof<uint32> || t = typeof<int32> || t = typeof<float32> then 4
        elif t = typeof<uint64> || t = typeof<int64> || t = typeof<float> then 8
        else invalidArg "T" $"Unsupported scalar buffer pixel type: {t.Name}"

    let getConstBuffer<'T> (image: itk.simple.Image) =
        let t = typeof<'T>
        if t = typeof<uint8> then image.GetConstBufferAsUInt8()
        elif t = typeof<int8> then image.GetConstBufferAsInt8()
        elif t = typeof<uint16> then image.GetConstBufferAsUInt16()
        elif t = typeof<int16> then image.GetConstBufferAsInt16()
        elif t = typeof<uint32> then image.GetConstBufferAsUInt32()
        elif t = typeof<int32> then image.GetConstBufferAsInt32()
        elif t = typeof<uint64> then image.GetConstBufferAsUInt64()
        elif t = typeof<int64> then image.GetConstBufferAsInt64()
        elif t = typeof<float32> then image.GetConstBufferAsFloat()
        elif t = typeof<float> then image.GetConstBufferAsDouble()
        else invalidArg "T" $"Unsupported scalar buffer pixel type: {t.Name}"

    let copyScalarPixels<'T> (image: itk.simple.Image) pixelCount =
        if image.GetPixelID() <> fromType<'T> then
            invalidArg "image" $"Expected {fromType<'T>} image buffer, got {image.GetPixelID()}."
        if image.GetNumberOfComponentsPerPixel() <> 1u then
            invalidArg "image" $"Expected a scalar image buffer, got {image.GetNumberOfComponentsPerPixel()} components per pixel."

        let byteCount = pixelCount * scalarComponentByteSize<'T>
        let bytes = Array.zeroCreate<byte> byteCount
        Marshal.Copy(getConstBuffer<'T> image, bytes, 0, byteCount)
        let pixels = Array.zeroCreate<'T> pixelCount
        Buffer.BlockCopy(bytes, 0, pixels, 0, byteCount)
        pixels

    let importScalarImage<'T> (size: uint list) (pixels: 'T[]) =
        use importer = new itk.simple.ImportImageFilter()
        importer.SetSize(size |> toVectorUInt32)

        let handle = GCHandle.Alloc(pixels, GCHandleType.Pinned)
        try
            setImportBuffer<'T> importer (handle.AddrOfPinnedObject())
            use imported = importer.Execute()
            use cast = new itk.simple.CastImageFilter()
            cast.SetOutputPixelType(fromType<'T>)
            cast.Execute(imported)
        finally
            handle.Free()

    let private deepCopyItkImage (itkImg: itk.simple.Image) : itk.simple.Image =
        let copy = new itk.simple.Image(itkImg)
        copy.MakeUnique()
        copy

    /// <summary>
    /// Creates a shallow SimpleITK image wrapper for an image whose pixel type already matches <typeparamref name="'T" />.
    /// The returned SimpleITK image shares the same pixel container until SimpleITK copy-on-write forces uniqueness.
    /// No cast, deep copy, or disposal of <paramref name="itkImg" /> is performed.
    /// </summary>
    let aliasSimpleITKImage<'T> (itkImg: itk.simple.Image) : itk.simple.Image =
        let expectedId = fromType<'T>
        if itkImg.GetPixelID() = expectedId then
            new itk.simple.Image(itkImg)
        else
            invalidArg "itkImg" $"Expected {expectedId} image for aliasing, got {itkImg.GetPixelID()}."

    /// <summary>
    /// Creates an independent SimpleITK image with pixel type <typeparamref name="'T" />.
    /// If <paramref name="itkImg" /> already has the requested pixel type, a shallow copy is first made and then
    /// <c>MakeUnique</c> is called to force a deep pixel-buffer copy. If the pixel type differs, SimpleITK's cast filter
    /// is used, which allocates a new output image. The argument is not disposed.
    /// </summary>
    let ofCastITK<'T> (itkImg: itk.simple.Image) : itk.simple.Image =
        let expectedId = fromType<'T>
        if itkImg.GetPixelID() = expectedId then
            deepCopyItkImage itkImg
        else
            use cast = new itk.simple.CastImageFilter()
            cast.SetOutputPixelType(expectedId)
            cast.Execute(itkImg)

    let private identityDirection dim =
        [ for row in 0 .. dim - 1 do
            for col in 0 .. dim - 1 do
                if row = col then 1.0 else 0.0 ]

    let canonicalizeSimpleItkImage (image: itk.simple.Image) =
        let dim = int (image.GetDimension())
        image.SetSpacing(List.replicate dim 1.0 |> toVectorFloat64)
        image.SetOrigin(List.replicate dim 0.0 |> toVectorFloat64)
        image.SetDirection(identityDirection dim |> toVectorFloat64)
        image.GetMetaDataKeys()
        |> Seq.toArray
        |> Array.iter (fun key -> image.EraseMetaData(key) |> ignore)
        image

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
            | _ -> failwith $"Unsupported dimensionality {size.Length}"

    let setBoxedPixel (sitkImg: itk.simple.Image) (t: itk.simple.PixelIDValueEnum) (u: itk.simple.VectorUInt32) (value: obj) : unit =
        match value with
        | :? (uint8 list) as v -> sitkImg.SetPixelAsVectorUInt8(u, toVectorUInt8 v)
        | :? (int8 list) as v -> sitkImg.SetPixelAsVectorInt8(u, toVectorInt8 v)
        | :? (uint16 list) as v -> sitkImg.SetPixelAsVectorUInt16(u, toVectorUInt16 v)
        | :? (int16 list) as v -> sitkImg.SetPixelAsVectorInt16(u, toVectorInt16 v)
        | :? (uint32 list) as v -> sitkImg.SetPixelAsVectorUInt32(u, toVectorUInt32 v)
        | :? (int32 list) as v -> sitkImg.SetPixelAsVectorInt32(u, toVectorInt32 v)
        | :? (uint64 list) as v -> sitkImg.SetPixelAsVectorUInt64(u, toVectorUInt64 v)
        | :? (int64 list) as v -> sitkImg.SetPixelAsVectorInt64(u, toVectorInt64 v)
        | :? (float32 list) as v -> sitkImg.SetPixelAsVectorFloat32(u, toVectorFloat32 v)
        | :? (float list) as v -> sitkImg.SetPixelAsVectorFloat64(u, toVectorFloat64 v)
        | _ ->
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
        elif    t = fromType<ComplexFloat32> || t = fromType<System.Numerics.Complex> then
            let pid = img.GetPixelID()
            if not (isComplexPixelId pid) then
                failwithf "Unsupported complex backing pixel type: %O" pid
            use realFilter = new itk.simple.ComplexToRealImageFilter()
            use imagFilter = new itk.simple.ComplexToImaginaryImageFilter()
            let getComponent (componentImg: itk.simple.Image) =
                let componentPid = componentImg.GetPixelID()
                if componentPid = itk.simple.PixelIDValueEnum.sitkFloat32 then
                    componentImg.GetPixelAsFloat u |> float
                elif componentPid = itk.simple.PixelIDValueEnum.sitkFloat64 then
                    componentImg.GetPixelAsDouble u
                else
                    failwithf "Unsupported real/imaginary complex component type: %O" componentPid
            use realComponent = realFilter.Execute(img)
            use imagComponent = imagFilter.Execute(img)
            let re = getComponent realComponent
            let im = getComponent imagComponent
            if t = fromType<ComplexFloat32> then
                ComplexFloat32(float32 re, float32 im) |> box
            else
                System.Numerics.Complex(re, im) |> box
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
        elif    t = fromType<ComplexFloat32>          then box ComplexFloat32.Zero
        elif    t = fromType<System.Numerics.Complex> then box (System.Numerics.Complex(0.0, 0.0))
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

    let getFloatPixel (img: itk.simple.Image) (u: itk.simple.VectorUInt32) =
        let pid = img.GetPixelID()
        if pid = itk.simple.PixelIDValueEnum.sitkFloat32 then
            img.GetPixelAsFloat u |> float
        elif pid = itk.simple.PixelIDValueEnum.sitkFloat64 then
            img.GetPixelAsDouble u
        else
            failwithf "Unsupported real/imaginary complex component type: %O" pid

    let setFloatPixel (img: itk.simple.Image) (u: itk.simple.VectorUInt32) (value: float) =
        let pid = img.GetPixelID()
        if pid = itk.simple.PixelIDValueEnum.sitkFloat32 then
            img.SetPixelAsFloat(u, float32 value)
        elif pid = itk.simple.PixelIDValueEnum.sitkFloat64 then
            img.SetPixelAsDouble(u, value)
        else
            failwithf "Unsupported real/imaginary complex component type: %O" pid

    let private ensureNativeComplex (name: string) (img: itk.simple.Image) =
        if not (isComplexPixelId (img.GetPixelID())) then
            invalidArg "img" $"%s{name}: expected a native complex image, got %O{img.GetPixelID()}."

    let extractComplexRealImage img =
        ensureNativeComplex "Re" img
        use filter = new itk.simple.ComplexToRealImageFilter()
        filter.Execute(img)

    let extractComplexImagImage img =
        ensureNativeComplex "Im" img
        use filter = new itk.simple.ComplexToImaginaryImageFilter()
        filter.Execute(img)

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
        elif    t = fromType<ComplexFloat32>          then
            let acc = unbox<ComplexFloat32> acc
            let k = unbox<ComplexFloat32> k
            let p = unbox<ComplexFloat32> p
            box (ComplexFloat32(acc.Real + k.Real * p.Real - k.Imaginary * p.Imaginary, acc.Imaginary + k.Real * p.Imaginary + k.Imaginary * p.Real))
        elif    t = fromType<System.Numerics.Complex> then box ((unbox acc : System.Numerics.Complex) + (unbox k : System.Numerics.Complex) * (unbox p : System.Numerics.Complex))
        else failwithf "mulAdd: unsupported pixel type %A" t

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
    elif t = typeof<ComplexFloat32> then 8u
    elif t = typeof<System.Numerics.Complex> then 16u
    else 8u // guessing here

let getBytesPerSItkComponent t =
    if   t = itk.simple.PixelIDValueEnum.sitkUInt8 then 1u
    elif t = itk.simple.PixelIDValueEnum.sitkInt8 then 1u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorUInt8 then 1u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorInt8 then 1u
    elif t = itk.simple.PixelIDValueEnum.sitkUInt16 then 2u
    elif t = itk.simple.PixelIDValueEnum.sitkInt16 then 2u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorUInt16 then 2u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorInt16 then 2u
    elif t = itk.simple.PixelIDValueEnum.sitkUInt32 then 4u
    elif t = itk.simple.PixelIDValueEnum.sitkInt32 then 4u
    elif t = itk.simple.PixelIDValueEnum.sitkFloat32 then 4u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorUInt32 then 4u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorInt32 then 4u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorFloat32 then 4u
    elif t = itk.simple.PixelIDValueEnum.sitkComplexFloat32 then 8u
    elif t = itk.simple.PixelIDValueEnum.sitkUInt64 then 8u
    elif t = itk.simple.PixelIDValueEnum.sitkInt64 then 8u
    elif t = itk.simple.PixelIDValueEnum.sitkFloat64 then 8u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorUInt64 then 8u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorInt64 then 8u
    elif t = itk.simple.PixelIDValueEnum.sitkVectorFloat64 then 8u
    elif t = itk.simple.PixelIDValueEnum.sitkComplexFloat64 then 16u
    elif t = itk.simple.PixelIDValueEnum.sitkLabelUInt8 then 1u
    elif t = itk.simple.PixelIDValueEnum.sitkLabelUInt16 then 2u
    elif t = itk.simple.PixelIDValueEnum.sitkLabelUInt32 then 4u
    elif t = itk.simple.PixelIDValueEnum.sitkLabelUInt64 then 8u
    else 8u // guessing here

type ImageFacts =
    { Backend: string
      PixelType: string
      ComponentBytes: uint64
      ComponentsPerPixel: uint64
      Size: uint64 list
      VoxelCount: uint64
      MemoryBytes: uint64 }

module ImageFacts =
    let private product values =
        values |> List.fold (fun acc value -> acc * value) 1UL

    let create backend pixelType componentBytes componentsPerPixel size =
        let voxelCount = product size
        { Backend = backend
          PixelType = pixelType
          ComponentBytes = componentBytes
          ComponentsPerPixel = componentsPerPixel
          Size = size
          VoxelCount = voxelCount
          MemoryBytes = voxelCount * componentBytes * componentsPerPixel }

    let forType<'T> size componentsPerPixel =
        create
            "generic"
            typeof<'T>.Name
            (getBytesPerComponent typeof<'T> |> uint64)
            (uint64 componentsPerPixel)
            (size |> List.map uint64)

    let ofSimpleITK (sitk: itk.simple.Image) =
        create
            "SimpleITK"
            (pixelIdToString (sitk.GetPixelID()))
            (getBytesPerSItkComponent (sitk.GetPixelID()) |> uint64)
            (uint64 (sitk.GetNumberOfComponentsPerPixel()))
            (sitk.GetSize() |> fromVectorUInt32 |> List.map uint64)

    let memoryBytesForType<'T> (nVoxels: uint64) componentsPerPixel =
        (forType<'T> [uint nVoxels] componentsPerPixel).MemoryBytes

    let sliceBytesForType<'T> width height =
        (forType<'T> [width; height] 1u).MemoryBytes
    

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
    | :? System.Numerics.Complex as c -> c.Real = 1.0 && c.Imaginary = 0.0
    | _ -> failwithf "Don't know the value of 1 for %A" (typeof<'T>)

let private syncRoot = obj() // The following 3 names may be accessed in parallel, so lock when writing
let mutable private totalImages = 0 // count how many images with references > 0, must be outside to be shared by all Image<*>
let mutable private peakTotalImages = 0 // count how many images with references > 0, must be outside to be shared by all Image<*>
let incTotalImages () =
    lock syncRoot (fun () -> totalImages <- totalImages + 1)
    lock syncRoot (fun () -> peakTotalImages <- max peakTotalImages totalImages)
let decTotalImages () =
    lock syncRoot (fun () -> totalImages <- totalImages - 1)

let mutable private memUsed = 0u 
let mutable private peakMemUsed = 0u 
let private incMemUsed mem =
    lock syncRoot (fun () -> memUsed <- memUsed + mem)
    lock syncRoot (fun () -> peakMemUsed <- max peakMemUsed memUsed)
let private decMemUsed mem =
    lock syncRoot (fun () -> memUsed <- memUsed - mem)

let private currentRssBytes () =
    let p = System.Diagnostics.Process.GetCurrentProcess()
    p.Refresh()
    uint64 p.WorkingSet64

let mutable private rssBaselineBytes = 0UL
let mutable private peakRssDeltaBytes = 0UL
let mutable private debugLevel = 0u

let private resetRssProbe () =
    let current = currentRssBytes()
    lock syncRoot (fun () ->
        rssBaselineBytes <- current
        peakRssDeltaBytes <- 0UL)

let private rssDeltaBytes () =
    let current = currentRssBytes()
    if current > rssBaselineBytes then current - rssBaselineBytes else 0UL

let private sampleRssDeltaBytes () =
    let delta = rssDeltaBytes()
    peakRssDeltaBytes <- max peakRssDeltaBytes delta
    delta, peakRssDeltaBytes

let private printDebugMessage str =
    lock syncRoot (fun () ->
       if debugLevel >= 2u then
           let rssDelta, rssPeakDelta = sampleRssDeltaBytes()
           printfn "%8d KB / %8d KB RSS %8d KB / %8d KB %3d / %3d Images %s" (memUsed/1024u) (peakMemUsed/1024u) (rssDelta/1024UL) (rssPeakDelta/1024UL) totalImages peakTotalImages str
       else
           printfn "%8d KB / %8d KB %3d / %3d Images %s" (memUsed/1024u) (peakMemUsed/1024u) totalImages peakTotalImages str) (*(String.replicate totalImages "*")*)
let mutable private debug = false

type PooledImageDebugCounters =
    { Rents: int64
      Returns: int64
      Live: int64
      PeakLive: int64 }

let mutable private pooledRents = 0L
let mutable private pooledReturns = 0L
let mutable private pooledLive = 0L
let mutable private pooledPeakLive = 0L

let private notePooledRent () =
    lock syncRoot (fun () ->
        pooledRents <- pooledRents + 1L
        pooledLive <- pooledLive + 1L
        pooledPeakLive <- max pooledPeakLive pooledLive)

let private notePooledReturn () =
    lock syncRoot (fun () ->
        pooledReturns <- pooledReturns + 1L
        pooledLive <- pooledLive - 1L)

let resetPooledImageDebugCounters () =
    lock syncRoot (fun () ->
        pooledRents <- 0L
        pooledReturns <- 0L
        pooledLive <- 0L
        pooledPeakLive <- 0L)

let getPooledImageDebugCounters () =
    lock syncRoot (fun () ->
        { Rents = pooledRents
          Returns = pooledReturns
          Live = pooledLive
          PeakLive = pooledPeakLive })

let private poisonPooledBuffersOnReturn () =
    match Environment.GetEnvironmentVariable("STACKPROCESSING_ARRAYPOOL_POISON_ON_RETURN") with
    | null -> false
    | value -> value.Equals("1", StringComparison.OrdinalIgnoreCase) || value.Equals("true", StringComparison.OrdinalIgnoreCase)

let private poisonPooledBuffer<'T> logicalLength (buffer: 'T[]) =
    if poisonPooledBuffersOnReturn() then
        match box buffer with
        | :? (uint8[]) as typed -> Array.Fill(typed, 0xCCuy, 0, logicalLength)
        | :? (int8[]) as typed -> Array.Fill(typed, -52y, 0, logicalLength)
        | :? (uint16[]) as typed -> Array.Fill(typed, 0xCCCCus, 0, logicalLength)
        | :? (int16[]) as typed -> Array.Fill(typed, -13108s, 0, logicalLength)
        | :? (uint32[]) as typed -> Array.Fill(typed, 0xCCCCCCCCu, 0, logicalLength)
        | :? (int32[]) as typed -> Array.Fill(typed, -858993460, 0, logicalLength)
        | :? (uint64[]) as typed -> Array.Fill(typed, 0xCCCCCCCCCCCCCCCCUL, 0, logicalLength)
        | :? (int64[]) as typed -> Array.Fill(typed, -3689348814741910324L, 0, logicalLength)
        | :? (float32[]) as typed -> Array.Fill(typed, Single.NaN, 0, logicalLength)
        | :? (float[]) as typed -> Array.Fill(typed, Double.NaN, 0, logicalLength)
        | _ -> ()

let private pooledMemoryEstimate<'T> logicalLength =
    let bytes = uint64 logicalLength * uint64 (scalarComponentByteSize<'T>)
    if bytes > uint64 System.UInt32.MaxValue then
        System.UInt32.MaxValue
    else
        uint32 bytes

type private PooledBufferOwner<'T>(buffer: 'T[], logicalLength: int) =
    let mutable refCount = 1
    let mutable returned = false
    do
        incMemUsed (pooledMemoryEstimate<'T> logicalLength)
        notePooledRent()

    member _.Buffer = buffer
    member _.LogicalLength = logicalLength
    member _.AddRef() =
        lock syncRoot (fun () ->
            if returned then invalidOp "Cannot retain a pooled buffer after it has been returned."
            refCount <- refCount + 1)
    member _.Release() =
        let shouldReturn =
            lock syncRoot (fun () ->
                if returned then
                    false
                else
                    refCount <- refCount - 1
                    if refCount < 0 then invalidOp "Pooled buffer owner reference count became negative."
                    if refCount = 0 then
                        returned <- true
                        true
                    else
                        false)
        if shouldReturn then
            decMemUsed (pooledMemoryEstimate<'T> logicalLength)
            poisonPooledBuffer logicalLength buffer
            ArrayPool<'T>.Shared.Return(buffer)
            notePooledReturn()

[<StructuredFormatDisplay("{Display}")>] // Prevent fsi printing information about its members such as img
type Image<'T when 'T : equality>(sz: uint list, ?optionalNumberComponents: uint, ?optionalName: string, ?optionalIndex: int, ?optionalQuiet: bool) =
    do if sz.Length > 3 then invalidArg "sz" $"Image supports at most 3 dimensions; got {sz.Length}."
    let isComplexType = typeof<'T> = typeof<System.Numerics.Complex> || typeof<'T> = typeof<ComplexFloat32>
    let numberCompDefault = 1u
    let numberComp = defaultArg optionalNumberComponents numberCompDefault
    do if isComplexType && numberComp <> 1u then invalidArg "optionalNumberComponents" "Complex pixel type requires 1 native component."
    let now = System.DateTime.UtcNow.ToString("HH:mm:ss.ffffff")
    let name = 
        match optionalName with
        | Some str -> $"{str} {now}"
        | _ -> now
    let idx = defaultArg optionalIndex 0 
    let quiet = defaultArg optionalQuiet false

    let isListType = typeof<'T>.IsGenericType && typeof<'T>.GetGenericTypeDefinition() = typedefof<list<_>>
    let initialComponents = if isListType then max 2u numberComp else numberComp
    let initialLength =
        sz
        |> List.fold (fun acc value -> acc * int64 value) 1L
        |> fun voxels -> voxels * int64 initialComponents
    do if initialLength > int64 Int32.MaxValue then invalidArg "sz" $"Image buffer is too large for a single managed array: {initialLength} elements."
    let initialBuffer = ArrayPool<'T>.Shared.Rent(int initialLength)
    do if initialLength > 0L then Array.Clear(initialBuffer, 0, int initialLength)
    let mutable owner = PooledBufferOwner(initialBuffer, int initialLength)
    let mutable pooledSize = sz
    let mutable pooledComponents = initialComponents
    // count how many references there is to this image.

    let clampStartStop (img: Image<'T>) start0 stop0 start1 stop1 start2 stop2 = 
        let x0 = [start0; start1; start2] |> List.map (Option.defaultValue 0)
        let x1 =
            (img.GetSize(), [stop0; stop1; stop2]) 
            ||> List.zip 
            |> List.map (fun (sz, v) -> (Option.defaultValue (int sz)) v) 
        x0,x1

    let mutable nReferences = 1 
 
    do incTotalImages()
    do if debug && not quiet then printDebugMessage $"Created {name} ({pooledSize}, {typeof<'T>.Name}, {pooledComponents}->{Image<'T>.pooledMemoryEstimate owner.LogicalLength})"
    let now = System.DateTime.UtcNow.ToString("HH:mm:ss.ffffff'Z'")

    static member setDebugLevel level =
        debugLevel <- level
        if level >= 2u then resetRssProbe()
        debug <- level > 0u
    static member setDebug d =
        Image<'T>.setDebugLevel(if d then 1u else 0u)
    member this.Name = name
    member val index = idx with get, set
    member private this.SetPooled1D (owner: PooledBufferOwner<'T>, size: uint list, components: uint) : unit =
        if owner.LogicalLength < 0 then invalidArg "owner" "Logical length must be non-negative."
        if components <> 1u then invalidArg "components" "Pooled image storage currently supports scalar images only."
        this.ReleasePooledOwner()
        this.SetPooledOwner(owner, size, components)
    member private this.SetPooledOwner(newOwner: PooledBufferOwner<'T>, size: uint list, components: uint) : unit =
        owner <- newOwner
        pooledSize <- size
        pooledComponents <- components
    member private this.ReleasePooledOwner() =
        owner.Release()
    // add a use of this image
    member this.getNReferences() = nReferences
    member this.incRefCount() = 
        if debug then printDebugMessage $"Increased reference to {this.Name}"
        nReferences<-nReferences+1
    // release a use of this image
    member this.decRefCount() =
        if debug then printDebugMessage $"Decreased reference to {this.Name}"
        nReferences<-nReferences-1
        if nReferences = 0 then
            decTotalImages()
            owner.Release()
            if debug then printDebugMessage $"Disposed of {this.Name}"

    static member memoryEstimateSItk (sitk : itk.simple.Image) = 
        let facts = ImageFacts.ofSimpleITK sitk
        if facts.MemoryBytes > uint64 System.UInt32.MaxValue then
            System.UInt32.MaxValue
        else
            uint32 facts.MemoryBytes

    static member private pooledMemoryEstimate logicalLength =
        pooledMemoryEstimate<'T> logicalLength

    static member memoryEstimate (width: uint) (height: uint) =
        ImageFacts.sliceBytesForType<'T> width height

    member this.GetSize () = pooledSize
    member this.GetFacts () =
        ImageFacts.create "ArrayPool" typeof<'T>.Name (uint64 (scalarComponentByteSize<'T>)) (uint64 pooledComponents) (pooledSize |> List.map uint64)
    member this.GetMemoryBytes () = this.GetFacts().MemoryBytes
    member this.GetDepth() =
        match this.GetSize() with
        | _ :: _ :: depth :: _ -> depth
        | _ -> 1u
    member this.GetDimensions() = uint32 (this.GetSize().Length)
    member this.GetHeight() =
        match this.GetSize() with
        | _ :: height :: _ -> height
        | _ -> 0u
    member this.GetWidth() =
        match this.GetSize() with
        | width :: _ -> width
        | _ -> 0u
    member this.GetNumberOfComponentsPerPixel() = pooledComponents
    member internal this.TryGetPooled1D () =
        Some(owner.Buffer, owner.LogicalLength, pooledSize, pooledComponents)
    member this.TryMapPooled1D<'U when 'U : equality> (name: string, f: 'T -> 'U) : Image<'U> option =
        match this.TryGetPooled1D() with
        | Some(input, logicalLength, size, 1u) ->
            let output = ArrayPool<'U>.Shared.Rent(logicalLength)
            try
                for i in 0 .. logicalLength - 1 do
                    output[i] <- f input[i]
                Some(Image<'U>.ofPooled1D(output, logicalLength, size, name, this.index))
            with
            | _ ->
                poisonPooledBuffer logicalLength output
                ArrayPool<'U>.Shared.Return(output)
                reraise()
        | _ -> None
    member this.TryMap2Pooled1D<'U, 'V when 'U : equality and 'V : equality> (other: Image<'U>, name: string, f: 'T -> 'U -> 'V) : Image<'V> option =
        match this.TryGetPooled1D(), other.TryGetPooled1D() with
        | Some(inputA, logicalLengthA, sizeA, 1u), Some(inputB, logicalLengthB, sizeB, 1u)
            when logicalLengthA = logicalLengthB && sizeA = sizeB ->
            let output = ArrayPool<'V>.Shared.Rent(logicalLengthA)
            try
                for i in 0 .. logicalLengthA - 1 do
                    output[i] <- f inputA[i] inputB[i]
                Some(Image<'V>.ofPooled1D(output, logicalLengthA, sizeA, name, this.index))
            with
            | _ ->
                poisonPooledBuffer logicalLengthA output
                ArrayPool<'V>.Shared.Return(output)
                reraise()
        | _ -> None
    member this.TryFoldPooled1D<'S> (f: 'S -> 'T -> 'S, acc0: 'S) : 'S option =
        match this.TryGetPooled1D() with
        | Some(input, logicalLength, _, 1u) ->
            let mutable acc = acc0
            for i in 0 .. logicalLength - 1 do
                acc <- f acc input[i]
            Some acc
        | _ -> None
    member this.TryFold2Pooled1D<'S, 'U when 'U : equality> (other: Image<'U>, f: 'S -> 'T -> 'U -> 'S, acc0: 'S) : 'S option =
        match this.TryGetPooled1D(), other.TryGetPooled1D() with
        | Some(inputA, logicalLengthA, sizeA, 1u), Some(inputB, logicalLengthB, sizeB, 1u)
            when logicalLengthA = logicalLengthB && sizeA = sizeB ->
            let mutable acc = acc0
            for i in 0 .. logicalLengthA - 1 do
                acc <- f acc inputA[i] inputB[i]
            Some acc
        | _ -> None
    member this.TryIterPooled1D (f: 'T -> unit) : bool =
        match this.TryGetPooled1D() with
        | Some(input, logicalLength, _, 1u) ->
            for i in 0 .. logicalLength - 1 do
                f input[i]
            true
        | _ -> false

    override this.ToString() = 
        let sz = this.GetSize()
        let szStr = List.fold (fun acc elm -> acc + $"x{elm}") (sz |> List.head |> string) (List.tail sz)
        let comp = this.GetNumberOfComponentsPerPixel()
        let vecStr = if comp = 1u then "Scalar" else sprintf $"{comp}-Vector "
        sprintf "%s %s<%s=ArrayPool>" szStr vecStr (typeof<'T>.Name)
    member this.Display = this.ToString() // related to [<StructuredFormatDisplay>]

    interface System.IEquatable<Image<'T>> with
        member this.Equals(other: Image<'T>) = Image<'T>.eq(this, other)

    interface System.IComparable with
        member this.CompareTo(obj: obj) =
            match obj with
            | :? Image<'T> as other -> this.CompareTo(other)
            | _ -> invalidArg "obj" "Expected Image<'T>"

    /// <summary>
    /// Creates a safe, independent <c>Image&lt;'T&gt;</c> from a SimpleITK image.
    /// The resulting image does not share its pixel buffer with <paramref name="itkImg" />. Matching pixel types are
    /// deep-copied; non-matching pixel types are converted with SimpleITK's cast filter. Physical metadata is normalized
    /// to StackProcessing defaults. The argument is borrowed and is not disposed.
    /// </summary>
    static member ofSimpleITK (itkImg: itk.simple.Image, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName ""
        let index = defaultArg optionalIndex 0

        let itkImgCast = ofCastITK<'T> itkImg |> canonicalizeSimpleItkImage
        try
            let isComplexType = typeof<'T> = typeof<System.Numerics.Complex> || typeof<'T> = typeof<ComplexFloat32>
            if isComplexType && not (isComplexCompatibleImage itkImgCast) then
                invalidArg "itkImg" "Complex pixel type requires a native complex image."
            if itkImgCast.GetNumberOfComponentsPerPixel() <> 1u then
                invalidArg "itkImg" "ArrayPool-backed Image currently supports scalar SimpleITK conversion only."
            let size = itkImgCast.GetSize() |> fromVectorUInt32
            let logicalLength = size |> List.fold (fun acc value -> acc * int value) 1
            let pixels = copyScalarPixels<'T> itkImgCast logicalLength
            let buffer = ArrayPool<'T>.Shared.Rent(logicalLength)
            Array.Copy(pixels, buffer, logicalLength)
            Image<'T>.ofPooled1D(buffer, logicalLength, size, name, index)
        with
        | _ ->
            itkImgCast.Dispose()
            reraise()

    /// <summary>
    /// Creates an aliasing <c>Image&lt;'T&gt;</c> from a SimpleITK image whose pixel type already matches <c>'T</c>.
    /// The returned image uses a shallow SimpleITK copy and may share the same pixel container as
    /// <paramref name="itkImg" /> until SimpleITK copy-on-write forces uniqueness. No cast, deep copy, metadata
    /// canonicalization, or disposal of the argument is performed. This is intended for internal hot paths where
    /// aliasing is acceptable and explicit.
    /// </summary>
    static member ofSimpleITKAlias (itkImg: itk.simple.Image, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName ""
        let index = defaultArg optionalIndex 0
        Image<'T>.ofSimpleITK(itkImg, name, index)

    /// <summary>
    /// Creates an aliasing <c>Image&lt;'T&gt;</c> by taking over a SimpleITK image whose pixel type already matches <c>'T</c>.
    /// No SimpleITK wrapper copy, deep copy, cast, or metadata canonicalization is performed. The returned image stores
    /// <paramref name="itkImg" /> directly and will dispose it when the image reference count reaches zero. The caller
    /// must not dispose or continue using <paramref name="itkImg" /> after a successful call.
    /// </summary>
    static member private ofSimpleITKAliasTransfer (itkImg: itk.simple.Image, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName ""
        let index = defaultArg optionalIndex 0
        let expectedId = fromType<'T>
        if itkImg.GetPixelID() <> expectedId then
            invalidArg "itkImg" $"Expected {expectedId} image for alias transfer, got {itkImg.GetPixelID()}."
        Image<'T>.ofSimpleITK(itkImg, name, index)

    /// <summary>
    /// Creates an <c>Image&lt;'T&gt;</c> from a temporary SimpleITK image and consumes that temporary.
    /// If the pixel type already matches <c>'T</c>, the SimpleITK wrapper is transferred directly into the returned
    /// image with no copy or cast; the returned image will dispose it when its reference count reaches zero. If a cast is
    /// needed, the result is deep-copied through <c>ofSimpleITK</c> and <paramref name="itkImg" /> is disposed before
    /// returning. The caller must not dispose or continue using <paramref name="itkImg" /> after calling this function.
    /// </summary>
    static member ofSimpleITKNDispose (itkImg: itk.simple.Image, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName ""
        let index = defaultArg optionalIndex 0
        try
            Image<'T>.ofSimpleITK(itkImg, name, index)
        finally
            itkImg.Dispose()

    /// <summary>
    /// Creates an <c>Image&lt;'T&gt;</c> backed by an ArrayPool-rented one-dimensional scalar buffer.
    /// Ownership of <paramref name="buffer" /> is transferred to the image and returned to the pool when the image is
    /// released. Only the first <paramref name="logicalLength" /> values are part of the image.
    /// </summary>
    static member internal ofPooled1D (buffer: 'T[], logicalLength: int, size: uint list, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName ""
        let index = defaultArg optionalIndex 0
        let expectedLength =
            size
            |> List.fold (fun acc value -> acc * int64 value) 1L
        if expectedLength <> int64 logicalLength then
            invalidArg "logicalLength" $"Logical length {logicalLength} does not match image size {size}."
        let image = new Image<'T>([0u;0u], 1u, name, index, true)
        let owner = PooledBufferOwner(buffer, logicalLength)
        image.SetPooled1D(owner, size, 1u)
        if debug then printDebugMessage $"Created pooled {image.Name} ({size}, {typeof<'T>.Name}->{Image<'T>.pooledMemoryEstimate logicalLength})"
        image

    member this.toSimpleITK () : itk.simple.Image =
        if isComplexType then
            invalidOp "ArrayPool-backed Image currently supports scalar toSimpleITK conversion only."
        if pooledComponents <> 1u then
            invalidOp "ArrayPool-backed Image currently supports scalar toSimpleITK conversion only."
        importScalarImage pooledSize owner.Buffer

    member this.copy (?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName "copy"
        let index = defaultArg optionalIndex this.index
        let buffer = owner.Buffer
        let logicalLength = owner.LogicalLength
        let output = ArrayPool<'T>.Shared.Rent(logicalLength)
        Array.Copy(buffer, output, logicalLength)
        Image<'T>.ofPooled1D(output, logicalLength, pooledSize, name, index)

    member this.castTo<'S when 'S: equality> () : Image<'S> = Image<'S>.ofSimpleITK(this.toSimpleITK(),"cast",this.index)
    member this.toUInt8 ()   : Image<uint8>   = Image<uint8>.ofSimpleITK(this.toSimpleITK(),"toUInt8",this.index)
    member this.toInt8 ()    : Image<int8>    = Image<int8>.ofSimpleITK(this.toSimpleITK(),"toInt8",this.index)
    member this.toUInt16 ()  : Image<uint16>  = Image<uint16>.ofSimpleITK(this.toSimpleITK(),"toUInt16",this.index)
    member this.toInt16 ()   : Image<int16>   = Image<int16>.ofSimpleITK(this.toSimpleITK(),"toInt16",this.index)
    member this.toUInt ()    : Image<uint>    = Image<uint>.ofSimpleITK(this.toSimpleITK(),"toUInt",this.index)
    member this.toInt ()     : Image<int>     = Image<int>.ofSimpleITK(this.toSimpleITK(),"toInt",this.index)
    member this.toUInt64 ()  : Image<uint64>  = Image<uint64>.ofSimpleITK(this.toSimpleITK(),"toUInt64",this.index)
    member this.toInt64 ()   : Image<int64>   = Image<int64>.ofSimpleITK(this.toSimpleITK(),"toInt64",this.index)
    member this.toFloat32 () : Image<float32> = Image<float32>.ofSimpleITK(this.toSimpleITK(),"toFloat32",this.index)
    member this.toFloat ()   : Image<float>   = Image<float>.ofSimpleITK(this.toSimpleITK(),"toFloat",this.index)
    member this.toComplexFloat32 () : Image<ComplexFloat32> = Image<ComplexFloat32>.ofSimpleITK(this.toSimpleITK(),"toComplexFloat32",this.index)
    member this.toComplex () : Image<System.Numerics.Complex> = Image<System.Numerics.Complex>.ofSimpleITK(this.toSimpleITK(),"toComplex",this.index)
    member this.toVectorUInt8 ()   : Image<uint8 list>   = Image<uint8 list>.ofSimpleITK(this.toSimpleITK(),"toVectorUInt8",this.index)
    member this.toVectorInt8 ()    : Image<int8 list>    = Image<int8 list>.ofSimpleITK(this.toSimpleITK(),"toVectorInt8",this.index)
    member this.toVectorUInt16 ()  : Image<uint16 list>  = Image<uint16 list>.ofSimpleITK(this.toSimpleITK(),"toVectorUInt16",this.index)
    member this.toVectorInt16 ()   : Image<int16 list>   = Image<int16 list>.ofSimpleITK(this.toSimpleITK(),"toVectorInt16",this.index)
    member this.toVectorUInt32 ()  : Image<uint32 list>  = Image<uint32 list>.ofSimpleITK(this.toSimpleITK(),"toVectorUInt32",this.index)
    member this.toVectorInt32 ()   : Image<int32 list>   = Image<int32 list>.ofSimpleITK(this.toSimpleITK(),"toVectorInt32",this.index)
    member this.toVectorUInt64 ()  : Image<uint64 list>  = Image<uint64 list>.ofSimpleITK(this.toSimpleITK(),"toVectorUInt64",this.index)
    member this.toVectorInt64 ()   : Image<int64 list>   = Image<int64 list>.ofSimpleITK(this.toSimpleITK(),"toVectorInt64",this.index)
    member this.toVectorFloat32 () : Image<float32 list> = Image<float32 list>.ofSimpleITK(this.toSimpleITK(),"toVectorFloat32",this.index)
    member this.toVectorFloat64 () : Image<float list>   = Image<float list>.ofSimpleITK(this.toSimpleITK(),"toVectorFloat64",this.index)

    static member ofArray2D (arr: 'T[,], ?name:string, ?index:int) : Image<'T> =
        let _name = defaultArg name ""
        let _index = defaultArg index 0
        let sz = [arr.GetLength(0); arr.GetLength(1)] |> List.map uint
        if isScalarImportSupported<'T> then
            let width = arr.GetLength(0)
            let height = arr.GetLength(1)
            let pixels = ArrayPool<'T>.Shared.Rent(width * height)
            let mutable offset = 0
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    pixels[offset] <- arr[x, y]
                    offset <- offset + 1

            Image<'T>.ofPooled1D(pixels, width * height, sz, _name, _index)
        else
            let img = new Image<'T>(sz,1u,_name,_index)
            img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1]])
            img

    static member constant2D (width: uint, height: uint, value: 'T, ?name: string, ?index: int) : Image<'T> =
        if width = 0u then invalidArg "width" "constant2D requires a positive width."
        if height = 0u then invalidArg "height" "constant2D requires a positive height."

        let _name = defaultArg name "constant2D"
        let _index = defaultArg index 0
        Array2D.create (int width) (int height) value
        |> fun values -> Image<'T>.ofArray2D(values, _name, _index)

    static member coordinateAxis2D (width: uint, height: uint, axis: int, ?name: string, ?index: int) : Image<'T> =
        if width = 0u then invalidArg "width" "coordinateAxis2D requires a positive width."
        if height = 0u then invalidArg "height" "coordinateAxis2D requires a positive height."
        if axis < 0 || axis > 1 then invalidArg "axis" "coordinateAxis2D axis must be 0 or 1."

        let _name = defaultArg name "coordinateAxis2D"
        let _index = defaultArg index 0
        Array2D.init (int width) (int height) (fun x y ->
            let value = if axis = 0 then x else y
            Convert.ChangeType(value, typeof<'T>) :?> 'T)
        |> fun values -> Image<'T>.ofArray2D(values, _name, _index)

    static member polygonMask (width: uint, height: uint, polygon: (float * float) list, ?name: string, ?index: int) : Image<uint8> =
        if width = 0u then invalidArg "width" "polygonMask requires a positive width."
        if height = 0u then invalidArg "height" "polygonMask requires a positive height."
        if polygon.Length < 3 then invalidArg "polygon" "polygonMask requires at least three polygon vertices."

        let _name = defaultArg name "polygonMask"
        let _index = defaultArg index 0
        let eps = 1e-9

        let pointOnSegment px py (x0, y0) (x1, y1) =
            let cross = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)
            abs cross <= eps
            && px >= min x0 x1 - eps && px <= max x0 x1 + eps
            && py >= min y0 y1 - eps && py <= max y0 y1 + eps

        let contains px py =
            let rec loop previous remaining inside =
                match remaining with
                | [] -> inside
                | current :: tail ->
                    if pointOnSegment px py previous current then
                        true
                    else
                        let x0, y0 = previous
                        let x1, y1 = current
                        let crosses = (y0 > py) <> (y1 > py)
                        let inside' =
                            if crosses then
                                let xCross = (x1 - x0) * (py - y0) / (y1 - y0) + x0
                                if px < xCross then not inside else inside
                            else
                                inside
                        loop current tail inside'

            loop (List.last polygon) polygon false

        let pixels =
            Array2D.init (int width) (int height) (fun x y ->
                let px = float x + 0.5
                let py = float y + 0.5
                if contains px py then 1uy else 0uy)

        Image<uint8>.ofArray2D(pixels, _name, _index)

    static member ofArray3D (arr: 'T[,,], ?name:string, ?index:int) : Image<'T> =
        let _name = defaultArg name ""
        let _index = defaultArg index 0
        let sz = [arr.GetLength(0); arr.GetLength(1); arr.GetLength(2)] |> List.map uint
        if isScalarImportSupported<'T> then
            let width = arr.GetLength(0)
            let height = arr.GetLength(1)
            let depth = arr.GetLength(2)
            let pixels = ArrayPool<'T>.Shared.Rent(width * height * depth)
            let mutable offset = 0
            for z in 0 .. depth - 1 do
                for y in 0 .. height - 1 do
                    for x in 0 .. width - 1 do
                        pixels[offset] <- arr[x, y, z]
                        offset <- offset + 1

            Image<'T>.ofPooled1D(pixels, width * height * depth, sz, _name, _index)
        else
            let img = new Image<'T>(sz,1u,_name,_index)
            img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1],int idxLst[2]])
            img

    static member ofArray4D (arr: 'T[,,,], ?name:string, ?index:int) : Image<'T> =
        let _name = defaultArg name ""
        let _index = defaultArg index 0
        let sz = [arr.GetLength(0); arr.GetLength(1); arr.GetLength(2); arr.GetLength(3)] |> List.map uint
        if isScalarImportSupported<'T> then
            let width = arr.GetLength(0)
            let height = arr.GetLength(1)
            let depth = arr.GetLength(2)
            let length = arr.GetLength(3)
            // SimpleITK's ImportImageFilter only supports 2D/3D buffers here, so import 3D chunks and join them into a hidden 4D image.
            use filter = new itk.simple.JoinSeriesImageFilter()
            filter.SetOrigin(0.0) |> ignore
            filter.SetSpacing(1.0) |> ignore
            use v = new itk.simple.VectorOfImage()
            let chunks = ResizeArray<itk.simple.Image>()
            try
                for t in 0 .. length - 1 do
                    let pixels = Array.zeroCreate<'T> (width * height * depth)
                    let mutable offset = 0
                    for z in 0 .. depth - 1 do
                        for y in 0 .. height - 1 do
                            for x in 0 .. width - 1 do
                                pixels[offset] <- arr[x, y, z, t]
                                offset <- offset + 1
                    let chunk = importScalarImage [ uint width; uint height; uint depth ] pixels
                    chunks.Add chunk
                    v.Add chunk

                Image<'T>.ofSimpleITKNDispose(filter.Execute(v), _name, _index)
            finally
                chunks |> Seq.iter (fun chunk -> chunk.Dispose())
        else
            let img = new Image<'T>(sz,1u,_name,_index)
            img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1],int idxLst[2],int idxLst[3]])
            img

    static member ofArray3DVector (arr: 'S[,,], ?name:string, ?index:int) : Image<'S list> =
        let _name = defaultArg name ""
        let _index = defaultArg index 0
        let w = arr.GetLength(0)
        let h = arr.GetLength(1)
        let c = arr.GetLength(2)
        let componentCount = max 2 c
        let components =
            [ for k in 0 .. componentCount - 1 ->
                Array2D.init w h (fun i0 i1 ->
                    if k < c then arr[i0, i1, k] else Unchecked.defaultof<'S>)
                |> fun values -> Image<'S>.ofArray2D(values, $"{_name}.Component{k}", _index + k) ]
        use filter = new itk.simple.ComposeImageFilter()
        use v = new itk.simple.VectorOfImage()
        try
            components |> List.iter (fun image -> v.Add(image.toSimpleITK()))
            Image<'S list>.ofSimpleITKNDispose(filter.Execute(v), _name, _index)
        finally
            components |> List.iter (fun image -> image.decRefCount())

    static member ofArray3DComplex (arr: float[,,], ?name:string) : Image<System.Numerics.Complex> =
        let _name = defaultArg name ""
        let w = arr.GetLength(0)
        let h = arr.GetLength(1)
        let c = arr.GetLength(2)
        if c <> 2 then invalidArg "arr" "ofArray3DComplex expects last dimension size 2 (real, imag)."
        Array2D.init w h (fun i0 i1 -> System.Numerics.Complex(arr[i0, i1, 0], arr[i0, i1, 1]))
        |> fun values -> Image<System.Numerics.Complex>.ofComplexArray2D(values, _name)

    static member ofComplexArray2D (arr: System.Numerics.Complex[,], ?name:string, ?index:int) : Image<System.Numerics.Complex> =
        let _name = defaultArg name "ofComplexArray2D"
        let _index = defaultArg index 0
        let real = Array2D.init (arr.GetLength 0) (arr.GetLength 1) (fun x y -> arr[x, y].Real)
        let imag = Array2D.init (arr.GetLength 0) (arr.GetLength 1) (fun x y -> arr[x, y].Imaginary)
        let realImg = Image<float>.ofArray2D(real, $"{_name}.Re", _index)
        let imagImg = Image<float>.ofArray2D(imag, $"{_name}.Im", _index)
        let result = Image<float>.ofImagePairToComplex realImg imagImg
        realImg.decRefCount()
        imagImg.decRefCount()
        result

    static member ofComplexFloat32Array2D (arr: ComplexFloat32[,], ?name:string, ?index:int) : Image<ComplexFloat32> =
        let _name = defaultArg name "ofComplexFloat32Array2D"
        let _index = defaultArg index 0
        let real = Array2D.init (arr.GetLength 0) (arr.GetLength 1) (fun x y -> arr[x, y].Real)
        let imag = Array2D.init (arr.GetLength 0) (arr.GetLength 1) (fun x y -> arr[x, y].Imaginary)
        let realImg = Image<float32>.ofArray2D(real, $"{_name}.Re", _index)
        let imagImg = Image<float32>.ofArray2D(imag, $"{_name}.Im", _index)
        let result = Image<float32>.ofImagePairToComplexFloat32 realImg imagImg
        realImg.decRefCount()
        imagImg.decRefCount()
        result

    static member ofComplexFloat32Array3D (arr: ComplexFloat32[,,], ?name:string, ?index:int) : Image<ComplexFloat32> =
        let _name = defaultArg name "ofComplexFloat32Array3D"
        let _index = defaultArg index 0
        let real = Array3D.init (arr.GetLength 0) (arr.GetLength 1) (arr.GetLength 2) (fun x y z -> arr[x, y, z].Real)
        let imag = Array3D.init (arr.GetLength 0) (arr.GetLength 1) (arr.GetLength 2) (fun x y z -> arr[x, y, z].Imaginary)
        let realImg = Image<float32>.ofArray3D(real, $"{_name}.Re", _index)
        let imagImg = Image<float32>.ofArray3D(imag, $"{_name}.Im", _index)
        let result = Image<float32>.ofImagePairToComplexFloat32 realImg imagImg
        realImg.decRefCount()
        imagImg.decRefCount()
        result

    static member ofComplexArray3D (arr: System.Numerics.Complex[,,], ?name:string, ?index:int) : Image<System.Numerics.Complex> =
        let _name = defaultArg name "ofComplexArray3D"
        let _index = defaultArg index 0
        let real = Array3D.init (arr.GetLength 0) (arr.GetLength 1) (arr.GetLength 2) (fun x y z -> arr[x, y, z].Real)
        let imag = Array3D.init (arr.GetLength 0) (arr.GetLength 1) (arr.GetLength 2) (fun x y z -> arr[x, y, z].Imaginary)
        let realImg = Image<float>.ofArray3D(real, $"{_name}.Re", _index)
        let imagImg = Image<float>.ofArray3D(imag, $"{_name}.Im", _index)
        let result = Image<float>.ofImagePairToComplex realImg imagImg
        realImg.decRefCount()
        imagImg.decRefCount()
        result

    member this.toArray2D (): 'T[,] =
        let sz = this.GetSize() |> List.map int
        match this.TryGetPooled1D() with
        | Some(buffer, logicalLength, _, 1u) when this.GetDimensions() = 2u && logicalLength = sz[0] * sz[1] ->
            let width = sz[0]
            Array2D.init sz[0] sz[1] (fun x y -> buffer[y * width + x])
        | _ when this.GetDimensions() = 2u && isScalarImportSupported<'T> && this.toSimpleITK().GetPixelID() = fromType<'T> ->
            let width = sz[0]
            let height = sz[1]
            let pixels = copyScalarPixels<'T> (this.toSimpleITK()) (width * height)
            Array2D.init width height (fun x y -> pixels[y * width + x])
        | _ ->
            Array2D.init sz[0] sz[1] (fun i0 i1 -> this.Get([uint i0; uint i1]))

    member this.toComplexArray2D () : System.Numerics.Complex[,] =
        if typeof<'T> <> typeof<System.Numerics.Complex> then
            invalidOp "toComplexArray2D: image pixel type must be System.Numerics.Complex."
        if this.GetDimensions() <> 2u then
            invalidOp $"toComplexArray2D: image must be 2D, got {this.GetDimensions()}D."

        let realItk = extractComplexRealImage (this.toSimpleITK())
        let imagItk = extractComplexImagImage (this.toSimpleITK())
        let realImg = Image<float>.ofSimpleITKNDispose(realItk, "toComplexArray2D.Re", this.index)
        let imagImg = Image<float>.ofSimpleITKNDispose(imagItk, "toComplexArray2D.Im", this.index)
        let real = realImg.toArray2D()
        let imag = imagImg.toArray2D()
        realImg.decRefCount()
        imagImg.decRefCount()
        Array2D.init (real.GetLength 0) (real.GetLength 1) (fun x y -> System.Numerics.Complex(real[x, y], imag[x, y]))

    member this.toComplexFloat32Array2D () : ComplexFloat32[,] =
        if typeof<'T> <> typeof<ComplexFloat32> then
            invalidOp "toComplexFloat32Array2D: image pixel type must be ComplexFloat32."
        if this.GetDimensions() <> 2u then
            invalidOp $"toComplexFloat32Array2D: image must be 2D, got {this.GetDimensions()}D."

        let realItk = extractComplexRealImage (this.toSimpleITK())
        let imagItk = extractComplexImagImage (this.toSimpleITK())
        let realImg = Image<float32>.ofSimpleITKNDispose(realItk, "toComplexFloat32Array2D.Re", this.index)
        let imagImg = Image<float32>.ofSimpleITKNDispose(imagItk, "toComplexFloat32Array2D.Im", this.index)
        let real = realImg.toArray2D()
        let imag = imagImg.toArray2D()
        realImg.decRefCount()
        imagImg.decRefCount()
        Array2D.init (real.GetLength 0) (real.GetLength 1) (fun x y -> ComplexFloat32(real[x, y], imag[x, y]))

    static member toArray3DVector<'S when 'S: equality> (img: Image<'S list>) : 'S[,,] =
        if img.GetDimensions() <> 2u then
            failwith $"toArray3DVector: image must be 2D, got {img.GetDimensions()}D"
        use filter = new itk.simple.VectorIndexSelectionCastImageFilter()
        let components =
            [ for i in 0 .. int (img.GetNumberOfComponentsPerPixel()) - 1 ->
                filter.SetIndex(uint i)
                let scalarItk = filter.Execute(img.toSimpleITK())
                Image<'S>.ofSimpleITKNDispose(scalarItk, $"toArray3DVector.Component{i}", img.index + i) ]
        try
            let values = components |> List.map (fun image -> image.toArray2D())
            let width = values.Head.GetLength(0)
            let height = values.Head.GetLength(1)
            Array3D.init width height values.Length (fun i0 i1 k -> values[k][i0, i1])
        finally
            components |> List.iter (fun image -> image.decRefCount())

    member this.toArray3D (): 'T[,,] =
        let sz = this.GetSize() |> List.map int
        match this.TryGetPooled1D() with
        | Some(buffer, logicalLength, _, 1u) when this.GetDimensions() = 3u && logicalLength = sz[0] * sz[1] * sz[2] ->
            let width = sz[0]
            let height = sz[1]
            Array3D.init sz[0] sz[1] sz[2] (fun x y z -> buffer[(z * height + y) * width + x])
        | _ when this.GetDimensions() = 3u && isScalarImportSupported<'T> && this.toSimpleITK().GetPixelID() = fromType<'T> ->
            let width = sz[0]
            let height = sz[1]
            let depth = sz[2]
            let pixels = copyScalarPixels<'T> (this.toSimpleITK()) (width * height * depth)
            Array3D.init width height depth (fun x y z -> pixels[(z * height + y) * width + x])
        | _ ->
            Array3D.init sz[0] sz[1] sz[2] (fun i0 i1 i2 -> this.Get([uint i0; uint i1; uint i2]))

    member this.toArray4D (): 'T[,,,] =
        let sz = this.GetSize() |> List.map int
        if this.GetDimensions() = 4u && isScalarImportSupported<'T> && this.toSimpleITK().GetPixelID() = fromType<'T> then
            let width = sz[0]
            let height = sz[1]
            let depth = sz[2]
            let length = sz[3]
            let pixels = copyScalarPixels<'T> (this.toSimpleITK()) (width * height * depth * length)
            Array4D.init width height depth length (fun x y z t -> pixels[((t * depth + z) * height + y) * width + x])
        else
            Array4D.init sz[0] sz[1] sz[2] sz[3] (fun i0 i1 i2 i3 -> this.Get([uint i0; uint i1; uint i2; uint i3]))

    member this.toComplexArray3D () : System.Numerics.Complex[,,] =
        if typeof<'T> <> typeof<System.Numerics.Complex> then
            invalidOp "toComplexArray3D: image pixel type must be System.Numerics.Complex."
        if this.GetDimensions() <> 3u then
            invalidOp $"toComplexArray3D: image must be 3D, got {this.GetDimensions()}D."

        let realItk = extractComplexRealImage (this.toSimpleITK())
        let imagItk = extractComplexImagImage (this.toSimpleITK())
        let realImg = Image<float>.ofSimpleITKNDispose(realItk, "toComplexArray3D.Re", this.index)
        let imagImg = Image<float>.ofSimpleITKNDispose(imagItk, "toComplexArray3D.Im", this.index)
        let real = realImg.toArray3D()
        let imag = imagImg.toArray3D()
        realImg.decRefCount()
        imagImg.decRefCount()
        Array3D.init (real.GetLength 0) (real.GetLength 1) (real.GetLength 2) (fun x y z -> System.Numerics.Complex(real[x, y, z], imag[x, y, z]))

    member this.toComplexFloat32Array3D () : ComplexFloat32[,,] =
        if typeof<'T> <> typeof<ComplexFloat32> then
            invalidOp "toComplexFloat32Array3D: image pixel type must be ComplexFloat32."
        if this.GetDimensions() <> 3u then
            invalidOp $"toComplexFloat32Array3D: image must be 3D, got {this.GetDimensions()}D."

        let realItk = extractComplexRealImage (this.toSimpleITK())
        let imagItk = extractComplexImagImage (this.toSimpleITK())
        let realImg = Image<float32>.ofSimpleITKNDispose(realItk, "toComplexFloat32Array3D.Re", this.index)
        let imagImg = Image<float32>.ofSimpleITKNDispose(imagItk, "toComplexFloat32Array3D.Im", this.index)
        let real = realImg.toArray3D()
        let imag = imagImg.toArray3D()
        realImg.decRefCount()
        imagImg.decRefCount()
        Array3D.init (real.GetLength 0) (real.GetLength 1) (real.GetLength 2) (fun x y z -> ComplexFloat32(real[x, y, z], imag[x, y, z]))

    // Make a multicomponent image of a list
    static member ofImageList (images: Image<'S> list) : Image<'S list> =
        match images with
        | []
        | [ _ ] ->
            invalidArg "images" "At least two images are required for ComposeImageFilter."
        | first :: _ ->
            let expectedSize = first.GetSize()
            images
            |> List.iter (fun img ->
                if img.GetSize() <> expectedSize then
                    invalidArg "images" $"Image dimensions must match: {img.GetSize()} vs {expectedSize}.")

            use filter = new itk.simple.ComposeImageFilter()
            use v = new itk.simple.VectorOfImage()
            images |> List.iter (fun image -> v.Add(image.toSimpleITK()))
            Image<'S list>.ofSimpleITKNDispose(filter.Execute(v), "ofImageList", first.index)

    static member ofImagePairToComplex (realImg: Image<float>) (imagImg: Image<float>) : Image<System.Numerics.Complex> =
        if realImg.GetSize() <> imagImg.GetSize() then
            invalidArg "imagImg" $"Image dimensions must match: {realImg.GetSize()} vs {imagImg.GetSize()}"

        use filter = new itk.simple.RealAndImaginaryToComplexImageFilter()
        Image<System.Numerics.Complex>.ofSimpleITKNDispose(filter.Execute(realImg.toSimpleITK(), imagImg.toSimpleITK()), "ofImagePairToComplex", realImg.index)

    static member ofImagePairToComplexFloat32 (realImg: Image<float32>) (imagImg: Image<float32>) : Image<ComplexFloat32> =
        if realImg.GetSize() <> imagImg.GetSize() then
            invalidArg "imagImg" $"Image dimensions must match: {realImg.GetSize()} vs {imagImg.GetSize()}"

        use filter = new itk.simple.RealAndImaginaryToComplexImageFilter()
        Image<ComplexFloat32>.ofSimpleITKNDispose(filter.Execute(realImg.toSimpleITK(), imagImg.toSimpleITK()), "ofImagePairToComplexFloat32", realImg.index)

    member this.toImageList () : Image<'S> list =
        use filter = new itk.simple.VectorIndexSelectionCastImageFilter()
        let n = this.toSimpleITK().GetNumberOfComponentsPerPixel() |> int
        List.init n (fun i ->
            filter.SetIndex(uint i)
            let scalarItk = filter.Execute(this.toSimpleITK())
            Image<'S>.ofSimpleITKNDispose(scalarItk,"toImageList",this.index+i)
        )

    static member ofFile(filename: string, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName filename
        let index = defaultArg optionalIndex 0

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        let itkImg = reader.Execute()
        let mutable consumed = false
        try
            let numComp = itkImg.GetNumberOfComponentsPerPixel()
            let tType = typeof<'T>
            let isComplexType = tType = typeof<System.Numerics.Complex> || tType = typeof<ComplexFloat32>
            let isVectorType =
                (tType.IsGenericType && tType.GetGenericTypeDefinition() = typedefof<list<_>>)
                || tType.IsArray

            // Validate number of components matches expectations
            match isComplexType, isVectorType, numComp with
            | true, _, _ when isComplexCompatibleImage itkImg ->
                let image = Image<'T>.ofSimpleITKNDispose(itkImg, name, index)
                consumed <- true
                image
            | true, _, n when n >= 2u ->
                let vector = Image<float list>.ofSimpleITK(itkImg, name + ".vector", index)
                let parts = vector.toImageList()
                if parts.Length < 2 then
                    vector.decRefCount()
                    parts |> List.iter (fun part -> part.decRefCount())
                    failwithf "Pixel type '%O' expects native complex or a vector with real and imaginary components, but image has %d component(s)." tType n
                let complex =
                    if tType = typeof<ComplexFloat32> then
                        let real = parts[0].toFloat32()
                        let imag = parts[1].toFloat32()
                        try
                            Image<float32>.ofImagePairToComplexFloat32 real imag |> box
                        finally
                            real.decRefCount()
                            imag.decRefCount()
                    else
                        Image<float>.ofImagePairToComplex parts[0] parts[1] |> box
                vector.decRefCount()
                parts |> List.iter (fun part -> part.decRefCount())
                complex |> unbox<Image<'T>>
            | true, _, n ->
                failwithf "Pixel type '%O' expects native complex or a vector with real and imaginary components, but got %O with %d component(s)." tType (itkImg.GetPixelID()) n
            | false, true, n when n < 2u ->
                failwithf "Pixel type '%O' expects a vector (>=2 components), but image has %d component(s)." tType n
            | false, false, n when n > 1u ->
                failwithf "Pixel type '%O' expects a scalar (1 component), but image has %d component(s)." tType n
            | _ ->
                let image = Image<'T>.ofSimpleITKNDispose(itkImg,name,index)
                consumed <- true
                image
        finally
            if not consumed then itkImg.Dispose()

    static member ofFileVector<'S when 'S: equality> (filename: string, ?optionalName: string, ?optionalIndex: int) : Image<'S list> =
        let name = defaultArg optionalName filename
        let index = defaultArg optionalIndex 0

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        let itkImg = reader.Execute()
        let mutable consumed = false
        try
            let numComp = itkImg.GetNumberOfComponentsPerPixel()
            if numComp < 2u then
                failwithf "Vector pixel type expects >=2 components, but image has %d component(s)." numComp
            let image = Image<'S list>.ofSimpleITKNDispose(itkImg, name, index)
            consumed <- true
            image
        finally
            if not consumed then itkImg.Dispose()

    static member ofFileComplex (filename: string, ?optionalName: string, ?optionalIndex: int) : Image<System.Numerics.Complex> =
        Image<System.Numerics.Complex>.ofFile(filename, ?optionalName = optionalName, ?optionalIndex = optionalIndex)

    static member ofFileComplexFloat32 (filename: string, ?optionalName: string, ?optionalIndex: int) : Image<ComplexFloat32> =
        Image<ComplexFloat32>.ofFile(filename, ?optionalName = optionalName, ?optionalIndex = optionalIndex)

    member this.toFile(filename: string, ?optionalFormat: string) =
        use writer = new itk.simple.ImageFileWriter()
        writer.SetFileName(filename)
        writer.SetUseCompression(false)
        match optionalFormat with
        | Some fmt -> writer.SetImageIO(fmt)
        | None -> ()
        writer.Execute(this.toSimpleITK())

    member this.toFileVector(filename: string, ?optionalFormat: string) =
        let isListType = typeof<'T>.IsGenericType && typeof<'T>.GetGenericTypeDefinition() = typedefof<list<_>>
        if not isListType then
            failwith "toFileVector: image pixel type must be a list for vector components."
        if this.GetNumberOfComponentsPerPixel() < 2u then
            failwithf "toFileVector: expected >=2 components per pixel, got %d." (this.GetNumberOfComponentsPerPixel())
        this.toFile(filename, ?optionalFormat = optionalFormat)

    member this.toFileComplex(filename: string, ?optionalFormat: string) =
        if typeof<'T> <> typeof<System.Numerics.Complex> && typeof<'T> <> typeof<ComplexFloat32> then
            failwith "toFileComplex: image pixel type must be System.Numerics.Complex or ComplexFloat32."
        if not (isComplexCompatibleImage (this.toSimpleITK())) then
            failwithf "toFileComplex: expected a native complex image, got %O with %d component(s)." (this.toSimpleITK().GetPixelID()) (this.GetNumberOfComponentsPerPixel())
        this.toFile(filename, ?optionalFormat = optionalFormat)

    // Arithmatic
    static member (+) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "add", f1.index)
    static member (-) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "subtract", f1.index)
    static member ( * ) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "multiply", f1.index)
    static member (/) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "divide", f1.index)

    static member maximumImage (f1: Image<'T>) (f2: Image<'T>) =
        use filter = new itk.simple.MaximumImageFilter()
        Image<'T>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "maximumImage", f1.index)

    static member minimumImage (f1: Image<'T>) (f2: Image<'T>) =
        use filter = new itk.simple.MinimumImageFilter()
        Image<'T>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "minimumImage", f1.index)

    static member getMinMax (img: Image<'T>) =
        use filter = new itk.simple.MinimumMaximumImageFilter()
        filter.Execute (img.toSimpleITK())
        (filter.GetMinimum(),filter.GetMaximum())

    // Collection type
    static member map (f:'T->'T) (im1: Image<'T>) : Image<'T> =
        match im1.TryMapPooled1D("map", f) with
        | Some output -> output
        | None ->
        match im1.GetDimensions() with
        | 2u ->
            im1.toArray2D()
            |> Array2D.map f
            |> fun values -> Image<'T>.ofArray2D(values, "map", im1.index)
        | 3u ->
            im1.toArray3D()
            |> Array3D.map f
            |> fun values -> Image<'T>.ofArray3D(values, "map", im1.index)
        | 4u ->
            let values = im1.toArray4D()
            Array4D.init (values.GetLength 0) (values.GetLength 1) (values.GetLength 2) (values.GetLength 3) (fun i0 i1 i2 i3 ->
                f values[i0, i1, i2, i3])
            |> fun output -> Image<'T>.ofArray4D(output, "map", im1.index)
        | _ ->
            let sz = im1.GetSize()
            let comp = im1.GetNumberOfComponentsPerPixel()
            let im = new Image<'T>(sz,comp)
            sz
            |> flatIndices
            |> Seq.iter (fun idx -> im1.Get idx |> f |> (im.Set idx))
            im

    static member mapi (f:uint list->'T->'T) (im1: Image<'T>) : Image<'T> =
        match im1.GetDimensions() with
        | 2u ->
            let values = im1.toArray2D()
            Array2D.init (values.GetLength 0) (values.GetLength 1) (fun i0 i1 ->
                f [ uint i0; uint i1 ] values[i0, i1])
            |> fun output -> Image<'T>.ofArray2D(output, "mapi", im1.index)
        | 3u ->
            let values = im1.toArray3D()
            Array3D.init (values.GetLength 0) (values.GetLength 1) (values.GetLength 2) (fun i0 i1 i2 ->
                f [ uint i0; uint i1; uint i2 ] values[i0, i1, i2])
            |> fun output -> Image<'T>.ofArray3D(output, "mapi", im1.index)
        | 4u ->
            let values = im1.toArray4D()
            Array4D.init (values.GetLength 0) (values.GetLength 1) (values.GetLength 2) (values.GetLength 3) (fun i0 i1 i2 i3 ->
                f [ uint i0; uint i1; uint i2; uint i3 ] values[i0, i1, i2, i3])
            |> fun output -> Image<'T>.ofArray4D(output, "mapi", im1.index)
        | _ ->
            let sz = im1.GetSize()
            let comp = im1.GetNumberOfComponentsPerPixel()
            let im = new Image<'T>(sz,comp)
            sz
            |> flatIndices
            |> Seq.iter (fun idx -> im1.Get idx |> f idx |> (im.Set idx))
            im

    static member iter (f:'T->unit) (im1: Image<'T>) : unit = 
        if im1.TryIterPooled1D(f) then
            ()
        else
            match im1.GetDimensions() with
            | 2u -> im1.toArray2D() |> Seq.cast<'T> |> Seq.iter f
            | 3u -> im1.toArray3D() |> Seq.cast<'T> |> Seq.iter f
            | 4u -> im1.toArray4D() |> Seq.cast<'T> |> Seq.iter f
            | _ ->
                let sz = im1.GetSize()
                sz
                |> flatIndices
                |> Seq.iter (fun idx -> im1.Get idx |> f)

    static member iteri (f:uint list->'T->unit) (im1: Image<'T>) : unit = 
        match im1.GetDimensions() with
        | 2u ->
            let values = im1.toArray2D()
            for i0 in 0 .. values.GetLength 0 - 1 do
                for i1 in 0 .. values.GetLength 1 - 1 do
                    f [ uint i0; uint i1 ] values[i0, i1]
        | 3u ->
            let values = im1.toArray3D()
            for i0 in 0 .. values.GetLength 0 - 1 do
                for i1 in 0 .. values.GetLength 1 - 1 do
                    for i2 in 0 .. values.GetLength 2 - 1 do
                        f [ uint i0; uint i1; uint i2 ] values[i0, i1, i2]
        | 4u ->
            let values = im1.toArray4D()
            for i0 in 0 .. values.GetLength 0 - 1 do
                for i1 in 0 .. values.GetLength 1 - 1 do
                    for i2 in 0 .. values.GetLength 2 - 1 do
                        for i3 in 0 .. values.GetLength 3 - 1 do
                            f [ uint i0; uint i1; uint i2; uint i3 ] values[i0, i1, i2, i3]
        | _ ->
            let sz = im1.GetSize()
            sz
            |> flatIndices
            |> Seq.iter (fun idx -> im1.Get idx |> f idx)

    static member fold (f:'S->'T->'S) (acc0: 'S) (im1: Image<'T>) : 'S = 
        match im1.TryFoldPooled1D(f, acc0) with
        | Some acc -> acc
        | None ->
            match im1.GetDimensions() with
            | 2u -> im1.toArray2D() |> Seq.cast<'T> |> Seq.fold f acc0
            | 3u -> im1.toArray3D() |> Seq.cast<'T> |> Seq.fold f acc0
            | 4u -> im1.toArray4D() |> Seq.cast<'T> |> Seq.fold f acc0
            | _ ->
                let sz = im1.GetSize()
                sz
                |> flatIndices
                |> Seq.fold (fun acc idx -> im1.Get idx |> f acc) acc0

    static member fold2 (f:'S->'T->'T->'S) (acc0: 'S) (im1: Image<'T>) (im2: Image<'T>) : 'S = 
        let sz1 = im1.GetSize()
        let sz2 = im2.GetSize()
        if List.exists2 (<>) sz1 sz2 then failwith "[Image.fold2] cannot fold over 2 images of unequal sizes"
        match im1.TryFold2Pooled1D(im2, f, acc0) with
        | Some acc -> acc
        | None ->
            match im1.GetDimensions(), im2.GetDimensions() with
            | 2u, 2u ->
                let v1 = im1.toArray2D()
                let v2 = im2.toArray2D()
                let mutable acc = acc0
                for i0 in 0 .. v1.GetLength 0 - 1 do
                    for i1 in 0 .. v1.GetLength 1 - 1 do
                        acc <- f acc v1[i0, i1] v2[i0, i1]
                acc
            | 3u, 3u ->
                let v1 = im1.toArray3D()
                let v2 = im2.toArray3D()
                let mutable acc = acc0
                for i0 in 0 .. v1.GetLength 0 - 1 do
                    for i1 in 0 .. v1.GetLength 1 - 1 do
                        for i2 in 0 .. v1.GetLength 2 - 1 do
                            acc <- f acc v1[i0, i1, i2] v2[i0, i1, i2]
                acc
            | 4u, 4u ->
                let v1 = im1.toArray4D()
                let v2 = im2.toArray4D()
                let mutable acc = acc0
                for i0 in 0 .. v1.GetLength 0 - 1 do
                    for i1 in 0 .. v1.GetLength 1 - 1 do
                        for i2 in 0 .. v1.GetLength 2 - 1 do
                            for i3 in 0 .. v1.GetLength 3 - 1 do
                                acc <- f acc v1[i0, i1, i2, i3] v2[i0, i1, i2, i3]
                acc
            | _ ->
                sz1
                |> flatIndices
                |> Seq.fold (fun acc idx -> (im1.Get idx, im2.Get idx) ||> f acc) acc0

    static member foldi (f:uint list->'S->'T->'S) (acc0: 'S) (im1: Image<'T>) : 'S =
        match im1.GetDimensions() with
        | 2u ->
            let values = im1.toArray2D()
            let mutable acc = acc0
            for i0 in 0 .. values.GetLength 0 - 1 do
                for i1 in 0 .. values.GetLength 1 - 1 do
                    acc <- f [ uint i0; uint i1 ] acc values[i0, i1]
            acc
        | 3u ->
            let values = im1.toArray3D()
            let mutable acc = acc0
            for i0 in 0 .. values.GetLength 0 - 1 do
                for i1 in 0 .. values.GetLength 1 - 1 do
                    for i2 in 0 .. values.GetLength 2 - 1 do
                        acc <- f [ uint i0; uint i1; uint i2 ] acc values[i0, i1, i2]
            acc
        | 4u ->
            let values = im1.toArray4D()
            let mutable acc = acc0
            for i0 in 0 .. values.GetLength 0 - 1 do
                for i1 in 0 .. values.GetLength 1 - 1 do
                    for i2 in 0 .. values.GetLength 2 - 1 do
                        for i3 in 0 .. values.GetLength 3 - 1 do
                            acc <- f [ uint i0; uint i1; uint i2; uint i3 ] acc values[i0, i1, i2, i3]
            acc
        | _ ->
            let sz = im1.GetSize()
            sz
            |> flatIndices
            |> Seq.fold (fun acc idx -> im1.Get idx |> f idx acc) acc0

    static member zip (imLst: Image<'T> list) : Image<'T list> =
        let nComp = imLst.Length
        if nComp < 2 then 
            failwith "can't zip list of less than 2 elements"
        else
            Image<'T>.ofImageList imLst

    static member unzip (im: Image<'T list>) : Image<'T> list =
        im.toImageList()

    member this.Get (coords: uint list) : 'T =
        if pooledComponents <> 1u || coords.Length <> pooledSize.Length then
            invalidOp "Get currently supports scalar ArrayPool-backed images only."
        let offset =
            (pooledSize, coords)
            ||> List.zip
            |> List.fold (fun (stride, acc) (dim, coord) -> stride * int dim, acc + int coord * stride) (1, 0)
            |> snd
        owner.Buffer[offset]

    // Slicing is available as https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/arrays
    member this.GetSlice (start0: int option, stop0: int option, start1: int option, stop1: int option, start2: int option, stop2: int option) : Image<'T> =
        use filter = new itk.simple.SliceImageFilter()
        let x0, x1Inner = clampStartStop this start0 stop0 start1 stop1 start2 stop2
        let x1 = List.map ((+) 1) x1Inner // SetStop does not include coordinates
        filter.SetStart(x0 |> toVectorInt32)
        filter.SetStop(x1 |> toVectorInt32)
        let img = filter.Execute (this.toSimpleITK())
        Image<'T>.ofSimpleITKNDispose(img,"GetSlice")

    member this.Set (coords: uint list) (value: 'T) : unit =
        if pooledComponents <> 1u || coords.Length <> pooledSize.Length then
            invalidOp "Set currently supports scalar ArrayPool-backed images only."
        let offset =
            (pooledSize, coords)
            ||> List.zip
            |> List.fold (fun (stride, acc) (dim, coord) -> stride * int dim, acc + int coord * stride) (1, 0)
            |> snd
        owner.Buffer[offset] <- value

    member this.SetSlice (start0: int option, stop0: int option, start1: int option, stop1: int option, start2: int option, stop2: int option) (src: Image<'T>): unit=
        use filter = new itk.simple.PasteImageFilter()
        let x0, x1 = clampStartStop this start0 stop0 start1 stop1 start2 stop2
        let sz = (x0,x1) ||> List.zip |> List.map (fun (a,b) -> b-a+1 |> uint)
        filter.SetDestinationIndex(x0 |> toVectorInt32)
        filter.SetSourceIndex([0;0;0] |> toVectorInt32)
        filter.SetSourceSize(sz |> toVectorUInt32)
        use pasted = filter.Execute (this.toSimpleITK(),src.toSimpleITK())
        let replacement = Image<'T>.ofSimpleITKNDispose(pasted, "SetSlice", this.index)
        match replacement.TryGetPooled1D() with
        | Some(buffer, logicalLength, size, components) ->
            let newBuffer = ArrayPool<'T>.Shared.Rent(logicalLength)
            Array.Copy(buffer, newBuffer, logicalLength)
            this.SetPooled1D(PooledBufferOwner(newBuffer, logicalLength), size, components)
        | None -> invalidOp "SetSlice expected an ArrayPool-backed replacement."
        replacement.decRefCount()

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
        use filter = new itk.simple.EqualImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isEqual", f1.index)
    static member eq (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isEqual(f1, f2)).forAll equalOne

    static member isNotEqual (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.NotEqualImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isNotEqual", f1.index)
    static member neq (f1: Image<'S>, f2: Image<'S>) =
        (Image<float>.isNotEqual(f1, f2)).forAll equalOne

    static member isLessThan (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.LessImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isLessThan", f1.index)
    static member lt (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isLessThan(f1, f2)).forAll equalOne

    static member isLessThanEqual (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.LessEqualImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isLessThanEqual", f1.index)
    static member lte (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isLessThanEqual(f1, f2)).forAll equalOne

    static member isGreater (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.GreaterImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isGreater", f1.index)
    static member gt (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isGreater(f1, f2)).forAll equalOne

    // greater than or equal
    static member isGreaterEqual (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.GreaterEqualImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isGreaterEqual", f1.index)
    static member gte (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isGreaterEqual(f1, f2)).forAll equalOne

    // Power (no direct operator for ** in .NET) - provide a named method instead
    static member Pow (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.PowImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"Pow", f1.index)

    // Bitwise AND ( &&& )
    static member op_BitwiseAnd (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.AndImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"op_BitwiseAnd", f1.index)

    // Bitwise XOR ( ^^^ )
    static member op_ExclusiveOr (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.XorImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"op_ExclusiveOr", f1.index)

    // Bitwise OR ( ||| )
    static member op_BitwiseOr (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.OrImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"op_BitwiseOr", f1.index)

    // Unary bitwise NOT ( ~~~ )
    static member op_LogicalNot (f: Image<'S>) =
        use filter = new itk.simple.InvertIntensityImageFilter()
        Image<'S>.ofSimpleITKNDispose(filter.Execute(f.toSimpleITK()),"op_LogicalNot", f.index)

let Re (img: Image<System.Numerics.Complex>) : Image<float> =
    let realItk = extractComplexRealImage (img.toSimpleITK())
    Image<float>.ofSimpleITKNDispose(realItk, "Re", img.index)

let Im (img: Image<System.Numerics.Complex>) : Image<float> =
    let imagItk = extractComplexImagImage (img.toSimpleITK())
    Image<float>.ofSimpleITKNDispose(imagItk, "Im", img.index)

let modulus (img: Image<System.Numerics.Complex>) : Image<float> =
    use filter = new itk.simple.ComplexToModulusImageFilter()
    let modulusItk = filter.Execute(img.toSimpleITK())
    Image<float>.ofSimpleITKNDispose(modulusItk, "modulus", img.index)

let arg (img: Image<System.Numerics.Complex>) : Image<float> =
    use filter = new itk.simple.ComplexToPhaseImageFilter()
    let phaseItk = filter.Execute(img.toSimpleITK())
    Image<float>.ofSimpleITKNDispose(phaseItk, "arg", img.index)

let toComplex (realImg: Image<float>) (imagImg: Image<float>) : Image<System.Numerics.Complex> =
    Image<float>.ofImagePairToComplex realImg imagImg

let polarToComplex (modulusImg: Image<float>) (argImg: Image<float>) : Image<System.Numerics.Complex> =
    use filter = new itk.simple.MagnitudeAndPhaseToComplexImageFilter()
    let complexItk = filter.Execute(modulusImg.toSimpleITK(), argImg.toSimpleITK())
    Image<System.Numerics.Complex>.ofSimpleITKNDispose(complexItk, "polarToComplex", modulusImg.index)

let conjugate (img: Image<System.Numerics.Complex>) : Image<System.Numerics.Complex> =
    let realImg = Re img
    let imagImg = Im img
    use negate = new itk.simple.MultiplyImageFilter()
    let negImagItk = negate.Execute(imagImg.toSimpleITK(), -1.0)
    let negImagImg = Image<float>.ofSimpleITKNDispose(negImagItk, "conjugate.Im", img.index)
    try
        toComplex realImg negImagImg
    finally
        realImg.decRefCount()
        imagImg.decRefCount()
        negImagImg.decRefCount()
