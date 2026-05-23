module Image
open FSharp.Collections
open System
open System.Runtime.InteropServices

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
        | [re; im] -> System.Numerics.Complex(float re, float im)
        | [re] -> System.Numerics.Complex(float re, 0.0)
        | [] -> System.Numerics.Complex(0.0, 0.0)
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

    let ofCastItk<'T> (itkImg: itk.simple.Image) : itk.simple.Image =
        let expectedId = fromType<'T>
        if typeof<'T> = typeof<System.Numerics.Complex> && isComplexCompatibleImage itkImg then
            new itk.simple.Image(itkImg)
        elif itkImg.GetPixelID() = expectedId then
            new itk.simple.Image(itkImg) // Preserve independent wrapper ownership when no cast is needed.
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
        elif    t = fromType<System.Numerics.Complex>                 then
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
            let re = realFilter.Execute(img) |> getComponent
            let im = imagFilter.Execute(img) |> getComponent
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

[<StructuredFormatDisplay("{Display}")>] // Prevent fsi printing information about its members such as img
type Image<'T when 'T : equality>(sz: uint list, ?optionalNumberComponents: uint, ?optionalName: string, ?optionalIndex: int, ?optionalQuiet: bool) =
    do if sz.Length > 3 then invalidArg "sz" $"Image supports at most 3 dimensions; got {sz.Length}."
    let isComplexType = typeof<'T> = typeof<System.Numerics.Complex>
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

    let itkId = fromType<'T>
    let isListType = typeof<'T>.IsGenericType && typeof<'T>.GetGenericTypeDefinition() = typedefof<list<_>>
    let mutable img = 
        if isListType then new itk.simple.Image(sz |> toVectorUInt32, itkId, max 2u numberComp)
        else new itk.simple.Image(sz |> toVectorUInt32, itkId, numberComp)
    do incMemUsed (Image<_>.memoryEstimateSItk img)
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
    do if debug && not quiet then printDebugMessage $"Created {name} ({img.GetSize()|> fromVectorUInt32}, {fromType<'T>} {img.GetPixelID()}, {img.GetNumberOfComponentsPerPixel()}->{Image<'T>.memoryEstimateSItk img})"
    let now = System.DateTime.UtcNow.ToString("HH:mm:ss.ffffff'Z'")

    static member setDebugLevel level =
        debugLevel <- level
        if level >= 2u then resetRssProbe()
        debug <- level > 0u
    static member setDebug d =
        Image<'T>.setDebugLevel(if d then 1u else 0u)
    member this.Image = img
    member this.Name = name
    member val index = idx with get, set
    member private this.SetImg (itkImg: itk.simple.Image) : unit =
        img <- itkImg
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
            decMemUsed (Image<_>.memoryEstimateSItk img)
            if debug then printDebugMessage $"Disposed of {this.Name}"
            img.Dispose()
            img <- new itk.simple.Image([0u;0u] |> toVectorUInt32, itkId)

    static member memoryEstimateSItk (sitk : itk.simple.Image) = 
        let facts = ImageFacts.ofSimpleITK sitk
        if facts.MemoryBytes > uint64 System.UInt32.MaxValue then
            System.UInt32.MaxValue
        else
            uint32 facts.MemoryBytes

    static member memoryEstimate (width: uint) (height: uint) =
        ImageFacts.sliceBytesForType<'T> width height

    member this.GetSize () = img.GetSize() |> fromVectorUInt32
    member this.GetFacts () = ImageFacts.ofSimpleITK img
    member this.GetMemoryBytes () = this.GetFacts().MemoryBytes
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
        sprintf "%s %s<%s=%A>" szStr vecStr (typeof<'T>.Name) (img.GetPixelID())
    member this.Display = this.ToString() // related to [<StructuredFormatDisplay>]

    interface System.IEquatable<Image<'T>> with
        member this.Equals(other: Image<'T>) = Image<'T>.eq(this, other)

    interface System.IComparable with
        member this.CompareTo(obj: obj) =
            match obj with
            | :? Image<'T> as other -> this.CompareTo(other)
            | _ -> invalidArg "obj" "Expected Image<'T>"

    static member ofSimpleITK (itkImg: itk.simple.Image, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName ""
        let index = defaultArg optionalIndex 0

        let itkImgCast = ofCastItk<'T> itkImg |> canonicalizeSimpleItkImage
        if typeof<'T> = typeof<System.Numerics.Complex> && not (isComplexCompatibleImage itkImgCast) then
            invalidArg "itkImg" "Complex pixel type requires a native complex image."
        let img = new Image<'T>([0u;0u],itkImgCast.GetNumberOfComponentsPerPixel(),name,index, true)

        img.SetImg itkImgCast
        incMemUsed (Image<_>.memoryEstimateSItk img.Image)
        if debug then printDebugMessage $"Created {img.Name} ({itkImgCast.GetSize()|> fromVectorUInt32}, {fromType<'T>} {itkImgCast.GetPixelID()}, {itkImgCast.GetNumberOfComponentsPerPixel()}->{Image<'T>.memoryEstimateSItk itkImgCast})"
        img

    member this.toSimpleITK () : itk.simple.Image =
        if typeof<'T> = typeof<System.Numerics.Complex> && not (isComplexCompatibleImage img) then
            invalidOp "Complex pixel type requires a native complex image."
        img

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
            let pixels = Array.zeroCreate<'T> (width * height)
            let mutable offset = 0
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    pixels[offset] <- arr[x, y]
                    offset <- offset + 1

            Image<'T>.ofSimpleITK(importScalarImage sz pixels, _name, _index)
        else
            let img = new Image<'T>(sz,1u,_name,_index)
            img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1]])
            img

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
            let pixels = Array.zeroCreate<'T> (width * height * depth)
            let mutable offset = 0
            for z in 0 .. depth - 1 do
                for y in 0 .. height - 1 do
                    for x in 0 .. width - 1 do
                        pixels[offset] <- arr[x, y, z]
                        offset <- offset + 1

            Image<'T>.ofSimpleITK(importScalarImage sz pixels, _name, _index)
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

                Image<'T>.ofSimpleITK(filter.Execute(v), _name, _index)
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
            Image<'S list>.ofSimpleITK(filter.Execute(v), _name, _index)
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
        if this.GetDimensions() = 2u && isScalarImportSupported<'T> && this.Image.GetPixelID() = fromType<'T> then
            let width = sz[0]
            let height = sz[1]
            let pixels = copyScalarPixels<'T> this.Image (width * height)
            Array2D.init width height (fun x y -> pixels[y * width + x])
        else
            Array2D.init sz[0] sz[1] (fun i0 i1 -> this.Get([uint i0; uint i1]))

    member this.toComplexArray2D () : System.Numerics.Complex[,] =
        if typeof<'T> <> typeof<System.Numerics.Complex> then
            invalidOp "toComplexArray2D: image pixel type must be System.Numerics.Complex."
        if this.GetDimensions() <> 2u then
            invalidOp $"toComplexArray2D: image must be 2D, got {this.GetDimensions()}D."

        use realItk = extractComplexRealImage this.Image
        use imagItk = extractComplexImagImage this.Image
        let realImg = Image<float>.ofSimpleITK(realItk, "toComplexArray2D.Re", this.index)
        let imagImg = Image<float>.ofSimpleITK(imagItk, "toComplexArray2D.Im", this.index)
        let real = realImg.toArray2D()
        let imag = imagImg.toArray2D()
        realImg.decRefCount()
        imagImg.decRefCount()
        Array2D.init (real.GetLength 0) (real.GetLength 1) (fun x y -> System.Numerics.Complex(real[x, y], imag[x, y]))

    static member toArray3DVector<'S when 'S: equality> (img: Image<'S list>) : 'S[,,] =
        if img.GetDimensions() <> 2u then
            failwith $"toArray3DVector: image must be 2D, got {img.GetDimensions()}D"
        use filter = new itk.simple.VectorIndexSelectionCastImageFilter()
        let components =
            [ for i in 0 .. int (img.GetNumberOfComponentsPerPixel()) - 1 ->
                filter.SetIndex(uint i)
                let scalarItk = filter.Execute(img.toSimpleITK())
                Image<'S>.ofSimpleITK(scalarItk, $"toArray3DVector.Component{i}", img.index + i) ]
        try
            let values = components |> List.map (fun image -> image.toArray2D())
            let width = values.Head.GetLength(0)
            let height = values.Head.GetLength(1)
            Array3D.init width height values.Length (fun i0 i1 k -> values[k][i0, i1])
        finally
            components |> List.iter (fun image -> image.decRefCount())

    member this.toArray3D (): 'T[,,] =
        let sz = this.GetSize() |> List.map int
        if this.GetDimensions() = 3u && isScalarImportSupported<'T> && this.Image.GetPixelID() = fromType<'T> then
            let width = sz[0]
            let height = sz[1]
            let depth = sz[2]
            let pixels = copyScalarPixels<'T> this.Image (width * height * depth)
            Array3D.init width height depth (fun x y z -> pixels[(z * height + y) * width + x])
        else
            Array3D.init sz[0] sz[1] sz[2] (fun i0 i1 i2 -> this.Get([uint i0; uint i1; uint i2]))

    member this.toArray4D (): 'T[,,,] =
        let sz = this.GetSize() |> List.map int
        if this.GetDimensions() = 4u && isScalarImportSupported<'T> && this.Image.GetPixelID() = fromType<'T> then
            let width = sz[0]
            let height = sz[1]
            let depth = sz[2]
            let length = sz[3]
            let pixels = copyScalarPixels<'T> this.Image (width * height * depth * length)
            Array4D.init width height depth length (fun x y z t -> pixels[((t * depth + z) * height + y) * width + x])
        else
            Array4D.init sz[0] sz[1] sz[2] sz[3] (fun i0 i1 i2 i3 -> this.Get([uint i0; uint i1; uint i2; uint i3]))

    member this.toComplexArray3D () : System.Numerics.Complex[,,] =
        if typeof<'T> <> typeof<System.Numerics.Complex> then
            invalidOp "toComplexArray3D: image pixel type must be System.Numerics.Complex."
        if this.GetDimensions() <> 3u then
            invalidOp $"toComplexArray3D: image must be 3D, got {this.GetDimensions()}D."

        use realItk = extractComplexRealImage this.Image
        use imagItk = extractComplexImagImage this.Image
        let realImg = Image<float>.ofSimpleITK(realItk, "toComplexArray3D.Re", this.index)
        let imagImg = Image<float>.ofSimpleITK(imagItk, "toComplexArray3D.Im", this.index)
        let real = realImg.toArray3D()
        let imag = imagImg.toArray3D()
        realImg.decRefCount()
        imagImg.decRefCount()
        Array3D.init (real.GetLength 0) (real.GetLength 1) (real.GetLength 2) (fun x y z -> System.Numerics.Complex(real[x, y, z], imag[x, y, z]))

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
            Image<'S list>.ofSimpleITK(filter.Execute(v), "ofImageList", first.index)

    static member ofImagePairToComplex (realImg: Image<float>) (imagImg: Image<float>) : Image<System.Numerics.Complex> =
        if realImg.GetSize() <> imagImg.GetSize() then
            invalidArg "imagImg" $"Image dimensions must match: {realImg.GetSize()} vs {imagImg.GetSize()}"

        use filter = new itk.simple.RealAndImaginaryToComplexImageFilter()
        Image<System.Numerics.Complex>.ofSimpleITK(filter.Execute(realImg.toSimpleITK(), imagImg.toSimpleITK()), "ofImagePairToComplex", realImg.index)

    member this.toImageList () : Image<'S> list =
        use filter = new itk.simple.VectorIndexSelectionCastImageFilter()
        let n = this.Image.GetNumberOfComponentsPerPixel() |> int
        List.init n (fun i ->
            filter.SetIndex(uint i)
            let scalarItk = filter.Execute(this.Image)
            Image<'S>.ofSimpleITK(scalarItk,"toImageList",this.index+i)
        )

    static member ofFile(filename: string, ?optionalName: string, ?optionalIndex: int) : Image<'T> =
        let name = defaultArg optionalName filename
        let index = defaultArg optionalIndex 0

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        let itkImg = reader.Execute()
        let numComp = itkImg.GetNumberOfComponentsPerPixel()
        let tType = typeof<'T>
        let isComplexType = tType = typeof<System.Numerics.Complex>
        let isVectorType =
            (tType.IsGenericType && tType.GetGenericTypeDefinition() = typedefof<list<_>>)
            || tType.IsArray

        // Validate number of components matches expectations
        match isComplexType, isVectorType, numComp with
        | true, _, _ when isComplexCompatibleImage itkImg ->
            Image<'T>.ofSimpleITK(itkImg, name, index)
        | true, _, n when n >= 2u ->
            let vector = Image<float list>.ofSimpleITK(itkImg, name + ".vector", index)
            let parts = vector.toImageList()
            if parts.Length < 2 then
                vector.decRefCount()
                parts |> List.iter (fun part -> part.decRefCount())
                failwithf "Pixel type '%O' expects native complex or a vector with real and imaginary components, but image has %d component(s)." tType n
            let complex = Image<float>.ofImagePairToComplex parts[0] parts[1]
            vector.decRefCount()
            parts |> List.iter (fun part -> part.decRefCount())
            complex |> box |> unbox<Image<'T>>
        | true, _, n ->
            failwithf "Pixel type '%O' expects native complex or a vector with real and imaginary components, but got %O with %d component(s)." tType (itkImg.GetPixelID()) n
        | false, true, n when n < 2u ->
            failwithf "Pixel type '%O' expects a vector (>=2 components), but image has %d component(s)." tType n
        | false, false, n when n > 1u ->
            failwithf "Pixel type '%O' expects a scalar (1 component), but image has %d component(s)." tType n
        | _ ->
            Image<'T>.ofSimpleITK(itkImg,name,index)

    static member ofFileVector<'S when 'S: equality> (filename: string, ?optionalName: string, ?optionalIndex: int) : Image<'S list> =
        let name = defaultArg optionalName filename
        let index = defaultArg optionalIndex 0

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        let itkImg = reader.Execute()
        let numComp = itkImg.GetNumberOfComponentsPerPixel()
        if numComp < 2u then
            failwithf "Vector pixel type expects >=2 components, but image has %d component(s)." numComp
        Image<'S list>.ofSimpleITK(itkImg, name, index)

    static member ofFileComplex (filename: string, ?optionalName: string, ?optionalIndex: int) : Image<System.Numerics.Complex> =
        Image<System.Numerics.Complex>.ofFile(filename, ?optionalName = optionalName, ?optionalIndex = optionalIndex)

    member this.toFile(filename: string, ?optionalFormat: string) =
        use writer = new itk.simple.ImageFileWriter()
        writer.SetFileName(filename)
        match optionalFormat with
        | Some fmt -> writer.SetImageIO(fmt)
        | None -> ()
        writer.Execute(this.Image)

    member this.toFileVector(filename: string, ?optionalFormat: string) =
        let isListType = typeof<'T>.IsGenericType && typeof<'T>.GetGenericTypeDefinition() = typedefof<list<_>>
        if not isListType then
            failwith "toFileVector: image pixel type must be a list for vector components."
        if this.GetNumberOfComponentsPerPixel() < 2u then
            failwithf "toFileVector: expected >=2 components per pixel, got %d." (this.GetNumberOfComponentsPerPixel())
        this.toFile(filename, ?optionalFormat = optionalFormat)

    member this.toFileComplex(filename: string, ?optionalFormat: string) =
        if typeof<'T> <> typeof<System.Numerics.Complex> then
            failwith "toFileComplex: image pixel type must be System.Numerics.Complex."
        if not (isComplexCompatibleImage this.Image) then
            failwithf "toFileComplex: expected a native complex image, got %O with %d component(s)." (this.Image.GetPixelID()) (this.GetNumberOfComponentsPerPixel())
        this.toFile(filename, ?optionalFormat = optionalFormat)

    // Arithmatic
    static member (+) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.AddImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "add")
    static member (-) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.SubtractImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "subtract")
    static member ( * ) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.MultiplyImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "multiply")
    static member (/) (f1: Image<'T>, f2: Image<'T>) =
        use filter = new itk.simple.DivideImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "divide")

    static member maximumImage (f1: Image<'T>) (f2: Image<'T>) =
        use filter = new itk.simple.MaximumImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "maximumImage")

    static member minimumImage (f1: Image<'T>) (f2: Image<'T>) =
        use filter = new itk.simple.MinimumImageFilter()
        Image<'T>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()), "minimumImage")

    static member getMinMax (img: Image<'T>) =
        use filter = new itk.simple.MinimumMaximumImageFilter()
        filter.Execute (img.toSimpleITK())
        (filter.GetMinimum(),filter.GetMaximum())

    // Collection type
    static member map (f:'T->'T) (im1: Image<'T>) : Image<'T> =
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
        let u = coords |> toVectorUInt32
        if typeof<'T> = typeof<System.Numerics.Complex> then
            let pid = this.Image.GetPixelID()
            if not (isComplexPixelId pid) then
                failwithf "Unsupported complex backing pixel type: %O" pid
            use realFilter = new itk.simple.ComplexToRealImageFilter()
            use imagFilter = new itk.simple.ComplexToImaginaryImageFilter()
            let re = realFilter.Execute(this.Image) |> fun img -> getFloatPixel img u
            let im = imagFilter.Execute(this.Image) |> fun img -> getFloatPixel img u
            let c = System.Numerics.Complex(re, im)
            unbox<'T> c
        elif typeof<'T> = typeof<uint8 list> then
            this.Image.GetPixelAsVectorUInt8(u) |> fromVectorUInt8 |> box :?> 'T
        elif typeof<'T> = typeof<int8 list> then
            this.Image.GetPixelAsVectorInt8(u) |> fromVectorInt8 |> box :?> 'T
        elif typeof<'T> = typeof<uint16 list> then
            this.Image.GetPixelAsVectorUInt16(u) |> fromVectorUInt16 |> box :?> 'T
        elif typeof<'T> = typeof<int16 list> then
            this.Image.GetPixelAsVectorInt16(u) |> fromVectorInt16 |> box :?> 'T
        elif typeof<'T> = typeof<uint32 list> then
            this.Image.GetPixelAsVectorUInt32(u) |> fromVectorUInt32 |> box :?> 'T
        elif typeof<'T> = typeof<int32 list> then
            this.Image.GetPixelAsVectorInt32(u) |> fromVectorInt32 |> box :?> 'T
        elif typeof<'T> = typeof<uint64 list> then
            this.Image.GetPixelAsVectorUInt64(u) |> fromVectorUInt64 |> box :?> 'T
        elif typeof<'T> = typeof<int64 list> then
            this.Image.GetPixelAsVectorInt64(u) |> fromVectorInt64 |> box :?> 'T
        elif typeof<'T> = typeof<float32 list> then
            this.Image.GetPixelAsVectorFloat32(u) |> fromVectorFloat32 |> box :?> 'T
        elif typeof<'T> = typeof<float list> then
            this.Image.GetPixelAsVectorFloat64(u) |> fromVectorFloat64 |> box :?> 'T
        else
            let t = fromType<'T>
            let raw = getBoxedPixel this.Image t u
            raw :?> 'T

    // Slicing is available as https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/arrays
    member this.GetSlice (start0: int option, stop0: int option, start1: int option, stop1: int option, start2: int option, stop2: int option) : Image<'T> =
        use filter = new itk.simple.SliceImageFilter()
        let x0, x1Inner = clampStartStop this start0 stop0 start1 stop1 start2 stop2
        let x1 = List.map ((+) 1) x1Inner // SetStop does not include coordinates
        filter.SetStart(x0 |> toVectorInt32)
        filter.SetStop(x1 |> toVectorInt32)
        let img = filter.Execute (this.toSimpleITK())
        let res = Image<'T>.ofSimpleITK(img,"GetSlice")
        res

    member this.Set (coords: uint list) (value: 'T) : unit =
        let u = toVectorUInt32 coords
        if typeof<'T> = typeof<System.Numerics.Complex> then
            let c = unbox<System.Numerics.Complex> value
            let pid = this.Image.GetPixelID()
            if not (isComplexPixelId pid) then
                failwithf "Unsupported complex backing pixel type: %O" pid
            use realFilter = new itk.simple.ComplexToRealImageFilter()
            use imagFilter = new itk.simple.ComplexToImaginaryImageFilter()
            use compose = new itk.simple.RealAndImaginaryToComplexImageFilter()
            use realImg = realFilter.Execute(this.Image)
            use imagImg = imagFilter.Execute(this.Image)
            setFloatPixel realImg u c.Real
            setFloatPixel imagImg u c.Imaginary
            img <- compose.Execute(realImg, imagImg)
        else
            let t = fromType<'T>
            setBoxedPixel this.Image t u value

    member this.SetSlice (start0: int option, stop0: int option, start1: int option, stop1: int option, start2: int option, stop2: int option) (src: Image<'T>): unit=
        use filter = new itk.simple.PasteImageFilter()
        let x0, x1 = clampStartStop this start0 stop0 start1 stop1 start2 stop2
        let sz = (x0,x1) ||> List.zip |> List.map (fun (a,b) -> b-a+1 |> uint)
        filter.SetDestinationIndex(x0 |> toVectorInt32)
        filter.SetSourceIndex([0;0;0] |> toVectorInt32)
        filter.SetSourceSize(sz |> toVectorUInt32)
        img <- filter.Execute (img,src.toSimpleITK())

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
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isEqual")
    static member eq (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isEqual(f1, f2)).forAll equalOne

    static member isNotEqual (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.NotEqualImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isNotEqual")
    static member neq (f1: Image<'S>, f2: Image<'S>) =
        (Image<float>.isNotEqual(f1, f2)).forAll equalOne

    static member isLessThan (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.LessImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isLessThan")
    static member lt (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isLessThan(f1, f2)).forAll equalOne

    static member isLessThanEqual (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.LessEqualImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isLessThanEqual")
    static member lte (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isLessThanEqual(f1, f2)).forAll equalOne

    static member isGreater (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.GreaterImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isGreater")
    static member gt (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isGreater(f1, f2)).forAll equalOne

    // greater than or equal
    static member isGreaterEqual (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.GreaterEqualImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"isGreaterEqual")
    static member gte (f1: Image<'S>, f2: Image<'S>) =
        (Image<'S>.isGreaterEqual(f1, f2)).forAll equalOne

    // Power (no direct operator for ** in .NET) - provide a named method instead
    static member Pow (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.PowImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"Pow")

    // Bitwise AND ( &&& )
    static member op_BitwiseAnd (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.AndImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"op_BitwiseAnd")

    // Bitwise XOR ( ^^^ )
    static member op_ExclusiveOr (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.XorImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"op_ExclusiveOr")

    // Bitwise OR ( ||| )
    static member op_BitwiseOr (f1: Image<'S>, f2: Image<'S>) =
        use filter = new itk.simple.OrImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f1.toSimpleITK(), f2.toSimpleITK()),"op_BitwiseOr")

    // Unary bitwise NOT ( ~~~ )
    static member op_LogicalNot (f: Image<'S>) =
        use filter = new itk.simple.InvertIntensityImageFilter()
        Image<'S>.ofSimpleITK(filter.Execute(f.toSimpleITK()),"op_LogicalNot")

let Re (img: Image<System.Numerics.Complex>) : Image<float> =
    use realItk = extractComplexRealImage (img.toSimpleITK())
    Image<float>.ofSimpleITK(realItk, "Re", img.index)

let Im (img: Image<System.Numerics.Complex>) : Image<float> =
    use imagItk = extractComplexImagImage (img.toSimpleITK())
    Image<float>.ofSimpleITK(imagItk, "Im", img.index)

let modulus (img: Image<System.Numerics.Complex>) : Image<float> =
    use filter = new itk.simple.ComplexToModulusImageFilter()
    let modulusItk = filter.Execute(img.toSimpleITK())
    Image<float>.ofSimpleITK(modulusItk, "modulus", img.index)

let arg (img: Image<System.Numerics.Complex>) : Image<float> =
    use filter = new itk.simple.ComplexToPhaseImageFilter()
    let phaseItk = filter.Execute(img.toSimpleITK())
    Image<float>.ofSimpleITK(phaseItk, "arg", img.index)

let toComplex (realImg: Image<float>) (imagImg: Image<float>) : Image<System.Numerics.Complex> =
    Image<float>.ofImagePairToComplex realImg imagImg

let polarToComplex (modulusImg: Image<float>) (argImg: Image<float>) : Image<System.Numerics.Complex> =
    use filter = new itk.simple.MagnitudeAndPhaseToComplexImageFilter()
    let complexItk = filter.Execute(modulusImg.toSimpleITK(), argImg.toSimpleITK())
    Image<System.Numerics.Complex>.ofSimpleITK(complexItk, "polarToComplex", modulusImg.index)

let conjugate (img: Image<System.Numerics.Complex>) : Image<System.Numerics.Complex> =
    let realImg = Re img
    let imagImg = Im img
    use negate = new itk.simple.MultiplyImageFilter()
    let negImagItk = negate.Execute(imagImg.toSimpleITK(), -1.0)
    let negImagImg = Image<float>.ofSimpleITK(negImagItk, "conjugate.Im", img.index)
    try
        toComplex realImg negImagImg
    finally
        realImg.decRefCount()
        imagImg.decRefCount()
        negImagImg.decRefCount()
