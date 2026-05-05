module Image
open FSharp.Collections

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

    let ofCastItk<'T> (itkImg: itk.simple.Image) : itk.simple.Image =
        let expectedId = fromType<'T>
        if typeof<'T> = typeof<System.Numerics.Complex> && isComplexCompatibleImage itkImg then
            itkImg
        elif itkImg.GetPixelID() = expectedId then
            itkImg // No casting needed
        else
            use cast = new itk.simple.CastImageFilter()
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
        if isListType && numberComp < 2u then new itk.simple.Image(sz |> toVectorUInt32, itkId, 2u)
        elif isListType && numberComp > 1u then  new itk.simple.Image(sz |> toVectorUInt32, itkId)
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
        printfn $"Added debugging of Image class"
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
        let noComponent = sitk.GetNumberOfComponentsPerPixel()
        let bytesPerComponent = getBytesPerSItkComponent (sitk.GetPixelID())
        let size = sitk.GetSize() |> fromVectorUInt32
        let res = bytesPerComponent * (size |> List.reduce ( * ));
        res

    static member memoryEstimate (width: uint) (height: uint) =
        let bytesPerComponent = getBytesPerSItkComponent (fromType<'T>)
        uint64 (bytesPerComponent * width * height);

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

        let itkImgCast = ofCastItk<'T> itkImg
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

    static member ofArray2D (arr: 'T[,], ?name:string) : Image<'T> =
        let _name = defaultArg name ""
        let sz = [arr.GetLength(0); arr.GetLength(1)] |> List.map uint
        let img = new Image<'T>(sz,1u,_name)
        img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1]])
        img

    static member ofArray3D (arr: 'T[,,], ?name:string) : Image<'T> =
        let _name = defaultArg name ""
        let sz = [arr.GetLength(0); arr.GetLength(1); arr.GetLength(2)] |> List.map uint
        let img = new Image<'T>(sz,1u,_name)
        img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1],int idxLst[2]])
        img

    static member ofArray3DVector (arr: 'S[,,], ?name:string) : Image<'S list> =
        let _name = defaultArg name ""
        let w = arr.GetLength(0)
        let h = arr.GetLength(1)
        let c = arr.GetLength(2)
        let img = new Image<'S list>([uint w; uint h], uint c, _name)
        for i0 in 0 .. w - 1 do
            for i1 in 0 .. h - 1 do
                let comps = [ for k in 0 .. c - 1 -> arr[i0, i1, k] ]
                img.Set [uint i0; uint i1] comps
        img

    static member ofArray3DComplex (arr: float[,,], ?name:string) : Image<System.Numerics.Complex> =
        let _name = defaultArg name ""
        let w = arr.GetLength(0)
        let h = arr.GetLength(1)
        let c = arr.GetLength(2)
        if c <> 2 then invalidArg "arr" "ofArray3DComplex expects last dimension size 2 (real, imag)."
        let img = new Image<System.Numerics.Complex>([uint w; uint h], 1u, _name)
        for i0 in 0 .. w - 1 do
            for i1 in 0 .. h - 1 do
                let cplx = System.Numerics.Complex(arr[i0, i1, 0], arr[i0, i1, 1])
                img.Set [uint i0; uint i1] cplx
        img

    static member ofArray4D (arr: 'T[,,,], ?name:string) : Image<'T> =
        let _name = defaultArg name ""
        let sz = [arr.GetLength(0); arr.GetLength(1); arr.GetLength(2); arr.GetLength(3)] |> List.map uint
        let img = new Image<'T>(sz,1u,_name)
        img |> Image.iteri (fun idxLst _ -> img.Set idxLst arr[int idxLst[0],int idxLst[1],int idxLst[2],int idxLst[3]])
        img

    static member ofArray4DVector (arr: 'S[,,,], ?name:string) : Image<'S list> =
        let _name = defaultArg name ""
        let w = arr.GetLength(0)
        let h = arr.GetLength(1)
        let d = arr.GetLength(2)
        let c = arr.GetLength(3)
        let img = new Image<'S list>([uint w; uint h; uint d], uint c, _name)
        for i0 in 0 .. w - 1 do
            for i1 in 0 .. h - 1 do
                for i2 in 0 .. d - 1 do
                    let comps = [ for k in 0 .. c - 1 -> arr[i0, i1, i2, k] ]
                    img.Set [uint i0; uint i1; uint i2] comps
        img

    static member ofArray4DComplex (arr: float[,,,], ?name:string) : Image<System.Numerics.Complex> =
        let _name = defaultArg name ""
        let w = arr.GetLength(0)
        let h = arr.GetLength(1)
        let d = arr.GetLength(2)
        let c = arr.GetLength(3)
        if c <> 2 then invalidArg "arr" "ofArray4DComplex expects last dimension size 2 (real, imag)."
        let img = new Image<System.Numerics.Complex>([uint w; uint h; uint d], 1u, _name)
        for i0 in 0 .. w - 1 do
            for i1 in 0 .. h - 1 do
                for i2 in 0 .. d - 1 do
                    let cplx = System.Numerics.Complex(arr[i0, i1, i2, 0], arr[i0, i1, i2, 1])
                    img.Set [uint i0; uint i1; uint i2] cplx
        img

    member this.toArray2D (): 'T[,] =
        let sz = this.GetSize() |> List.map int
        Array2D.init sz[0] sz[1] (fun i0 i1 -> this.Get([uint i0; uint i1]))

    static member toArray3DVector<'S when 'S: equality> (img: Image<'S list>) : 'S[,,] =
        if img.GetDimensions() <> 2u then
            failwith $"toArray3DVector: image must be 2D, got {img.GetDimensions()}D"
        let sz = img.GetSize() |> List.map int
        let comps = img.GetNumberOfComponentsPerPixel() |> int
        Array3D.init sz[0] sz[1] comps (fun i0 i1 k ->
            let lst = img.Get([uint i0; uint i1])
            lst[k])

    member this.toArray3D (): 'T[,,] =
        let sz = this.GetSize() |> List.map int
        Array3D.init sz[0] sz[1] sz[2] (fun i0 i1 i2 -> this.Get([uint i0; uint i1; uint i2]))

    member this.toArray4D (): 'T[,,,] =
        let sz = this.GetSize() |> List.map int
        Array4D.init sz[0] sz[1] sz[2] sz[3] (fun i0 i1 i2 i3 -> this.Get([uint i0; uint i1; uint i2; uint i3]))

    static member toArray4DVector<'S when 'S: equality> (img: Image<'S list>) : 'S[,,,] =
        if img.GetDimensions() <> 3u then
            failwith $"toArray4DVector: image must be 3D, got {img.GetDimensions()}D"
        let sz = img.GetSize() |> List.map int
        let comps = img.GetNumberOfComponentsPerPixel() |> int
        Array4D.init sz[0] sz[1] sz[2] comps (fun i0 i1 i2 k ->
            let lst = img.Get([uint i0; uint i1; uint i2])
            lst[k])

    // Make a multicomponent image of a list
    static member ofImageList (images: Image<'S> list) : Image<'S list> =
        let itkImages = images |> List.map (fun img -> img.toSimpleITK())
        use filter = new itk.simple.ComposeImageFilter()
        match itkImages with // seems no other way than unrolling them manually
        | [i1; i2] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2),"ofImageList",images[0].index)
        | [i1; i2; i3] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2, i3),"ofImageList",images[0].index)
        | [i1; i2; i3; i4] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2, i3, i4),"ofImageList",images[0].index)
        | [i1; i2; i3; i4; i5] ->
            Image<'S list>.ofSimpleITK(filter.Execute(i1, i2, i3, i4, i5),"ofImageList",images[0].index)
        | [] ->
            invalidArg "images" "At least two images are required for ComposeImageFilter."
        | _ ->
            invalidArg "images" "ofImageList supports up to 5 images."

    static member ofImagePairToComplex (realImg: Image<float>) (imagImg: Image<float>) : Image<System.Numerics.Complex> =
        if realImg.GetSize() <> imagImg.GetSize() then
            invalidArg "imagImg" $"Image dimensions must match: {realImg.GetSize()} vs {imagImg.GetSize()}"

        let result = new Image<System.Numerics.Complex>(realImg.GetSize(), 1u, "ofImagePairToComplex", realImg.index)
        realImg.GetSize()
        |> flatIndices
        |> Seq.iter (fun idx ->
            result.Set idx (System.Numerics.Complex(realImg.Get idx, imagImg.Get idx)))
        result

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
        | true, _, _ when not (isComplexCompatibleImage itkImg) ->
            failwithf "Pixel type '%O' expects a native complex image, but got %O with %d component(s)." tType (itkImg.GetPixelID()) numComp
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
        let name = defaultArg optionalName filename
        let index = defaultArg optionalIndex 0

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        let itkImg = reader.Execute()
        if not (isComplexCompatibleImage itkImg) then
            failwithf "Complex pixel type expects a native complex image, but got %O with %d component(s)." (itkImg.GetPixelID()) (itkImg.GetNumberOfComponentsPerPixel())
        Image<System.Numerics.Complex>.ofSimpleITK(itkImg, name, index)

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

    static member fold2 (f:'S->'T->'T->'S) (acc0: 'S) (im1: Image<'T>) (im2: Image<'T>) : 'S = 
        let sz1 = im1.GetSize()
        let sz2 = im2.GetSize()
        if List.exists2 (<>) sz1 sz2 then failwith "[Image.fold2] cannot fold over 2 images of unequal sizes"
        sz1
        |> flatIndices
        |> Seq.fold (fun acc idx -> (im1.Get idx, im2.Get idx) ||> f acc) acc0

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
