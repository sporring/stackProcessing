open System
open System.Buffers
open System.Collections.Generic
open System.Diagnostics
open System.Globalization
open System.IO
open System.Runtime.InteropServices

type PixelType =
    | UInt8
    | UInt16
    | Float32

type Shape =
    { Width: int
      Height: int
      Depth: int }

let private invariant = CultureInfo.InvariantCulture

let private parseArgs (args: string array) =
    let rec loop i (acc: Map<string,string>) =
        if i >= args.Length then acc
        elif args[i].StartsWith("--", StringComparison.Ordinal) then
            if i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal) then
                loop (i + 1) (acc.Add(args[i].Substring(2), "true"))
            else
                loop (i + 2) (acc.Add(args[i].Substring(2), args[i + 1]))
        else
            loop (i + 1) acc
    loop 0 Map.empty

let private optional name fallback opts =
    opts |> Map.tryFind name |> Option.defaultValue fallback

let private parseShape (text: string) =
    let parts = text.Split('x', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
    if parts.Length <> 3 then invalidArg "shape" $"Expected WxHxD, got {text}."
    { Width = Int32.Parse(parts[0], invariant)
      Height = Int32.Parse(parts[1], invariant)
      Depth = Int32.Parse(parts[2], invariant) }

let private parsePixelType (text: string) =
    match text.Trim().ToLowerInvariant() with
    | "uint8" -> UInt8
    | "uint16" -> UInt16
    | "float32" -> Float32
    | _ -> invalidArg "pixel-type" $"Unsupported pixel type {text}."

let private shapeText s = $"{s.Width}x{s.Height}x{s.Depth}"

let private length s =
    let n = int64 s.Width * int64 s.Height * int64 s.Depth
    if n > int64 Int32.MaxValue then invalidArg "shape" $"Too many elements for CLR array: {n}."
    int n

let private vectorUInt32 values = new itk.simple.VectorUInt32(values |> Seq.map uint32 |> Seq.toList)
let private vectorDouble values = new itk.simple.VectorDouble(values |> Seq.toList)

let private identityDirection dim =
    [ for row in 0 .. dim - 1 do
        for col in 0 .. dim - 1 do
            if row = col then 1.0 else 0.0 ]

let private setImportBuffer<'T> (importer: itk.simple.ImportImageFilter) (buffer: nativeint) =
    let t = typeof<'T>
    if t = typeof<uint8> then importer.SetBufferAsUInt8(buffer)
    elif t = typeof<uint16> then importer.SetBufferAsUInt16(buffer)
    elif t = typeof<float32> then importer.SetBufferAsFloat(buffer)
    else invalidArg "T" $"Unsupported import type {t.Name}."

let private importedImage<'T> shape (source: 'T[]) =
    use importer = new itk.simple.ImportImageFilter()
    importer.SetSize(vectorUInt32 [ shape.Width; shape.Height; shape.Depth ])
    importer.SetSpacing(vectorDouble [ 1.0; 1.0; 1.0 ])
    importer.SetOrigin(vectorDouble [ 0.0; 0.0; 0.0 ])
    importer.SetDirection(vectorDouble (identityDirection 3))
    let handle = GCHandle.Alloc(source, GCHandleType.Pinned)
    try
        setImportBuffer<'T> importer (handle.AddrOfPinnedObject())
        use imported = importer.Execute()
        let copy = new itk.simple.Image(imported)
        copy.MakeUnique()
        copy
    finally
        handle.Free()

let private fillUInt8 (buffer: uint8[]) n =
    for i in 0 .. n - 1 do
        buffer[i] <- byte ((i * 17 + i / 7) &&& 0xff)

let private fillUInt16 (buffer: uint16[]) n =
    for i in 0 .. n - 1 do
        buffer[i] <- uint16 ((i * 17 + i / 7) &&& 0xffff)

let private fillFloat32 (buffer: float32[]) n =
    for i in 0 .. n - 1 do
        buffer[i] <- float32 ((i * 17 + i / 7) &&& 0xff)

let private measure action =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let before = GC.GetTotalAllocatedBytes(true)
    let sw = Stopwatch.StartNew()
    action()
    sw.Stop()
    let after = GC.GetTotalAllocatedBytes(true)
    sw.Elapsed.TotalSeconds, after - before

let private index width height x y z =
    (z * height + y) * width + x

let private stencil3x3x3Offsets width height =
    let sliceStride = width * height
    [| for dz in -1 .. 1 do
        for dy in -1 .. 1 do
            for dx in -1 .. 1 do
                yield dz * sliceStride + dy * width + dx |]

let inline private insertionSort27 (scratch: ^T[]) =
    let mutable i = 1
    while i < 27 do
        let value = scratch[i]
        let mutable j = i - 1
        while j >= 0 && scratch[j] > value do
            scratch[j + 1] <- scratch[j]
            j <- j - 1
        scratch[j + 1] <- value
        i <- i + 1

let private median3x3x3UInt8 shape (input: uint8[]) (output: uint8[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    let scratch = Array.zeroCreate<uint8> 27
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let baseIndex = index w h x y z
                let mutable k = 0
                while k < 27 do
                    scratch[k] <- input[baseIndex + offsets[k]]
                    k <- k + 1
                insertionSort27 scratch
                output[baseIndex] <- scratch[13]

let private addSparse (hist: SortedDictionary<int, int>) value =
    match hist.TryGetValue(value) with
    | true, count -> hist[value] <- count + 1
    | false, _ -> hist[value] <- 1

let private removeSparse (hist: SortedDictionary<int, int>) value =
    match hist.TryGetValue(value) with
    | true, 1 -> hist.Remove(value) |> ignore
    | true, count -> hist[value] <- count - 1
    | false, _ -> failwith $"Sparse histogram underflow for value {value}."

let private medianSparse (hist: SortedDictionary<int, int>) =
    let mutable cumulative = 0
    let mutable result = 0
    let mutable found = false
    use mutable e = hist.GetEnumerator()
    while not found && e.MoveNext() do
        cumulative <- cumulative + e.Current.Value
        if cumulative >= 14 then
            result <- e.Current.Key
            found <- true
    result

let private medianSparseAt kth (hist: SortedDictionary<int, int>) =
    let mutable cumulative = 0
    let mutable result = 0
    let mutable found = false
    use mutable e = hist.GetEnumerator()
    while not found && e.MoveNext() do
        cumulative <- cumulative + e.Current.Value
        if cumulative >= kth then
            result <- e.Current.Key
            found <- true
    result

[<AllowNullLiteral>]
type private SparseOrderNode(key: int) =
    member val Key = key
    member val Count = 1 with get, set
    member val Total = 1 with get, set
    member val Priority = uint32 key * 2654435761u + 1013904223u
    member val Left: SparseOrderNode = null with get, set
    member val Right: SparseOrderNode = null with get, set

let inline private nodeTotal (node: SparseOrderNode) =
    if isNull node then 0 else node.Total

let inline private refreshNode (node: SparseOrderNode) =
    node.Total <- node.Count + nodeTotal node.Left + nodeTotal node.Right
    node

let private rotateRight (node: SparseOrderNode) =
    let left = node.Left
    node.Left <- left.Right
    left.Right <- refreshNode node
    refreshNode left

let private rotateLeft (node: SparseOrderNode) =
    let right = node.Right
    node.Right <- right.Left
    right.Left <- refreshNode node
    refreshNode right

let rec private insertOrderNode (node: SparseOrderNode) key =
    if isNull node then
        SparseOrderNode(key)
    elif key = node.Key then
        node.Count <- node.Count + 1
        refreshNode node
    elif key < node.Key then
        node.Left <- insertOrderNode node.Left key
        if node.Left.Priority < node.Priority then rotateRight node else refreshNode node
    else
        node.Right <- insertOrderNode node.Right key
        if node.Right.Priority < node.Priority then rotateLeft node else refreshNode node

let rec private mergeOrderNodes (left: SparseOrderNode) (right: SparseOrderNode) =
    if isNull left then right
    elif isNull right then left
    elif left.Priority < right.Priority then
        left.Right <- mergeOrderNodes left.Right right
        refreshNode left
    else
        right.Left <- mergeOrderNodes left right.Left
        refreshNode right

let rec private removeOrderNode (node: SparseOrderNode) key =
    if isNull node then
        failwith $"Sparse order tree underflow for value {key}."
    elif key = node.Key then
        if node.Count > 1 then
            node.Count <- node.Count - 1
            refreshNode node
        else
            mergeOrderNodes node.Left node.Right
    elif key < node.Key then
        node.Left <- removeOrderNode node.Left key
        refreshNode node
    else
        node.Right <- removeOrderNode node.Right key
        refreshNode node

let rec private kthOrderNode (node: SparseOrderNode) k =
    if isNull node then
        failwith "Sparse order tree median lookup on an empty tree."
    let leftTotal = nodeTotal node.Left
    if k <= leftTotal then
        kthOrderNode node.Left k
    elif k <= leftTotal + node.Count then
        node.Key
    else
        kthOrderNode node.Right (k - leftTotal - node.Count)

let private median3x3x3UInt8SparseRollingX shape (input: uint8[]) (output: uint8[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    Array.Clear(output, 0, w * h * d)
    let hist = SortedDictionary<int, int>()
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            hist.Clear()
            for dz in -1 .. 1 do
                for dy in -1 .. 1 do
                    for xx in 0 .. 2 do
                        addSparse hist (int input[index w h xx (y + dy) (z + dz)])
            output[index w h 1 y z] <- byte (medianSparse hist)
            for x in 2 .. w - 2 do
                let removeX = x - 2
                let addX = x + 1
                for dz in -1 .. 1 do
                    for dy in -1 .. 1 do
                        removeSparse hist (int input[index w h removeX (y + dy) (z + dz)])
                        addSparse hist (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- byte (medianSparse hist)

let private median3x3x3UInt8SparseOrderRollingX shape (input: uint8[]) (output: uint8[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            let mutable root: SparseOrderNode = null
            for dz in -1 .. 1 do
                for dy in -1 .. 1 do
                    for xx in 0 .. 2 do
                        root <- insertOrderNode root (int input[index w h xx (y + dy) (z + dz)])
            output[index w h 1 y z] <- byte (kthOrderNode root 14)
            for x in 2 .. w - 2 do
                let removeX = x - 2
                let addX = x + 1
                for dz in -1 .. 1 do
                    for dy in -1 .. 1 do
                        root <- removeOrderNode root (int input[index w h removeX (y + dy) (z + dz)])
                        root <- insertOrderNode root (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- byte (kthOrderNode root 14)

let private median3x3x3UInt16 shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    let scratch = Array.zeroCreate<uint16> 27
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let baseIndex = index w h x y z
                let mutable k = 0
                while k < 27 do
                    scratch[k] <- input[baseIndex + offsets[k]]
                    k <- k + 1
                insertionSort27 scratch
                output[baseIndex] <- scratch[13]

let private median3x3x3UInt16SparseRollingX shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    Array.Clear(output, 0, w * h * d)
    let hist = SortedDictionary<int, int>()
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            hist.Clear()
            for dz in -1 .. 1 do
                for dy in -1 .. 1 do
                    for xx in 0 .. 2 do
                        addSparse hist (int input[index w h xx (y + dy) (z + dz)])
            output[index w h 1 y z] <- uint16 (medianSparse hist)
            for x in 2 .. w - 2 do
                let removeX = x - 2
                let addX = x + 1
                for dz in -1 .. 1 do
                    for dy in -1 .. 1 do
                        removeSparse hist (int input[index w h removeX (y + dy) (z + dz)])
                        addSparse hist (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- uint16 (medianSparse hist)

let private median3x3x3UInt16SparseOrderRollingX shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            let mutable root: SparseOrderNode = null
            for dz in -1 .. 1 do
                for dy in -1 .. 1 do
                    for xx in 0 .. 2 do
                        root <- insertOrderNode root (int input[index w h xx (y + dy) (z + dz)])
            output[index w h 1 y z] <- uint16 (kthOrderNode root 14)
            for x in 2 .. w - 2 do
                let removeX = x - 2
                let addX = x + 1
                for dz in -1 .. 1 do
                    for dy in -1 .. 1 do
                        root <- removeOrderNode root (int input[index w h removeX (y + dy) (z + dz)])
                        root <- insertOrderNode root (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- uint16 (kthOrderNode root 14)

let private medianSparseRollingXUInt8 radius shape (input: uint8[]) (output: uint8[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let side = radius * 2 + 1
    let kth = (side * side * side) / 2 + 1
    Array.Clear(output, 0, w * h * d)
    let hist = SortedDictionary<int, int>()
    for z in radius .. d - radius - 1 do
        for y in radius .. h - radius - 1 do
            hist.Clear()
            for dz in -radius .. radius do
                for dy in -radius .. radius do
                    for xx in 0 .. side - 1 do
                        addSparse hist (int input[index w h xx (y + dy) (z + dz)])
            output[index w h radius y z] <- byte (medianSparseAt kth hist)
            for x in radius + 1 .. w - radius - 1 do
                let removeX = x - radius - 1
                let addX = x + radius
                for dz in -radius .. radius do
                    for dy in -radius .. radius do
                        removeSparse hist (int input[index w h removeX (y + dy) (z + dz)])
                        addSparse hist (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- byte (medianSparseAt kth hist)

let private medianSparseOrderRollingXUInt8 radius shape (input: uint8[]) (output: uint8[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let side = radius * 2 + 1
    let kth = (side * side * side) / 2 + 1
    Array.Clear(output, 0, w * h * d)
    for z in radius .. d - radius - 1 do
        for y in radius .. h - radius - 1 do
            let mutable root: SparseOrderNode = null
            for dz in -radius .. radius do
                for dy in -radius .. radius do
                    for xx in 0 .. side - 1 do
                        root <- insertOrderNode root (int input[index w h xx (y + dy) (z + dz)])
            output[index w h radius y z] <- byte (kthOrderNode root kth)
            for x in radius + 1 .. w - radius - 1 do
                let removeX = x - radius - 1
                let addX = x + radius
                for dz in -radius .. radius do
                    for dy in -radius .. radius do
                        root <- removeOrderNode root (int input[index w h removeX (y + dy) (z + dz)])
                        root <- insertOrderNode root (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- byte (kthOrderNode root kth)

let private medianSparseRollingXUInt16 radius shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let side = radius * 2 + 1
    let kth = (side * side * side) / 2 + 1
    Array.Clear(output, 0, w * h * d)
    let hist = SortedDictionary<int, int>()
    for z in radius .. d - radius - 1 do
        for y in radius .. h - radius - 1 do
            hist.Clear()
            for dz in -radius .. radius do
                for dy in -radius .. radius do
                    for xx in 0 .. side - 1 do
                        addSparse hist (int input[index w h xx (y + dy) (z + dz)])
            output[index w h radius y z] <- uint16 (medianSparseAt kth hist)
            for x in radius + 1 .. w - radius - 1 do
                let removeX = x - radius - 1
                let addX = x + radius
                for dz in -radius .. radius do
                    for dy in -radius .. radius do
                        removeSparse hist (int input[index w h removeX (y + dy) (z + dz)])
                        addSparse hist (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- uint16 (medianSparseAt kth hist)

let private medianSparseOrderRollingXUInt16 radius shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let side = radius * 2 + 1
    let kth = (side * side * side) / 2 + 1
    Array.Clear(output, 0, w * h * d)
    for z in radius .. d - radius - 1 do
        for y in radius .. h - radius - 1 do
            let mutable root: SparseOrderNode = null
            for dz in -radius .. radius do
                for dy in -radius .. radius do
                    for xx in 0 .. side - 1 do
                        root <- insertOrderNode root (int input[index w h xx (y + dy) (z + dz)])
            output[index w h radius y z] <- uint16 (kthOrderNode root kth)
            for x in radius + 1 .. w - radius - 1 do
                let removeX = x - radius - 1
                let addX = x + radius
                for dz in -radius .. radius do
                    for dy in -radius .. radius do
                        root <- removeOrderNode root (int input[index w h removeX (y + dy) (z + dz)])
                        root <- insertOrderNode root (int input[index w h addX (y + dy) (z + dz)])
                output[index w h x y z] <- uint16 (kthOrderNode root kth)

let private median3x3x3Float32 shape (input: float32[]) (output: float32[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    let scratch = Array.zeroCreate<float32> 27
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let baseIndex = index w h x y z
                let mutable k = 0
                while k < 27 do
                    scratch[k] <- input[baseIndex + offsets[k]]
                    k <- k + 1
                insertionSort27 scratch
                output[baseIndex] <- scratch[13]

let private measureItkMedian<'T> radius shape (input: 'T[]) =
    use image = importedImage shape input
    use filter = new itk.simple.MedianImageFilter()
    filter.SetRadius(vectorUInt32 [ radius; radius; radius ])
    use warmup = filter.Execute(image)
    measure (fun () ->
        use result = filter.Execute(image)
        if result.GetSize().Count = 0 then failwith "unreachable")

let private writeHeaderIfNeeded output =
    if not (File.Exists output) then
        File.WriteAllText(output, "backend,variant,pixelType,shape,repeat,exitCode,seconds,managedAllocatedBytes,error\n")

let private appendRow output backend variant pixel shape repeat exitCode (seconds: string) (allocated: string) (error: string) =
    writeHeaderIfNeeded output
    let q (s: string) = "\"" + s.Replace("\"", "\"\"") + "\""
    let fields =
        [ backend; variant; pixel; shapeText shape; string repeat; string exitCode; seconds; allocated; error.Replace("\n", " ") ]
        |> List.map q
    File.AppendAllText(output, String.Join(",", fields) + "\n")

let private runCase output radius shape pixel repeat =
    let n = length shape
    let diameter = radius * 2 + 1
    let windowText = $"{diameter}x{diameter}x{diameter}"
    match pixel with
    | UInt8 ->
        let input = ArrayPool<uint8>.Shared.Rent(n)
        let outputBuffer = ArrayPool<uint8>.Shared.Rent(n)
        try
            fillUInt8 input n
            if radius = 1 then
                let seconds, allocated = measure (fun () -> median3x3x3UInt8 shape input outputBuffer)
                appendRow output "arraypool" "flat-insertion-3x3x3" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
                let seconds, allocated = measure (fun () -> median3x3x3UInt8SparseRollingX shape input outputBuffer)
                appendRow output "arraypool" "sparse-rollingx-3x3x3" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
                let seconds, allocated = measure (fun () -> median3x3x3UInt8SparseOrderRollingX shape input outputBuffer)
                appendRow output "arraypool" "sparse-order-rollingx-3x3x3" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            else
                let seconds, allocated = measure (fun () -> medianSparseRollingXUInt8 radius shape input outputBuffer)
                appendRow output "arraypool" $"sparse-rollingx-{windowText}" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
                let seconds, allocated = measure (fun () -> medianSparseOrderRollingXUInt8 radius shape input outputBuffer)
                appendRow output "arraypool" $"sparse-order-rollingx-{windowText}" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            try
                let seconds, allocated = measureItkMedian<uint8> radius shape input
                appendRow output "simpleitk" $"MedianImageFilter-{windowText}" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            with ex ->
                appendRow output "simpleitk" $"MedianImageFilter-{windowText}" "UInt8" shape repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)
        finally
            ArrayPool<uint8>.Shared.Return(input)
            ArrayPool<uint8>.Shared.Return(outputBuffer)
    | UInt16 ->
        let input = ArrayPool<uint16>.Shared.Rent(n)
        let outputBuffer = ArrayPool<uint16>.Shared.Rent(n)
        try
            fillUInt16 input n
            if radius = 1 then
                let seconds, allocated = measure (fun () -> median3x3x3UInt16 shape input outputBuffer)
                appendRow output "arraypool" "flat-insertion-3x3x3" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
                let seconds, allocated = measure (fun () -> median3x3x3UInt16SparseRollingX shape input outputBuffer)
                appendRow output "arraypool" "sparse-rollingx-3x3x3" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
                let seconds, allocated = measure (fun () -> median3x3x3UInt16SparseOrderRollingX shape input outputBuffer)
                appendRow output "arraypool" "sparse-order-rollingx-3x3x3" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            else
                let seconds, allocated = measure (fun () -> medianSparseRollingXUInt16 radius shape input outputBuffer)
                appendRow output "arraypool" $"sparse-rollingx-{windowText}" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
                let seconds, allocated = measure (fun () -> medianSparseOrderRollingXUInt16 radius shape input outputBuffer)
                appendRow output "arraypool" $"sparse-order-rollingx-{windowText}" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            try
                let seconds, allocated = measureItkMedian<uint16> radius shape input
                appendRow output "simpleitk" $"MedianImageFilter-{windowText}" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            with ex ->
                appendRow output "simpleitk" $"MedianImageFilter-{windowText}" "UInt16" shape repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)
        finally
            ArrayPool<uint16>.Shared.Return(input)
            ArrayPool<uint16>.Shared.Return(outputBuffer)
    | Float32 ->
        let input = ArrayPool<float32>.Shared.Rent(n)
        let outputBuffer = ArrayPool<float32>.Shared.Rent(n)
        try
            fillFloat32 input n
            if radius = 1 then
                let seconds, allocated = measure (fun () -> median3x3x3Float32 shape input outputBuffer)
                appendRow output "arraypool" "flat-insertion-3x3x3" "Float32" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            try
                let seconds, allocated = measureItkMedian<float32> radius shape input
                appendRow output "simpleitk" $"MedianImageFilter-{windowText}" "Float32" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            with ex ->
                appendRow output "simpleitk" $"MedianImageFilter-{windowText}" "Float32" shape repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)
        finally
            ArrayPool<float32>.Shared.Return(input)
            ArrayPool<float32>.Shared.Return(outputBuffer)

[<EntryPoint>]
let main args =
    try
        let opts = parseArgs args
        let output = optional "output" "benchmarks/results/in-memory-median.csv" opts
        let shapes =
            optional "shapes" "128x128x128" opts
            |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
            |> Array.map parseShape
        let pixelTypes =
            optional "pixel-types" "UInt8,UInt16,Float32" opts
            |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
            |> Array.map parsePixelType
        let repeat = optional "repeat" "3" opts |> int
        let radius = optional "radius" "1" opts |> int
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))) |> ignore
        if File.Exists output then File.Delete output
        for shape in shapes do
            for pixel in pixelTypes do
                for r in 1 .. repeat do
                    runCase output radius shape pixel r
                    printfn "wrote %s %A repeat %d" (shapeText shape) pixel r
        0
    with ex ->
        eprintfn "%s: %s" (ex.GetType().Name) ex.Message
        2
