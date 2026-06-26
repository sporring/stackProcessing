namespace ChunkCore

open System
open System.Buffers
open System.Collections.Generic
open System.Numerics
open System.Runtime.InteropServices
open Chunk

module ChunkFunctions =

    type DenseHistogram =
        | UInt8Counts of uint64[]
        | Int8Counts of uint64[]
        | UInt16Counts of uint64[]
        | Int16Counts of uint64[]

    type LeftEdgeHistogram =
        { LeftEdges: float[]
          Counts: uint64[] }

    type ResampleInterpolation =
        | NearestNeighbor
        | Linear

    module ResampleInterpolation =
        let parse (value: string) =
            match value.Trim().ToLowerInvariant().Replace("_", "").Replace("-", "").Replace(" ", "") with
            | "nearest"
            | "nearestneighbor"
            | "nn" -> NearestNeighbor
            | "linear" -> Linear
            | _ -> failwith $"Unknown chunk resampling interpolation '{value}'. Use NearestNeighbor or Linear."

        let toNative = function
            | NearestNeighbor -> 0
            | Linear -> 1

    let finiteDiffKernel1D order =
        // Coefficients mirror ImageFunctions.finiteDiffFilter*.
        if order = 1u then [| 0.5f; 0.0f; -0.5f |]
        elif order = 2u then [| 1.0f; -2.0f; 1.0f |]
        elif order = 3u then [| 0.5f; -1.0f; 0.0f; 1.0f; -0.5f |]
        elif order = 4u then [| 1.0f; -4.0f; 6.0f; -4.0f; 1.0f |]
        elif order = 5u then [| 0.5f; -2.0f; 2.5f; 0.0f; -2.5f; 2.0f; -0.5f |]
        elif order = 6u then [| 1.0f; -6.0f; 15.0f; -20.0f; 15.0f; -6.0f; 1.0f |]
        else
            invalidArg "order" "Chunk finite-difference kernels are implemented for derivative order 1 <= order <= 6."

    let finiteDiffKernel = finiteDiffKernel1D

    let inline private nextGaussian (rng: Random) mean stddev =
        let u1 = max Double.Epsilon (rng.NextDouble())
        let u2 = rng.NextDouble()
        mean + stddev * sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)

    let private nextPoisson (rng: Random) lambda =
        if lambda <= 0.0 then
            0.0
        elif lambda < 30.0 then
            let threshold = exp (-lambda)
            let mutable k = 0
            let mutable p = 1.0
            while p > threshold do
                k <- k + 1
                p <- p * rng.NextDouble()
            float (k - 1)
        else
            let sample = nextGaussian rng lambda (sqrt lambda)
            max 0.0 (Math.Round sample)

    let private checkedIntDimension name value =
        if value > uint64 Int32.MaxValue then
            invalidArg name $"Chunk dimension must fit in Int32 for managed indexing, got {value}."
        int value

    let private checkedUIntToInt name value =
        if value > uint Int32.MaxValue then
            invalidArg name $"Chunk extent must fit in Int32 for managed indexing, got {value}."
        int value

    let private axisSize axis (width, height, depth) =
        match axis with
        | 0 -> width
        | 1 -> height
        | 2 -> depth
        | _ -> invalidArg "axis" $"Chunk axis must be 0, 1, or 2, got {axis}."

    let private setAxisSize axis value (width, height, depth) =
        match axis with
        | 0 -> value, height, depth
        | 1 -> width, value, depth
        | 2 -> width, height, value
        | _ -> invalidArg "axis" $"Chunk axis must be 0, 1, or 2, got {axis}."

    let private validatePermutation (order: int[]) =
        if isNull order then
            nullArg "order"
        if order.Length <> 3 then
            invalidArg "order" $"Chunk axis permutation expects exactly three axes, got {order.Length}."
        let seen = Array.zeroCreate<bool> 3
        for axis in order do
            if axis < 0 || axis > 2 then
                invalidArg "order" $"Chunk axis permutation entries must be 0, 1, or 2, got {axis}."
            if seen[axis] then
                let orderText = String.Join(", ", order)
                invalidArg "order" $"Chunk axis permutation must contain each axis exactly once, got [{orderText}]."
            seen[axis] <- true

    let private chunkDimensionsInt<'T when 'T: equality> (chunk: Chunk<'T>) =
        let width, height, depth = chunk.Size
        checkedIntDimension "width" width,
        checkedIntDimension "height" height,
        checkedIntDimension "depth" depth

    let private createTypedLike<'T when 'T: equality> size =
        Chunk.create<'T> size

    let private fillSpan value (span: Span<'T>) =
        for i in 0 .. span.Length - 1 do
            span[i] <- value

    module LowLevelNative =
        [<Literal>]
        let LibraryPath = "lowlevel"

        [<DllImport(LibraryPath, EntryPoint = "sp_median_uint16_nth_slab")>]
        extern void medianUInt16NthSlab(
            nativeint slices,
            nativeint output,
            int width,
            int height,
            int windowLength,
            int radius,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_median_uint8_nth_slab")>]
        extern void medianUInt8NthSlab(
            nativeint slices,
            nativeint output,
            int width,
            int height,
            int windowLength,
            int radius,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_median_int32_nth_slab")>]
        extern void medianInt32NthSlab(
            nativeint slices,
            nativeint output,
            int width,
            int height,
            int windowLength,
            int radius,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_median_float32_nth_slab")>]
        extern void medianFloat32NthSlab(
            nativeint slices,
            nativeint output,
            int width,
            int height,
            int windowLength,
            int radius,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_slab")>]
        extern void convolveFloat32Slab(
            nativeint slices,
            nativeint output,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelWidth,
            int kernelHeight,
            int kernelDepth,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_slices")>]
        extern void convolveFloat32Slices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelWidth,
            int kernelHeight,
            int kernelDepth,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint8_slices")>]
        extern void convolveUInt8Slices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelWidth,
            int kernelHeight,
            int kernelDepth,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int8_slices")>]
        extern void convolveInt8Slices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelWidth,
            int kernelHeight,
            int kernelDepth,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint16_slices")>]
        extern void convolveUInt16Slices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelWidth,
            int kernelHeight,
            int kernelDepth,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int16_slices")>]
        extern void convolveInt16Slices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelWidth,
            int kernelHeight,
            int kernelDepth,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int32_slices")>]
        extern void convolveInt32Slices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelWidth,
            int kernelHeight,
            int kernelDepth,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint8_x_slices")>]
        extern void convolveUInt8XSlices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelLength,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint8_y_slices")>]
        extern void convolveUInt8YSlices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelLength,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint8_z_slices")>]
        extern void convolveUInt8ZSlices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int windowLength,
            int kernelLength,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int8_x_slices")>]
        extern void convolveInt8XSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int8_y_slices")>]
        extern void convolveInt8YSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int8_z_slices")>]
        extern void convolveInt8ZSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint16_x_slices")>]
        extern void convolveUInt16XSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint16_y_slices")>]
        extern void convolveUInt16YSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint16_z_slices")>]
        extern void convolveUInt16ZSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int32_x_slices")>]
        extern void convolveInt32XSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int32_y_slices")>]
        extern void convolveInt32YSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_int32_z_slices")>]
        extern void convolveInt32ZSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_x_slices")>]
        extern void convolveFloat32XSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_y_slices")>]
        extern void convolveFloat32YSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_z_slices")>]
        extern void convolveFloat32ZSlices(nativeint slices, nativeint outputs, nativeint kernel, int width, int height, int windowLength, int kernelLength, int outputStart, int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_vector_components_slices")>]
        extern int convolveFloat32VectorComponentsSlices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int components,
            int windowLength,
            int kernelLength,
            int outputStart,
            int outputCount,
            int axis)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_vector_components_x_slices")>]
        extern int convolveFloat32VectorComponentsXSlices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int components,
            int windowLength,
            int kernelLength,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_vector_components_y_slices")>]
        extern int convolveFloat32VectorComponentsYSlices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int components,
            int windowLength,
            int kernelLength,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_vector_components_z_slices")>]
        extern int convolveFloat32VectorComponentsZSlices(
            nativeint slices,
            nativeint outputs,
            nativeint kernel,
            int width,
            int height,
            int components,
            int windowLength,
            int kernelLength,
            int outputStart,
            int outputCount)

        [<DllImport(LibraryPath, EntryPoint = "sp_signed_distance_band_uint8_slices")>]
        extern int signedDistanceBandUInt8Slices(
            nativeint slices,
            nativeint outputs,
            int width,
            int height,
            int windowLength,
            int outputStart,
            int outputCount,
            float32 bandRadius)

        [<DllImport(LibraryPath, EntryPoint = "sp_resample_2d")>]
        extern int resample2D(
            nativeint input,
            nativeint output,
            int pixelType,
            int inputWidth,
            int inputHeight,
            int outputWidth,
            int outputHeight,
            double spacingX,
            double spacingY,
            int interpolation)

        [<DllImport(LibraryPath, EntryPoint = "sp_resize_3d_pair_slice")>]
        extern int resize3DPairSlice(
            nativeint lower,
            nativeint upper,
            nativeint output,
            int pixelType,
            int inputWidth,
            int inputHeight,
            int outputWidth,
            int outputHeight,
            double spacingX,
            double spacingY,
            double zFraction,
            int interpolation)

        [<DllImport(LibraryPath, EntryPoint = "sp_euler_2d")>]
        extern int euler2D(
            nativeint input,
            nativeint output,
            int pixelType,
            int width,
            int height,
            double centerX,
            double centerY,
            double angle,
            double dx,
            double dy,
            int inverse,
            int interpolation)

        let ensureAvailable () =
            let mutable handle = nativeint 0
            let searchPath = Nullable(DllImportSearchPath.AssemblyDirectory)
            if NativeLibrary.TryLoad(LibraryPath, typeof<ChunkLayout>.Assembly, searchPath, &handle) then
                NativeLibrary.Free(handle)
            else
                invalidOp "Native StackProcessing helper 'lowlevel' was not found. Build it with lowlevel/build.sh or lowlevel/build.ps1 so the platform library is placed in the solution lib directory and copied to the application output."

    let copyChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        let output = create<'T> chunk.Size
        try
            chunk.Bytes.AsSpan(0, chunk.ByteLength).CopyTo(output.Bytes.AsSpan(0, output.ByteLength))
            output
        with
        | _ ->
            decRef output
            reraise()

    let padChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (lowerX: uint)
        (upperX: uint)
        (lowerY: uint)
        (upperY: uint)
        (lowerZ: uint)
        (upperZ: uint)
        (value: 'T)
        (chunk: Chunk<'T>)
        =
        let width, height, depth = chunkDimensionsInt chunk
        let lx = checkedUIntToInt "lowerX" lowerX
        let ly = checkedUIntToInt "lowerY" lowerY
        let lz = checkedUIntToInt "lowerZ" lowerZ
        let ux = checkedUIntToInt "upperX" upperX
        let uy = checkedUIntToInt "upperY" upperY
        let uz = checkedUIntToInt "upperZ" upperZ
        let outputWidth = width + lx + ux
        let outputHeight = height + ly + uy
        let outputDepth = depth + lz + uz
        let output = create<'T> (uint64 outputWidth, uint64 outputHeight, uint64 outputDepth)
        try
            let inputPixels = span<'T> chunk
            let outputPixels = span<'T> output
            fillSpan value outputPixels

            for z in 0 .. depth - 1 do
                for y in 0 .. height - 1 do
                    for x in 0 .. width - 1 do
                        let inputIndex = Chunk.toIndex width height x y z
                        let outputIndex = Chunk.toIndex outputWidth outputHeight (x + lx) (y + ly) (z + lz)
                        outputPixels[outputIndex] <- inputPixels[inputIndex]

            output
        with
        | _ ->
            decRef output
            reraise()

    let cropChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (lowerX: uint)
        (upperX: uint)
        (lowerY: uint)
        (upperY: uint)
        (lowerZ: uint)
        (upperZ: uint)
        (chunk: Chunk<'T>)
        =
        let width, height, depth = chunkDimensionsInt chunk
        let lx = checkedUIntToInt "lowerX" lowerX
        let ly = checkedUIntToInt "lowerY" lowerY
        let lz = checkedUIntToInt "lowerZ" lowerZ
        let ux = checkedUIntToInt "upperX" upperX
        let uy = checkedUIntToInt "upperY" upperY
        let uz = checkedUIntToInt "upperZ" upperZ
        if lx + ux >= width then
            invalidArg "lowerX/upperX" $"Chunk crop removes the full X axis: lower={lowerX}, upper={upperX}, width={width}."
        if ly + uy >= height then
            invalidArg "lowerY/upperY" $"Chunk crop removes the full Y axis: lower={lowerY}, upper={upperY}, height={height}."
        if lz + uz >= depth then
            invalidArg "lowerZ/upperZ" $"Chunk crop removes the full Z axis: lower={lowerZ}, upper={upperZ}, depth={depth}."

        let outputWidth = width - lx - ux
        let outputHeight = height - ly - uy
        let outputDepth = depth - lz - uz
        let output = create<'T> (uint64 outputWidth, uint64 outputHeight, uint64 outputDepth)
        try
            let inputPixels = span<'T> chunk
            let outputPixels = span<'T> output

            for z in 0 .. outputDepth - 1 do
                for y in 0 .. outputHeight - 1 do
                    for x in 0 .. outputWidth - 1 do
                        let inputIndex = Chunk.toIndex width height (x + lx) (y + ly) (z + lz)
                        let outputIndex = Chunk.toIndex outputWidth outputHeight x y z
                        outputPixels[outputIndex] <- inputPixels[inputIndex]

            output
        with
        | _ ->
            decRef output
            reraise()

    let squeezeChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        let width, height, depth = chunk.Size
        let compacted = ResizeArray<uint64>(3)
        if width <> 1UL then compacted.Add width
        if height <> 1UL then compacted.Add height
        if depth <> 1UL then compacted.Add depth
        while compacted.Count < 3 do
            compacted.Add 1UL
        let outputSize = compacted[0], compacted[1], compacted[2]
        let output = create<'T> outputSize
        try
            chunk.Bytes.AsSpan(0, chunk.ByteLength).CopyTo(output.Bytes.AsSpan(0, output.ByteLength))
            output
        with
        | _ ->
            decRef output
            reraise()

    let concatenateChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        axis
        (a: Chunk<'T>)
        (b: Chunk<'T>)
        =
        let aWidth, aHeight, aDepth = chunkDimensionsInt a
        let bWidth, bHeight, bDepth = chunkDimensionsInt b
        let aSize = aWidth, aHeight, aDepth
        let bSize = bWidth, bHeight, bDepth

        for checkAxis in 0 .. 2 do
            if checkAxis <> axis && axisSize checkAxis aSize <> axisSize checkAxis bSize then
                invalidArg "b" $"Chunk concatenate along axis {axis} expects equal non-concatenated dimensions, got {a.Size} and {b.Size}."

        let outputAxisSize = axisSize axis aSize + axisSize axis bSize
        let outputWidth, outputHeight, outputDepth = setAxisSize axis outputAxisSize aSize
        let output = create<'T> (uint64 outputWidth, uint64 outputHeight, uint64 outputDepth)
        try
            let aPixels = span<'T> a
            let bPixels = span<'T> b
            let outputPixels = span<'T> output
            let aAxisSize = axisSize axis aSize

            for z in 0 .. outputDepth - 1 do
                for y in 0 .. outputHeight - 1 do
                    for x in 0 .. outputWidth - 1 do
                        let outputIndex = Chunk.toIndex outputWidth outputHeight x y z
                        let coordinate = [| x; y; z |]
                        if coordinate[axis] < aAxisSize then
                            let inputIndex = Chunk.toIndex aWidth aHeight coordinate[0] coordinate[1] coordinate[2]
                            outputPixels[outputIndex] <- aPixels[inputIndex]
                        else
                            coordinate[axis] <- coordinate[axis] - aAxisSize
                            let inputIndex = Chunk.toIndex bWidth bHeight coordinate[0] coordinate[1] coordinate[2]
                            outputPixels[outputIndex] <- bPixels[inputIndex]

            output
        with
        | _ ->
            decRef output
            reraise()

    let permuteAxesChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (order: int[])
        (chunk: Chunk<'T>)
        =
        validatePermutation order
        let inputWidth, inputHeight, inputDepth = chunkDimensionsInt chunk
        let inputSize = [| inputWidth; inputHeight; inputDepth |]
        let outputWidth = inputSize[order[0]]
        let outputHeight = inputSize[order[1]]
        let outputDepth = inputSize[order[2]]
        let output = create<'T> (uint64 outputWidth, uint64 outputHeight, uint64 outputDepth)
        try
            let inputPixels = span<'T> chunk
            let outputPixels = span<'T> output

            for z in 0 .. outputDepth - 1 do
                for y in 0 .. outputHeight - 1 do
                    for x in 0 .. outputWidth - 1 do
                        let outputCoord = [| x; y; z |]
                        let inputCoord = Array.zeroCreate<int> 3
                        for outputAxis in 0 .. 2 do
                            inputCoord[order[outputAxis]] <- outputCoord[outputAxis]
                        let inputIndex = Chunk.toIndex inputWidth inputHeight inputCoord[0] inputCoord[1] inputCoord[2]
                        let outputIndex = Chunk.toIndex outputWidth outputHeight x y z
                        outputPixels[outputIndex] <- inputPixels[inputIndex]

            output
        with
        | _ ->
            decRef output
            reraise()

    let fftXYFloat32ToComplex64InterleavedChunk (input: Chunk<float32>) =
        let width64, height64, depth64 = input.Size
        if depth64 <> 1UL then
            invalidArg "input" $"Chunk FFT XY expects 2D slice chunks with depth 1, got {input.Size}."
        if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg "input" $"Chunk FFT XY slice dimensions must fit Int32, got {input.Size}."

        let width = int width64
        let height = int height64
        let output = create<float32> (2UL * width64, height64, 1UL)

        try
            let inputSpan = span<float32> input
            let outputSpan = span<float32> output
            let mutable i = 0
            let mutable j = 0
            while i < inputSpan.Length do
                outputSpan[j] <- inputSpan[i]
                outputSpan[j + 1] <- 0.0f
                i <- i + 1
                j <- j + 2

            NativeSp.ensureAvailable ()
            let mutable outputHandle = Unchecked.defaultof<GCHandle>
            let mutable outputPinned = false
            try
                outputHandle <- GCHandle.Alloc(output.Bytes, GCHandleType.Pinned)
                outputPinned <- true
                NativeSp.fftwfComplexXYInplace(outputHandle.AddrOfPinnedObject(), width, height, 0)
                |> NativeSp.checkStatus "fftwf xy complex"
            finally
                if outputPinned then
                    outputHandle.Free()

            output
        with
        | _ ->
            decRef output
            reraise()

    let invFftXYComplex64InterleavedToFloat32Chunk (input: Chunk<float32>) =
        let width64, height64, depth64 = input.Size
        if depth64 <> 1UL then
            invalidArg "input" $"Chunk inverse FFT XY expects 2D complex64-interleaved chunks with depth 1, got {input.Size}."
        if width64 % 2UL <> 0UL then
            invalidArg "input" $"Chunk inverse FFT XY expects even interleaved width, got {input.Size}."
        if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg "input" $"Chunk inverse FFT XY slice dimensions must fit Int32, got {input.Size}."

        let logicalWidth64 = width64 / 2UL
        let logicalWidth = int logicalWidth64
        let height = int height64
        let scratch = create<float32> input.Size
        let output = create<float32> (logicalWidth64, height64, 1UL)

        try
            try
                input.Bytes.AsSpan(0, input.ByteLength).CopyTo(scratch.Bytes.AsSpan(0, scratch.ByteLength))

                NativeSp.ensureAvailable ()
                let mutable scratchHandle = Unchecked.defaultof<GCHandle>
                let mutable scratchPinned = false
                try
                    scratchHandle <- GCHandle.Alloc(scratch.Bytes, GCHandleType.Pinned)
                    scratchPinned <- true
                    NativeSp.invFftwfComplexXYInplace(scratchHandle.AddrOfPinnedObject(), logicalWidth, height)
                    |> NativeSp.checkStatus "inverse fftwf xy complex"
                finally
                    if scratchPinned then
                        scratchHandle.Free()

                let scratchSpan = span<float32> scratch
                let outputSpan = span<float32> output
                let mutable j = 0
                for i in 0 .. outputSpan.Length - 1 do
                    outputSpan[i] <- scratchSpan[j]
                    j <- j + 2

                output
            with
            | _ ->
                decRef output
                reraise()
        finally
            decRef scratch

    let fftRealXYFloat32ToHermitianPackedComplex64InterleavedChunk (input: Chunk<float32>) =
        let width64, height64, depth64 = input.Size
        if depth64 <> 1UL then
            invalidArg "input" $"Chunk real FFT XY expects 2D slice chunks with depth 1, got {input.Size}."
        if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg "input" $"Chunk real FFT XY slice dimensions must fit Int32, got {input.Size}."

        let width = int width64
        let height = int height64
        let packedComplexWidth64 = width64 / 2UL + 1UL
        let output = create<float32> (2UL * packedComplexWidth64, height64, 1UL)
        try
            NativeSp.ensureAvailable ()
            let mutable inputHandle = Unchecked.defaultof<GCHandle>
            let mutable outputHandle = Unchecked.defaultof<GCHandle>
            try
                inputHandle <- GCHandle.Alloc(input.Bytes, GCHandleType.Pinned)
                outputHandle <- GCHandle.Alloc(output.Bytes, GCHandleType.Pinned)
                NativeSp.fftwfRealXYToComplex(inputHandle.AddrOfPinnedObject(), outputHandle.AddrOfPinnedObject(), width, height)
                |> NativeSp.checkStatus "fftwf real xy to compact complex"
            finally
                if outputHandle.IsAllocated then
                    outputHandle.Free()
                if inputHandle.IsAllocated then
                    inputHandle.Free()

            { LogicalSize = (width64, height64, 1UL)
              Layout = HermitianPackedComplex64Interleaved(0, width64)
              Chunk = output }
        with
        | _ ->
            decRef output
            reraise()

    let invFftXYHermitianPackedComplex64InterleavedToFloat32Chunk (input: SpectralChunk) =
        let width64, height64, depth64 = input.LogicalSize
        if depth64 <> 1UL then
            invalidArg "input" $"Chunk inverse real FFT XY expects per-slice spectral chunks with logical depth 1, got {input.LogicalSize}."
        if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg "input" $"Chunk inverse real FFT XY slice dimensions must fit Int32, got {input.LogicalSize}."
        match input.Layout with
        | HermitianPackedComplex64Interleaved(0, realSize) when realSize = width64 -> ()
        | layout -> invalidArg "input" $"Chunk inverse real FFT XY expects HermitianPackedComplex64Interleaved(0, {width64}), got {layout}."

        let width = int width64
        let height = int height64
        let packedComplexWidth64 = width64 / 2UL + 1UL
        let expectedPackedSize = (2UL * packedComplexWidth64, height64, 1UL)
        if input.Chunk.Size <> expectedPackedSize then
            invalidArg "input" $"Chunk inverse real FFT XY expects packed chunk size {expectedPackedSize}, got {input.Chunk.Size}."

        let output = create<float32> (width64, height64, 1UL)
        try
            NativeSp.ensureAvailable ()
            let mutable inputHandle = Unchecked.defaultof<GCHandle>
            let mutable outputHandle = Unchecked.defaultof<GCHandle>
            let mutable plan = nativeint 0
            try
                inputHandle <- GCHandle.Alloc(input.Chunk.Bytes, GCHandleType.Pinned)
                outputHandle <- GCHandle.Alloc(output.Bytes, GCHandleType.Pinned)
                plan <- NativeSp.fftwfComplexXYToRealPlanCreate(inputHandle.AddrOfPinnedObject(), outputHandle.AddrOfPinnedObject(), width, height)
                if plan = nativeint 0 then
                    invalidOp "fftwf compact complex xy to real plan creation failed in native helper."
                NativeSp.fftwfComplexXYToRealPlanExecute(plan, inputHandle.AddrOfPinnedObject(), outputHandle.AddrOfPinnedObject())
                |> NativeSp.checkStatus "inverse fftwf compact complex xy to real"
            finally
                if plan <> nativeint 0 then
                    NativeSp.fftwfComplexXYToRealPlanDestroy(plan)
                if outputHandle.IsAllocated then
                    outputHandle.Free()
                if inputHandle.IsAllocated then
                    inputHandle.Free()

            output
        with
        | _ ->
            decRef output
            reraise()

    type FftXYAndZPlanCache() =
        let mutable xyPlan = nativeint 0
        let mutable zPlan = nativeint 0
        let mutable width = 0
        let mutable height = 0
        let mutable depth = 0
        let mutable buffer: float32[] = Array.empty
        let mutable bufferLength = 0
        let mutable handle = Unchecked.defaultof<GCHandle>
        let mutable disposed = false

        let destroyPlans () =
            if zPlan <> nativeint 0 then
                NativeSp.fftwfComplexZPlanDestroy(zPlan)
                zPlan <- nativeint 0
            if xyPlan <> nativeint 0 then
                NativeSp.fftwfComplexXYPlanDestroy(xyPlan)
                xyPlan <- nativeint 0

        let releaseBuffer () =
            if handle.IsAllocated then
                handle.Free()
            if bufferLength > 0 then
                ArrayPool<float32>.Shared.Return(buffer)
            buffer <- Array.empty
            bufferLength <- 0

        let ensureNotDisposed () =
            if disposed then
                invalidOp "FFT XY+Z plan cache has been disposed."

        member private _.Ensure(logicalWidth: int, logicalHeight: int, logicalDepth: int) =
            ensureNotDisposed()
            if logicalWidth <= 0 || logicalHeight <= 0 || logicalDepth <= 0 then
                invalidArg "size" $"FFT XY+Z plan cache expects positive dimensions, got {logicalWidth}x{logicalHeight}x{logicalDepth}."

            let required = 2 * logicalWidth * logicalHeight * logicalDepth
            if logicalWidth <> width || logicalHeight <> height || logicalDepth <> depth || required > bufferLength then
                destroyPlans()
                releaseBuffer()

                buffer <- ArrayPool<float32>.Shared.Rent(required)
                bufferLength <- required
                handle <- GCHandle.Alloc(buffer, GCHandleType.Pinned)
                let basePointer = handle.AddrOfPinnedObject()

                NativeSp.ensureAvailable()
                xyPlan <- NativeSp.fftwfComplexXYPlanCreate(basePointer, logicalWidth, logicalHeight, 0)
                if xyPlan = nativeint 0 then
                    releaseBuffer()
                    invalidOp "fftwf xy complex plan creation failed in native helper."

                zPlan <- NativeSp.fftwfComplexZPlanCreate(basePointer, logicalWidth, logicalHeight, logicalDepth, 0)
                if zPlan = nativeint 0 then
                    destroyPlans()
                    releaseBuffer()
                    invalidOp "fftwf z complex plan creation failed in native helper."

                width <- logicalWidth
                height <- logicalHeight
                depth <- logicalDepth

        member this.ForwardFloat32SlicesToComplex64Interleaved(input: IReadOnlyList<Chunk<float32>>) =
            ensureNotDisposed()
            if isNull input then
                nullArg "input"
            if input.Count = 0 then
                [||]
            else
                let width64, height64, depth64 = input[0].Size
                if depth64 <> 1UL then
                    invalidArg "input" $"FFT XY+Z expects 2D slice chunks with depth 1, got {input[0].Size}."
                if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue || input.Count > Int32.MaxValue then
                    invalidArg "input" $"FFT XY+Z dimensions must fit Int32, got {width64}x{height64}x{input.Count}."

                let logicalWidth = int width64
                let logicalHeight = int height64
                let logicalDepth = input.Count
                let planeValues = 2 * logicalWidth * logicalHeight
                let outputs = Array.zeroCreate<Chunk<float32>> logicalDepth

                try
                    for z in 0 .. logicalDepth - 1 do
                        let chunk = input[z]
                        if chunk.Size <> (width64, height64, 1UL) then
                            invalidArg "input" $"FFT XY+Z expects all slices to have size {(width64, height64, 1UL)}, got {chunk.Size} at z={z}."

                    this.Ensure(logicalWidth, logicalHeight, logicalDepth)

                    for z in 0 .. logicalDepth - 1 do
                        let inputPixels = span<float32> input[z]
                        let baseOffset = z * planeValues
                        let mutable src = 0
                        let mutable dst = baseOffset
                        while src < inputPixels.Length do
                            buffer[dst] <- inputPixels[src]
                            buffer[dst + 1] <- 0.0f
                            src <- src + 1
                            dst <- dst + 2

                    let basePointer = handle.AddrOfPinnedObject()
                    for z in 0 .. logicalDepth - 1 do
                        let planePointer = IntPtr.Add(basePointer, z * planeValues * sizeof<float32>)
                        NativeSp.fftwfComplexXYPlanExecute(xyPlan, planePointer)
                        |> NativeSp.checkStatus "fftwf xy complex planned"

                    NativeSp.fftwfComplexZPlanExecute(zPlan, basePointer)
                    |> NativeSp.checkStatus "fftwf z complex planned"

                    for z in 0 .. logicalDepth - 1 do
                        let output = create<float32> (2UL * width64, height64, 1UL)
                        outputs[z] <- output
                        buffer.AsSpan(z * planeValues, planeValues).CopyTo(span<float32> output)

                    outputs
                with
                | _ ->
                    for output in outputs do
                        if not (isNull (box output)) then
                            decRef output
                    reraise()

        interface IDisposable with
            member _.Dispose() =
                if not disposed then
                    disposed <- true
                    destroyPlans()
                    releaseBuffer()

    type FftRealXYAndZPlanCache() =
        let mutable realXYPlan = nativeint 0
        let mutable zPlan = nativeint 0
        let mutable width = 0
        let mutable height = 0
        let mutable depth = 0
        let mutable complexWidth = 0
        let mutable buffer: float32[] = Array.empty
        let mutable bufferLength = 0
        let mutable handle = Unchecked.defaultof<GCHandle>
        let mutable disposed = false

        let destroyPlans () =
            if zPlan <> nativeint 0 then
                NativeSp.fftwfComplexZPlanDestroy(zPlan)
                zPlan <- nativeint 0
            if realXYPlan <> nativeint 0 then
                NativeSp.fftwfRealXYPlanDestroy(realXYPlan)
                realXYPlan <- nativeint 0

        let releaseBuffer () =
            if handle.IsAllocated then
                handle.Free()
            if bufferLength > 0 then
                ArrayPool<float32>.Shared.Return(buffer)
            buffer <- Array.empty
            bufferLength <- 0

        let ensureNotDisposed () =
            if disposed then
                invalidOp "FFT real-XY+Z plan cache has been disposed."

        member private _.Ensure(logicalWidth: int, logicalHeight: int, logicalDepth: int, sampleRealPointer: nativeint) =
            ensureNotDisposed()
            if logicalWidth <= 0 || logicalHeight <= 0 || logicalDepth <= 0 then
                invalidArg "size" $"FFT real-XY+Z plan cache expects positive dimensions, got {logicalWidth}x{logicalHeight}x{logicalDepth}."
            if sampleRealPointer = nativeint 0 then
                invalidArg "sampleRealPointer" "FFT real-XY+Z plan cache expects a non-null sample real pointer."

            let logicalComplexWidth = logicalWidth / 2 + 1
            let required = 2 * logicalComplexWidth * logicalHeight * logicalDepth
            if logicalWidth <> width || logicalHeight <> height || logicalDepth <> depth || required > bufferLength then
                destroyPlans()
                releaseBuffer()

                buffer <- ArrayPool<float32>.Shared.Rent(required)
                bufferLength <- required
                handle <- GCHandle.Alloc(buffer, GCHandleType.Pinned)
                let basePointer = handle.AddrOfPinnedObject()

                NativeSp.ensureAvailable()
                realXYPlan <- NativeSp.fftwfRealXYPlanCreate(sampleRealPointer, basePointer, logicalWidth, logicalHeight)
                if realXYPlan = nativeint 0 then
                    releaseBuffer()
                    invalidOp "fftwf real xy plan creation failed in native helper."

                zPlan <- NativeSp.fftwfComplexZPlanCreate(basePointer, logicalComplexWidth, logicalHeight, logicalDepth, 0)
                if zPlan = nativeint 0 then
                    destroyPlans()
                    releaseBuffer()
                    invalidOp "fftwf z complex plan creation failed in native helper."

                width <- logicalWidth
                height <- logicalHeight
                depth <- logicalDepth
                complexWidth <- logicalComplexWidth

        member this.ForwardFloat32SlicesToComplex64Interleaved(input: IReadOnlyList<Chunk<float32>>) =
            ensureNotDisposed()
            if isNull input then
                nullArg "input"
            if input.Count = 0 then
                [||]
            else
                let width64, height64, depth64 = input[0].Size
                if depth64 <> 1UL then
                    invalidArg "input" $"FFT real-XY+Z expects 2D slice chunks with depth 1, got {input[0].Size}."
                if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue || input.Count > Int32.MaxValue then
                    invalidArg "input" $"FFT real-XY+Z dimensions must fit Int32, got {width64}x{height64}x{input.Count}."

                let logicalWidth = int width64
                let logicalHeight = int height64
                let logicalDepth = input.Count
                let logicalComplexWidth = logicalWidth / 2 + 1
                let planeValues = 2 * logicalComplexWidth * logicalHeight
                let outputs = Array.zeroCreate<SpectralChunk> logicalDepth
                let mutable firstHandle = Unchecked.defaultof<GCHandle>

                try
                    for z in 0 .. logicalDepth - 1 do
                        let chunk = input[z]
                        if chunk.Size <> (width64, height64, 1UL) then
                            invalidArg "input" $"FFT real-XY+Z expects all slices to have size {(width64, height64, 1UL)}, got {chunk.Size} at z={z}."

                    firstHandle <- GCHandle.Alloc(input[0].Bytes, GCHandleType.Pinned)
                    this.Ensure(logicalWidth, logicalHeight, logicalDepth, firstHandle.AddrOfPinnedObject())

                    let basePointer = handle.AddrOfPinnedObject()
                    for z in 0 .. logicalDepth - 1 do
                        let mutable inputHandle = Unchecked.defaultof<GCHandle>
                        try
                            inputHandle <- if z = 0 then firstHandle else GCHandle.Alloc(input[z].Bytes, GCHandleType.Pinned)
                            let outputPointer = IntPtr.Add(basePointer, z * planeValues * sizeof<float32>)
                            NativeSp.fftwfRealXYPlanExecute(realXYPlan, inputHandle.AddrOfPinnedObject(), outputPointer)
                            |> NativeSp.checkStatus "fftwf real xy planned"
                        finally
                            if z <> 0 && inputHandle.IsAllocated then
                                inputHandle.Free()

                    if firstHandle.IsAllocated then
                        firstHandle.Free()

                    NativeSp.fftwfComplexZPlanExecute(zPlan, basePointer)
                    |> NativeSp.checkStatus "fftwf z complex planned after real xy"

                    for z in 0 .. logicalDepth - 1 do
                        let output = create<float32> (uint64 (2 * logicalComplexWidth), height64, 1UL)
                        outputs[z] <-
                            { LogicalSize = (width64, height64, uint64 logicalDepth)
                              Layout = HermitianPackedComplex64Interleaved(0, width64)
                              Chunk = output }
                        buffer.AsSpan(z * planeValues, planeValues).CopyTo(span<float32> output)

                    outputs
                with
                | _ ->
                    if firstHandle.IsAllocated then
                        firstHandle.Free()
                    for output in outputs do
                        if not (isNull (box output)) then
                            decRef output.Chunk
                    reraise()

        interface IDisposable with
            member _.Dispose() =
                if not disposed then
                    disposed <- true
                    destroyPlans()
                    releaseBuffer()

    type InvFftRealXYAndZPlanCache() =
        let mutable c2rXYPlan = nativeint 0
        let mutable zPlan = nativeint 0
        let mutable width = 0
        let mutable height = 0
        let mutable depth = 0
        let mutable complexWidth = 0
        let mutable buffer: float32[] = Array.empty
        let mutable bufferLength = 0
        let mutable sampleReal: float32[] = Array.empty
        let mutable sampleRealLength = 0
        let mutable handle = Unchecked.defaultof<GCHandle>
        let mutable sampleRealHandle = Unchecked.defaultof<GCHandle>
        let mutable disposed = false

        let destroyPlans () =
            if zPlan <> nativeint 0 then
                NativeSp.fftwfComplexZPlanDestroy(zPlan)
                zPlan <- nativeint 0
            if c2rXYPlan <> nativeint 0 then
                NativeSp.fftwfComplexXYToRealPlanDestroy(c2rXYPlan)
                c2rXYPlan <- nativeint 0

        let releaseBuffers () =
            if handle.IsAllocated then
                handle.Free()
            if sampleRealHandle.IsAllocated then
                sampleRealHandle.Free()
            if bufferLength > 0 then
                ArrayPool<float32>.Shared.Return(buffer)
            if sampleRealLength > 0 then
                ArrayPool<float32>.Shared.Return(sampleReal)
            buffer <- Array.empty
            sampleReal <- Array.empty
            bufferLength <- 0
            sampleRealLength <- 0

        let ensureNotDisposed () =
            if disposed then
                invalidOp "Inverse FFT real-XY+Z plan cache has been disposed."

        member private _.Ensure(logicalWidth: int, logicalHeight: int, logicalDepth: int) =
            ensureNotDisposed()
            if logicalWidth <= 0 || logicalHeight <= 0 || logicalDepth <= 0 then
                invalidArg "size" $"Inverse FFT real-XY+Z plan cache expects positive dimensions, got {logicalWidth}x{logicalHeight}x{logicalDepth}."

            let logicalComplexWidth = logicalWidth / 2 + 1
            let required = 2 * logicalComplexWidth * logicalHeight * logicalDepth
            let requiredReal = logicalWidth * logicalHeight
            if logicalWidth <> width || logicalHeight <> height || logicalDepth <> depth || required > bufferLength || requiredReal > sampleRealLength then
                destroyPlans()
                releaseBuffers()

                buffer <- ArrayPool<float32>.Shared.Rent(required)
                bufferLength <- required
                handle <- GCHandle.Alloc(buffer, GCHandleType.Pinned)

                sampleReal <- ArrayPool<float32>.Shared.Rent(requiredReal)
                sampleRealLength <- requiredReal
                sampleRealHandle <- GCHandle.Alloc(sampleReal, GCHandleType.Pinned)

                let basePointer = handle.AddrOfPinnedObject()
                NativeSp.ensureAvailable()
                zPlan <- NativeSp.fftwfComplexZPlanCreate(basePointer, logicalComplexWidth, logicalHeight, logicalDepth, 1)
                if zPlan = nativeint 0 then
                    releaseBuffers()
                    invalidOp "inverse fftwf z complex plan creation failed in native helper."

                c2rXYPlan <- NativeSp.fftwfComplexXYToRealPlanCreate(basePointer, sampleRealHandle.AddrOfPinnedObject(), logicalWidth, logicalHeight)
                if c2rXYPlan = nativeint 0 then
                    destroyPlans()
                    releaseBuffers()
                    invalidOp "inverse fftwf xy complex-to-real plan creation failed in native helper."

                width <- logicalWidth
                height <- logicalHeight
                depth <- logicalDepth
                complexWidth <- logicalComplexWidth

        member this.InverseHermitianPackedToFloat32Slices(input: IReadOnlyList<SpectralChunk>) =
            ensureNotDisposed()
            if isNull input then
                nullArg "input"
            if input.Count = 0 then
                [||]
            else
                let first = input[0]
                let width64, height64, logicalDepth64 = first.LogicalSize
                if logicalDepth64 <> uint64 input.Count then
                    invalidArg "input" $"Inverse FFT real-XY+Z expects LogicalSize depth to match window length, got {logicalDepth64} and {input.Count}."
                if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue || input.Count > Int32.MaxValue then
                    invalidArg "input" $"Inverse FFT real-XY+Z dimensions must fit Int32, got {width64}x{height64}x{input.Count}."
                match first.Layout with
                | HermitianPackedComplex64Interleaved(0, realSize) when realSize = width64 -> ()
                | layout -> invalidArg "input" $"Inverse FFT real-XY+Z expects HermitianPackedComplex64Interleaved(0, {width64}), got {layout}."

                let logicalWidth = int width64
                let logicalHeight = int height64
                let logicalDepth = input.Count
                let logicalComplexWidth = logicalWidth / 2 + 1
                let packedSize = (uint64 (2 * logicalComplexWidth), height64, 1UL)
                let planeValues = 2 * logicalComplexWidth * logicalHeight
                let outputs = Array.zeroCreate<Chunk<float32>> logicalDepth

                try
                    for z in 0 .. logicalDepth - 1 do
                        let spectral = input[z]
                        if spectral.LogicalSize <> (width64, height64, uint64 logicalDepth) then
                            invalidArg "input" $"Inverse FFT real-XY+Z expects matching logical sizes, got {spectral.LogicalSize} at z={z}."
                        if spectral.Layout <> first.Layout then
                            invalidArg "input" $"Inverse FFT real-XY+Z expects matching spectral layouts, got {spectral.Layout} at z={z}."
                        if spectral.Chunk.Size <> packedSize then
                            invalidArg "input" $"Inverse FFT real-XY+Z expects packed chunk size {packedSize}, got {spectral.Chunk.Size} at z={z}."

                    this.Ensure(logicalWidth, logicalHeight, logicalDepth)

                    for z in 0 .. logicalDepth - 1 do
                        let source = span<float32> input[z].Chunk
                        let target = buffer.AsSpan(z * planeValues, planeValues)
                        source.CopyTo(target)

                    let basePointer = handle.AddrOfPinnedObject()
                    NativeSp.fftwfComplexZPlanExecute(zPlan, basePointer)
                    |> NativeSp.checkStatus "inverse fftwf z complex planned before real xy"

                    for z in 0 .. logicalDepth - 1 do
                        let output = create<float32> (width64, height64, 1UL)
                        outputs[z] <- output
                        let planePointer = IntPtr.Add(basePointer, z * planeValues * sizeof<float32>)
                        let mutable outputHandle = Unchecked.defaultof<GCHandle>
                        try
                            outputHandle <- GCHandle.Alloc(output.Bytes, GCHandleType.Pinned)
                            NativeSp.fftwfComplexXYToRealPlanExecute(c2rXYPlan, planePointer, outputHandle.AddrOfPinnedObject())
                            |> NativeSp.checkStatus "inverse fftwf xy complex-to-real planned"
                        finally
                            if outputHandle.IsAllocated then
                                outputHandle.Free()

                    outputs
                with
                | _ ->
                    for output in outputs do
                        if not (isNull (box output)) then
                            decRef output
                    reraise()

        interface IDisposable with
            member _.Dispose() =
                if not disposed then
                    disposed <- true
                    destroyPlans()
                    releaseBuffers()

    let fftShiftXYComplex64InterleavedChunk (input: Chunk<float32>) =
        let interleavedWidth64, height64, depth64 = input.Size
        if depth64 <> 1UL then
            invalidArg "input" $"Chunk FFT shift XY expects 2D complex64-interleaved chunks with depth 1, got {input.Size}."
        if interleavedWidth64 % 2UL <> 0UL then
            invalidArg "input" $"Chunk FFT shift XY expects even interleaved width, got {input.Size}."
        if interleavedWidth64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg "input" $"Chunk FFT shift XY slice dimensions must fit Int32, got {input.Size}."

        let logicalWidth = int (interleavedWidth64 / 2UL)
        let height = int height64
        let shiftX = logicalWidth / 2
        let shiftY = height / 2
        let output = create<float32> input.Size

        try
            let inputPixels = span<float32> input
            let outputPixels = span<float32> output

            for y in 0 .. height - 1 do
                let sourceY = (y - shiftY + height) % height
                for x in 0 .. logicalWidth - 1 do
                    let sourceX = (x - shiftX + logicalWidth) % logicalWidth
                    let source = 2 * (sourceY * logicalWidth + sourceX)
                    let target = 2 * (y * logicalWidth + x)
                    outputPixels[target] <- inputPixels[source]
                    outputPixels[target + 1] <- inputPixels[source + 1]

            output
        with
        | _ ->
            decRef output
            reraise()

    let mapFloat32Vector name (scalarOp: float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32>) (chunk: Chunk<float32>) =
        let output = create<float32> chunk.Size
        try
            let input = span<float32> chunk
            let outputSpan = span<float32> output
            let width = Vector<float32>.Count
            let vectorEnd = input.Length - (input.Length % width)
            let mutable i = 0
            while i < vectorEnd do
                let result = vectorOp (Vector<float32>(input.Slice(i, width)))
                result.CopyTo(outputSpan.Slice(i, width))
                i <- i + width
            while i < input.Length do
                outputSpan[i] <- scalarOp input[i]
                i <- i + 1
            output
        with
        | _ ->
            decRef output
            reraise()

    let map2Float32Vector name (scalarOp: float32 -> float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32> -> Vector<float32>) (a: Chunk<float32>) (b: Chunk<float32>) =
        if a.Size <> b.Size then
            invalidArg "b" $"ChunkFunctions.{name} expects chunks with identical sizes, got {a.Size} and {b.Size}."
        let output = create<float32> a.Size
        try
            let aSpan = span<float32> a
            let bSpan = span<float32> b
            let outputSpan = span<float32> output
            let width = Vector<float32>.Count
            let vectorEnd = aSpan.Length - (aSpan.Length % width)
            let mutable i = 0
            while i < vectorEnd do
                let result = vectorOp (Vector<float32>(aSpan.Slice(i, width))) (Vector<float32>(bSpan.Slice(i, width)))
                result.CopyTo(outputSpan.Slice(i, width))
                i <- i + width
            while i < aSpan.Length do
                outputSpan[i] <- scalarOp aSpan[i] bSpan[i]
                i <- i + 1
            output
        with
        | _ ->
            decRef output
            reraise()

    let validateSameSize name (a: Chunk<'T>) (b: Chunk<'U>) =
        if a.Size <> b.Size then
            invalidArg "b" $"ChunkFunctions.{name} expects chunks with identical sizes, got {a.Size} and {b.Size}."

    let map2Chunk<'T, 'U, 'V when 'T: equality and 'U: equality and 'V: equality
                                    and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                                    and 'U: (new: unit -> 'U) and 'U: struct and 'U :> ValueType
                                    and 'V: (new: unit -> 'V) and 'V: struct and 'V :> ValueType>
        name
        f
        (a: Chunk<'T>)
        (b: Chunk<'U>) =
        validateSameSize name a b
        let output = create<'V> a.Size
        try
            let aSpan = span<'T> a
            let bSpan = span<'U> b
            let outputSpan = span<'V> output
            let mutable i = 0
            while i < aSpan.Length do
                outputSpan[i] <- f aSpan[i] bSpan[i]
                i <- i + 1
            output
        with
        | _ ->
            decRef output
            reraise()

    let inline sum<'T when 'T: equality
                        and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                        and 'T: (static member ( + ) : 'T * 'T -> 'T)
                        and 'T: (static member Zero : 'T)> (chunk: Chunk<'T>) : 'T =
        fold (fun acc value -> acc + value) LanguagePrimitives.GenericZero chunk

    let inline prod<'T when 'T: equality
                         and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                         and 'T: (static member ( * ) : 'T * 'T -> 'T)
                         and 'T: (static member One : 'T)> (chunk: Chunk<'T>) : 'T =
        fold (fun acc value -> acc * value) LanguagePrimitives.GenericOne chunk

    let inline minMax<'T when 'T: equality and 'T: comparison
                           and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        let values = span<'T> chunk
        if values.Length = 0 then
            invalidArg "chunk" "ChunkFunctions.minMax cannot reduce an empty chunk."
        let mutable mn = values[0]
        let mutable mx = values[0]
        let mutable i = 1
        while i < values.Length do
            let value = values[i]
            if value < mn then mn <- value
            if value > mx then mx <- value
            i <- i + 1
        mn, mx

    let inline getMinMax chunk = minMax chunk

    type ChunkStats =
        { NumPixels: uint64
          Mean: float
          Std: float
          Min: float
          Max: float
          Sum: float
          Var: float }

    let zeroStats =
        { NumPixels = 0UL
          Mean = 0.0
          Std = 0.0
          Min = Double.PositiveInfinity
          Max = Double.NegativeInfinity
          Sum = 0.0
          Var = 0.0 }

    let addStats (a: ChunkStats) (b: ChunkStats) =
        if a.NumPixels = 0UL then
            b
        elif b.NumPixels = 0UL then
            a
        else
            let nA = float a.NumPixels
            let nB = float b.NumPixels
            let n = nA + nB
            let delta = b.Mean - a.Mean
            let mean = a.Mean + delta * nB / n
            let m2A = a.Var * max 0.0 (nA - 1.0)
            let m2B = b.Var * max 0.0 (nB - 1.0)
            let m2 = m2A + m2B + delta * delta * nA * nB / n
            let var = if n > 1.0 then m2 / (n - 1.0) else 0.0

            { NumPixels = a.NumPixels + b.NumPixels
              Mean = mean
              Std = sqrt var
              Min = min a.Min b.Min
              Max = max a.Max b.Max
              Sum = a.Sum + b.Sum
              Var = var }

    let private statsFromMoments count mean m2 minimum maximum sum =
        let var = if count > 1UL then m2 / float (count - 1UL) else 0.0
        { NumPixels = count
          Mean = mean
          Std = sqrt var
          Min = minimum
          Max = maximum
          Sum = sum
          Var = var }

    let private statsFromSumSq count minimum maximum sum sumSq =
        let n = float count
        let mean = sum / n
        let var = if count > 1UL then max 0.0 ((sumSq - sum * sum / n) / (n - 1.0)) else 0.0
        { NumPixels = count
          Mean = mean
          Std = sqrt var
          Min = minimum
          Max = maximum
          Sum = sum
          Var = var }

    let private sumVectorUInt32 (v: Vector<uint32>) =
        let mutable sum = 0.0
        let mutable lane = 0
        while lane < Vector<uint32>.Count do
            sum <- sum + float v[lane]
            lane <- lane + 1
        sum

    let private sumVectorUInt64 (v: Vector<uint64>) =
        let mutable sum = 0.0
        let mutable lane = 0
        while lane < Vector<uint64>.Count do
            sum <- sum + float v[lane]
            lane <- lane + 1
        sum

    let private computeStatsUInt8Vector (values: Span<uint8>) =
        let width = Vector<byte>.Count
        let vectorEnd = values.Length - values.Length % width
        let flushEvery = 4096
        let mutable sum0 = Vector<uint32>.Zero
        let mutable sum1 = Vector<uint32>.Zero
        let mutable sum2 = Vector<uint32>.Zero
        let mutable sum3 = Vector<uint32>.Zero
        let mutable sq0 = Vector<uint32>.Zero
        let mutable sq1 = Vector<uint32>.Zero
        let mutable sq2 = Vector<uint32>.Zero
        let mutable sq3 = Vector<uint32>.Zero
        let mutable minAcc = Vector<byte>(Byte.MaxValue)
        let mutable maxAcc = Vector<byte>(Byte.MinValue)
        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable vectorsSinceFlush = 0

        let flush () =
            sum <- sum + sumVectorUInt32 sum0 + sumVectorUInt32 sum1 + sumVectorUInt32 sum2 + sumVectorUInt32 sum3
            sumSq <- sumSq + sumVectorUInt32 sq0 + sumVectorUInt32 sq1 + sumVectorUInt32 sq2 + sumVectorUInt32 sq3
            sum0 <- Vector<uint32>.Zero
            sum1 <- Vector<uint32>.Zero
            sum2 <- Vector<uint32>.Zero
            sum3 <- Vector<uint32>.Zero
            sq0 <- Vector<uint32>.Zero
            sq1 <- Vector<uint32>.Zero
            sq2 <- Vector<uint32>.Zero
            sq3 <- Vector<uint32>.Zero
            vectorsSinceFlush <- 0

        let mutable i = 0
        while i < vectorEnd do
            let v = Vector<byte>(values.Slice(i, width))
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            let mutable lo16 = Vector<uint16>.Zero
            let mutable hi16 = Vector<uint16>.Zero
            Vector.Widen(v, &lo16, &hi16)
            let mutable lo0 = Vector<uint32>.Zero
            let mutable lo1 = Vector<uint32>.Zero
            let mutable hi0 = Vector<uint32>.Zero
            let mutable hi1 = Vector<uint32>.Zero
            Vector.Widen(lo16, &lo0, &lo1)
            Vector.Widen(hi16, &hi0, &hi1)
            sum0 <- sum0 + lo0
            sum1 <- sum1 + lo1
            sum2 <- sum2 + hi0
            sum3 <- sum3 + hi1
            sq0 <- sq0 + lo0 * lo0
            sq1 <- sq1 + lo1 * lo1
            sq2 <- sq2 + hi0 * hi0
            sq3 <- sq3 + hi1 * hi1
            vectorsSinceFlush <- vectorsSinceFlush + 1
            if vectorsSinceFlush = flushEvery then flush()
            i <- i + width

        flush()

        let mutable minimum = float Byte.MaxValue
        let mutable maximum = float Byte.MinValue
        let mutable lane = 0
        while lane < Vector<byte>.Count do
            let minValue = float minAcc[lane]
            let maxValue = float maxAcc[lane]
            if minValue < minimum then minimum <- minValue
            if maxValue > maximum then maximum <- maxValue
            lane <- lane + 1

        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1

        statsFromSumSq (uint64 values.Length) minimum maximum sum sumSq

    let private computeStatsUInt16Vector (values: Span<uint16>) =
        let width = Vector<uint16>.Count
        let vectorEnd = values.Length - values.Length % width
        let flushEvery = 4096
        let mutable sum0 = Vector<uint32>.Zero
        let mutable sum1 = Vector<uint32>.Zero
        let mutable sq0 = Vector<uint64>.Zero
        let mutable sq1 = Vector<uint64>.Zero
        let mutable sq2 = Vector<uint64>.Zero
        let mutable sq3 = Vector<uint64>.Zero
        let mutable minAcc = Vector<uint16>(UInt16.MaxValue)
        let mutable maxAcc = Vector<uint16>(UInt16.MinValue)
        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable vectorsSinceFlush = 0

        let flush () =
            sum <- sum + sumVectorUInt32 sum0 + sumVectorUInt32 sum1
            sumSq <- sumSq + sumVectorUInt64 sq0 + sumVectorUInt64 sq1 + sumVectorUInt64 sq2 + sumVectorUInt64 sq3
            sum0 <- Vector<uint32>.Zero
            sum1 <- Vector<uint32>.Zero
            sq0 <- Vector<uint64>.Zero
            sq1 <- Vector<uint64>.Zero
            sq2 <- Vector<uint64>.Zero
            sq3 <- Vector<uint64>.Zero
            vectorsSinceFlush <- 0

        let mutable i = 0
        while i < vectorEnd do
            let v = Vector<uint16>(values.Slice(i, width))
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            let mutable lo = Vector<uint32>.Zero
            let mutable hi = Vector<uint32>.Zero
            Vector.Widen(v, &lo, &hi)
            sum0 <- sum0 + lo
            sum1 <- sum1 + hi
            let loSq = lo * lo
            let hiSq = hi * hi
            let mutable loSq0 = Vector<uint64>.Zero
            let mutable loSq1 = Vector<uint64>.Zero
            let mutable hiSq0 = Vector<uint64>.Zero
            let mutable hiSq1 = Vector<uint64>.Zero
            Vector.Widen(loSq, &loSq0, &loSq1)
            Vector.Widen(hiSq, &hiSq0, &hiSq1)
            sq0 <- sq0 + loSq0
            sq1 <- sq1 + loSq1
            sq2 <- sq2 + hiSq0
            sq3 <- sq3 + hiSq1
            vectorsSinceFlush <- vectorsSinceFlush + 1
            if vectorsSinceFlush = flushEvery then flush()
            i <- i + width

        flush()

        let mutable minimum = float UInt16.MaxValue
        let mutable maximum = float UInt16.MinValue
        let mutable lane = 0
        while lane < Vector<uint16>.Count do
            let minValue = float minAcc[lane]
            let maxValue = float maxAcc[lane]
            if minValue < minimum then minimum <- minValue
            if maxValue > maximum then maximum <- maxValue
            lane <- lane + 1

        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1

        statsFromSumSq (uint64 values.Length) minimum maximum sum sumSq

    let private computeStatsFloat32VectorAccurate (values: Span<float32>) =
        let width = Vector<float32>.Count
        let vectorEnd = values.Length - values.Length % width
        let mutable sum0 = Vector<float>.Zero
        let mutable sum1 = Vector<float>.Zero
        let mutable sumSq0 = Vector<float>.Zero
        let mutable sumSq1 = Vector<float>.Zero
        let mutable minAcc = Vector<float32>(Single.PositiveInfinity)
        let mutable maxAcc = Vector<float32>(Single.NegativeInfinity)
        let mutable i = 0

        while i < vectorEnd do
            let v = Vector<float32>(values.Slice(i, width))
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            let mutable lo = Vector<float>.Zero
            let mutable hi = Vector<float>.Zero
            Vector.Widen(v, &lo, &hi)
            sum0 <- sum0 + lo
            sum1 <- sum1 + hi
            sumSq0 <- sumSq0 + lo * lo
            sumSq1 <- sumSq1 + hi * hi
            i <- i + width

        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable lane = 0
        while lane < Vector<float>.Count do
            sum <- sum + sum0[lane] + sum1[lane]
            sumSq <- sumSq + sumSq0[lane] + sumSq1[lane]
            lane <- lane + 1

        let mutable minimum = Double.PositiveInfinity
        let mutable maximum = Double.NegativeInfinity
        lane <- 0
        while lane < Vector<float32>.Count do
            let minValue = float minAcc[lane]
            let maxValue = float maxAcc[lane]
            if minValue < minimum then minimum <- minValue
            if maxValue > maximum then maximum <- maxValue
            lane <- lane + 1

        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1

        statsFromSumSq (uint64 values.Length) minimum maximum sum sumSq

    let private computeStatsFloat32VectorStable (values: Span<float32>) =
        let width = Vector<float32>.Count
        let vectorEnd = values.Length - values.Length % width
        let mutable sum0 = Vector<float>.Zero
        let mutable sum1 = Vector<float>.Zero
        let mutable minAcc = Vector<float32>(Single.PositiveInfinity)
        let mutable maxAcc = Vector<float32>(Single.NegativeInfinity)
        let mutable i = 0

        while i < vectorEnd do
            let v = Vector<float32>(values.Slice(i, width))
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            let mutable lo = Vector<float>.Zero
            let mutable hi = Vector<float>.Zero
            Vector.Widen(v, &lo, &hi)
            sum0 <- sum0 + lo
            sum1 <- sum1 + hi
            i <- i + width

        let mutable sum = 0.0
        let mutable minimum = Double.PositiveInfinity
        let mutable maximum = Double.NegativeInfinity
        let mutable lane = 0
        while lane < Vector<float>.Count do
            sum <- sum + sum0[lane] + sum1[lane]
            lane <- lane + 1

        lane <- 0
        while lane < Vector<float32>.Count do
            let minValue = float minAcc[lane]
            let maxValue = float maxAcc[lane]
            if minValue < minimum then minimum <- minValue
            if maxValue > maximum then maximum <- maxValue
            lane <- lane + 1

        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1

        let count = uint64 values.Length
        let mean = sum / float values.Length
        let meanVector = Vector<float>(mean)
        let mutable m2Acc0 = Vector<float>.Zero
        let mutable m2Acc1 = Vector<float>.Zero
        i <- 0

        while i < vectorEnd do
            let v = Vector<float32>(values.Slice(i, width))
            let mutable lo = Vector<float>.Zero
            let mutable hi = Vector<float>.Zero
            Vector.Widen(v, &lo, &hi)
            let delta0 = lo - meanVector
            let delta1 = hi - meanVector
            m2Acc0 <- m2Acc0 + delta0 * delta0
            m2Acc1 <- m2Acc1 + delta1 * delta1
            i <- i + width

        let mutable m2 = 0.0
        lane <- 0
        while lane < Vector<float>.Count do
            m2 <- m2 + m2Acc0[lane] + m2Acc1[lane]
            lane <- lane + 1

        while i < values.Length do
            let delta = float values[i] - mean
            m2 <- m2 + delta * delta
            i <- i + 1

        statsFromMoments count mean m2 minimum maximum sum

    let private computeStatsFloat64Vector (values: Span<float>) =
        let width = Vector<float>.Count
        let vectorEnd = values.Length - values.Length % width
        let mutable sumAcc = Vector<float>.Zero
        let mutable sumSqAcc = Vector<float>.Zero
        let mutable minAcc = Vector<float>(Double.PositiveInfinity)
        let mutable maxAcc = Vector<float>(Double.NegativeInfinity)
        let mutable i = 0

        while i < vectorEnd do
            let v = Vector<float>(values.Slice(i, width))
            sumAcc <- sumAcc + v
            sumSqAcc <- sumSqAcc + v * v
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            i <- i + width

        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable minimum = Double.PositiveInfinity
        let mutable maximum = Double.NegativeInfinity
        let mutable lane = 0
        while lane < Vector<float>.Count do
            sum <- sum + sumAcc[lane]
            sumSq <- sumSq + sumSqAcc[lane]
            if minAcc[lane] < minimum then minimum <- minAcc[lane]
            if maxAcc[lane] > maximum then maximum <- maxAcc[lane]
            lane <- lane + 1

        while i < values.Length do
            let value = values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1

        statsFromSumSq (uint64 values.Length) minimum maximum sum sumSq

    let inline private computeStatsTyped (values: Span< ^T>) =
        let first = float values[0]
        let mutable minimum = first
        let mutable maximum = first
        let mutable sum = first
        let mutable sumSq = first * first
        let mutable i = 1

        while i < values.Length do
            let value = float values[i]
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            sum <- sum + value
            sumSq <- sumSq + value * value
            i <- i + 1

        let count = uint64 values.Length
        let n = float values.Length
        let mean = sum / n
        let var = if values.Length > 1 then max 0.0 ((sumSq - sum * sum / n) / (n - 1.0)) else 0.0
        { NumPixels = count
          Mean = mean
          Std = sqrt var
          Min = minimum
          Max = maximum
          Sum = sum
          Var = var }

    let computeStats<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'T>)
        =
        let values = span<'T> chunk
        if values.Length = 0 then
            zeroStats
        else
            let t = typeof<'T>
            if t = typeof<uint8> then
                computeStatsUInt8Vector (MemoryMarshal.Cast<'T, uint8>(values))
            elif t = typeof<int8> then
                computeStatsTyped (MemoryMarshal.Cast<'T, int8>(values))
            elif t = typeof<uint16> then
                computeStatsUInt16Vector (MemoryMarshal.Cast<'T, uint16>(values))
            elif t = typeof<int16> then
                computeStatsTyped (MemoryMarshal.Cast<'T, int16>(values))
            elif t = typeof<uint32> then
                computeStatsTyped (MemoryMarshal.Cast<'T, uint32>(values))
            elif t = typeof<int32> then
                computeStatsTyped (MemoryMarshal.Cast<'T, int32>(values))
            elif t = typeof<float32> then
                computeStatsFloat32VectorStable (MemoryMarshal.Cast<'T, float32>(values))
            elif t = typeof<float> then
                computeStatsFloat64Vector (MemoryMarshal.Cast<'T, float>(values))
            else
                let first = Convert.ToDouble(box values[0])
                let mutable count = 1UL
                let mutable mean = first
                let mutable m2 = 0.0
                let mutable minimum = first
                let mutable maximum = first
                let mutable sum = first
                let mutable i = 1
                while i < values.Length do
                    let value = Convert.ToDouble(box values[i])
                    count <- count + 1UL
                    let n = float count
                    let delta = value - mean
                    mean <- mean + delta / n
                    m2 <- m2 + delta * (value - mean)
                    if value < minimum then minimum <- value
                    if value > maximum then maximum <- value
                    sum <- sum + value
                    i <- i + 1
                statsFromMoments count mean m2 minimum maximum sum

    let inline private clampRoundToByte (value: float32) =
        if Single.IsNaN value || value <= 0.0f then
            0uy
        elif value >= 255.0f then
            255uy
        else
            uint8 (MathF.Round value)

    let inline private clampRoundToSByte (value: float32) =
        if Single.IsNaN value then
            0y
        elif value <= float32 SByte.MinValue then
            SByte.MinValue
        elif value >= float32 SByte.MaxValue then
            SByte.MaxValue
        else
            int8 (MathF.Round value)

    let inline private clampRoundToUInt16 (value: float32) =
        if Single.IsNaN value || value <= 0.0f then
            0us
        elif value >= 65535.0f then
            65535us
        else
            uint16 (MathF.Round value)

    let inline private clampRoundToInt16 (value: float32) =
        if Single.IsNaN value then
            0s
        elif value <= float32 Int16.MinValue then
            Int16.MinValue
        elif value >= float32 Int16.MaxValue then
            Int16.MaxValue
        else
            int16 (MathF.Round value)

    let inline private clampRoundToInt32 (value: float32) =
        if Single.IsNaN value then
            0
        elif value <= float32 Int32.MinValue then
            Int32.MinValue
        elif value >= float32 Int32.MaxValue then
            Int32.MaxValue
        else
            int32 (MathF.Round value)

    let inline private clampRoundToUInt32 (value: float32) =
        if Single.IsNaN value || value <= 0.0f then
            0u
        elif value >= float32 UInt32.MaxValue then
            UInt32.MaxValue
        else
            uint32 (MathF.Round value)

    let inline private clampRoundDoubleToSByte (value: double) =
        if Double.IsNaN value then
            0y
        elif value <= double SByte.MinValue then
            SByte.MinValue
        elif value >= double SByte.MaxValue then
            SByte.MaxValue
        else
            sbyte (Math.Round value)

    let inline private clampRoundDoubleToByte (value: double) =
        if Double.IsNaN value || value <= 0.0 then
            0uy
        elif value >= double Byte.MaxValue then
            Byte.MaxValue
        else
            byte (Math.Round value)

    let inline private clampRoundDoubleToInt16 (value: double) =
        if Double.IsNaN value then
            0s
        elif value <= double Int16.MinValue then
            Int16.MinValue
        elif value >= double Int16.MaxValue then
            Int16.MaxValue
        else
            int16 (Math.Round value)

    let inline private clampRoundDoubleToUInt16 (value: double) =
        if Double.IsNaN value || value <= 0.0 then
            0us
        elif value >= double UInt16.MaxValue then
            UInt16.MaxValue
        else
            uint16 (Math.Round value)

    let inline private clampRoundDoubleToInt32 (value: double) =
        if Double.IsNaN value then
            0
        elif value <= double Int32.MinValue then
            Int32.MinValue
        elif value >= double Int32.MaxValue then
            Int32.MaxValue
        else
            int32 (Math.Round value)

    let inline private clampRoundDoubleToUInt32 (value: double) =
        if Double.IsNaN value || value <= 0.0 then
            0u
        elif value >= double UInt32.MaxValue then
            UInt32.MaxValue
        else
            uint32 (Math.Round value)

    let inline private clampRoundDoubleToInt64 (value: double) =
        if Double.IsNaN value then
            0L
        elif value <= double Int64.MinValue then
            Int64.MinValue
        elif value >= double Int64.MaxValue then
            Int64.MaxValue
        else
            int64 (Math.Round value)

    let inline private clampRoundDoubleToUInt64 (value: double) =
        if Double.IsNaN value || value <= 0.0 then
            0UL
        elif value >= double UInt64.MaxValue then
            UInt64.MaxValue
        else
            uint64 (Math.Round value)

    let private byteVectorToSingleVectors (source: ReadOnlySpan<byte>) =
        let bytes = MemoryMarshal.Read<Vector<byte>>(source)
        let mutable lo16 = Vector<uint16>.Zero
        let mutable hi16 = Vector<uint16>.Zero
        Vector.Widen(bytes, &lo16, &hi16)
        let mutable loLo32 = Vector<uint32>.Zero
        let mutable loHi32 = Vector<uint32>.Zero
        let mutable hiLo32 = Vector<uint32>.Zero
        let mutable hiHi32 = Vector<uint32>.Zero
        Vector.Widen(lo16, &loLo32, &loHi32)
        Vector.Widen(hi16, &hiLo32, &hiHi32)
        Vector.ConvertToSingle(loLo32), Vector.ConvertToSingle(loHi32), Vector.ConvertToSingle(hiLo32), Vector.ConvertToSingle(hiHi32)

    let private uint16VectorToSingleVectors (source: ReadOnlySpan<uint16>) =
        let values = Vector<uint16>(source)
        let mutable lo32 = Vector<uint32>.Zero
        let mutable hi32 = Vector<uint32>.Zero
        Vector.Widen(values, &lo32, &hi32)
        Vector.ConvertToSingle(lo32), Vector.ConvertToSingle(hi32)

    let private int8VectorToSingleVectors (source: ReadOnlySpan<sbyte>) =
        let values = Vector<sbyte>(source)
        let mutable lo16 = Vector<int16>.Zero
        let mutable hi16 = Vector<int16>.Zero
        Vector.Widen(values, &lo16, &hi16)
        let mutable loLo32 = Vector<int32>.Zero
        let mutable loHi32 = Vector<int32>.Zero
        let mutable hiLo32 = Vector<int32>.Zero
        let mutable hiHi32 = Vector<int32>.Zero
        Vector.Widen(lo16, &loLo32, &loHi32)
        Vector.Widen(hi16, &hiLo32, &hiHi32)
        Vector.ConvertToSingle(loLo32), Vector.ConvertToSingle(loHi32), Vector.ConvertToSingle(hiLo32), Vector.ConvertToSingle(hiHi32)

    let private int16VectorToSingleVectors (source: ReadOnlySpan<int16>) =
        let values = Vector<int16>(source)
        let mutable lo32 = Vector<int32>.Zero
        let mutable hi32 = Vector<int32>.Zero
        Vector.Widen(values, &lo32, &hi32)
        Vector.ConvertToSingle(lo32), Vector.ConvertToSingle(hi32)

    let private thresholdUInt8Vector (threshold: byte) (inputPixels: Span<byte>) (outputPixels: Span<byte>) =
        let width = Vector<byte>.Count
        let vectorEnd = inputPixels.Length - inputPixels.Length % width
        let thresholdVector = Vector<byte>(threshold)
        let oneVector = Vector<byte>(1uy)
        let mutable i = 0

        while i < vectorEnd do
            let values = Vector<byte>(inputPixels.Slice(i, width))
            let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
            let selected: Vector<byte> = Vector.BitwiseAnd(mask, oneVector)
            let mutable lane = 0
            while lane < width do
                outputPixels[i + lane] <- selected[lane]
                lane <- lane + 1
            i <- i + width

        while i < inputPixels.Length do
            outputPixels[i] <- if inputPixels[i] >= threshold then 1uy else 0uy
            i <- i + 1

    let private thresholdUInt16Vector (threshold: uint16) (inputPixels: Span<uint16>) (outputPixels: Span<uint16>) =
        let width = Vector<uint16>.Count
        let vectorEnd = inputPixels.Length - inputPixels.Length % width
        let thresholdVector = Vector<uint16>(threshold)
        let oneVector = Vector<uint16>(1us)
        let mutable i = 0

        while i < vectorEnd do
            let values = Vector<uint16>(inputPixels.Slice(i, width))
            let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
            Vector.BitwiseAnd(mask, oneVector).CopyTo(outputPixels.Slice(i, width))
            i <- i + width

        while i < inputPixels.Length do
            outputPixels[i] <- if inputPixels[i] >= threshold then 1us else 0us
            i <- i + 1

    let private thresholdFloat32Vector (threshold: float32) (inputPixels: Span<float32>) (outputPixels: Span<float32>) =
        let width = Vector<float32>.Count
        let vectorEnd = inputPixels.Length - inputPixels.Length % width
        let thresholdVector = Vector<float32>(threshold)
        let oneVector = Vector<float32>(1.0f)
        let zeroVector = Vector<float32>.Zero
        let mutable i = 0

        while i < vectorEnd do
            let values = Vector<float32>(inputPixels.Slice(i, width))
            let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
            Vector.ConditionalSelect(mask, oneVector, zeroVector).CopyTo(outputPixels.Slice(i, width))
            i <- i + width

        while i < inputPixels.Length do
            outputPixels[i] <- if inputPixels[i] >= threshold then 1.0f else 0.0f
            i <- i + 1

    let castUInt8SpanToFloat32 (inputPixels: ReadOnlySpan<byte>) (outputPixels: Span<float32>) =
        if outputPixels.Length < inputPixels.Length then
            invalidArg "outputPixels" $"Output span length {outputPixels.Length} is too small for {inputPixels.Length} widened UInt8 values."

        let byteVectorWidth = Vector<byte>.Count
        let floatVectorWidth = Vector<float32>.Count
        let vectorEnd = inputPixels.Length - (inputPixels.Length % byteVectorWidth)
        let mutable i = 0

        while i < vectorEnd do
            let inputSlice = inputPixels.Slice(i, byteVectorWidth)
            let a, b, c, d = byteVectorToSingleVectors inputSlice
            a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
            b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
            c.CopyTo(outputPixels.Slice(i + 2 * floatVectorWidth, floatVectorWidth))
            d.CopyTo(outputPixels.Slice(i + 3 * floatVectorWidth, floatVectorWidth))
            i <- i + byteVectorWidth

        while i < inputPixels.Length do
            outputPixels[i] <- float32 inputPixels[i]
            i <- i + 1

    let private clampRoundFloat32ToUInt32Vector (maximum: float32) (values: Vector<float32>) =
        let zero = Vector<float32>.Zero
        let maximumV = Vector<float32>(maximum)
        let finiteValues = Vector.ConditionalSelect(Vector.Equals(values, values), values, zero)
        Vector.ConvertToUInt32(Vector.Round(Vector.Min(maximumV, Vector.Max(zero, finiteValues))))

    let private castFloat32ToUInt16Vector (inputPixels: Span<float32>) (outputPixels: Span<uint16>) =
        let floatWidth = Vector<float32>.Count
        let outputWidth = Vector<uint16>.Count
        let vectorEnd = inputPixels.Length - inputPixels.Length % outputWidth
        let mutable i = 0

        while i < vectorEnd do
            let lo = clampRoundFloat32ToUInt32Vector 65535.0f (Vector<float32>(inputPixels.Slice(i, floatWidth)))
            let hi = clampRoundFloat32ToUInt32Vector 65535.0f (Vector<float32>(inputPixels.Slice(i + floatWidth, floatWidth)))
            Vector.NarrowWithSaturation(lo, hi).CopyTo(outputPixels.Slice(i, outputWidth))
            i <- i + outputWidth

        while i < inputPixels.Length do
            outputPixels[i] <- clampRoundToUInt16 inputPixels[i]
            i <- i + 1

    let private castFloat32ToUInt8Vector (inputPixels: Span<float32>) (outputPixels: Span<byte>) =
        let floatWidth = Vector<float32>.Count
        let uint16Width = Vector<uint16>.Count
        let byteWidth = Vector<byte>.Count
        let vectorEnd = inputPixels.Length - inputPixels.Length % byteWidth
        let mutable i = 0

        while i < vectorEnd do
            let a = clampRoundFloat32ToUInt32Vector 255.0f (Vector<float32>(inputPixels.Slice(i, floatWidth)))
            let b = clampRoundFloat32ToUInt32Vector 255.0f (Vector<float32>(inputPixels.Slice(i + floatWidth, floatWidth)))
            let c = clampRoundFloat32ToUInt32Vector 255.0f (Vector<float32>(inputPixels.Slice(i + 2 * floatWidth, floatWidth)))
            let d = clampRoundFloat32ToUInt32Vector 255.0f (Vector<float32>(inputPixels.Slice(i + 3 * floatWidth, floatWidth)))
            let lo16: Vector<uint16> = Vector.NarrowWithSaturation(a, b)
            let hi16: Vector<uint16> = Vector.NarrowWithSaturation(c, d)
            let mutable packed: Vector<byte> = Vector.NarrowWithSaturation(lo16, hi16)
            MemoryMarshal.Write(outputPixels.Slice(i, byteWidth), &packed)
            i <- i + byteWidth

        while i < inputPixels.Length do
            outputPixels[i] <- clampRoundToByte inputPixels[i]
            i <- i + 1

    let thresholdNativeChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (threshold: double) (chunk: Chunk<'T>) =
        let output = Chunk.create<'T> chunk.Size
        try
            let t = typeof<'T>
            if t = typeof<uint8> then
                let threshold = byte (Math.Clamp(Math.Ceiling(threshold), 0.0, 255.0))
                let inputPixels = chunk.Bytes.AsSpan(0, chunk.ByteLength)
                let outputPixels = output.Bytes.AsSpan(0, output.ByteLength)
                thresholdUInt8Vector threshold inputPixels outputPixels
            elif t = typeof<int8> then
                let threshold = sbyte (Math.Clamp(Math.Ceiling(threshold), float SByte.MinValue, float SByte.MaxValue))
                let inputPixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let outputPixels = MemoryMarshal.Cast<byte, sbyte>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- if inputPixels[i] >= threshold then 1y else 0y
                    i <- i + 1
            elif t = typeof<uint16> then
                let threshold = uint16 (Math.Clamp(Math.Ceiling(threshold), 0.0, float UInt16.MaxValue))
                let inputPixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let outputPixels = MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))
                thresholdUInt16Vector threshold inputPixels outputPixels
            elif t = typeof<int16> then
                let threshold = int16 (Math.Clamp(Math.Ceiling(threshold), float Int16.MinValue, float Int16.MaxValue))
                let inputPixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let outputPixels = MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- if inputPixels[i] >= threshold then 1s else 0s
                    i <- i + 1
            elif t = typeof<int32> then
                let threshold = int32 (Math.Clamp(Math.Ceiling(threshold), float Int32.MinValue, float Int32.MaxValue))
                let inputPixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let outputPixels = MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- if inputPixels[i] >= threshold then 1 else 0
                    i <- i + 1
            elif t = typeof<float32> then
                let threshold = float32 threshold
                let inputPixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let outputPixels = MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))
                thresholdFloat32Vector threshold inputPixels outputPixels
            else
                invalidArg "T" $"ChunkFunctions.thresholdNative supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."
            output
        with
        | _ ->
            Chunk.decRef output
            reraise()
    let castChunkToUInt8<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        let output = Chunk.create<uint8> chunk.Size
        try
            let t = typeof<'T>
            let outputPixels = output.Bytes.AsSpan(0, output.ByteLength)
            if t = typeof<uint8> then
                chunk.Bytes.AsSpan(0, chunk.ByteLength).CopyTo(outputPixels)
            elif t = typeof<int8> then
                let inputPixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    let value = inputPixels[i]
                    outputPixels[i] <- if value <= 0y then 0uy else uint8 value
                    i <- i + 1
            elif t = typeof<uint16> then
                let inputPixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    let value = inputPixels[i]
                    outputPixels[i] <- if value >= 255us then 255uy else uint8 value
                    i <- i + 1
            elif t = typeof<int16> then
                let inputPixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    let value = inputPixels[i]
                    outputPixels[i] <- if value <= 0s then 0uy elif value >= 255s then 255uy else uint8 value
                    i <- i + 1
            elif t = typeof<int32> then
                let inputPixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    let value = inputPixels[i]
                    outputPixels[i] <- if value <= 0 then 0uy elif value >= 255 then 255uy else uint8 value
                    i <- i + 1
            elif t = typeof<uint32> then
                let inputPixels = MemoryMarshal.Cast<byte, uint32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    let value = inputPixels[i]
                    outputPixels[i] <- if value >= 255u then 255uy else uint8 value
                    i <- i + 1
            elif t = typeof<int64> then
                let inputPixels = MemoryMarshal.Cast<byte, int64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    let value = inputPixels[i]
                    outputPixels[i] <- if value <= 0L then 0uy elif value >= 255L then 255uy else uint8 value
                    i <- i + 1
            elif t = typeof<uint64> then
                let inputPixels = MemoryMarshal.Cast<byte, uint64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    let value = inputPixels[i]
                    outputPixels[i] <- if value >= 255UL then 255uy else uint8 value
                    i <- i + 1
            elif t = typeof<float32> then
                let inputPixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                castFloat32ToUInt8Vector inputPixels outputPixels
            elif t = typeof<float> then
                let inputPixels = MemoryMarshal.Cast<byte, float>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- clampRoundToByte (float32 inputPixels[i])
                    i <- i + 1
            else
                invalidArg "T" $"ChunkFunctions.castToUInt8 supports real numeric chunks, got {t.Name}."
            output
        with
        | _ ->
            Chunk.decRef output
            reraise()
    let castChunkToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        let output = Chunk.create<float32> chunk.Size
        try
            let outputPixels = Chunk.span<float32> output
            let t = typeof<'T>
            if t = typeof<float32> then
                let inputPixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                inputPixels.CopyTo(outputPixels)
            elif t = typeof<uint8> then
                let inputPixels = ReadOnlySpan<byte>(chunk.Bytes, 0, chunk.ByteLength)
                castUInt8SpanToFloat32 inputPixels outputPixels
            elif t = typeof<int8> then
                let inputPixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let int8VectorWidth = Vector<sbyte>.Count
                let floatVectorWidth = Vector<float32>.Count
                let vectorEnd = inputPixels.Length - (inputPixels.Length % int8VectorWidth)
                let mutable i = 0
                while i < vectorEnd do
                    let mutable inputPart = inputPixels.Slice(i, int8VectorWidth)
                    let inputSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(inputPart), int8VectorWidth)
                    let a, b, c, d = int8VectorToSingleVectors inputSlice
                    a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
                    b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
                    c.CopyTo(outputPixels.Slice(i + 2 * floatVectorWidth, floatVectorWidth))
                    d.CopyTo(outputPixels.Slice(i + 3 * floatVectorWidth, floatVectorWidth))
                    i <- i + int8VectorWidth
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<uint16> then
                let inputPixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let uint16VectorWidth = Vector<uint16>.Count
                let floatVectorWidth = Vector<float32>.Count
                let vectorEnd = inputPixels.Length - (inputPixels.Length % uint16VectorWidth)
                let mutable i = 0
                while i < vectorEnd do
                    let mutable inputPart = inputPixels.Slice(i, uint16VectorWidth)
                    let inputSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(inputPart), uint16VectorWidth)
                    let a, b = uint16VectorToSingleVectors inputSlice
                    a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
                    b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
                    i <- i + uint16VectorWidth
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<int16> then
                let inputPixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let int16VectorWidth = Vector<int16>.Count
                let floatVectorWidth = Vector<float32>.Count
                let vectorEnd = inputPixels.Length - (inputPixels.Length % int16VectorWidth)
                let mutable i = 0
                while i < vectorEnd do
                    let mutable inputPart = inputPixels.Slice(i, int16VectorWidth)
                    let inputSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(inputPart), int16VectorWidth)
                    let a, b = int16VectorToSingleVectors inputSlice
                    a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
                    b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
                    i <- i + int16VectorWidth
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<int32> then
                let inputPixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let vectorWidth = Vector<int32>.Count
                let vectorEnd = inputPixels.Length - (inputPixels.Length % vectorWidth)
                let mutable i = 0
                while i < vectorEnd do
                    Vector.ConvertToSingle(Vector<int32>(inputPixels.Slice(i, vectorWidth))).CopyTo(outputPixels.Slice(i, vectorWidth))
                    i <- i + vectorWidth
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<uint32> then
                let inputPixels = MemoryMarshal.Cast<byte, uint32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<int64> then
                let inputPixels = MemoryMarshal.Cast<byte, int64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<uint64> then
                let inputPixels = MemoryMarshal.Cast<byte, uint64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<float> then
                let inputPixels = MemoryMarshal.Cast<byte, float>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- float32 inputPixels[i]
                    i <- i + 1
            else
                invalidArg "T" $"ChunkFunctions.castToFloat32 supports real numeric chunks, got {t.Name}."
            output
        with
        | _ ->
            Chunk.decRef output
            reraise()
    let castFloat32ToChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<float32>) =
        let output = Chunk.create<'T> chunk.Size
        try
            let inputPixels = Chunk.span<float32> chunk
            let t = typeof<'T>
            if t = typeof<float32> then
                let outputPixels = MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))
                inputPixels.CopyTo(outputPixels)
            elif t = typeof<uint8> then
                let outputPixels = output.Bytes.AsSpan(0, output.ByteLength)
                castFloat32ToUInt8Vector inputPixels outputPixels
            elif t = typeof<int8> then
                let outputPixels = MemoryMarshal.Cast<byte, sbyte>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- clampRoundToSByte inputPixels[i]
                    i <- i + 1
            elif t = typeof<uint16> then
                let outputPixels = MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))
                castFloat32ToUInt16Vector inputPixels outputPixels
            elif t = typeof<int16> then
                let outputPixels = MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- clampRoundToInt16 inputPixels[i]
                    i <- i + 1
            elif t = typeof<int32> then
                let outputPixels = MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- clampRoundToInt32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<uint32> then
                let outputPixels = MemoryMarshal.Cast<byte, uint32>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- clampRoundToUInt32 inputPixels[i]
                    i <- i + 1
            elif t = typeof<int64> then
                let outputPixels = MemoryMarshal.Cast<byte, int64>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- clampRoundDoubleToInt64 (double inputPixels[i])
                    i <- i + 1
            elif t = typeof<uint64> then
                let outputPixels = MemoryMarshal.Cast<byte, uint64>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- clampRoundDoubleToUInt64 (double inputPixels[i])
                    i <- i + 1
            elif t = typeof<float> then
                let outputPixels = MemoryMarshal.Cast<byte, float>(output.Bytes.AsSpan(0, output.ByteLength))
                let mutable i = 0
                while i < inputPixels.Length do
                    outputPixels[i] <- double inputPixels[i]
                    i <- i + 1
            else
                invalidArg "T" $"ChunkFunctions.castFromFloat32 supports real numeric chunks, got {t.Name}."
            output
        with
        | _ ->
            Chunk.decRef output
            reraise()

    let private readRealAsDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'T>)
        index
        =
        let t = typeof<'T>
        if t = typeof<uint8> then
            double chunk.Bytes[index]
        elif t = typeof<int8> then
            let pixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<uint16> then
            let pixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<int16> then
            let pixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<uint32> then
            let pixels = MemoryMarshal.Cast<byte, uint32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<int32> then
            let pixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<uint64> then
            let pixels = MemoryMarshal.Cast<byte, uint64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<int64> then
            let pixels = MemoryMarshal.Cast<byte, int64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<float32> then
            let pixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            double pixels[index]
        elif t = typeof<float> then
            let pixels = MemoryMarshal.Cast<byte, float>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            pixels[index]
        else
            invalidArg "T" $"ChunkFunctions.cast supports real numeric chunks, got {t.Name}."

    let private writeNumericValue<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (output: Chunk<'T>)
        index
        real
        =
        let t = typeof<'T>
        if t = typeof<uint8> then
            output.Bytes[index] <- clampRoundDoubleToByte real
        elif t = typeof<int8> then
            let pixels = MemoryMarshal.Cast<byte, sbyte>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- clampRoundDoubleToSByte real
        elif t = typeof<uint16> then
            let pixels = MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- clampRoundDoubleToUInt16 real
        elif t = typeof<int16> then
            let pixels = MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- clampRoundDoubleToInt16 real
        elif t = typeof<uint32> then
            let pixels = MemoryMarshal.Cast<byte, uint32>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- clampRoundDoubleToUInt32 real
        elif t = typeof<int32> then
            let pixels = MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- clampRoundDoubleToInt32 real
        elif t = typeof<uint64> then
            let pixels = MemoryMarshal.Cast<byte, uint64>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- clampRoundDoubleToUInt64 real
        elif t = typeof<int64> then
            let pixels = MemoryMarshal.Cast<byte, int64>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- clampRoundDoubleToInt64 real
        elif t = typeof<float32> then
            let pixels = MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- float32 real
        elif t = typeof<float> then
            let pixels = MemoryMarshal.Cast<byte, float>(output.Bytes.AsSpan(0, output.ByteLength))
            pixels[index] <- real
        else
            invalidArg "T" $"ChunkFunctions.cast supports real numeric chunks, got {t.Name}."

    let castChunk<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                            and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'S>)
        =
        if typeof<'S> = typeof<'T> then
            unbox (box (copyChunk chunk))
        else
            let output = Chunk.create<'T> chunk.Size
            try
                let count = chunk.ByteLength / Marshal.SizeOf<'S>()
                for i in 0 .. count - 1 do
                    writeNumericValue output i (readRealAsDouble chunk i)
                output
            with
            | _ ->
                Chunk.decRef output
                reraise()

    let private writeFloatToTypedOutput<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (output: Chunk<'T>)
        index
        value
        =
        let t = typeof<'T>
        if t = typeof<float32> then
            let outputPixels = MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))
            outputPixels[index] <- value
        elif t = typeof<uint8> then
            output.Bytes[index] <- clampRoundToByte value
        elif t = typeof<int8> then
            let outputPixels = MemoryMarshal.Cast<byte, sbyte>(output.Bytes.AsSpan(0, output.ByteLength))
            outputPixels[index] <- clampRoundToSByte value
        elif t = typeof<uint16> then
            let outputPixels = MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))
            outputPixels[index] <- clampRoundToUInt16 value
        elif t = typeof<int16> then
            let outputPixels = MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength))
            outputPixels[index] <- clampRoundToInt16 value
        elif t = typeof<int32> then
            let outputPixels = MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))
            outputPixels[index] <- clampRoundToInt32 value
        else
            invalidArg "T" $"ChunkFunctions noise supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."

    let private copyTypedValue<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (input: Chunk<'T>)
        (output: Chunk<'T>)
        index
        =
        let t = typeof<'T>
        if t = typeof<float32> then
            let inputPixels = MemoryMarshal.Cast<byte, float32>(input.Bytes.AsSpan(0, input.ByteLength))
            writeFloatToTypedOutput output index inputPixels[index]
        elif t = typeof<uint8> then
            output.Bytes[index] <- input.Bytes[index]
        elif t = typeof<int8> then
            let inputPixels = MemoryMarshal.Cast<byte, sbyte>(input.Bytes.AsSpan(0, input.ByteLength))
            writeFloatToTypedOutput output index (float32 inputPixels[index])
        elif t = typeof<uint16> then
            let inputPixels = MemoryMarshal.Cast<byte, uint16>(input.Bytes.AsSpan(0, input.ByteLength))
            writeFloatToTypedOutput output index (float32 inputPixels[index])
        elif t = typeof<int16> then
            let inputPixels = MemoryMarshal.Cast<byte, int16>(input.Bytes.AsSpan(0, input.ByteLength))
            writeFloatToTypedOutput output index (float32 inputPixels[index])
        elif t = typeof<int32> then
            let inputPixels = MemoryMarshal.Cast<byte, int32>(input.Bytes.AsSpan(0, input.ByteLength))
            writeFloatToTypedOutput output index (float32 inputPixels[index])
        else
            invalidArg "T" $"ChunkFunctions noise supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."

    let private typedValueAsFloat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'T>)
        index
        =
        let t = typeof<'T>
        if t = typeof<float32> then
            let pixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            pixels[index]
        elif t = typeof<uint8> then
            float32 chunk.Bytes[index]
        elif t = typeof<int8> then
            let pixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            float32 pixels[index]
        elif t = typeof<uint16> then
            let pixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            float32 pixels[index]
        elif t = typeof<int16> then
            let pixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            float32 pixels[index]
        elif t = typeof<int32> then
            let pixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            float32 pixels[index]
        else
            invalidArg "T" $"ChunkFunctions noise supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."

    let private saltAndPepperValues<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
        let t = typeof<'T>
        if t = typeof<float32> then 0.0f, 1.0f
        elif t = typeof<uint8> then 0.0f, 255.0f
        elif t = typeof<int8> then float32 SByte.MinValue, float32 SByte.MaxValue
        elif t = typeof<uint16> then 0.0f, 65535.0f
        elif t = typeof<int16> then float32 Int16.MinValue, float32 Int16.MaxValue
        elif t = typeof<int32> then float32 Int32.MinValue, float32 Int32.MaxValue
        else
            invalidArg "T" $"ChunkFunctions salt-and-pepper noise supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."

    let addNormalNoiseChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        mean
        stddev
        (chunk: Chunk<'T>)
        =
        if stddev <= 0.0 then
            copyChunk chunk
        else
            let output = Chunk.create<'T> chunk.Size
            try
                let rng = Random.Shared
                let count = chunk.ByteLength / Marshal.SizeOf<'T>()
                for i in 0 .. count - 1 do
                    let value = typedValueAsFloat chunk i + float32 (nextGaussian rng mean stddev)
                    writeFloatToTypedOutput output i value
                output
            with
            | _ ->
                Chunk.decRef output
                reraise()

    let addSaltAndPepperNoiseChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        probability
        (chunk: Chunk<'T>)
        =
        if probability <= 0.0 then
            copyChunk chunk
        else
            let probability = min 1.0 probability
            let pepper, salt = saltAndPepperValues<'T>
            let output = Chunk.create<'T> chunk.Size
            try
                let rng = Random.Shared
                let count = chunk.ByteLength / Marshal.SizeOf<'T>()
                for i in 0 .. count - 1 do
                    let sample = rng.NextDouble()
                    if sample < probability then
                        let value = if rng.NextDouble() < 0.5 then pepper else salt
                        writeFloatToTypedOutput output i value
                    else
                        copyTypedValue chunk output i
                output
            with
            | _ ->
                Chunk.decRef output
                reraise()

    let addShotNoiseChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        scale
        (chunk: Chunk<'T>)
        =
        if scale <= 0.0 then
            copyChunk chunk
        else
            let output = Chunk.create<'T> chunk.Size
            try
                let rng = Random.Shared
                let count = chunk.ByteLength / Marshal.SizeOf<'T>()
                for i in 0 .. count - 1 do
                    let value = typedValueAsFloat chunk i
                    let lambda = max 0.0 (double value / scale)
                    let noisy = float32 (nextPoisson rng lambda * scale)
                    writeFloatToTypedOutput output i noisy
                output
            with
            | _ ->
                Chunk.decRef output
                reraise()

    let private nativePixelType<'T> =
        let t = typeof<'T>
        if t = typeof<uint8> then 1
        elif t = typeof<int8> then 2
        elif t = typeof<uint16> then 3
        elif t = typeof<int16> then 4
        elif t = typeof<int32> then 5
        elif t = typeof<float32> then 6
        else
            invalidArg "T" $"Chunk native 2D resampling supports UInt8, Int8, UInt16, Int16, Int32, and Float32 chunks, got {t.Name}."

    let private validateNative2DSlice name (chunk: Chunk<'T>) =
        let width64, height64, depth64 = chunk.Size
        if depth64 <> 1UL then
            invalidArg name $"Chunk native 2D resampling expects 2D slice chunks with depth 1, got {chunk.Size}."
        if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg name $"Chunk native 2D resampling dimensions must fit Int32, got {chunk.Size}."
        int width64, int height64

    let private runNative2DUnary<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        name
        outputSize
        nativeCall
        (chunk: Chunk<'T>)
        =
        LowLevelNative.ensureAvailable ()
        let output = Chunk.create<'T> outputSize
        let mutable inputHandle = Unchecked.defaultof<GCHandle>
        let mutable outputHandle = Unchecked.defaultof<GCHandle>
        let mutable inputPinned = false
        let mutable outputPinned = false
        try
            try
                inputHandle <- GCHandle.Alloc(chunk.Bytes, GCHandleType.Pinned)
                inputPinned <- true
                outputHandle <- GCHandle.Alloc(output.Bytes, GCHandleType.Pinned)
                outputPinned <- true
                let status = nativeCall (inputHandle.AddrOfPinnedObject()) (outputHandle.AddrOfPinnedObject())
                if status <> 0 then
                    invalidOp $"{name} failed in native helper with status {status}."
                output
            finally
                if outputPinned then outputHandle.Free()
                if inputPinned then inputHandle.Free()
        with
        | _ ->
            Chunk.decRef output
            reraise()

    let resample2DNativeChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        interpolation
        outputWidth
        outputHeight
        spacingX
        spacingY
        (chunk: Chunk<'T>)
        =
        if outputWidth = 0u then invalidArg "outputWidth" "Chunk resample2D expects a positive output width."
        if outputHeight = 0u then invalidArg "outputHeight" "Chunk resample2D expects a positive output height."
        if spacingX <= 0.0 then invalidArg "spacingX" "Chunk resample2D expects positive X spacing."
        if spacingY <= 0.0 then invalidArg "spacingY" "Chunk resample2D expects positive Y spacing."

        let inputWidth, inputHeight = validateNative2DSlice "chunk" chunk
        let outputWidthI = checkedUIntToInt "outputWidth" outputWidth
        let outputHeightI = checkedUIntToInt "outputHeight" outputHeight
        let pixelType = nativePixelType<'T>
        let interpolation = ResampleInterpolation.toNative interpolation
        runNative2DUnary
            "sp_resample_2d"
            (uint64 outputWidth, uint64 outputHeight, 1UL)
            (fun input output ->
                LowLevelNative.resample2D(input, output, pixelType, inputWidth, inputHeight, outputWidthI, outputHeightI, spacingX, spacingY, interpolation))
            chunk

    let resize3DPairSliceNativeChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        interpolation
        outputWidth
        outputHeight
        spacingX
        spacingY
        zFraction
        (lower: Chunk<'T>)
        (upper: Chunk<'T>)
        =
        let inputWidth, inputHeight = validateNative2DSlice "lower" lower
        let upperWidth, upperHeight = validateNative2DSlice "upper" upper
        if inputWidth <> upperWidth || inputHeight <> upperHeight then
            invalidArg "upper" $"Chunk resize3D expects matching source slice sizes, got lower={lower.Size}, upper={upper.Size}."
        if outputWidth = 0u then invalidArg "outputWidth" "Chunk resize3D expects a positive output width."
        if outputHeight = 0u then invalidArg "outputHeight" "Chunk resize3D expects a positive output height."
        if spacingX <= 0.0 then invalidArg "spacingX" "Chunk resize3D expects positive X spacing."
        if spacingY <= 0.0 then invalidArg "spacingY" "Chunk resize3D expects positive Y spacing."
        if zFraction < 0.0 || zFraction > 1.0 then
            invalidArg "zFraction" $"Chunk resize3D expects zFraction in [0, 1], got {zFraction}."

        let outputWidthI = checkedUIntToInt "outputWidth" outputWidth
        let outputHeightI = checkedUIntToInt "outputHeight" outputHeight
        let pixelType = nativePixelType<'T>
        let interpolation = ResampleInterpolation.toNative interpolation
        LowLevelNative.ensureAvailable ()
        let output = Chunk.create<'T> (uint64 outputWidth, uint64 outputHeight, 1UL)
        let mutable lowerHandle = Unchecked.defaultof<GCHandle>
        let mutable upperHandle = Unchecked.defaultof<GCHandle>
        let mutable outputHandle = Unchecked.defaultof<GCHandle>
        let mutable lowerPinned = false
        let mutable upperPinned = false
        let mutable outputPinned = false

        try
            try
                lowerHandle <- GCHandle.Alloc(lower.Bytes, GCHandleType.Pinned)
                lowerPinned <- true
                upperHandle <- GCHandle.Alloc(upper.Bytes, GCHandleType.Pinned)
                upperPinned <- true
                outputHandle <- GCHandle.Alloc(output.Bytes, GCHandleType.Pinned)
                outputPinned <- true

                LowLevelNative.resize3DPairSlice(
                    lowerHandle.AddrOfPinnedObject(),
                    upperHandle.AddrOfPinnedObject(),
                    outputHandle.AddrOfPinnedObject(),
                    pixelType,
                    inputWidth,
                    inputHeight,
                    outputWidthI,
                    outputHeightI,
                    spacingX,
                    spacingY,
                    zFraction,
                    interpolation)
                |> NativeSp.checkStatus "sp_resize_3d_pair_slice"

                output
            finally
                if outputPinned then outputHandle.Free()
                if upperPinned then upperHandle.Free()
                if lowerPinned then lowerHandle.Free()
        with
        | _ ->
            Chunk.decRef output
            reraise()

    let euler2DTransformNativeChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (centerX, centerY, angle)
        (dx, dy)
        (chunk: Chunk<'T>)
        =
        let width, height = validateNative2DSlice "chunk" chunk
        let pixelType = nativePixelType<'T>
        runNative2DUnary
            "sp_euler_2d.transform"
            chunk.Size
            (fun input output ->
                LowLevelNative.euler2D(input, output, pixelType, width, height, centerX, centerY, angle, dx, dy, 1, ResampleInterpolation.toNative Linear))
            chunk

    let euler2DRotateNativeChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (centerX, centerY)
        angle
        (chunk: Chunk<'T>)
        =
        let width, height = validateNative2DSlice "chunk" chunk
        let pixelType = nativePixelType<'T>
        runNative2DUnary
            "sp_euler_2d.rotate"
            chunk.Size
            (fun input output ->
                LowLevelNative.euler2D(input, output, pixelType, width, height, centerX, centerY, angle, 0.0, 0.0, 0, ResampleInterpolation.toNative Linear))
            chunk

    let private validateOddKernel name (kernel: float32[]) =
        if isNull kernel then
            nullArg name
        if kernel.Length = 0 || kernel.Length % 2 = 0 then
            invalidArg name $"Chunk convolution expects a non-empty odd-length kernel, got {kernel.Length}."
        kernel.Length / 2

    let private validateUInt8Slice name (chunk: Chunk<uint8>) =
        let width64, height64, depth64 = chunk.Size
        if depth64 <> 1UL then
            invalidArg name $"Chunk convolution expects 2D slice chunks with depth 1, got {chunk.Size}."
        if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg name $"Chunk convolution dimensions must fit Int32, got {chunk.Size}."
        int width64, int height64

    let private validateUInt8Window name (window: Chunk<uint8>[]) =
        if isNull window then
            nullArg name
        if window.Length = 0 then
            invalidArg name "Chunk convolution expects at least one input slice."
        let width, height = validateUInt8Slice $"{name}[0]" window[0]
        for i in 1 .. window.Length - 1 do
            let otherWidth, otherHeight = validateUInt8Slice $"{name}[{i}]" window[i]
            if otherWidth <> width || otherHeight <> height then
                invalidArg name $"Chunk convolution expects all input slices to have size {width}x{height}, got {window[i].Size} at index {i}."
        width, height

    let private flattenKernel (kernel: float32[,,]) =
        let width = kernel.GetLength(0)
        let height = kernel.GetLength(1)
        let depth = kernel.GetLength(2)
        if width < 1 || height < 1 || depth < 1 then
            invalidArg "kernel" $"Chunk convolution expects a non-empty kernel, got {width}x{height}x{depth}."
        let values = Array.zeroCreate<float32> (width * height * depth)
        let mutable i = 0
        for z in 0 .. depth - 1 do
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    values[i] <- kernel[x, y, z]
                    i <- i + 1
        width, height, depth, values

    let convolveNativeUInt8Kernel (kernel: float32[,,]) outputStart outputCount (window: Chunk<uint8>[]) =
        if outputStart < 0 then
            invalidArg "outputStart" $"Chunk native convolution expects non-negative outputStart, got {outputStart}."
        if outputCount < 1 then
            invalidArg "outputCount" $"Chunk native convolution expects positive outputCount, got {outputCount}."

        let width, height = validateUInt8Window "window" window
        let kernelWidth, kernelHeight, kernelDepth, nativeKernel = flattenKernel kernel
        LowLevelNative.ensureAvailable ()

        let outputs =
            Array.init outputCount (fun _ -> create<uint8> (uint64 width, uint64 height, 1UL))

        let inputHandles = Array.zeroCreate<GCHandle> window.Length
        let outputHandles = Array.zeroCreate<GCHandle> outputs.Length
        let mutable retainedInputHandles = 0
        let mutable retainedOutputHandles = 0
        let mutable inputPointerHandle = Unchecked.defaultof<GCHandle>
        let mutable inputPointersPinned = false
        let mutable outputPointerHandle = Unchecked.defaultof<GCHandle>
        let mutable outputPointersPinned = false
        let mutable kernelHandle = Unchecked.defaultof<GCHandle>
        let mutable kernelPinned = false

        try
            try
                let inputPointers = Array.zeroCreate<nativeint> window.Length
                for i in 0 .. window.Length - 1 do
                    inputHandles[i] <- GCHandle.Alloc(window[i].Bytes, GCHandleType.Pinned)
                    retainedInputHandles <- retainedInputHandles + 1
                    inputPointers[i] <- inputHandles[i].AddrOfPinnedObject()

                let outputPointers = Array.zeroCreate<nativeint> outputs.Length
                for i in 0 .. outputs.Length - 1 do
                    outputHandles[i] <- GCHandle.Alloc(outputs[i].Bytes, GCHandleType.Pinned)
                    retainedOutputHandles <- retainedOutputHandles + 1
                    outputPointers[i] <- outputHandles[i].AddrOfPinnedObject()

                inputPointerHandle <- GCHandle.Alloc(inputPointers, GCHandleType.Pinned)
                inputPointersPinned <- true
                outputPointerHandle <- GCHandle.Alloc(outputPointers, GCHandleType.Pinned)
                outputPointersPinned <- true
                kernelHandle <- GCHandle.Alloc(nativeKernel, GCHandleType.Pinned)
                kernelPinned <- true

                LowLevelNative.convolveUInt8Slices(
                    inputPointerHandle.AddrOfPinnedObject(),
                    outputPointerHandle.AddrOfPinnedObject(),
                    kernelHandle.AddrOfPinnedObject(),
                    width,
                    height,
                    window.Length,
                    kernelWidth,
                    kernelHeight,
                    kernelDepth,
                    outputStart,
                    outputCount)

                outputs |> Array.toList
            with
            | _ ->
                outputs |> Array.iter decRef
                reraise()
        finally
            if kernelPinned then
                kernelHandle.Free()
            if outputPointersPinned then
                outputPointerHandle.Free()
            if inputPointersPinned then
                inputPointerHandle.Free()
            for i in 0 .. retainedOutputHandles - 1 do
                outputHandles[i].Free()
            for i in 0 .. retainedInputHandles - 1 do
                inputHandles[i].Free()

    let private kernelX (kernel: float32[]) =
        let values = Array3D.zeroCreate<float32> kernel.Length 1 1
        for i in 0 .. kernel.Length - 1 do
            values[i, 0, 0] <- kernel.[i]
        values

    let private kernelY (kernel: float32[]) =
        let values = Array3D.zeroCreate<float32> 1 kernel.Length 1
        for i in 0 .. kernel.Length - 1 do
            values[0, i, 0] <- kernel.[i]
        values

    let private kernelZ (kernel: float32[]) =
        let values = Array3D.zeroCreate<float32> 1 1 kernel.Length
        for i in 0 .. kernel.Length - 1 do
            values[0, 0, i] <- kernel.[i]
        values

    let convolveNativeXUInt8 (kernel: float32[]) (chunk: Chunk<uint8>) =
        validateOddKernel "kernel" kernel |> ignore
        match convolveNativeUInt8Kernel (kernelX kernel) 0 1 [| chunk |] with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRef
            invalidOp $"Native X convolution unexpectedly returned {outputs.Length} outputs."

    let convolveNativeYUInt8 (kernel: float32[]) (chunk: Chunk<uint8>) =
        validateOddKernel "kernel" kernel |> ignore
        match convolveNativeUInt8Kernel (kernelY kernel) 0 1 [| chunk |] with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRef
            invalidOp $"Native Y convolution unexpectedly returned {outputs.Length} outputs."

    let convolveNativeZUInt8 (kernel: float32[]) (window: Chunk<uint8>[]) =
        let radius = validateOddKernel "kernel" kernel
        if window.Length <> kernel.Length then
            invalidArg "window" $"Native Z convolution expects one window slice per kernel tap, got {window.Length} slices and {kernel.Length} taps."
        match convolveNativeUInt8Kernel (kernelZ kernel) radius 1 window with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRef
            invalidOp $"Native Z convolution unexpectedly returned {outputs.Length} outputs."

    type private NativeAxis =
        | NativeX
        | NativeY
        | NativeZ

    let private validateScalarSlice<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> name (chunk: Chunk<'T>) =
        let width64, height64, depth64 = chunk.Size
        if depth64 <> 1UL then
            invalidArg name $"Chunk convolution expects 2D slice chunks with depth 1, got {chunk.Size}."
        if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
            invalidArg name $"Chunk convolution dimensions must fit Int32, got {chunk.Size}."
        int width64, int height64

    let private validateScalarWindow<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> name (window: Chunk<'T>[]) =
        if isNull window then
            nullArg name
        if window.Length = 0 then
            invalidArg name "Chunk convolution expects at least one input slice."
        let width, height = validateScalarSlice $"{name}[0]" window[0]
        for i in 1 .. window.Length - 1 do
            let otherWidth, otherHeight = validateScalarSlice $"{name}[{i}]" window[i]
            if otherWidth <> width || otherHeight <> height then
                invalidArg name $"Chunk convolution expects all input slices to have size {width}x{height}, got {window[i].Size} at index {i}."
        width, height

    let private invokeNativeAxis<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        axis
        slices
        outputs
        kernel
        width
        height
        windowLength
        kernelLength
        outputStart
        outputCount
        =
        let t = typeof<'T>
        match axis with
        | NativeX when t = typeof<uint8> -> LowLevelNative.convolveUInt8XSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeY when t = typeof<uint8> -> LowLevelNative.convolveUInt8YSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeZ when t = typeof<uint8> -> LowLevelNative.convolveUInt8ZSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeX when t = typeof<int8> -> LowLevelNative.convolveInt8XSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeY when t = typeof<int8> -> LowLevelNative.convolveInt8YSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeZ when t = typeof<int8> -> LowLevelNative.convolveInt8ZSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeX when t = typeof<uint16> -> LowLevelNative.convolveUInt16XSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeY when t = typeof<uint16> -> LowLevelNative.convolveUInt16YSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeZ when t = typeof<uint16> -> LowLevelNative.convolveUInt16ZSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeX when t = typeof<int32> -> LowLevelNative.convolveInt32XSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeY when t = typeof<int32> -> LowLevelNative.convolveInt32YSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeZ when t = typeof<int32> -> LowLevelNative.convolveInt32ZSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeX when t = typeof<float32> -> LowLevelNative.convolveFloat32XSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeY when t = typeof<float32> -> LowLevelNative.convolveFloat32YSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | NativeZ when t = typeof<float32> -> LowLevelNative.convolveFloat32ZSlices(slices, outputs, kernel, width, height, windowLength, kernelLength, outputStart, outputCount)
        | _ ->
            invalidArg "T" $"Chunk native axis convolution supports UInt8, Int8, UInt16, Int32, and Float32 chunks, got {t.Name}."

    let private convolveNativeAxis<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        axis
        (kernel: float32[])
        outputStart
        outputCount
        (window: Chunk<'T>[])
        =
        validateOddKernel "kernel" kernel |> ignore
        if outputStart < 0 then
            invalidArg "outputStart" $"Chunk native axis convolution expects non-negative outputStart, got {outputStart}."
        if outputCount < 1 then
            invalidArg "outputCount" $"Chunk native axis convolution expects positive outputCount, got {outputCount}."

        let width, height = validateScalarWindow "window" window
        LowLevelNative.ensureAvailable ()

        let outputs =
            Array.init outputCount (fun _ -> create<'T> (uint64 width, uint64 height, 1UL))

        let inputHandles = Array.zeroCreate<GCHandle> window.Length
        let outputHandles = Array.zeroCreate<GCHandle> outputs.Length
        let mutable retainedInputHandles = 0
        let mutable retainedOutputHandles = 0
        let mutable inputPointerHandle = Unchecked.defaultof<GCHandle>
        let mutable inputPointersPinned = false
        let mutable outputPointerHandle = Unchecked.defaultof<GCHandle>
        let mutable outputPointersPinned = false
        let mutable kernelHandle = Unchecked.defaultof<GCHandle>
        let mutable kernelPinned = false

        try
            try
                let inputPointers = Array.zeroCreate<nativeint> window.Length
                for i in 0 .. window.Length - 1 do
                    inputHandles[i] <- GCHandle.Alloc(window[i].Bytes, GCHandleType.Pinned)
                    retainedInputHandles <- retainedInputHandles + 1
                    inputPointers[i] <- inputHandles[i].AddrOfPinnedObject()

                let outputPointers = Array.zeroCreate<nativeint> outputs.Length
                for i in 0 .. outputs.Length - 1 do
                    outputHandles[i] <- GCHandle.Alloc(outputs[i].Bytes, GCHandleType.Pinned)
                    retainedOutputHandles <- retainedOutputHandles + 1
                    outputPointers[i] <- outputHandles[i].AddrOfPinnedObject()

                inputPointerHandle <- GCHandle.Alloc(inputPointers, GCHandleType.Pinned)
                inputPointersPinned <- true
                outputPointerHandle <- GCHandle.Alloc(outputPointers, GCHandleType.Pinned)
                outputPointersPinned <- true
                kernelHandle <- GCHandle.Alloc(kernel, GCHandleType.Pinned)
                kernelPinned <- true

                invokeNativeAxis<'T>
                    axis
                    (inputPointerHandle.AddrOfPinnedObject())
                    (outputPointerHandle.AddrOfPinnedObject())
                    (kernelHandle.AddrOfPinnedObject())
                    width
                    height
                    window.Length
                    kernel.Length
                    outputStart
                    outputCount

                outputs |> Array.toList
            with
            | _ ->
                outputs |> Array.iter decRef
                reraise()
        finally
            if kernelPinned then
                kernelHandle.Free()
            if outputPointersPinned then
                outputPointerHandle.Free()
            if inputPointersPinned then
                inputPointerHandle.Free()
            for i in 0 .. retainedOutputHandles - 1 do
                outputHandles[i].Free()
            for i in 0 .. retainedInputHandles - 1 do
                inputHandles[i].Free()

    let convolveNativeX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (kernel: float32[]) (chunk: Chunk<'T>) =
        match convolveNativeAxis<'T> NativeX kernel 0 1 [| chunk |] with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRef
            invalidOp $"Native specialized X convolution unexpectedly returned {outputs.Length} outputs."

    let convolveNativeY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (kernel: float32[]) (chunk: Chunk<'T>) =
        match convolveNativeAxis<'T> NativeY kernel 0 1 [| chunk |] with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRef
            invalidOp $"Native specialized Y convolution unexpectedly returned {outputs.Length} outputs."

    let convolveNativeZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (kernel: float32[]) (window: Chunk<'T>[]) =
        let radius = validateOddKernel "kernel" kernel
        if window.Length <> kernel.Length then
            invalidArg "window" $"Native specialized Z convolution expects one window slice per kernel tap, got {window.Length} slices and {kernel.Length} taps."
        match convolveNativeAxis<'T> NativeZ kernel radius 1 window with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRef
            invalidOp $"Native specialized Z convolution unexpectedly returned {outputs.Length} outputs."

    let convolveNativeXUInt8Specialized (kernel: float32[]) (chunk: Chunk<uint8>) =
        convolveNativeX<uint8> kernel chunk

    let convolveNativeYUInt8Specialized (kernel: float32[]) (chunk: Chunk<uint8>) =
        convolveNativeY<uint8> kernel chunk

    let convolveNativeZUInt8Specialized (kernel: float32[]) (window: Chunk<uint8>[]) =
        convolveNativeZ<uint8> kernel window

    let private validateVectorSliceFloat32 name (vector: VectorChunk<float32>) =
        let components = vectorComponentCount vector
        if components = 0u then
            invalidArg name "Native vector-component convolution expects at least one vector component."
        if components > uint32 Int32.MaxValue then
            invalidArg name $"Native vector-component convolution component count must fit Int32, got {components}."
        let width, height, depth = vector.SpatialSize
        if depth <> 1UL then
            invalidArg name $"Native vector-component convolution expects 2D vector slices with depth 1, got {vector.SpatialSize}."
        if width > uint64 Int32.MaxValue || height > uint64 Int32.MaxValue then
            invalidArg name $"Native vector-component convolution dimensions must fit Int32, got {vector.SpatialSize}."
        vector.Components
        |> Array.iteri (fun i chunk ->
            if chunk.Size <> vector.SpatialSize then
                invalidArg name $"Native vector-component convolution component {i} size {chunk.Size} does not match spatial size {vector.SpatialSize}.")
        int width, int height, int components

    let private validateVectorWindowFloat32 name (window: VectorChunk<float32>[]) =
        if isNull window then
            nullArg name
        if window.Length = 0 then
            invalidArg name "Native vector-component convolution expects at least one input slice."

        let width, height, components = validateVectorSliceFloat32 $"{name}[0]" window[0]
        let spatialSize = window[0].SpatialSize
        let componentCount = vectorComponentCount window[0]
        for i in 1 .. window.Length - 1 do
            let otherWidth, otherHeight, otherComponents = validateVectorSliceFloat32 $"{name}[{i}]" window[i]
            if otherWidth <> width || otherHeight <> height || otherComponents <> components ||
               window[i].SpatialSize <> spatialSize || vectorComponentCount window[i] <> componentCount then
                invalidArg name $"Native vector-component convolution expects matching vector slice sizes and component counts, got {window[i].SpatialSize} with {vectorComponentCount window[i]} components at index {i}."
        width, height, components

    let private convolveVectorComponentsNativeAxisFloat32
        axis
        (kernel: float32[])
        outputStart
        outputCount
        (window: VectorChunk<float32>[])
        =
        validateOddKernel "kernel" kernel |> ignore
        if outputStart < 0 then
            invalidArg "outputStart" $"Native vector-component convolution expects non-negative outputStart, got {outputStart}."
        if outputCount < 1 then
            invalidArg "outputCount" $"Native vector-component convolution expects positive outputCount, got {outputCount}."
        if axis < 0 || axis > 2 then
            invalidArg "axis" $"Native vector-component convolution axis must be 0, 1, or 2, got {axis}."

        let _width, _height, components = validateVectorWindowFloat32 "window" window
        let spatialSize = window[0].SpatialSize
        let componentOutputs =
            Array.init outputCount (fun _ -> Array.zeroCreate<Chunk<float32>> components)

        try
            match axis with
            | 0 ->
                if window.Length <> 1 || outputStart <> 0 || outputCount <> 1 then
                    invalidArg "window" "Vector component X convolution expects one input slice and one output slice."
                for c in 0 .. components - 1 do
                    componentOutputs[0][c] <- convolveNativeX<float32> kernel window[0].Components[c]
            | 1 ->
                if window.Length <> 1 || outputStart <> 0 || outputCount <> 1 then
                    invalidArg "window" "Vector component Y convolution expects one input slice and one output slice."
                for c in 0 .. components - 1 do
                    componentOutputs[0][c] <- convolveNativeY<float32> kernel window[0].Components[c]
            | 2 ->
                for c in 0 .. components - 1 do
                    let componentWindow = Array.init window.Length (fun i -> window[i].Components[c])
                    let outputs = convolveNativeAxis<float32> NativeZ kernel outputStart outputCount componentWindow
                    outputs |> List.iteri (fun i chunk -> componentOutputs[i][c] <- chunk)
            | _ ->
                invalidArg "axis" $"Native vector-component convolution axis must be 0, 1, or 2, got {axis}."

            componentOutputs
            |> Array.map (fun chunks ->
                { SpatialSize = spatialSize
                  Components = chunks })
            |> Array.toList
        with
        | _ ->
            componentOutputs
            |> Array.iter (fun chunks ->
                chunks
                |> Array.iter (fun chunk ->
                    if not (isNull (box chunk)) then
                        decRef chunk))
            reraise()

    let convolveVectorComponentsNativeXFloat32 (kernel: float32[]) (vector: VectorChunk<float32>) =
        match convolveVectorComponentsNativeAxisFloat32 0 kernel 0 1 [| vector |] with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRefVector
            invalidOp $"Native vector-component X convolution unexpectedly returned {outputs.Length} outputs."

    let convolveVectorComponentsNativeYFloat32 (kernel: float32[]) (vector: VectorChunk<float32>) =
        match convolveVectorComponentsNativeAxisFloat32 1 kernel 0 1 [| vector |] with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRefVector
            invalidOp $"Native vector-component Y convolution unexpectedly returned {outputs.Length} outputs."

    let convolveVectorComponentsNativeZFloat32 (kernel: float32[]) (window: VectorChunk<float32>[]) =
        let radius = validateOddKernel "kernel" kernel
        if window.Length <> kernel.Length then
            invalidArg "window" $"Native vector-component Z convolution expects one window slice per kernel tap, got {window.Length} slices and {kernel.Length} taps."
        match convolveVectorComponentsNativeAxisFloat32 2 kernel radius 1 window with
        | [ output ] -> output
        | outputs ->
            outputs |> List.iter decRefVector
            invalidOp $"Native vector-component Z convolution unexpectedly returned {outputs.Length} outputs."

    let signedDistanceBandNativeUInt8 (bandRadius: uint) outputStart outputCount (window: Chunk<uint8>[]) =
        if bandRadius = 0u then
            invalidArg "bandRadius" "Chunk signed distance band requires a positive band radius."
        if isNull window || window.Length = 0 then
            invalidArg "window" "Chunk signed distance band requires a non-empty window."
        if outputStart < 0 || outputCount < 0 || outputStart + outputCount > window.Length then
            invalidArg "outputStart" $"Chunk signed distance band emit range ({outputStart}, {outputCount}) exceeds window length {window.Length}."

        let width, height, depth = window[0].Size
        if depth <> 1UL then
            invalidArg "window" $"Chunk signed distance band expects 2D slice chunks with depth 1, got {window[0].Size}."
        let widthI = checkedIntDimension "width" width
        let heightI = checkedIntDimension "height" height
        for i in 1 .. window.Length - 1 do
            if window[i].Size <> (width, height, 1UL) then
                invalidArg "window" $"Chunk signed distance band expects all slices to have size {(width, height, 1UL)}, got {window[i].Size} at window index {i}."

        LowLevelNative.ensureAvailable ()

        let outputs =
            Array.init outputCount (fun _ -> create<float32> (width, height, 1UL))

        let inputHandles = Array.zeroCreate<GCHandle> window.Length
        let outputHandles = Array.zeroCreate<GCHandle> outputs.Length
        let mutable retainedInputHandles = 0
        let mutable retainedOutputHandles = 0
        let mutable inputPointerHandle = Unchecked.defaultof<GCHandle>
        let mutable inputPointersPinned = false
        let mutable outputPointerHandle = Unchecked.defaultof<GCHandle>
        let mutable outputPointersPinned = false

        try
            try
                let inputPointers = Array.zeroCreate<nativeint> window.Length
                for i in 0 .. window.Length - 1 do
                    inputHandles[i] <- GCHandle.Alloc(window[i].Bytes, GCHandleType.Pinned)
                    retainedInputHandles <- retainedInputHandles + 1
                    inputPointers[i] <- inputHandles[i].AddrOfPinnedObject()

                let outputPointers = Array.zeroCreate<nativeint> outputs.Length
                for i in 0 .. outputs.Length - 1 do
                    outputHandles[i] <- GCHandle.Alloc(outputs[i].Bytes, GCHandleType.Pinned)
                    retainedOutputHandles <- retainedOutputHandles + 1
                    outputPointers[i] <- outputHandles[i].AddrOfPinnedObject()

                inputPointerHandle <- GCHandle.Alloc(inputPointers, GCHandleType.Pinned)
                inputPointersPinned <- true
                outputPointerHandle <- GCHandle.Alloc(outputPointers, GCHandleType.Pinned)
                outputPointersPinned <- true

                NativeSp.checkStatus
                    "sp_signed_distance_band_uint8_slices"
                    (LowLevelNative.signedDistanceBandUInt8Slices(
                        inputPointerHandle.AddrOfPinnedObject(),
                        outputPointerHandle.AddrOfPinnedObject(),
                        widthI,
                        heightI,
                        window.Length,
                        outputStart,
                        outputCount,
                        float32 bandRadius))

                outputs |> Array.toList
            with
            | _ ->
                outputs |> Array.iter decRef
                reraise()
        finally
            if outputPointersPinned then
                outputPointerHandle.Free()
            if inputPointersPinned then
                inputPointerHandle.Free()
            for i in 0 .. retainedOutputHandles - 1 do
                outputHandles[i].Free()
            for i in 0 .. retainedInputHandles - 1 do
                inputHandles[i].Free()

    let private validateDerivativeWindow name (window: Chunk<float32>[]) =
        let firstOrder = finiteDiffKernel1D 1u
        if window.Length <> firstOrder.Length then
            invalidArg "window" $"{name} expects a smoothed 3-slice window, got {window.Length} slices."
        let center = window[firstOrder.Length / 2]
        for i in 0 .. window.Length - 1 do
            if window[i].Size <> center.Size then
                invalidArg "window" $"{name} expects all window chunks to have size {center.Size}, got {window[i].Size} at slice {i}."
        center

    let private sumFloat32Chunks3 name (a: Chunk<float32>) (b: Chunk<float32>) (c: Chunk<float32>) =
        validateSameSize name a b
        validateSameSize name a c
        let output = create<float32> a.Size
        try
            let aPixels = span<float32> a
            let bPixels = span<float32> b
            let cPixels = span<float32> c
            let outputPixels = span<float32> output
            let mutable i = 0
            while i < outputPixels.Length do
                outputPixels[i] <- aPixels[i] + bPixels[i] + cPixels[i]
                i <- i + 1
            output
        with
        | _ ->
            decRef output
            reraise()

    let private firstDerivativeX (chunk: Chunk<float32>) =
        convolveNativeX<float32> (finiteDiffKernel1D 1u) chunk

    let private firstDerivativeY (chunk: Chunk<float32>) =
        convolveNativeY<float32> (finiteDiffKernel1D 1u) chunk

    let private firstDerivativeZ (window: Chunk<float32>[]) =
        convolveNativeZ<float32> (finiteDiffKernel1D 1u) window

    let private secondDerivativeX (chunk: Chunk<float32>) =
        convolveNativeX<float32> (finiteDiffKernel1D 2u) chunk

    let private secondDerivativeY (chunk: Chunk<float32>) =
        convolveNativeY<float32> (finiteDiffKernel1D 2u) chunk

    let private secondDerivativeZ (window: Chunk<float32>[]) =
        convolveNativeZ<float32> (finiteDiffKernel1D 2u) window

    let private crossDerivativeXY (center: Chunk<float32>) =
        let dx = firstDerivativeX center
        try
            firstDerivativeY dx
        finally
            decRef dx

    let private crossDerivativeXZ (window: Chunk<float32>[]) =
        let dxWindow = window |> Array.map firstDerivativeX
        try
            firstDerivativeZ dxWindow
        finally
            dxWindow |> Array.iter decRef

    let private crossDerivativeYZ (window: Chunk<float32>[]) =
        let dyWindow = window |> Array.map firstDerivativeY
        try
            firstDerivativeZ dyWindow
        finally
            dyWindow |> Array.iter decRef

    let gradientVectorFromSmoothedNative (window: Chunk<float32>[]) =
        let center = validateDerivativeWindow "gradientVectorFromSmoothedNative" window
        let dx = firstDerivativeX center
        let dy = firstDerivativeY center
        let dz = firstDerivativeZ window
        try
            toVectorImage [ dx; dy; dz ]
        finally
            decRef dx
            decRef dy
            decRef dz

    let hessianUpperFromSmoothedNative (window: Chunk<float32>[]) =
        let center = validateDerivativeWindow "hessianUpperFromSmoothedNative" window
        let dxx = secondDerivativeX center
        let dxy = crossDerivativeXY center
        let dxz = crossDerivativeXZ window
        let dyy = secondDerivativeY center
        let dyz = crossDerivativeYZ window
        let dzz = secondDerivativeZ window
        try
            toVectorImage [ dxx; dxy; dxz; dyy; dyz; dzz ]
        finally
            decRef dxx
            decRef dxy
            decRef dxz
            decRef dyy
            decRef dyz
            decRef dzz

    let laplacianFromSmoothedNative (window: Chunk<float32>[]) =
        let center = validateDerivativeWindow "laplacianFromSmoothedNative" window
        let dxx = secondDerivativeX center
        let dyy = secondDerivativeY center
        let dzz = secondDerivativeZ window
        try
            sumFloat32Chunks3 "laplacianFromSmoothedNative" dxx dyy dzz
        finally
            decRef dxx
            decRef dyy
            decRef dzz

    let private sobelDerivative = [| -1.0f; 0.0f; 1.0f |]
    let private sobelSmooth = [| 0.25f; 0.5f; 0.25f |]

    let private sobelX (window: Chunk<float32>[]) =
        let zSmooth = convolveNativeZ<float32> sobelSmooth window
        try
            let ySmooth = convolveNativeY<float32> sobelSmooth zSmooth
            try
                convolveNativeX<float32> sobelDerivative ySmooth
            finally
                decRef ySmooth
        finally
            decRef zSmooth

    let private sobelY (window: Chunk<float32>[]) =
        let zSmooth = convolveNativeZ<float32> sobelSmooth window
        try
            let xSmooth = convolveNativeX<float32> sobelSmooth zSmooth
            try
                convolveNativeY<float32> sobelDerivative xSmooth
            finally
                decRef xSmooth
        finally
            decRef zSmooth

    let private sobelZ (window: Chunk<float32>[]) =
        let xySmooth =
            window
            |> Array.map (fun slice ->
                let xSmooth = convolveNativeX<float32> sobelSmooth slice
                try
                    convolveNativeY<float32> sobelSmooth xSmooth
                finally
                    decRef xSmooth)
        try
            convolveNativeZ<float32> sobelDerivative xySmooth
        finally
            xySmooth |> Array.iter decRef

    let sobelMagnitudeFromNativeFloat32 (window: Chunk<float32>[]) =
        validateDerivativeWindow "sobelMagnitudeFromNativeFloat32" window |> ignore
        let dx = sobelX window
        let dy = sobelY window
        let dz = sobelZ window
        let output = create<float32> dx.Size
        try
            try
                validateSameSize "sobelMagnitudeFromNativeFloat32" dx dy
                validateSameSize "sobelMagnitudeFromNativeFloat32" dx dz
                let dxPixels = span<float32> dx
                let dyPixels = span<float32> dy
                let dzPixels = span<float32> dz
                let outputPixels = span<float32> output
                let mutable i = 0
                while i < outputPixels.Length do
                    outputPixels[i] <- MathF.Sqrt(dxPixels[i] * dxPixels[i] + dyPixels[i] * dyPixels[i] + dzPixels[i] * dzPixels[i])
                    i <- i + 1
                output
            with
            | _ ->
                decRef output
                reraise()
        finally
            decRef dx
            decRef dy
            decRef dz

    let addCountsInto (target: uint64[]) (source: uint64[]) =
        if target.Length <> source.Length then
            invalidArg "source" $"Cannot add count arrays with different lengths: target has {target.Length}, source has {source.Length}."

        let width = Vector<uint64>.Count
        let vectorEnd = target.Length - (target.Length % width)
        let mutable i = 0
        while i < vectorEnd do
            let targetVector = Vector<uint64>(target.AsSpan(i, width))
            let sourceVector = Vector<uint64>(source.AsSpan(i, width))
            (targetVector + sourceVector).CopyTo(target.AsSpan(i, width))
            i <- i + width
        while i < target.Length do
            target[i] <- target[i] + source[i]
            i <- i + 1

    let histogramDictionaryBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (bytes: byte[])
        byteLength
        =
        let values = MemoryMarshal.Cast<byte, 'T>(bytes.AsSpan(0, byteLength))
        let counts = Dictionary<'T, uint64>()
        let mutable i = 0
        while i < values.Length do
            let mutable exists = false
            let count = &CollectionsMarshal.GetValueRefOrAddDefault(counts, values[i], &exists)
            count <- count + 1UL
            i <- i + 1
        counts

    let addDictionaryInto<'T when 'T: equality> (target: Dictionary<'T, uint64>) (source: Dictionary<'T, uint64>) =
        for pair in source do
            let mutable exists = false
            let count = &CollectionsMarshal.GetValueRefOrAddDefault(target, pair.Key, &exists)
            count <- count + pair.Value

    let dictionaryToMap<'T when 'T: comparison> (counts: Dictionary<'T, uint64>) =
        counts
        |> Seq.map (fun pair -> pair.Key, pair.Value)
        |> Map.ofSeq

    let histogramBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (bytes: byte[])
        byteLength
        =
        histogramDictionaryBytes<'T> bytes byteLength
        |> dictionaryToMap

    let private denseHistogramMap<'T when 'T: comparison> (counts: uint64[]) (keyOfIndex: int -> 'T) =
        let mutable histogram = Map.empty<'T, uint64>
        for index in 0 .. counts.Length - 1 do
            let count = counts[index]
            if count <> 0UL then
                histogram <- histogram.Add(keyOfIndex index, count)
        histogram

    let private denseCountsFromBytes<'Raw when 'Raw: (new: unit -> 'Raw) and 'Raw: struct and 'Raw :> ValueType>
        (bytes: byte[])
        byteLength
        binCount
        (indexOf: 'Raw -> int)
        =
        let values = MemoryMarshal.Cast<byte, 'Raw>(bytes.AsSpan(0, byteLength))
        let counts = Array.zeroCreate<uint64> binCount
        let mutable i = 0
        while i < values.Length do
            let index = indexOf values[i]
            counts[index] <- counts[index] + 1UL
            i <- i + 1
        counts

    let private addDenseCountsFromBytesInto<'Raw when 'Raw: (new: unit -> 'Raw) and 'Raw: struct and 'Raw :> ValueType>
        (counts: uint64[])
        (bytes: byte[])
        byteLength
        (indexOf: 'Raw -> int)
        =
        let values = MemoryMarshal.Cast<byte, 'Raw>(bytes.AsSpan(0, byteLength))
        let mutable i = 0
        while i < values.Length do
            let index = indexOf values[i]
            counts[index] <- counts[index] + 1UL
            i <- i + 1

    let histogramDenseCountsBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (bytes: byte[])
        byteLength
        =
        let t = typeof<'T>
        if t = typeof<uint8> then
            UInt8Counts(denseCountsFromBytes<uint8> bytes byteLength 256 int)
        elif t = typeof<int8> then
            let offset = -int SByte.MinValue
            Int8Counts(denseCountsFromBytes<int8> bytes byteLength 256 (fun value -> int value + offset))
        elif t = typeof<uint16> then
            UInt16Counts(denseCountsFromBytes<uint16> bytes byteLength 65536 int)
        elif t = typeof<int16> then
            let offset = -int Int16.MinValue
            Int16Counts(denseCountsFromBytes<int16> bytes byteLength 65536 (fun value -> int value + offset))
        else
            invalidArg "T" $"ChunkFunctions.histogramDense supports only 8-bit and 16-bit integer chunks, but got {t.Name}. Use ChunkFunctions.histogram or a binned histogram instead."

    let addDenseInto target source =
        match target, source with
        | UInt8Counts targetCounts, UInt8Counts sourceCounts
        | Int8Counts targetCounts, Int8Counts sourceCounts
        | UInt16Counts targetCounts, UInt16Counts sourceCounts
        | Int16Counts targetCounts, Int16Counts sourceCounts ->
            addCountsInto targetCounts sourceCounts
        | _ ->
            invalidArg "source" "Cannot add dense histograms with different integer domains."

    let emptyDenseCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
        let t = typeof<'T>
        if t = typeof<uint8> then
            UInt8Counts(Array.zeroCreate<uint64> 256)
        elif t = typeof<int8> then
            Int8Counts(Array.zeroCreate<uint64> 256)
        elif t = typeof<uint16> then
            UInt16Counts(Array.zeroCreate<uint64> 65536)
        elif t = typeof<int16> then
            Int16Counts(Array.zeroCreate<uint64> 65536)
        else
            invalidArg "T" $"ChunkFunctions.histogramDense supports only 8-bit and 16-bit integer chunks, but got {t.Name}. Use ChunkFunctions.histogram or a binned histogram instead."

    let addDenseChunkInto<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> target (chunk: Chunk<'T>) =
        let t = typeof<'T>
        match target with
        | UInt8Counts counts when t = typeof<uint8> ->
            addDenseCountsFromBytesInto<uint8> counts chunk.Bytes chunk.ByteLength int
        | Int8Counts counts when t = typeof<int8> ->
            let offset = -int SByte.MinValue
            addDenseCountsFromBytesInto<int8> counts chunk.Bytes chunk.ByteLength (fun value -> int value + offset)
        | UInt16Counts counts when t = typeof<uint16> ->
            addDenseCountsFromBytesInto<uint16> counts chunk.Bytes chunk.ByteLength int
        | Int16Counts counts when t = typeof<int16> ->
            let offset = -int Int16.MinValue
            addDenseCountsFromBytesInto<int16> counts chunk.Bytes chunk.ByteLength (fun value -> int value + offset)
        | _ ->
            invalidArg "target" $"Dense histogram accumulator does not match chunk pixel type {t.Name}."

    let denseToMap<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> dense =
        let t = typeof<'T>
        if t = typeof<uint8> then
            match dense with
            | UInt8Counts counts -> denseHistogramMap counts (fun index -> box (uint8 index) :?> 'T)
            | _ -> invalidArg "dense" $"Expected UInt8 dense histogram for {t.Name} map conversion."
        elif t = typeof<int8> then
            match dense with
            | Int8Counts counts ->
                let offset = -int SByte.MinValue
                denseHistogramMap counts (fun index -> box (int8 (index - offset)) :?> 'T)
            | _ -> invalidArg "dense" $"Expected Int8 dense histogram for {t.Name} map conversion."
        elif t = typeof<uint16> then
            match dense with
            | UInt16Counts counts -> denseHistogramMap counts (fun index -> box (uint16 index) :?> 'T)
            | _ -> invalidArg "dense" $"Expected UInt16 dense histogram for {t.Name} map conversion."
        elif t = typeof<int16> then
            match dense with
            | Int16Counts counts ->
                let offset = -int Int16.MinValue
                denseHistogramMap counts (fun index -> box (int16 (index - offset)) :?> 'T)
            | _ -> invalidArg "dense" $"Expected Int16 dense histogram for {t.Name} map conversion."
        else
            invalidArg "T" $"ChunkFunctions.denseToMap supports only 8-bit and 16-bit integer chunks, but got {t.Name}."

    let histogramDenseBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (bytes: byte[])
        byteLength
        =
        histogramDenseCountsBytes<'T> bytes byteLength
        |> denseToMap<'T>

    let private leftEdgeBin (edges: float[]) value =
        let search = Array.BinarySearch(edges, value)
        if search >= 0 then
            search
        else
            let insertion = ~~~search
            if insertion = 0 then 0
            elif insertion >= edges.Length then edges.Length - 1
                else insertion - 1

    let private histogramLeftEdgesFromBytes<'Raw when 'Raw: (new: unit -> 'Raw) and 'Raw: struct and 'Raw :> ValueType>
        (edges: float[])
        (bytes: byte[])
        byteLength
        (toFloat: 'Raw -> float)
        =
        let values = MemoryMarshal.Cast<byte, 'Raw>(bytes.AsSpan(0, byteLength))
        let counts = Array.zeroCreate<uint64> edges.Length
        let mutable i = 0
        while i < values.Length do
            let value = toFloat values[i]
            if not (Double.IsNaN value || Double.IsInfinity value) then
                let bin = leftEdgeBin edges value
                counts[bin] <- counts[bin] + 1UL
            i <- i + 1
        counts

    let validateLeftEdges (leftEdges: float seq) =
        let edges = leftEdges |> Seq.toArray
        if edges.Length = 0 then
            invalidArg "leftEdges" "Histogram left edges must contain at least one edge."

        for i in 0 .. edges.Length - 1 do
            if Double.IsNaN edges[i] || Double.IsInfinity edges[i] then
                invalidArg "leftEdges" "Histogram left edges must be finite values."
            if i > 0 && edges[i] <= edges[i - 1] then
                invalidArg "leftEdges" "Histogram left edges must be strictly increasing."
        edges

    let histogramLeftEdgeCountsBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (leftEdges: float seq)
        (bytes: byte[])
        byteLength
        =
        let edges = validateLeftEdges leftEdges
        let t = typeof<'T>
        let counts =
            if t = typeof<uint8> then
                histogramLeftEdgesFromBytes<uint8> edges bytes byteLength float
            elif t = typeof<int8> then
                histogramLeftEdgesFromBytes<int8> edges bytes byteLength float
            elif t = typeof<uint16> then
                histogramLeftEdgesFromBytes<uint16> edges bytes byteLength float
            elif t = typeof<int16> then
                histogramLeftEdgesFromBytes<int16> edges bytes byteLength float
            elif t = typeof<uint32> then
                histogramLeftEdgesFromBytes<uint32> edges bytes byteLength float
            elif t = typeof<int32> then
                histogramLeftEdgesFromBytes<int32> edges bytes byteLength float
            elif t = typeof<float32> then
                histogramLeftEdgesFromBytes<float32> edges bytes byteLength float
            elif t = typeof<float> then
                histogramLeftEdgesFromBytes<float> edges bytes byteLength id
            else
                let values = MemoryMarshal.Cast<byte, 'T>(bytes.AsSpan(0, byteLength))
                let counts = Array.zeroCreate<uint64> edges.Length
                let mutable i = 0
                while i < values.Length do
                    let value = Convert.ToDouble(box values[i])
                    if not (Double.IsNaN value || Double.IsInfinity value) then
                        let bin = leftEdgeBin edges value
                        counts[bin] <- counts[bin] + 1UL
                    i <- i + 1
                counts

        { LeftEdges = edges
          Counts = counts }

    let private leftEdgesEqual (left: float[]) (right: float[]) =
        if left.Length <> right.Length then
            false
        else
            let mutable equal = true
            let mutable i = 0
            while equal && i < left.Length do
                equal <- left[i] = right[i]
                i <- i + 1
            equal

    let addLeftEdgesInto target source =
        if not (leftEdgesEqual target.LeftEdges source.LeftEdges) then
            invalidArg "source" "Cannot add left-edge histograms with different bin edges."
        addCountsInto target.Counts source.Counts

    let leftEdgesToMap leftEdgeHistogram =
        let mutable output = Map.empty<float, uint64>
        for i in 0 .. leftEdgeHistogram.LeftEdges.Length - 1 do
            output <- output.Add(leftEdgeHistogram.LeftEdges[i], leftEdgeHistogram.Counts[i])
        output

    let histogramLeftEdgesBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (leftEdges: float seq)
        (bytes: byte[])
        byteLength
        =
        histogramLeftEdgeCountsBytes<'T> leftEdges bytes byteLength
        |> leftEdgesToMap

    let histogramDictionary<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'T>)
        =
        histogramDictionaryBytes<'T> chunk.Bytes chunk.ByteLength

    let addChunkIntoDictionary<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (counts: Dictionary<'T, uint64>)
        (chunk: Chunk<'T>)
        =
        let values = MemoryMarshal.Cast<byte, 'T>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
        let mutable i = 0
        while i < values.Length do
            let mutable exists = false
            let count = &CollectionsMarshal.GetValueRefOrAddDefault(counts, values[i], &exists)
            count <- count + 1UL
            i <- i + 1

    let histogram<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'T>)
        =
        histogramBytes<'T> chunk.Bytes chunk.ByteLength

    let histogramDenseCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'T>)
        =
        histogramDenseCountsBytes<'T> chunk.Bytes chunk.ByteLength

    let histogramDense<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (chunk: Chunk<'T>)
        =
        histogramDenseBytes<'T> chunk.Bytes chunk.ByteLength

    let histogramLeftEdgeCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        leftEdges
        (chunk: Chunk<'T>)
        =
        histogramLeftEdgeCountsBytes<'T> leftEdges chunk.Bytes chunk.ByteLength

    let histogramLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        leftEdges
        (chunk: Chunk<'T>)
        =
        histogramLeftEdgesBytes<'T> leftEdges chunk.Bytes chunk.ByteLength

    let private denseDomain = function
        | UInt8Counts counts -> counts, 0.0, 255.0
        | Int8Counts counts -> counts, float SByte.MinValue, float SByte.MaxValue
        | UInt16Counts counts -> counts, 0.0, float UInt16.MaxValue
        | Int16Counts counts -> counts, float Int16.MinValue, float Int16.MaxValue

    let private totalCounts (counts: uint64[]) =
        counts |> Array.fold (fun acc count -> acc + count) 0UL

    let private denseEqualizationLut dense =
        let counts, outputMinimum, outputMaximum = denseDomain dense
        let total = totalCounts counts
        if total = 0UL then
            invalidArg "histogram" "Histogram equalization needs at least one counted pixel."

        let scale = (outputMaximum - outputMinimum) / float total
        let lut = Array.zeroCreate<float> counts.Length
        let mutable cumulative = 0UL
        for i in 0 .. counts.Length - 1 do
            cumulative <- cumulative + counts[i]
            lut[i] <- outputMinimum + float cumulative * scale
        lut

    let private exactEqualizationLut<'T when 'T: comparison> (counts: Map<'T, uint64>) =
        if counts.IsEmpty then
            invalidArg "histogram" "Histogram equalization needs at least one counted pixel."

        let ordered = counts |> Seq.map (fun pair -> pair.Key, pair.Value) |> Seq.sortBy fst |> Seq.toArray
        let total = ordered |> Array.sumBy snd
        if total = 0UL then
            invalidArg "histogram" "Histogram equalization needs at least one counted pixel."

        let minimum = ordered[0] |> fst |> box |> Convert.ToDouble
        let maximum = ordered[ordered.Length - 1] |> fst |> box |> Convert.ToDouble
        let scale =
            if maximum = minimum then
                0.0
            else
                (maximum - minimum) / float total

        let lut = Dictionary<'T, float>()
        let mutable cumulative = 0UL
        for value, count in ordered do
            cumulative <- cumulative + count
            lut[value] <- minimum + float cumulative * scale
        lut

    let private leftEdgeEqualizationLut (histogram: LeftEdgeHistogram) =
        let edges = validateLeftEdges histogram.LeftEdges
        if histogram.Counts.Length <> edges.Length then
            invalidArg "histogram" $"Left-edge histogram has {edges.Length} edges but {histogram.Counts.Length} counts."
        let total = totalCounts histogram.Counts
        if total = 0UL then
            invalidArg "histogram" "Histogram equalization needs at least one counted pixel."

        let outputMinimum = edges[0]
        let outputMaximum = edges[edges.Length - 1]
        let scale =
            if outputMaximum = outputMinimum then
                0.0
            else
                (outputMaximum - outputMinimum) / float total
        let lut = Array.zeroCreate<float> edges.Length
        let mutable cumulative = 0UL
        for i in 0 .. edges.Length - 1 do
            cumulative <- cumulative + histogram.Counts[i]
            lut[i] <- outputMinimum + float cumulative * scale
        edges, lut

    let inline private equalizeDenseUInt8 (lut: float[]) (inputPixels: Span<uint8>) (outputPixels: Span<uint8>) =
        let mutable i = 0
        while i < inputPixels.Length do
            outputPixels[i] <- clampRoundToByte (float32 lut[int inputPixels[i]])
            i <- i + 1

    let inline private equalizeDenseInt8 (lut: float[]) (inputPixels: Span<int8>) (outputPixels: Span<int8>) =
        let offset = -int SByte.MinValue
        let mutable i = 0
        while i < inputPixels.Length do
            outputPixels[i] <- clampRoundToSByte (float32 lut[int inputPixels[i] + offset])
            i <- i + 1

    let inline private equalizeDenseUInt16 (lut: float[]) (inputPixels: Span<uint16>) (outputPixels: Span<uint16>) =
        let mutable i = 0
        while i < inputPixels.Length do
            outputPixels[i] <- clampRoundToUInt16 (float32 lut[int inputPixels[i]])
            i <- i + 1

    let inline private equalizeDenseInt16 (lut: float[]) (inputPixels: Span<int16>) (outputPixels: Span<int16>) =
        let offset = -int Int16.MinValue
        let mutable i = 0
        while i < inputPixels.Length do
            outputPixels[i] <- clampRoundToInt16 (float32 lut[int inputPixels[i] + offset])
            i <- i + 1

    let inline private equalizeLeftEdgesTyped
        (edges: float[])
        (lut: float[])
        (inputPixels: Span< ^T>)
        (outputPixels: Span< ^T>)
        (convert: float -> ^T)
        =
        let mutable i = 0
        while i < inputPixels.Length do
            let value = float inputPixels[i]
            if Double.IsNaN value || Double.IsInfinity value then
                outputPixels[i] <- inputPixels[i]
            else
                outputPixels[i] <- convert lut[leftEdgeBin edges value]
            i <- i + 1

    let private convertEqualizedGeneric<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: float) =
        let t = typeof<'T>
        if t = typeof<uint8> then
            box (clampRoundDoubleToByte value) :?> 'T
        elif t = typeof<int8> then
            box (clampRoundDoubleToSByte value) :?> 'T
        elif t = typeof<uint16> then
            box (clampRoundDoubleToUInt16 value) :?> 'T
        elif t = typeof<int16> then
            box (clampRoundDoubleToInt16 value) :?> 'T
        elif t = typeof<int32> then
            box (clampRoundDoubleToInt32 value) :?> 'T
        elif t = typeof<uint32> then
            box (clampRoundDoubleToUInt32 value) :?> 'T
        elif t = typeof<int64> then
            box (clampRoundDoubleToInt64 value) :?> 'T
        elif t = typeof<uint64> then
            box (clampRoundDoubleToUInt64 value) :?> 'T
        elif t = typeof<float32> then
            box (float32 value) :?> 'T
        elif t = typeof<float> then
            box value :?> 'T
        else
            Convert.ChangeType(value, t) :?> 'T

    let histogramEqualizationDense<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        dense
        (chunk: Chunk<'T>)
        =
        let lut = denseEqualizationLut dense
        let output = create<'T> chunk.Size
        try
            let t = typeof<'T>
            if t = typeof<uint8> then
                equalizeDenseUInt8 lut (MemoryMarshal.Cast<byte, uint8>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, uint8>(output.Bytes.AsSpan(0, output.ByteLength)))
            elif t = typeof<int8> then
                equalizeDenseInt8 lut (MemoryMarshal.Cast<byte, int8>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, int8>(output.Bytes.AsSpan(0, output.ByteLength)))
            elif t = typeof<uint16> then
                equalizeDenseUInt16 lut (MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength)))
            elif t = typeof<int16> then
                equalizeDenseInt16 lut (MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength)))
            else
                invalidArg "T" $"Dense histogram equalization supports UInt8, Int8, UInt16, and Int16 chunks, got {t.Name}."
            output
        with
        | _ ->
            decRef output
            reraise()

    let histogramEqualizationSparse<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        counts
        (chunk: Chunk<'T>)
        =
        let lut = exactEqualizationLut counts
        let output = create<'T> chunk.Size
        try
            let inputPixels = span<'T> chunk
            let outputPixels = span<'T> output
            for i in 0 .. inputPixels.Length - 1 do
                match lut.TryGetValue inputPixels[i] with
                | true, value -> outputPixels[i] <- convertEqualizedGeneric<'T> value
                | false, _ -> outputPixels[i] <- inputPixels[i]
            output
        with
        | _ ->
            decRef output
            reraise()

    let histogramEqualizationLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        histogram
        (chunk: Chunk<'T>)
        =
        let edges, lut = leftEdgeEqualizationLut histogram
        let output = create<'T> chunk.Size
        try
            let t = typeof<'T>
            if t = typeof<uint8> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, uint8>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, uint8>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToByte
            elif t = typeof<int8> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, int8>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, int8>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToSByte
            elif t = typeof<uint16> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToUInt16
            elif t = typeof<int16> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToInt16
            elif t = typeof<int32> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToInt32
            elif t = typeof<uint32> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, uint32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, uint32>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToUInt32
            elif t = typeof<int64> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, int64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, int64>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToInt64
            elif t = typeof<uint64> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, uint64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, uint64>(output.Bytes.AsSpan(0, output.ByteLength))) clampRoundDoubleToUInt64
            elif t = typeof<float32> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))) float32
            elif t = typeof<float> then
                equalizeLeftEdgesTyped edges lut (MemoryMarshal.Cast<byte, float>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) (MemoryMarshal.Cast<byte, float>(output.Bytes.AsSpan(0, output.ByteLength))) id
            else
                invalidArg "T" $"Left-edge histogram equalization supports real numeric chunks, got {t.Name}."
            output
        with
        | _ ->
            decRef output
            reraise()
