module StackFFT

open System
open System.IO
open FSharp.Control
open SlimPipeline
open StackCore

module ChunkKernel = ChunkCore.ChunkFunctions

let private chunkElementBytes<'T> =
    System.Runtime.InteropServices.Marshal.SizeOf<'T>()

let private chunkMemoryNeed<'T> nPixels =
    nPixels * uint64 (chunkElementBytes<'T>)

let private releaseUnaryChunk name f memoryNeed : Stage<Chunk<'T>, Chunk<'U>> =
    let mapper _debug chunk =
        try
            f chunk
        finally
            Chunk.decRef chunk

    Stage.map name mapper memoryNeed id

let private releaseBinaryChunk name f memoryNeed : Stage<Chunk<'T> * Chunk<'U>, Chunk<'V>> =
    let mapper _debug (a, b) =
        try
            f a b
        finally
            Chunk.decRef a
            Chunk.decRef b

    Stage.map name mapper memoryNeed id

let private validateSameSize name (a: Chunk<'T>) (b: Chunk<'U>) =
    if a.Size <> b.Size then
        invalidArg "b" $"{name}: chunk sizes differ: {a.Size} vs {b.Size}."

let private validateComplex64Interleaved name (chunk: Chunk<float32>) =
    let width, height, depth = chunk.Size
    if depth <> 1UL then
        invalidArg "chunk" $"{name} expects 2D complex64-interleaved chunks with depth 1, got {chunk.Size}."
    if width % 2UL <> 0UL then
        invalidArg "chunk" $"{name} expects even interleaved width, got {chunk.Size}."
    int (width / 2UL), int height

let private complex64FromRealImag (real: Chunk<float>) (imag: Chunk<float>) =
    validateSameSize "chunkToComplex64" real imag
    let width, height, depth = real.Size
    if depth <> 1UL then
        invalidArg "real" $"chunkToComplex64 expects 2D slice chunks with depth 1, got {real.Size}."

    let output = Chunk.create<float32> (2UL * width, height, 1UL)
    try
        let realPixels = Chunk.span real
        let imagPixels = Chunk.span imag
        let outputPixels = Chunk.span output
        let mutable j = 0
        for i in 0 .. realPixels.Length - 1 do
            outputPixels[j] <- float32 realPixels[i]
            outputPixels[j + 1] <- float32 imagPixels[i]
            j <- j + 2
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private complex64FromPolar (modulus: Chunk<float>) (argument: Chunk<float>) =
    validateSameSize "chunkPolarToComplex64" modulus argument
    let width, height, depth = modulus.Size
    if depth <> 1UL then
        invalidArg "modulus" $"chunkPolarToComplex64 expects 2D slice chunks with depth 1, got {modulus.Size}."

    let output = Chunk.create<float32> (2UL * width, height, 1UL)
    try
        let modulusPixels = Chunk.span modulus
        let argumentPixels = Chunk.span argument
        let outputPixels = Chunk.span output
        let mutable j = 0
        for i in 0 .. modulusPixels.Length - 1 do
            let r = modulusPixels[i]
            let theta = argumentPixels[i]
            outputPixels[j] <- float32 (r * Math.Cos theta)
            outputPixels[j + 1] <- float32 (r * Math.Sin theta)
            j <- j + 2
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let toComplex64 : Stage<Chunk<float> * Chunk<float>, Chunk<float32>> =
    releaseBinaryChunk "chunkToComplex64" complex64FromRealImag (fun n -> n * uint64 (2 * sizeof<float> + 2 * sizeof<float32>))

let polarToComplex64 : Stage<Chunk<float> * Chunk<float>, Chunk<float32>> =
    releaseBinaryChunk "chunkPolarToComplex64" complex64FromPolar (fun n -> n * uint64 (2 * sizeof<float> + 2 * sizeof<float32>))

let private complex64Part name selector (chunk: Chunk<float32>) =
    let logicalWidth, height = validateComplex64Interleaved name chunk
    let output = Chunk.create<float> (uint64 logicalWidth, uint64 height, 1UL)
    try
        let inputPixels = Chunk.span chunk
        let outputPixels = Chunk.span output
        let mutable j = 0
        for i in 0 .. outputPixels.Length - 1 do
            outputPixels[i] <- selector inputPixels[j] inputPixels[j + 1]
            j <- j + 2
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let complex64Real : Stage<Chunk<float32>, Chunk<float>> =
    releaseUnaryChunk "chunkComplex64Real" (complex64Part "chunkComplex64Real" (fun re _im -> float re)) (fun n -> n * uint64 (2 * sizeof<float32> + sizeof<float>))

let complex64Imag : Stage<Chunk<float32>, Chunk<float>> =
    releaseUnaryChunk "chunkComplex64Imag" (complex64Part "chunkComplex64Imag" (fun _re im -> float im)) (fun n -> n * uint64 (2 * sizeof<float32> + sizeof<float>))

let complex64Modulus : Stage<Chunk<float32>, Chunk<float>> =
    releaseUnaryChunk "chunkComplex64Modulus" (complex64Part "chunkComplex64Modulus" (fun re im -> Math.Sqrt(float re * float re + float im * float im))) (fun n -> n * uint64 (2 * sizeof<float32> + sizeof<float>))

let complex64Argument : Stage<Chunk<float32>, Chunk<float>> =
    releaseUnaryChunk "chunkComplex64Argument" (complex64Part "chunkComplex64Argument" (fun re im -> Math.Atan2(float im, float re))) (fun n -> n * uint64 (2 * sizeof<float32> + sizeof<float>))

let private complex64ConjugateChunk (chunk: Chunk<float32>) =
    let logicalWidth, height = validateComplex64Interleaved "chunkComplex64Conjugate" chunk
    let output = Chunk.create<float32> (uint64 (2 * logicalWidth), uint64 height, 1UL)
    try
        let inputPixels = Chunk.span chunk
        let outputPixels = Chunk.span output
        let mutable j = 0
        while j < inputPixels.Length do
            outputPixels[j] <- inputPixels[j]
            outputPixels[j + 1] <- -inputPixels[j + 1]
            j <- j + 2
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let complex64Conjugate : Stage<Chunk<float32>, Chunk<float32>> =
    releaseUnaryChunk "chunkComplex64Conjugate" complex64ConjugateChunk (fun n -> n * uint64 (4 * sizeof<float32>))

let fftShiftXYComplex64Interleaved : Stage<Chunk<float32>, Chunk<float32>> =
    releaseUnaryChunk
        "chunkFftShiftXYComplex64Interleaved"
        ChunkKernel.fftShiftXYComplex64InterleavedChunk
        (fun n -> n * uint64 (2 * sizeof<float32> + 2 * sizeof<float32>))

let fftShiftZComplex64InterleavedViaTempChunks : Stage<Chunk<float32>, Chunk<float32>> =
    let name = "chunkFftShiftZComplex64InterleavedViaTempChunks"
    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel =
        StageMemoryModel.fromSinglePeak
            (Custom name)
            (fun n -> n * uint64 (3 * sizeof<float32>))

    let apply _debug (input: AsyncSeq<Chunk<float32>>) =
        asyncSeq {
            let tempDir =
                Path.Combine(
                    Path.GetTempPath(),
                    $"stackprocessing-fftshift-{Guid.NewGuid():N}")

            Directory.CreateDirectory(tempDir) |> ignore
            let paths = ResizeArray<string>()
            let mutable expectedSize = ValueNone

            let cleanup () =
                for path in paths do
                    try
                        if File.Exists(path) then
                            File.Delete(path)
                    with _ ->
                        ()

                try
                    if Directory.Exists(tempDir) then
                        Directory.Delete(tempDir)
                with _ ->
                    ()

            try
                for chunk in input do
                    try
                        let interleavedWidth, _height, depth = chunk.Size
                        if depth <> 1UL then
                            invalidArg "chunk" $"{name} expects 2D complex64-interleaved chunks with depth 1, got {chunk.Size}."
                        if interleavedWidth % 2UL <> 0UL then
                            invalidArg "chunk" $"{name} expects even interleaved width, got {chunk.Size}."

                        match expectedSize with
                        | ValueNone ->
                            expectedSize <- ValueSome chunk.Size
                        | ValueSome size when size <> chunk.Size ->
                            invalidArg "chunk" $"{name} expects all slices to have the same size, got {chunk.Size} after {size}."
                        | ValueSome _ ->
                            ()

                        let path = Path.Combine(tempDir, $"{paths.Count:D12}.bin")
                        use stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1024 * 1024, FileOptions.SequentialScan)
                        stream.Write(chunk.Bytes, 0, chunk.ByteLength)
                        paths.Add(path)
                    finally
                        Chunk.decRef chunk

                let depth = paths.Count
                if depth > 0 then
                    let shiftZ = depth / 2
                    let start = depth - shiftZ
                    let size =
                        match expectedSize with
                        | ValueSome value -> value
                        | ValueNone -> invalidOp $"{name} internal error: missing chunk size for non-empty stream."

                    for outZ in 0 .. depth - 1 do
                        let sourceZ = (start + outZ) % depth
                        let path = paths[sourceZ]
                        let bytes = File.ReadAllBytes(path)
                        let output = Chunk.create<float32> size

                        try
                            if bytes.Length <> output.ByteLength then
                                invalidOp $"{name} expected {output.ByteLength} bytes in {path}, got {bytes.Length}."
                            bytes.AsSpan().CopyTo(output.Bytes.AsSpan(0, output.ByteLength))
                            yield output
                            File.Delete(path)
                        with
                        | ex ->
                            Chunk.decRef output
                            raise ex
            finally
                cleanup()
        }

    Stage.fromAsyncSeq name apply transition memoryModel id

let fftShift3DComplex64Interleaved : Stage<Chunk<float32>, Chunk<float32>> =
    fftShiftXYComplex64Interleaved --> fftShiftZComplex64InterleavedViaTempChunks

let fftXYFloat32ToComplex64Interleaved : Stage<Chunk<float32>, Chunk<float32>> =
    let mapper (input: Chunk<float32>) =
        ChunkKernel.fftXYFloat32ToComplex64InterleavedChunk input

    releaseUnaryChunk
        "chunkFftXYFloat32ToComplex64Interleaved"
        mapper
        (fun nPixels -> nPixels * uint64 (sizeof<float32> + 2 * sizeof<float32>))

let invFftXYComplex64InterleavedToFloat32 : Stage<Chunk<float32>, Chunk<float32>> =
    let mapper (input: Chunk<float32>) =
        ChunkKernel.invFftXYComplex64InterleavedToFloat32Chunk input

    releaseUnaryChunk
        "chunkInvFftXYComplex64InterleavedToFloat32"
        mapper
        (fun nPixels -> nPixels * uint64 (2 * sizeof<float32> + sizeof<float32>))

let fftRealXYFloat32ToHermitianPackedComplex64Interleaved : Stage<Chunk<float32>, SpectralChunk> =
    let mapper _debug (input: Chunk<float32>) =
        try
            ChunkKernel.fftRealXYFloat32ToHermitianPackedComplex64InterleavedChunk input
        finally
            Chunk.decRef input

    Stage.map
        "chunkFftRealXYFloat32ToHermitianPackedComplex64Interleaved"
        mapper
        (fun nPixels -> nPixels * uint64 (sizeof<float32> + 2 * sizeof<float32>))
        id

let invFftXYHermitianPackedComplex64InterleavedToFloat32 : Stage<SpectralChunk, Chunk<float32>> =
    let mapper _debug (input: SpectralChunk) =
        try
            ChunkKernel.invFftXYHermitianPackedComplex64InterleavedToFloat32Chunk input
        finally
            Chunk.decRef input.Chunk

    Stage.map
        "chunkInvFftXYHermitianPackedComplex64InterleavedToFloat32"
        mapper
        (fun nPixels -> nPixels * uint64 (2 * sizeof<float32> + sizeof<float32>))
        id

let fftXYFloat32ToComplex64InterleavedParallelCollect (workers: int) : Stage<Chunk<float32>, Chunk<float32>> =
    if workers < 1 then
        invalidArg "workers" $"Chunk FFT XY parallelCollect expects at least one worker, got {workers}."
    if workers = 1 then
        fftXYFloat32ToComplex64Interleaved
    else
        let mapper _debug (window: Window<Chunk<float32>>) =
            match window.Items with
            | [ chunk ] ->
                try
                    [ ChunkKernel.fftXYFloat32ToComplex64InterleavedChunk chunk ]
                finally
                    Chunk.decRef chunk
            | items ->
                for chunk in items do
                    Chunk.decRef chunk
                invalidArg "window" $"Chunk FFT XY parallelCollect expects singleton windows, got {items.Length} items."

        Stage.parallelCollect
            $"chunkFftXYFloat32ToComplex64Interleaved.parallelCollect.workers{workers}"
            1
            workers
            1
            0
            (fun _ chunk -> chunk)
            mapper
            (fun nPixels -> nPixels * uint64 (sizeof<float32> + 2 * sizeof<float32>))
            id

let invFftXYComplex64InterleavedToFloat32ParallelCollect (workers: int) : Stage<Chunk<float32>, Chunk<float32>> =
    if workers < 1 then
        invalidArg "workers" $"Chunk inverse FFT XY parallelCollect expects at least one worker, got {workers}."
    if workers = 1 then
        invFftXYComplex64InterleavedToFloat32
    else
        let mapper _debug (window: Window<Chunk<float32>>) =
            match window.Items with
            | [ chunk ] ->
                try
                    [ ChunkKernel.invFftXYComplex64InterleavedToFloat32Chunk chunk ]
                finally
                    Chunk.decRef chunk
            | items ->
                for chunk in items do
                    Chunk.decRef chunk
                invalidArg "window" $"Chunk inverse FFT XY parallelCollect expects singleton windows, got {items.Length} items."

        Stage.parallelCollect
            $"chunkInvFftXYComplex64InterleavedToFloat32.parallelCollect.workers{workers}"
            1
            workers
            1
            0
            (fun _ chunk -> chunk)
            mapper
            (fun nPixels -> nPixels * uint64 (2 * sizeof<float32> + sizeof<float32>))
            id

let fftXYThenZFloat32ToComplex64InterleavedPlanned (windowLength: int) : Stage<Chunk<float32>, Chunk<float32>> =
    if windowLength < 1 then
        invalidArg "windowLength" $"Chunk FFT XY+Z planned expects a positive window length, got {windowLength}."

    let name = $"chunkFftXYThenZFloat32ToComplex64Interleaved.planned.window{windowLength}"
    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel =
        StageMemoryModel.fromSinglePeak
            (Custom name)
            (fun nPixels -> nPixels * uint64 windowLength * uint64 (sizeof<float32> + 2 * sizeof<float32> + 2 * sizeof<float32>))

    let apply _debug (input: AsyncSeq<Chunk<float32>>) =
        asyncSeq {
            use plans = new ChunkKernel.FftXYAndZPlanCache()
            let batch = ResizeArray<Chunk<float32>>(windowLength)

            let releaseBatch () =
                for chunk in batch do
                    Chunk.decRef chunk
                batch.Clear()

            let processBatch () =
                try
                    let outputs = plans.ForwardFloat32SlicesToComplex64Interleaved(batch)
                    releaseBatch()
                    outputs
                with
                | ex ->
                    releaseBatch()
                    raise ex

            try
                for chunk in input do
                    batch.Add(chunk)
                    if batch.Count = windowLength then
                        yield! processBatch() |> AsyncSeq.ofSeq

                if batch.Count > 0 then
                    yield! processBatch() |> AsyncSeq.ofSeq
            finally
                releaseBatch()
        }

    Stage.fromAsyncSeq name apply transition memoryModel id

let fft3DFloat32ToComplex64Interleaved (windowLength: int) : Stage<Chunk<float32>, Chunk<float32>> =
    fftXYThenZFloat32ToComplex64InterleavedPlanned windowLength

let fft3DRealXYFloat32ToComplex64Interleaved (windowLength: int) : Stage<Chunk<float32>, SpectralChunk> =
    if windowLength < 1 then
        invalidArg "windowLength" $"Chunk FFT real-XY+Z expects a positive window length, got {windowLength}."

    let name = $"chunkFft3DRealXYFloat32ToComplex64Interleaved.window{windowLength}"
    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel =
        StageMemoryModel.fromSinglePeak
            (Custom name)
            (fun nPixels -> nPixels * uint64 windowLength * uint64 (sizeof<float32> + 2 * sizeof<float32> + 2 * sizeof<float32>))

    let apply _debug (input: AsyncSeq<Chunk<float32>>) =
        asyncSeq {
            use plans = new ChunkKernel.FftRealXYAndZPlanCache()
            let batch = ResizeArray<Chunk<float32>>(windowLength)

            let releaseBatch () =
                for chunk in batch do
                    Chunk.decRef chunk
                batch.Clear()

            let processBatch () =
                try
                    let outputs = plans.ForwardFloat32SlicesToComplex64Interleaved(batch)
                    releaseBatch()
                    outputs
                with
                | ex ->
                    releaseBatch()
                    raise ex

            try
                for chunk in input do
                    batch.Add(chunk)
                    if batch.Count = windowLength then
                        yield! processBatch() |> AsyncSeq.ofSeq

                if batch.Count > 0 then
                    yield! processBatch() |> AsyncSeq.ofSeq
            finally
                releaseBatch()
        }

    Stage.fromAsyncSeq name apply transition memoryModel id

let invFft3DRealXYComplex64InterleavedToFloat32 (windowLength: int) : Stage<SpectralChunk, Chunk<float32>> =
    if windowLength < 1 then
        invalidArg "windowLength" $"Chunk inverse FFT real-XY+Z expects a positive window length, got {windowLength}."

    let name = $"chunkInvFft3DRealXYComplex64InterleavedToFloat32.window{windowLength}"
    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel =
        StageMemoryModel.fromSinglePeak
            (Custom name)
            (fun nPixels -> nPixels * uint64 windowLength * uint64 (2 * sizeof<float32> + sizeof<float32>))

    let apply _debug (input: AsyncSeq<SpectralChunk>) =
        asyncSeq {
            use plans = new ChunkKernel.InvFftRealXYAndZPlanCache()
            let batch = ResizeArray<SpectralChunk>(windowLength)

            let releaseBatch () =
                for spectral in batch do
                    Chunk.decRef spectral.Chunk
                batch.Clear()

            let processBatch () =
                try
                    let outputs = plans.InverseHermitianPackedToFloat32Slices(batch)
                    releaseBatch()
                    outputs
                with
                | ex ->
                    releaseBatch()
                    raise ex

            try
                for spectral in input do
                    batch.Add(spectral)
                    if batch.Count = windowLength then
                        yield! processBatch() |> AsyncSeq.ofSeq

                if batch.Count > 0 then
                    yield! processBatch() |> AsyncSeq.ofSeq
            finally
                releaseBatch()
        }

    Stage.fromAsyncSeq name apply transition memoryModel id
