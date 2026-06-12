module StackFFT

open System
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

let fftXYFloat32ToComplex64Interleaved : Stage<Chunk<float32>, Chunk<float32>> =
    let mapper (input: Chunk<float32>) =
        ChunkKernel.fftXYFloat32ToComplex64InterleavedChunk input

    releaseUnaryChunk
        "chunkFftXYFloat32ToComplex64Interleaved"
        mapper
        (fun nPixels -> nPixels * uint64 (sizeof<float32> + 2 * sizeof<float32>))

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
