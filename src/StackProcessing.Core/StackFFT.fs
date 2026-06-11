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

