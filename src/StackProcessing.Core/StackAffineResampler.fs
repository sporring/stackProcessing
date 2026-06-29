module StackAffineResampler

open System
open System.Runtime.InteropServices
open FSharp.Control
open SlimPipeline
open TinyLinAlg
open StackCore

let inline floorToInt (x:float) = int (Math.Floor x)

// -------------------------
// Image geometry
// -------------------------
type ImageGeom =
    { W: int; H: int; D: int
      Origin: V3
      Spacing: V3
      Direction: M3 }

let indexToPhysical (g:ImageGeom) (i:int) (j:int) (k:int) : V3 =
    // p = origin + direction * ( [i*spx, j*spy, k*spz] )
    let v = v3 (float i * g.Spacing.x) (float j * g.Spacing.y) (float k * g.Spacing.z)
    v3Add g.Origin (m3v3Mul g.Direction v)

// Convert physical point to continuous index in image g:
// c = inv(Direction) * (p - origin) / spacing
let physicalToContIndex (g:ImageGeom) (invDir:M3) (p:V3) : V3 =
    let q = m3v3Mul invDir (v3Sub p g.Origin)
    v3 (q.x / g.Spacing.x) (q.y / g.Spacing.y) (q.z / g.Spacing.z)

let inline clamp (lo:int) (hi:int) (v:int) = if v < lo then lo elif v > hi then hi else v

let private validateChunkSlice<'T when 'T: equality> operatorName width height (chunk: Chunk<'T>) =
    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
    if chunkDepth <> 1UL then
        invalidArg "chunk" $"{operatorName} expects 2D slice chunks with depth 1, got {chunk.Size}."
    if chunkWidth <> uint64 width || chunkHeight <> uint64 height then
        invalidArg "chunk" $"{operatorName} expects chunks of size {width}x{height}x1, got {chunk.Size}."

let private getChunkVoxel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: int)
    (height: int)
    (background: 'T)
    (x: int)
    (y: int)
    (z: int)
    (slices: Chunk<'T>[])
    : 'T
    =
    if z < 0 || z >= slices.Length || x < 0 || x >= width || y < 0 || y >= height then
        background
    else
        let pixels = Chunk.span<'T> slices[z]
        pixels[flatIndex2 width x y]

let trilinearSampleChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: int)
    (height: int)
    (depth: int)
    (background: 'T)
    (lerp: 'T -> 'T -> float32 -> 'T)
    (c: V3)
    (slices: Chunk<'T>[])
    : 'T
    =
    if c.x < 0.0 || c.y < 0.0 || c.z < 0.0 || c.x >= float (width - 1) || c.y >= float (height - 1) || c.z >= float (depth - 1) then
        background
    else
        let x0 = floorToInt c.x
        let y0 = floorToInt c.y
        let z0 = floorToInt c.z
        let x1 = x0 + 1
        let y1 = y0 + 1
        let z1 = z0 + 1

        let fx = float32 (c.x - float x0)
        let fy = float32 (c.y - float y0)
        let fz = float32 (c.z - float z0)

        let c000 = getChunkVoxel width height background x0 y0 z0 slices
        let c100 = getChunkVoxel width height background x1 y0 z0 slices
        let c010 = getChunkVoxel width height background x0 y1 z0 slices
        let c110 = getChunkVoxel width height background x1 y1 z0 slices
        let c001 = getChunkVoxel width height background x0 y0 z1 slices
        let c101 = getChunkVoxel width height background x1 y0 z1 slices
        let c011 = getChunkVoxel width height background x0 y1 z1 slices
        let c111 = getChunkVoxel width height background x1 y1 z1 slices

        let c00 = lerp c000 c100 fx
        let c10 = lerp c010 c110 fx
        let c01 = lerp c001 c101 fx
        let c11 = lerp c011 c111 fx
        let c0 = lerp c00 c10 fy
        let c1 = lerp c01 c11 fy
        lerp c0 c1 fz

// -------------------------
// Required chunk computation for ONE output slice (k)
// Using 4 corners mapped to input continuous index bounds.
// Then expands integer bounds by +1 for trilinear footprint.
// -------------------------
let requiredChunksForSliceTrilinear (winsz:int) (inG:ImageGeom) (outG:ImageGeom) (affOutToIn: Affine) (k:int) : seq<int*int*int> =

    let invInDir = m3Inv inG.Direction

    // corners in output index space
    let corners =
        [| (0, 0, k)
           (outG.W - 1, 0, k)
           (0, outG.H - 1, k)
           (outG.W - 1, outG.H - 1, k) |]

    let mutable xmin = Double.PositiveInfinity
    let mutable xmax = Double.NegativeInfinity
    let mutable ymin = Double.PositiveInfinity
    let mutable ymax = Double.NegativeInfinity
    let mutable zmin = Double.PositiveInfinity
    let mutable zmax = Double.NegativeInfinity

    for (i,j,kk) in corners do
        let pOut = indexToPhysical outG i j kk
        let pIn  = affinePoint affOutToIn pOut
        let cIn  = physicalToContIndex inG invInDir pIn
        xmin <- min xmin cIn.x; xmax <- max xmax cIn.x
        ymin <- min ymin cIn.y; ymax <- max ymax cIn.y
        zmin <- min zmin cIn.z; zmax <- max zmax cIn.z

    // integer bounds needed for trilinear: floor(min) .. floor(max)+1  (inclusive)
    let x0 = clamp 0 (inG.W-1) (floorToInt xmin)
    let x1 = clamp 0 (inG.W-1) (floorToInt xmax + 1)
    let y0 = clamp 0 (inG.H-1) (floorToInt ymin)
    let y1 = clamp 0 (inG.H-1) (floorToInt ymax + 1)
    let z0 = clamp 0 (inG.D-1) (floorToInt zmin)
    let z1 = clamp 0 (inG.D-1) (floorToInt zmax + 1)

    let cx0, cx1 = (x0 / winsz), (x1 / winsz)
    let cy0, cy1 = (y0 / winsz), (y1 / winsz)
    let cz0, cz1 = (z0 / winsz), (z1 / winsz)

    seq {
        for cz in cz0 .. cz1 do
          for cy in cy0 .. cy1 do
            for cx in cx0 .. cx1 do
              yield (cx,cy,cz)
    }

// -------------------------
// Main: resample output slices (k = 0..outD-1) in a z-sweep.
// Uses affine incremental stepping for speed.
// Returns a sequence of float32[] slices, x-fastest, length = outW*outH.
// -------------------------
let resampleAffineChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (lerp: 'T -> 'T -> float32 -> 'T)
    (inG: ImageGeom)
    (outG: ImageGeom)
    (affOutToIn: Affine)
    (background: 'T)
    (input: Chunk<'T> list)
    : Chunk<'T> list
    =

    if input.Length <> inG.D then
        invalidArg "input" $"resampleAffineChunk expects {inG.D} input chunks from the input geometry, got {input.Length}."

    input |> List.iter (validateChunkSlice "resampleAffineChunk" inG.W inG.H)
    let slices = input |> List.toArray

    let invInDir = m3Inv inG.Direction

    let stepI_out = m3v3Mul outG.Direction (v3 outG.Spacing.x 0.0 0.0)
    let stepJ_out = m3v3Mul outG.Direction (v3 0.0 outG.Spacing.y 0.0)
    let stepK_out = m3v3Mul outG.Direction (v3 0.0 0.0 outG.Spacing.z)

    let stepI_in_phys = m3v3Mul affOutToIn.A stepI_out
    let stepJ_in_phys = m3v3Mul affOutToIn.A stepJ_out
    let stepK_in_phys = m3v3Mul affOutToIn.A stepK_out

    let stepI_in_cont =
        let q = m3v3Mul invInDir stepI_in_phys
        v3 (q.x / inG.Spacing.x) (q.y / inG.Spacing.y) (q.z / inG.Spacing.z)

    let stepJ_in_cont =
        let q = m3v3Mul invInDir stepJ_in_phys
        v3 (q.x / inG.Spacing.x) (q.y / inG.Spacing.y) (q.z / inG.Spacing.z)

    let stepK_in_cont =
        let q = m3v3Mul invInDir stepK_in_phys
        v3 (q.x / inG.Spacing.x) (q.y / inG.Spacing.y) (q.z / inG.Spacing.z)

    let outputs = ResizeArray<Chunk<'T>>(outG.D)

    try
        for k in 0 .. outG.D - 1 do
            let pOut00k = indexToPhysical outG 0 0 k
            let pIn00k = affinePoint affOutToIn pOut00k
            let c00k = physicalToContIndex inG invInDir pIn00k

            let output = Chunk.create<'T> (uint64 outG.W, uint64 outG.H, 1UL)
            try
                let outputPixels = Chunk.span<'T> output
                let mutable rowStart = c00k

                for j in 0 .. outG.H - 1 do
                    let mutable c = rowStart
                    for i in 0 .. outG.W - 1 do
                        outputPixels[flatIndex2 outG.W i j] <-
                            trilinearSampleChunkSlices inG.W inG.H inG.D background lerp c slices
                        c <- v3Add c stepI_in_cont
                    rowStart <- v3Add rowStart stepJ_in_cont

                outputs.Add output
            with
            | _ ->
                Chunk.decRef output
                reraise()

        outputs |> Seq.toList
    with
    | _ ->
        outputs |> Seq.iter Chunk.decRef
        reraise()

let resampleAffineChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (lerp: 'T -> 'T -> float32 -> 'T)
    (inG: ImageGeom)
    (outG: ImageGeom)
    (affOutToIn: Affine)
    (background: 'T)
    : Stage<Chunk<'T>, Chunk<'T>>
    =

    let name = $"resampleAffineChunk.{typeof<'T>.Name}"
    let elementBytes = uint64 (Marshal.SizeOf<'T>())
    let inputPixels = uint64 inG.W * uint64 inG.H * uint64 inG.D
    let outputPixels = uint64 outG.W * uint64 outG.H * uint64 outG.D
    let memoryNeed _ = (inputPixels + outputPixels) * elementBytes
    let elementTransformation _ = uint64 outG.D

    let apply _debug (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            let! chunks = input |> AsyncSeq.toListAsync
            try
                let outputs = resampleAffineChunkSlices lerp inG outG affOutToIn background chunks
                for output in outputs do
                    yield output
            finally
                chunks |> List.iter Chunk.decRef
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) memoryNeed elementTransformation pipe
    |> Stage.withSliceCardinality (SliceCardinality.reduceTo (uint64 outG.D))
