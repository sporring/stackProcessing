module StackAffineResampler

open System
open System.Collections.Generic
open System.IO
open FSharp.Control
open SlimPipeline
open TinyLinAlg
open StackIO

type Image<'S when 'S: equality> = Image.Image<'S>

let inline floorToInt (x:float) = int (Math.Floor x)

// -------------------------
// Image geometry
// -------------------------
type ImageGeom =
    { W: int; H: int; D: int
      Origin: V3
      Spacing: V3
      Direction: M3 } // ITK/SimpleITK direction matrix

let indexToPhysical (g:ImageGeom) (i:int) (j:int) (k:int) : V3 =
    // p = origin + direction * ( [i*spx, j*spy, k*spz] )
    let v = v3 (float i * g.Spacing.x) (float j * g.Spacing.y) (float k * g.Spacing.z)
    add g.Origin (mulMV g.Direction v)

// Convert physical point to continuous index in image g:
// c = inv(Direction) * (p - origin) / spacing
let physicalToContIndex (g:ImageGeom) (invDir:M3) (p:V3) : V3 =
    let q = mulMV invDir (sub p g.Origin)
    v3 (q.x / g.Spacing.x) (q.y / g.Spacing.y) (q.z / g.Spacing.z)

// -------------------------
// Chunk interface + cache
// -------------------------
let inline clamp (lo:int) (hi:int) (v:int) = if v < lo then lo elif v > hi then hi else v

let inline packKey (cx:int) (cy:int) (cz:int) : int64 =
    // pack into 21 bits each (good for sizes up to ~2 million chunks/axis)
    (int64 cx <<< 42) ||| (int64 cy <<< 21) ||| (int64 cz)

type ChunkData<'T> =
    { Pixels: 'T[,,]
      Width: int
      Height: int
      Depth: int }

type ChunkCache<'T when 'T: equality> = Dictionary<int64, ChunkData<'T>>

module ChunkCache =
    let create<'T when 'T: equality> () = Dictionary<int64, ChunkData<'T>>()

    let private readChunkData<'T when 'T: equality> inputDir suffix cx cy cz =
        let image = _readChunk<'T> inputDir suffix cx cy cz
        try
            let pixels =
                if image.GetDimensions() = 2u then
                    let plane = image.toArray2D()
                    Array3D.init (Array2D.length1 plane) (Array2D.length2 plane) 1 (fun x y _ -> plane[x, y])
                else
                    image.toArray3D()

            { Pixels = pixels
              Width = Array3D.length1 pixels
              Height = Array3D.length2 pixels
              Depth = Array3D.length3 pixels }
        finally
            image.decRefCount()

    let Get (inputDir: string) (suffix: string) (cx,cy,cz) (dict: ChunkCache<'T>) =
        let key = packKey cx cy cz
        match dict.TryGetValue key with
        | true, ch -> ch
        | _ ->
            let ch = readChunkData inputDir suffix cx cy cz
            dict.[key] <- ch
            ch

    let KeepOnly (required: HashSet<int64>) (dict: ChunkCache<'T>) =
        // evict everything not required
        let toRemove = ResizeArray<int64>()
        for kv in dict do
            if not (required.Contains kv.Key) then
                toRemove.Add kv.Key
        for k in toRemove do dict.Remove k |> ignore

    let Ensure (inputDir: string) (suffix: string) (required: seq<int*int*int>) (dict: ChunkCache<'T>) : HashSet<int64> =
        let set = HashSet<int64>()
        for (cx,cy,cz) in required do
            let key = packKey cx cy cz
            set.Add key |> ignore
            if not (dict.ContainsKey key) then
                dict.[key] <- readChunkData inputDir suffix cx cy cz
        set

// Global voxel fetch from chunk cache. Assumes 0 <= x < W etc.
let getVoxel (inputDir: string) (suffix: string) (winsz:int) (background: 'T) (x:int) (y:int) (z:int) (dict: ChunkCache<'T>) : 'T =
    let cx = x / winsz
    let cy = y / winsz
    let cz = z / winsz
    let lx = x - cx*winsz
    let ly = y - cy*winsz
    let lz = z - cz*winsz
    let ch = ChunkCache.Get inputDir suffix (cx,cy,cz) (dict: ChunkCache<'T>) 
    // handle edge chunks that may be smaller:
    if lx < 0 || ly < 0 || lz < 0 || lx >= ch.Width || ly >= ch.Height || lz >= ch.Depth then
        background // shouldn't happen if your chunk loader sizes match the image, but safe
    else
        ch.Pixels[lx, ly, lz]

// -------------------------
// Trilinear interpolation
// Matches "constant outside" behavior by returning background if the full 2x2x2 footprint isn't inside.
// i.e. requires 0 <= x < W-1, 0 <= y < H-1, 0 <= z < D-1 in continuous index.
// -------------------------
let trilinearSample (inputDir: string) (suffix: string) (winsz:int) (W:int) (H:int) (D:int) (background: 'T) (lerp: 'T->'T->float32->'T) (c:V3) (cache: ChunkCache<'T>) : 'T =
    // Need x0,x1 in [0..W-1] with x1=x0+1 => x0 in [0..W-2]
    if c.x < 0.0 || c.y < 0.0 || c.z < 0.0 || c.x >= float (W-1) || c.y >= float (H-1) || c.z >= float (D-1) then
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

        let c000 = getVoxel inputDir suffix winsz background x0 y0 z0 cache
        let c100 = getVoxel inputDir suffix winsz background x1 y0 z0 cache
        let c010 = getVoxel inputDir suffix winsz background x0 y1 z0 cache
        let c110 = getVoxel inputDir suffix winsz background x1 y1 z0 cache
        let c001 = getVoxel inputDir suffix winsz background x0 y0 z1 cache
        let c101 = getVoxel inputDir suffix winsz background x1 y0 z1 cache
        let c011 = getVoxel inputDir suffix winsz background x0 y1 z1 cache
        let c111 = getVoxel inputDir suffix winsz background x1 y1 z1 cache

        //let inline lerp (a: 'T) (b: 'T) (t: float32) = a + (b - a) * t

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

    let invInDir = inv3 inG.Direction

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
let resampleAffineFromChunks (inputDir: string) (suffix: string) (lerp: 'T->'T->float32->'T) 
    (winsz:int)
    (inG:ImageGeom) (outG:ImageGeom)
    (affOutToIn: Affine)   // output -> input
    (background: 'T) : seq<int * Image<'T>> =

    let cache = ChunkCache.create<'T> ()

    let invInDir = inv3 inG.Direction

    // Precompute output physical step vectors (in physical space)
    let stepI_out = mulMV outG.Direction (v3 outG.Spacing.x 0.0 0.0)
    let stepJ_out = mulMV outG.Direction (v3 0.0 outG.Spacing.y 0.0)
    let stepK_out = mulMV outG.Direction (v3 0.0 0.0 outG.Spacing.z)

    // Affine linear part applied to steps (center cancels for differences)
    let stepI_in_phys = mulMV affOutToIn.A stepI_out
    let stepJ_in_phys = mulMV affOutToIn.A stepJ_out
    let stepK_in_phys = mulMV affOutToIn.A stepK_out

    // Convert physical steps in input space to continuous-index steps
    let stepI_in_cont =
        let q = mulMV invInDir stepI_in_phys
        v3 (q.x / inG.Spacing.x) (q.y / inG.Spacing.y) (q.z / inG.Spacing.z)

    let stepJ_in_cont =
        let q = mulMV invInDir stepJ_in_phys
        v3 (q.x / inG.Spacing.x) (q.y / inG.Spacing.y) (q.z / inG.Spacing.z)

    let stepK_in_cont =
        let q = mulMV invInDir stepK_in_phys
        v3 (q.x / inG.Spacing.x) (q.y / inG.Spacing.y) (q.z / inG.Spacing.z)

    seq {
        for k in 0 .. outG.D - 1 do
            // Decide which chunks are needed for this slice, load them, evict others
            let need = requiredChunksForSliceTrilinear winsz inG outG affOutToIn k
            let requiredSet = ChunkCache.Ensure inputDir suffix need cache
            ChunkCache.KeepOnly requiredSet cache

            // Compute starting continuous input index for (i=0,j=0,k)
            let pOut00k = indexToPhysical outG 0 0 k
            let pIn00k  = affinePoint affOutToIn pOut00k
            let c00k    = physicalToContIndex inG invInDir pIn00k

            let slice = Array2D.zeroCreate<'T> outG.W outG.H

            let mutable rowStart = c00k
            for j in 0 .. outG.H - 1 do
                let mutable c = rowStart
                for i in 0 .. outG.W - 1 do
                    slice[i,j] <- trilinearSample inputDir suffix winsz inG.W inG.H inG.D background lerp c cache
                    c <- add c stepI_in_cont
                rowStart <- add rowStart stepJ_in_cont

            yield (k, Image<'T>.ofArray2D(slice, $"resampleAffine[{k}]", k))
    }

let resampleAffine
    (lerp: 'T -> 'T -> float32 -> 'T)
    (chunkSize: int option)
    (inG: ImageGeom)
    (outG: ImageGeom)
    (affOutToIn: Affine)
    (background: 'T)
    : Stage<Image<'T>, Image<'T>> =

    let name = $"resampleAffine.{typeof<'T>.Name}"
    let chunkSize = chunkSize |> Option.defaultValue 8 |> max 1
    let chunkSizeU = uint chunkSize
    let memoryNeed nPixels =
        let bytes = Image.ImageFacts.memoryBytesForType<'T> nPixels 1u
        bytes * uint64 chunkSizeU

    let elementTransformation _ = uint64 outG.D

    let apply debug (input: AsyncSeq<Image<'T>>) =
        asyncSeq {
            let workspaceRoot =
                Path.Combine(Path.GetTempPath(), $"stackprocessing-{name}-{Guid.NewGuid():N}")
            let chunkDir = Path.Combine(workspaceRoot, "chunks")
            let suffix = ".mha"

            try
                let writer =
                    writeChunks<'T> chunkDir suffix chunkSizeU chunkSizeU chunkSizeU

                let writerPipe = writer.Build()

                do!
                    writerPipe.Apply debug input
                    |> AsyncSeq.iterAsync (fun image ->
                        async {
                            image.decRefCount()
                        })

                for index, image in
                    resampleAffineFromChunks
                        chunkDir
                        suffix
                        lerp
                        chunkSize
                        inG
                        outG
                        affOutToIn
                        background do
                    image.index <- index
                    yield image
            finally
                deleteIfExists workspaceRoot
        }

    let pipe =
        { Name = name
          Apply = apply
          Profile = Streaming }

    Stage.fromPipe name (ProfileTransition.create Streaming Streaming) memoryNeed elementTransformation pipe
    |> Stage.withSliceCardinality (SliceCardinality.reduceTo (uint64 outG.D))
