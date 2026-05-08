module StackStitching

open System
open System.IO
open SlimPipeline
open StackCore
open StackManifest
open Image

type StitchPlanItem =
    { Id: string
      Path: string
      Suffix: string
      Size: uint64 list
      Spacing: float list
      ItemToWorld: ImageSetTransform
      WorldToItem: ImageSetTransform }

type StitchPlan =
    { Origin: float list
      Spacing: float list
      Size: uint64 list
      Items: StitchPlanItem list }

let private validateVector name values =
    if List.length values <> 3 then
        invalidArg name $"{name} must contain exactly 3 values."

let private matrixRows transform =
    validateAffineTransform transform |> ignore
    transform.Matrix

let private transformPoint (transform: ImageSetTransform) (x: float) (y: float) (z: float) =
    let m = matrixRows transform
    let tx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
    let ty = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
    let tz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
    tx, ty, tz

let private effectiveItemTransform (manifest: ImageSetManifest) (item: ImageSetItem) =
    match manifest.Grid, item.GridIndex with
    | Some grid, Some gridIndex ->
        composeTransforms item.TransformToWorld (gridIndexTransform grid gridIndex)
    | _ ->
        item.TransformToWorld

let private itemCorners (transform: ImageSetTransform) (item: ImageSetItem) =
    validateVector "size" item.Size
    let maxX = float item.Size[0] - 1.0
    let maxY = float item.Size[1] - 1.0
    let maxZ = float item.Size[2] - 1.0

    [ for x in [ 0.0; max 0.0 maxX ] do
        for y in [ 0.0; max 0.0 maxY ] do
            for z in [ 0.0; max 0.0 maxZ ] ->
                transformPoint transform x y z ]

let private minMax values =
    values |> List.min, values |> List.max

let createStitchPlan (manifest: ImageSetManifest) (itemIds: string list) =
    let selected =
        itemIds
        |> List.map (fun itemId ->
            manifest.Items
            |> List.tryFind (fun item -> item.Id = itemId)
            |> Option.defaultWith (fun () -> invalidArg "itemIds" $"The image-set manifest does not contain an item with id '{itemId}'."))

    if selected.IsEmpty then
        invalidArg "itemIds" "Stitching needs at least one image item."

    let firstSize = selected.Head.Size

    for item in selected do
        if item.Kind <> ImageSetItemKind.scalarImage && item.Kind <> ImageSetItemKind.vectorImage then
            invalidArg "itemIds" $"Stitching supports image manifest items only, but '{item.Id}' has kind '{item.Kind}'."
        validateVector "size" item.Size
        validateVector "spacing" item.Spacing
        if item.Size <> firstSize then
            invalidArg "itemIds" $"Stitching expects all selected image items to have equal size. Item '{item.Id}' has size {item.Size}, but the first item has size {firstSize}."
        if item.Size |> List.exists ((=) 0UL) then
            invalidArg "item.Size" $"Stitching item '{item.Id}' must have positive size."
        validateAffineTransform item.TransformToWorld |> ignore

    match manifest.Grid with
    | Some grid ->
        if grid.TileSize <> firstSize then
            invalidArg "manifest.Grid" $"Image-set grid tile size must match the stitched image size. Grid tile size was {grid.TileSize}, but image size was {firstSize}."

        for item in selected do
            if item.GridIndex.IsNone then
                invalidArg "itemIds" $"Image-set grid stitching expects item '{item.Id}' to have a grid index."
    | None -> ()

    let spacing = [ 1.0; 1.0; 1.0 ]
    let effectiveItems =
        selected
        |> List.map (fun item -> item, effectiveItemTransform manifest item)

    let corners = effectiveItems |> List.collect (fun (item, transform) -> itemCorners transform item)
    let xs = corners |> List.map (fun (x, _, _) -> x)
    let ys = corners |> List.map (fun (_, y, _) -> y)
    let zs = corners |> List.map (fun (_, _, z) -> z)
    let minX, maxX = minMax xs
    let minY, maxY = minMax ys
    let minZ, maxZ = minMax zs

    let size (minValue: float) (maxValue: float) (spacing: float) =
        uint64 (Math.Floor((maxValue - minValue) / spacing) + 1.0)

    { Origin = [ minX; minY; minZ ]
      Spacing = spacing
      Size =
        [ size minX maxX spacing[0]
          size minY maxY spacing[1]
          size minZ maxZ spacing[2] ]
      Items =
        effectiveItems
        |> List.map (fun (item, itemToWorld) ->
            { Id = item.Id
              Path = item.Path
              Suffix = item.Suffix
              Size = item.Size
              Spacing = spacing
              ItemToWorld = itemToWorld
              WorldToItem = itemToWorld |> transformToAffine |> StackRegistration.inverseAffine |> transformFromAffine }) }

let private outputPath (manifestPath: string) (path: string) =
    if Path.IsPathRooted path then
        path
    else
        Path.Combine(Path.GetDirectoryName(manifestPath), path)

let private stackFiles (path: string) (suffix: string) =
    if not (Directory.Exists path) then
        invalidArg "path" $"Stitch input directory does not exist: {path}"

    Directory.GetFiles(path, $"*{suffix}")
    |> Array.sort

let private convertFromDouble<'T> (value: float) : 'T =
    let target = typeof<'T>
    let rounded () = Math.Round(value: float)

    if target = typeof<float> then box value :?> 'T
    elif target = typeof<float32> then box (float32 value) :?> 'T
    elif target = typeof<uint8> then box (uint8 (max 0.0 (min 255.0 (rounded ())))) :?> 'T
    elif target = typeof<int8> then box (int8 (max -128.0 (min 127.0 (rounded ())))) :?> 'T
    elif target = typeof<uint16> then box (uint16 (max 0.0 (min 65535.0 (rounded ())))) :?> 'T
    elif target = typeof<int16> then box (int16 (max -32768.0 (min 32767.0 (rounded ())))) :?> 'T
    elif target = typeof<uint32> then box (uint32 (max 0.0 (min (float UInt32.MaxValue) (rounded ())))) :?> 'T
    elif target = typeof<int32> then box (int32 (max (float Int32.MinValue) (min (float Int32.MaxValue) (rounded ())))) :?> 'T
    elif target = typeof<uint64> then box (uint64 (max 0.0 (rounded ()))) :?> 'T
    elif target = typeof<int64> then box (int64 (rounded ())) :?> 'T
    else
        Convert.ChangeType(box value, target, Globalization.CultureInfo.InvariantCulture) :?> 'T

let private borderWeight (blendBorderVoxels: float) (size: uint64 list) (x: float) (y: float) (z: float) =
    if blendBorderVoxels <= 0.0 then
        1.0
    else
        let sx = float size[0]
        let sy = float size[1]
        let sz = float size[2]
        let distance =
            [ x + 1.0; sx - x; y + 1.0; sy - y; z + 1.0; sz - z ]
            |> List.min

        max 0.0 (min 1.0 (distance / blendBorderVoxels))

let stitchManifestImages<'T when 'T: equality>
    (manifestPath: string)
    (itemIds: string list)
    (blendBorderVoxels: float)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let manifest = readManifest manifestPath
    let plan = createStitchPlan manifest itemIds
    let width = uint plan.Size[0]
    let height = uint plan.Size[1]
    let depth = uint plan.Size[2]

    let resolvedItems =
        plan.Items
        |> List.map (fun item ->
            let path = outputPath manifestPath item.Path
            item, stackFiles path item.Suffix)

    let mapper (outputZ: int) =
        let values = Array2D.zeroCreate<'T> (int width) (int height)
        let sliceCache = Collections.Generic.Dictionary<string * int, Image<'T>>()

        let getSlice (itemId: string) (files: string[]) (z: int) : Image<'T> option =
            let key = (itemId, z)
            let mutable cached = Unchecked.defaultof<Image<'T>>
            if sliceCache.TryGetValue(key, &cached) then
                Some cached
            else
                if z < 0 || z >= files.Length then
                    None
                else
                    let slice = Image<'T>.ofFile(files[z])
                    sliceCache[key] <- slice
                    Some slice

        try
            for y in 0 .. int height - 1 do
                for x in 0 .. int width - 1 do
                    let worldX = plan.Origin[0] + float x * plan.Spacing[0]
                    let worldY = plan.Origin[1] + float y * plan.Spacing[1]
                    let worldZ = plan.Origin[2] + float outputZ * plan.Spacing[2]
                    let mutable valueSum = 0.0
                    let mutable weightSum = 0.0

                    for item, files in resolvedItems do
                        let sx, sy, sz = transformPoint item.WorldToItem worldX worldY worldZ
                        let ix = int (Math.Round sx)
                        let iy = int (Math.Round sy)
                        let iz = int (Math.Round sz)

                        if ix >= 0
                           && iy >= 0
                           && iz >= 0
                           && uint64 ix < item.Size[0]
                           && uint64 iy < item.Size[1]
                           && uint64 iz < item.Size[2] then
                            match getSlice item.Id files iz with
                            | Some sourceSlice ->
                                let weight = borderWeight blendBorderVoxels item.Size sx sy sz
                                if weight > 0.0 then
                                    valueSum <- valueSum + weight * Convert.ToDouble(box sourceSlice[ix, iy])
                                    weightSum <- weightSum + weight
                            | None -> ()

                    if weightSum > 0.0 then
                        values[x, y] <- convertFromDouble<'T> (valueSum / weightSum)

            Image<'T>.ofArray2D(values, $"stitch[{outputZ}]", outputZ)
        finally
            for slice in sliceCache.Values do
                slice.decRefCount()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed = fun _ -> Image<'T>.memoryEstimate width height
    let stage = Stage.init "stitchManifestImages" depth mapper transition memoryNeed id |> Some
    Plan.create stage pl.memAvail (Image<'T>.memoryEstimate width height) (uint64 width * uint64 height) (uint64 depth) pl.debug
