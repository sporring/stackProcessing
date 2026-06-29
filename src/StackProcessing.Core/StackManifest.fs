module StackManifest

open System
open System.IO
open System.Text.Json
open StackPoints
open StackRegistration
open TinyLinAlg

[<CLIMutable>]
type ImageSetCoordinateSystem =
    { Name: string
      Units: string }

[<CLIMutable>]
type ImageSetTransform =
    { Type: string
      Matrix: float list list }

[<CLIMutable>]
type ImageSetGrid =
    { TileSize: uint64 list }

[<CLIMutable>]
type ImageSetItem =
    { Id: string
      Kind: string
      Path: string
      Suffix: string
      Size: uint64 list
      Spacing: float list
      GridIndex: int list option
      TransformToWorld: ImageSetTransform
      Sources: string list }

[<CLIMutable>]
type ImageSetManifest =
    { Version: int
      CoordinateSystem: ImageSetCoordinateSystem
      Grid: ImageSetGrid option
      Items: ImageSetItem list }

module ImageSetItemKind =
    let scalarImage = "ScalarImage"
    let vectorImage = "VectorImage"
    let pointSet = "PointSet"
    let triangleMesh = "TriangleMesh"
    let matrix = "Matrix"

let private jsonOptions =
    JsonSerializerOptions(WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

let private validateVector name expected values =
    if List.length values <> expected then
        invalidArg name $"{name} must contain exactly {expected} values."

let private validateMatrixRows (matrix: float list list) =
    validateVector "matrix" 4 matrix
    matrix |> List.iter (validateVector "matrix row" 4)

let private validateAffineMatrixRows (matrix: float list list) =
    validateMatrixRows matrix

    let lastRow = matrix[3]
    if abs lastRow[0] > 1.0e-12
       || abs lastRow[1] > 1.0e-12
       || abs lastRow[2] > 1.0e-12
       || abs (lastRow[3] - 1.0) > 1.0e-12 then
        invalidArg "matrix" "Image-set transforms must have affine homogeneous last row [0, 0, 0, 1]."

let private vectorizedToRows (matrix: VectorizedMatrix) =
    if matrix.Rows <> 4u || matrix.Columns <> 4u then
        invalidArg "matrix" "Image-set transforms must be 4x4 homogeneous matrices."

    let values = unvectorizeMatrix matrix
    [ for row in 0 .. 3 ->
        [ for column in 0 .. 3 -> values[row, column] ] ]

let private rowsToVectorized (matrix: float list list) =
    validateAffineMatrixRows matrix
    let values = Array2D.zeroCreate<float> 4 4
    matrix
    |> List.iteri (fun row columns ->
        columns |> List.iteri (fun column value -> values[row, column] <- value))
    vectorizeMatrix values

let identityTransform =
    { Type = "affine4x4"
      Matrix =
          [ [ 1.0; 0.0; 0.0; 0.0 ]
            [ 0.0; 1.0; 0.0; 0.0 ]
            [ 0.0; 0.0; 1.0; 0.0 ]
            [ 0.0; 0.0; 0.0; 1.0 ] ] }

let transformFromMatrix matrix =
    { Type = "affine4x4"
      Matrix = vectorizedToRows matrix }

let transformToMatrix transform =
    if not (String.Equals(transform.Type, "affine4x4", StringComparison.OrdinalIgnoreCase)) then
        invalidArg "transform" $"Unsupported image-set transform type '{transform.Type}'."

    rowsToVectorized transform.Matrix

let transformFromAffine affine =
    affine |> toHomogeneousMatrix |> transformFromMatrix

let transformToAffine transform =
    transform |> transformToMatrix |> ofHomogeneousMatrix

let createManifest name units =
    { Version = 1
      CoordinateSystem = { Name = name; Units = units }
      Grid = None
      Items = [] }

let identityManifest name units =
    createManifest name units

let imageSetGrid tileSize =
    validateVector "tileSize" 3 tileSize

    if tileSize |> List.exists ((=) 0UL) then
        invalidArg "tileSize" "Image-set grid tile size values must be positive."

    { TileSize = tileSize }

let withGrid grid manifest =
    { manifest with Grid = Some grid }

let validateAffineTransform (transform: ImageSetTransform) =
    transformToMatrix transform |> ignore
    transform

let private multiplyRows (left: float list list) (right: float list list) =
    validateMatrixRows left
    validateMatrixRows right

    [ for row in 0 .. 3 ->
        [ for column in 0 .. 3 ->
            [ 0 .. 3 ]
            |> List.sumBy (fun k -> left[row][k] * right[k][column]) ] ]

let composeTransforms (left: ImageSetTransform) (right: ImageSetTransform) =
    validateAffineTransform left |> ignore
    validateAffineTransform right |> ignore

    { Type = "affine4x4"
      Matrix = multiplyRows left.Matrix right.Matrix }

let gridIndexTransform (grid: ImageSetGrid) (gridIndex: int list) =
    validateVector "gridIndex" 3 gridIndex

    let offset axis =
        float gridIndex[axis] * float grid.TileSize[axis]

    { Type = "affine4x4"
      Matrix =
        [ [ 1.0; 0.0; 0.0; offset 0 ]
          [ 0.0; 1.0; 0.0; offset 1 ]
          [ 0.0; 0.0; 1.0; offset 2 ]
          [ 0.0; 0.0; 0.0; 1.0 ] ] }

let spatialDataItem kind id path suffix size spacing transform sources =
    validateVector "size" 3 size
    validateVector "spacing" 3 spacing
    validateAffineTransform transform |> ignore

    { Id = id
      Kind = kind
      Path = path
      Suffix = suffix
      Size = size
      Spacing = spacing
      GridIndex = None
      TransformToWorld = transform
      Sources = sources }

let withGridIndex gridIndex item =
    validateVector "gridIndex" 3 gridIndex
    { item with GridIndex = Some gridIndex }

let scalarImageItem id path suffix size spacing transform sources =
    spatialDataItem ImageSetItemKind.scalarImage id path suffix size spacing transform sources

let gridImageItem id path suffix size spacing gridIndex transform sources =
    scalarImageItem id path suffix size spacing transform sources
    |> withGridIndex gridIndex

let vectorImageItem id path suffix size spacing transform sources =
    spatialDataItem ImageSetItemKind.vectorImage id path suffix size spacing transform sources

let pointSetItem id path suffix size spacing transform sources =
    spatialDataItem ImageSetItemKind.pointSet id path suffix size spacing transform sources

let triangleMeshItem id path suffix size spacing transform sources =
    spatialDataItem ImageSetItemKind.triangleMesh id path suffix size spacing transform sources

let matrixItem id path suffix transform sources =
    spatialDataItem ImageSetItemKind.matrix id path suffix [ 0UL; 0UL; 0UL ] [ 1.0; 1.0; 1.0 ] transform sources

let addItem item manifest =
    if manifest.Items |> List.exists (fun existing -> existing.Id = item.Id) then
        invalidArg "item" $"The image-set manifest already contains an item with id '{item.Id}'."

    { manifest with Items = manifest.Items @ [ item ] }

let replaceItemTransform itemId transform manifest =
    let mutable found = false
    let items =
        manifest.Items
        |> List.map (fun item ->
            if item.Id = itemId then
                found <- true
                { item with TransformToWorld = transform }
            else
                item)

    if not found then
        invalidArg "itemId" $"The image-set manifest does not contain an item with id '{itemId}'."

    { manifest with Items = items }

let imageMember id path suffix size spacing transform =
    scalarImageItem id path suffix size spacing transform []

let addImage memberInfo manifest =
    addItem memberInfo manifest

let replaceImageTransform imageId transform manifest =
    replaceItemTransform imageId transform manifest

let updateMovingItemTransformFromRegistration fixedItemId movingItemId fixedFromMoving manifest =
    let fixedItem =
        manifest.Items
        |> List.tryFind (fun item -> item.Id = fixedItemId)
        |> Option.defaultWith (fun () -> invalidArg "fixedItemId" $"The image-set manifest does not contain an item with id '{fixedItemId}'.")

    let updatedMovingTransform =
        composeTransforms fixedItem.TransformToWorld fixedFromMoving

    manifest
    |> replaceItemTransform movingItemId updatedMovingTransform

let writeManifest (path: string) manifest =
    let directory = Path.GetDirectoryName(path)
    if not (String.IsNullOrWhiteSpace directory) then
        Directory.CreateDirectory(directory) |> ignore

    let json = JsonSerializer.Serialize(manifest, jsonOptions)
    File.WriteAllText(path, json)

let readManifest (path: string) =
    let json = File.ReadAllText(path)
    JsonSerializer.Deserialize<ImageSetManifest>(json, jsonOptions)
