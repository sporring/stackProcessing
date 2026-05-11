namespace Studio.Graph

type NumericType =
    | Number
    | UInt8
    | Int8
    | UInt16
    | Int16
    | UInt32
    | Int32
    | UInt64
    | Int64
    | Float32
    | Float64
    | Complex

module NumericType = 
  let toString (tp:NumericType) : string = 
      match tp with
      | Number -> "Number"
      | UInt8 -> "UInt8"
      | Int8 -> "Int8"
      | UInt16 -> "UInt16"
      | Int16 -> "Int16"
      | UInt32 -> "UInt32"
      | Int32 -> "Int32"
      | UInt64 -> "UInt64"
      | Int64 -> "Int64"
      | Float32 -> "Float32"
      | Float64 -> "Float64"
      | Complex -> "Complex"

  let tryParse (value: string) =
      match value with
      | "Number" -> Some Number
      | "UInt8" -> Some UInt8
      | "Int8" -> Some Int8
      | "UInt16" -> Some UInt16
      | "Int16" -> Some Int16
      | "UInt32" -> Some UInt32
      | "Int32" -> Some Int32
      | "UInt64" -> Some UInt64
      | "Int64" -> Some Int64
      | "Float32" -> Some Float32
      | "Float64" -> Some Float64
      | "Complex" -> Some Complex
      | _ -> None

module ImageFileFormat =
    type Format =
        { Label: string
          Suffix: string
          SupportedTypes: NumericType list }

    let private commonScalarTypes =
        [ UInt8
          Int8
          UInt16
          Int16
          UInt32
          Int32
          UInt64
          Int64
          Float32
          Float64 ]

    let private tiffReadScalarTypes =
        [ UInt8
          Int8
          UInt16
          Int16
          Float32
          Float64 ]

    let private tiffWriteScalarTypes =
        [ UInt8
          Int8
          UInt16
          Int16
          Float32 ]

    let formats =
        [ { Label = "TIFF (.tiff)"; Suffix = ".tiff"; SupportedTypes = tiffWriteScalarTypes }
          { Label = "TIFF (.tif)"; Suffix = ".tif"; SupportedTypes = tiffWriteScalarTypes }
          { Label = "MetaImage (.mha)"; Suffix = ".mha"; SupportedTypes = commonScalarTypes }
          { Label = "MetaImage header (.mhd)"; Suffix = ".mhd"; SupportedTypes = commonScalarTypes }
          { Label = "NRRD (.nrrd)"; Suffix = ".nrrd"; SupportedTypes = commonScalarTypes }
          { Label = "NIfTI (.nii)"; Suffix = ".nii"; SupportedTypes = commonScalarTypes }
          { Label = "NIfTI compressed (.nii.gz)"; Suffix = ".nii.gz"; SupportedTypes = commonScalarTypes }
          { Label = "PNG (.png)"; Suffix = ".png"; SupportedTypes = [ UInt8; UInt16 ] }
          { Label = "JPEG (.jpg)"; Suffix = ".jpg"; SupportedTypes = [ UInt8 ] }
          { Label = "JPEG (.jpeg)"; Suffix = ".jpeg"; SupportedTypes = [ UInt8 ] }
          { Label = "BMP (.bmp)"; Suffix = ".bmp"; SupportedTypes = [ UInt8 ] } ]

    let readFormats =
        [ { Label = "TIFF (.tif or .tiff)"; Suffix = ".tiff"; SupportedTypes = tiffReadScalarTypes }
          { Label = "JPEG (.jpg or .jpeg)"; Suffix = ".jpg"; SupportedTypes = [ UInt8 ] } ]
        @ (formats
           |> List.filter (fun format ->
               format.Suffix <> ".tiff"
               && format.Suffix <> ".tif"
               && format.Suffix <> ".jpg"
               && format.Suffix <> ".jpeg"))

    let suffixes =
        formats |> List.map _.Suffix

    let private normalizeSuffix (suffix: string) =
        let trimmed = suffix.Trim()

        if trimmed.StartsWith(".", System.StringComparison.Ordinal) then
            trimmed.ToLowerInvariant()
        else
            "." + trimmed.ToLowerInvariant()

    let tryFind suffix =
        let suffix = normalizeSuffix suffix

        formats
        |> List.tryFind (fun format -> format.Suffix = suffix)

    let supportedTypes suffix =
        tryFind suffix
        |> Option.map _.SupportedTypes
        |> Option.defaultValue commonScalarTypes

    let readSupportedTypes suffix =
        let suffix = normalizeSuffix suffix

        readFormats
        |> List.tryFind (fun format -> format.Suffix = suffix)
        |> Option.map _.SupportedTypes
        |> Option.defaultWith (fun () -> supportedTypes suffix)

    let supports suffix numericType =
        supportedTypes suffix
        |> List.contains numericType

    let readSupports suffix numericType =
        readSupportedTypes suffix
        |> List.contains numericType

type BasicType = 
    | Numeric of NumericType
    | Bool
    | String
    | Map
    | Unit

module BasicType = 
  let toString (tp:BasicType) : string = 
      match tp with
      | Numeric Number -> "Number"
      | Numeric UInt8 -> "UInt8"
      | Numeric Int8 -> "Int8"
      | Numeric UInt16 -> "UInt16"
      | Numeric Int16 -> "Int16"
      | Numeric UInt32 -> "UInt32"
      | Numeric Int32 -> "Int32"
      | Numeric UInt64 -> "UInt64"
      | Numeric Int64 -> "Int64"
      | Numeric Float32 -> "Float32"
      | Numeric Float64 -> "Float64"
      | Numeric Complex -> "Complex"
      | Bool -> "Bool"
      | String -> "String"
      | Map -> "Map"
      | Unit -> "Unit"

  let tryParse (value: string) =
      match value with
      | "Number" -> Some(Numeric Number)
      | "UInt8" -> Some(Numeric UInt8)
      | "Int8" -> Some(Numeric Int8)
      | "UInt16" -> Some(Numeric UInt16)
      | "Int16" -> Some(Numeric Int16)
      | "UInt32" -> Some(Numeric UInt32)
      | "Int32" -> Some(Numeric Int32)
      | "UInt64" -> Some(Numeric UInt64)
      | "Int64" -> Some(Numeric Int64)
      | "Float32" -> Some(Numeric Float32)
      | "Float64" -> Some(Numeric Float64)
      | "Complex" -> Some(Numeric Complex)
      | "Bool" -> Some Bool
      | "String" -> Some String
      | "Map" -> Some Map
      | "Unit" -> Some Unit
      | _ -> None

type PortType = // all but unit are lists
    | Image of NumericType
    | Scalar of BasicType
    | Tuple of PortType * PortType
    | Custom of string
    | Any
    | Unit

type Port =
    { Name: string
      Type: PortType }

module PortType =
    let isRecordType portType =
        match portType with
        | Custom "Record"
        | Custom "ImageStats"
        | Custom "ObjectSizeStats"
        | Custom "StackInfo"
        | Custom "ChunkInfo" -> true
        | _ -> false

    let numericToImage (tp:NumericType) : PortType = 
        match tp with
        | Number -> PortType.Image(NumericType.Float64)
        | UInt8 -> PortType.Image(NumericType.UInt8)
        | Int8 -> PortType.Image(NumericType.Int8)
        | UInt16 -> PortType.Image(NumericType.UInt16)
        | Int16 -> PortType.Image(NumericType.Int16)
        | UInt32 -> PortType.Image(NumericType.UInt32)
        | Int32 -> PortType.Image(NumericType.Int32)
        | UInt64 -> PortType.Image(NumericType.UInt64)
        | Int64 -> PortType.Image(NumericType.Int64)
        | Float32 -> PortType.Image(NumericType.Float32)
        | Float64 -> PortType.Image(NumericType.Float64)
        | Complex -> PortType.Image(NumericType.Complex)

    let canConnect (outputType: PortType) (inputType: PortType) =
        match outputType, inputType with
        | Any, _
        | _, Any -> true
        | outputType, Custom "Record" when isRecordType outputType -> true
        | Custom "ColorImage", Image Number -> true
        | Image _, Image Number -> true
        | Image Number, Image _ -> true
        | _ -> outputType = inputType

type Parameter =
    { Key: string
      Label: string
      DefaultValue: string
      Type: BasicType }

type Function =
    { Id: string
      DisplayName: string
      Category: string
      Summary: string
      Description: string
      Aliases: string list
      Inputs: Port list
      Outputs: Port list
      Parameters: Parameter list }

type Position =
    { X: float
      Y: float }

type Node =
    { Id: string
      Function: Function
      ParameterValues: Map<string, string>
      Position: Position option }

type Edge =
    { FromNode: Node
      FromPort: int // index into GraphNode.FunctionDefinition.Inputs
      ToNode: Node
      ToPort: int  // index into GraphNode.FunctionDefinition.Outputs
      }

type PipelineGraph =
    { Version: int
      Nodes: Node list
      Edges: Edge list }

[<CLIMutable>]
type SavedParameter =
    { Key: string
      Value: string
      UseInput: bool }

[<CLIMutable>]
type SavedNode =
    { Id: string
      FunctionId: string
      X: float
      Y: float
      Parameters: SavedParameter array }

[<CLIMutable>]
type SavedEdge =
    { FromNode: string
      FromKind: string
      FromPort: int
      ToNode: string
      ToKind: string
      ToPort: int }

[<CLIMutable>]
type SavedGraph =
    { Version: int
      Nodes: SavedNode array
      Edges: SavedEdge array }

module FunctionDefinition =
    let matches (searchText: string) (definition: Function) =
        let contains (value: string) =
            value.Contains(searchText, System.StringComparison.OrdinalIgnoreCase)

        System.String.IsNullOrWhiteSpace(searchText)
        || contains definition.DisplayName
        || contains definition.Category
        || contains definition.Summary
        || contains definition.Description
        || (definition.Aliases |> List.exists contains)
