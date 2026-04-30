namespace Graph

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


type BasicType = 
    | Numeric of NumericType
    | Bool
    | String
    | Unit

type PortType = // all but unit are lists
    | Image of NumericType
    | Scalar of BasicType
    | Tuple of PortType * PortType
    | Source
    | Sink
    | Unit

type Port =
    { Name: string
      Type: PortType }

module PortType =
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
        | Image _, Image Number -> true
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

module FunctionDefinition =
    let matches (searchText: string) (definition: Function) =
        let contains (value: string) =
            value.Contains(searchText, System.StringComparison.OrdinalIgnoreCase)

        System.String.IsNullOrWhiteSpace(searchText)
        || contains definition.DisplayName
        || contains definition.Category
        || contains definition.Description
        || (definition.Aliases |> List.exists contains)
