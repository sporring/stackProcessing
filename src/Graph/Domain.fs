namespace Graph

type PixelType =
    | UInt8
    | Float32
    | Float64

type Dimensionality =
    | D2
    | D3

type ScalarType =
    | Int
    | Float
    | Bool
    | String

type PortType =
    | Image of PixelType * Dimensionality
    | Scalar of ScalarType
    | Tuple of PortType list
    | Unit

type PortDirection =
    | Input
    | Output

type PortDefinition =
    { Name: string
      Type: PortType
      Direction: PortDirection }

type ParameterDefinition =
    { Key: string
      Label: string
      DefaultValue: string
      Type: ScalarType }

type Streamability =
    | Source
    | Sink
    | Local
    | Global
    | Structural

type FunctionDefinition =
    { Id: string
      DisplayName: string
      Category: string
      Description: string
      Aliases: string list
      Inputs: PortDefinition list
      Outputs: PortDefinition list
      Parameters: ParameterDefinition list
      Streamability: Streamability
      Properties: Map<string, string> }

type GraphPosition =
    { X: float
      Y: float }

type GraphNode =
    { Id: string
      FunctionId: string
      Parameters: Map<string, string>
      Position: GraphPosition option }

type Edge =
    { FromNode: string
      FromPort: int
      ToNode: string
      ToPort: int }

type PipelineGraph =
    { Version: int
      Nodes: GraphNode list
      Edges: Edge list }

module FunctionDefinition =
    let matches (searchText: string) (definition: FunctionDefinition) =
        let contains (value: string) =
            value.Contains(searchText, System.StringComparison.OrdinalIgnoreCase)

        System.String.IsNullOrWhiteSpace(searchText)
        || contains definition.DisplayName
        || contains definition.Category
        || contains definition.Description
        || (definition.Aliases |> List.exists contains)

