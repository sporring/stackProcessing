namespace Graph

module BuiltInCatalog =
    let image3dFloat32 = PortType.Image(PixelType.Float32, Dimensionality.D3)

    let private input name portType =
        { Name = name
          Type = portType
          Direction = PortDirection.Input }

    let private output name portType =
        { Name = name
          Type = portType
          Direction = PortDirection.Output }

    let private parameter key label defaultValue parameterType =
        { Key = key
          Label = label
          DefaultValue = defaultValue
          Type = parameterType }

    let orderedFunctions =
        [ { Id = "Source"
            DisplayName = "source"
            Category = "Sources / Sinks"
            Description = "Begin a streaming StackProcessing pipeline with available memory."
            Aliases = [ "availableMemory"; "start"; "input" ]
            Inputs = []
            Outputs = [ output "OUT" image3dFloat32 ]
            Parameters = [ parameter "availableMemory" "Available memory" "availableMemory" ScalarType.String ]
            Streamability = Streamability.Source
            Properties = Map.empty }

          { Id = "Sink"
            DisplayName = "sink"
            Category = "Sources / Sinks"
            Description = "Run the built pipeline."
            Aliases = [ "execute"; "run"; "terminal" ]
            Inputs = [ input "IN" image3dFloat32 ]
            Outputs = []
            Parameters = []
            Streamability = Streamability.Sink
            Properties = Map.empty }

          { Id = "Read"
            DisplayName = "read"
            Category = "IO"
            Description = "Read a stack from chunked image files."
            Aliases = [ "input"; "load"; "tiff"; "file" ]
            Inputs = [ input "IN" PortType.Unit ]
            Outputs = [ output "OUT" image3dFloat32 ]
            Parameters =
                [ parameter "pixelType" "Pixel type" "float" ScalarType.String
                  parameter "input" "Input" "input" ScalarType.String
                  parameter "suffix" "Suffix" ".tiff" ScalarType.String ]
            Streamability = Streamability.Source
            Properties = Map.empty }

          { Id = "Write"
            DisplayName = "write"
            Category = "IO"
            Description = "Write a processed stack to image files."
            Aliases = [ "output"; "save"; "tiff"; "file" ]
            Inputs = [ input "IN" image3dFloat32 ]
            Outputs = [ output "OUT" image3dFloat32 ]
            Parameters =
                [ parameter "output" "Output" "output" ScalarType.String
                  parameter "suffix" "Suffix" ".tiff" ScalarType.String ]
            Streamability = Streamability.Sink
            Properties = Map.empty }

          { Id = "DiscreteGaussian"
            DisplayName = "discreteGaussian"
            Category = "Filters"
            Description = "Apply a Gaussian smoothing filter."
            Aliases = [ "gaussian"; "smooth"; "blur"; "filter" ]
            Inputs = [ input "IN" image3dFloat32 ]
            Outputs = [ output "OUT" image3dFloat32 ]
            Parameters =
                [ parameter "sigma" "Sigma" "1.0" ScalarType.Float
                  parameter "outputRegionMode" "Output region" "None" ScalarType.String
                  parameter "boundaryCondition" "Boundary" "None" ScalarType.String
                  parameter "windowSize" "Window size" "15" ScalarType.Int ]
            Streamability = Streamability.Local
            Properties = Map.ofList [ "halo", "3" ] }

          { Id = "Cast"
            DisplayName = "cast"
            Category = "Type conversions"
            Description = "Convert stream element type."
            Aliases = [ "convert"; "uint8"; "float"; "type" ]
            Inputs = [ input "IN" image3dFloat32 ]
            Outputs = [ output "OUT" (PortType.Image(PixelType.UInt8, Dimensionality.D3)) ]
            Parameters =
                [ parameter "sourceType" "Source type" "float" ScalarType.String
                  parameter "targetType" "Target type" "uint8" ScalarType.String ]
            Streamability = Streamability.Local
            Properties = Map.empty } ]

    let functions =
        orderedFunctions
        |> List.map (fun definition -> definition.Id, definition)
        |> Map.ofList

    let tryFind id =
        functions
        |> Map.tryFind id

    let find id =
        match tryFind id with
        | Some definition -> definition
        | None -> invalidArg (nameof id) $"Unknown graph function id: {id}"
