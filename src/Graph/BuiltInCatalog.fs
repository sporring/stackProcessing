namespace Graph

module BuiltInCatalog =
  let imageAny = PortType.Image NumericType.Number
  let imageUInt8 = PortType.Image NumericType.UInt8
  let imageInt8 = PortType.Image NumericType.Int8
  let imageUInt16 = PortType.Image NumericType.UInt16
  let imageInt16 = PortType.Image NumericType.Int16
  let imageUInt32 = PortType.Image NumericType.UInt32
  let imageInt32 = PortType.Image NumericType.Int32
  let imageUInt64 = PortType.Image NumericType.UInt64
  let imageInt64 = PortType.Image NumericType.Int64
  let imageFloat32 = PortType.Image NumericType.Float32
  let imageFloat64 = PortType.Image NumericType.Float64
  let imageComplex = PortType.Image NumericType.Complex

  let private makePort name portType =
      { Name = name
        Type = portType }

  let private makeParameter key label defaultValue parameterType =
      { Key = key
        Label = label
        DefaultValue = defaultValue
        Type = parameterType }

  let makeRead (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "Read" + str
      DisplayName = "read" + str
      Category = "IO"
      Description = "Read a stack from chunked image files."
      Aliases = [ "input"; "load"; "tiff"; "file" ]
      Inputs = [ makePort "FromSource" Source ]
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ makeParameter "NumericType" "Pixel type" "float" BasicType.String
            makeParameter "input" "Input" "input" BasicType.String
            makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

  let makeCast (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "Cast" + str
      DisplayName = "cast" + str
      Category = "Type conversions"
      Description = "Convert stream element type."
      Aliases = [ "convert"; "uint8"; "float"; "type" ]
      Inputs = [ makePort "IN" imageAny ]
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ makeParameter "sourceType" "Source type" "float" BasicType.String
            makeParameter "targetType" "Target type" "uint8" BasicType.String ] }

  let orderedFunctions =
      [ { Id = "Source"
          DisplayName = "source"
          Category = "Sources / Sinks"
          Description = "Begin a streaming StackProcessing pipeline with available memory."
          Aliases = [ "availableMemory"; "start"; "input" ]
          Inputs = []
          Outputs = [ makePort "Source" Source ]
          Parameters = [ makeParameter "availableMemory" "Available memory" "availableMemory" BasicType.String ] }

        { Id = "Sink"
          DisplayName = "sink"
          Category = "Sources / Sinks"
          Description = "Run the built pipeline."
          Aliases = [ "execute"; "run"; "terminal" ]
          Inputs = [ makePort "Sink" Sink ]
          Outputs = []
          Parameters = [] }

        makeRead UInt8
        makeRead Int8
        makeRead UInt16
        makeRead Int16
        makeRead UInt32
        makeRead Int32
        makeRead UInt64
        makeRead Int64
        makeRead Float32
        makeRead Float64

        { Id = "Write"
          DisplayName = "write"
          Category = "IO"
          Description = "Write a processed stack to image files."
          Aliases = [ "output"; "save"; "tiff"; "file" ]
          Inputs = [ makePort "IN" imageAny ]
          Outputs = [ makePort "ToSink" Sink ]
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

        { Id = "DiscreteGaussian"
          DisplayName = "discreteGaussian"
          Category = "Filters"
          Description = "Apply a Gaussian smoothing filter."
          Aliases = [ "gaussian"; "smooth"; "blur"; "filter" ]
          Inputs = [ makePort "IN" imageFloat64 ]
          Outputs = [ makePort "OUT" imageFloat64 ]
          Parameters =
              [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "outputRegionMode" "Output region" "None" BasicType.String
                makeParameter "boundaryCondition" "Boundary" "None" BasicType.String
                makeParameter "windowSize" "Window size" "15" (BasicType.Numeric Int32) ] }

        makeCast UInt8
        makeCast Int8
        makeCast UInt16
        makeCast Int16
        makeCast UInt32
        makeCast Int32
        makeCast UInt64
        makeCast Int64
        makeCast Float32
        makeCast Float64 ]

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
