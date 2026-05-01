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

  let private fsharpTypeName tp =
      match tp with
      | UInt8 -> "uint8"
      | Int8 -> "int8"
      | UInt16 -> "uint16"
      | Int16 -> "int16"
      | UInt32 -> "uint32"
      | Int32 -> "int32"
      | UInt64 -> "uint64"
      | Int64 -> "int64"
      | Float32 -> "float32"
      | Float64 -> "float"
      | Complex -> "System.Numerics.Complex"
      | Number -> "float"

  let private numericDefaultValue tp =
      match tp with
      | UInt8 -> "1uy"
      | Int8 -> "1y"
      | UInt16 -> "1us"
      | Int16 -> "1s"
      | UInt32 -> "1u"
      | Int32 -> "1"
      | UInt64 -> "1UL"
      | Int64 -> "1L"
      | Float32 -> "1.0f"
      | Float64
      | Number -> "1.0"
      | Complex -> "System.Numerics.Complex.One"

  let private scalarDefaultValue tp =
      match tp with
      | BasicType.Bool -> "true"
      | BasicType.String -> "value"
      | BasicType.Unit -> "()"
      | BasicType.Numeric numericType -> numericDefaultValue numericType

  let makeScalar (tp: BasicType) =
    let str = BasicType.toString tp
    { Id = "Scalar" + str
      DisplayName = str
      Category = "Scalars"
      Description = "Bind a scalar value for graph parameters."
      Aliases = [ "value"; "parameter"; "constant"; "let" ]
      Inputs = []
      Outputs = [ makePort "Value" (Scalar tp) ]
      Parameters =
          [ makeParameter "value" "Value" (scalarDefaultValue tp) tp ] }

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
          [ makeParameter "input" "Input" "input" BasicType.String
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
            makeParameter "targetType" "Target type" (fsharpTypeName tp) BasicType.String ] }

  let makeZero (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "Zero" + str
      DisplayName = "zero" + str
      Category = "Sources"
      Description = "Create a zero-valued synthetic stack."
      Aliases = [ "empty"; "synthetic"; "source" ]
      Inputs = [ makePort "FromSource" Source ]
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ makeParameter "width" "Width" "64u" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64u" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1u" (BasicType.Numeric UInt32) ] }

  let makeReadRandom (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "ReadRandom" + str
      DisplayName = "readRandom" + str
      Category = "IO"
      Description = "Read a randomized subset of stack files."
      Aliases = [ "random"; "input"; "tiff"; "file" ]
      Inputs = [ makePort "FromSource" Source ]
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ makeParameter "depth" "Depth" "1u" (BasicType.Numeric UInt32)
            makeParameter "input" "Input" "input" BasicType.String
            makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

  let makeReadChunks (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "ReadChunks" + str
      DisplayName = "readChunks" + str
      Category = "IO"
      Description = "Read stack chunks from image files."
      Aliases = [ "chunks"; "input"; "tiff"; "file" ]
      Inputs = [ makePort "FromSource" Source ]
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ makeParameter "input" "Input" "input" BasicType.String
            makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

  let makeScalarImageFunction operation displayName category tp =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = operation + str
      DisplayName = displayName + str
      Category = category
      Description = "Apply a scalar arithmetic operation to each image."
      Aliases = [ "scalar"; "arithmetic"; displayName ]
      Inputs = [ makePort "IN" pt ]
      Outputs = [ makePort "OUT" pt ]
      Parameters = [ makeParameter "value" "Value" (numericDefaultValue tp) (BasicType.Numeric tp) ] }

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

        makeScalar (Numeric UInt8)
        makeScalar (Numeric Int8)
        makeScalar (Numeric UInt16)
        makeScalar (Numeric Int16)
        makeScalar (Numeric UInt32)
        makeScalar (Numeric Int32)
        makeScalar (Numeric UInt64)
        makeScalar (Numeric Int64)
        makeScalar (Numeric Float32)
        makeScalar (Numeric Float64)
        makeScalar Bool
        makeScalar String

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

        makeReadRandom UInt8
        makeReadRandom Float64

        makeReadChunks UInt8

        makeZero UInt8
        makeZero Float64

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

        { Id = "WriteInChunks"
          DisplayName = "writeInChunks"
          Category = "IO"
          Description = "Write a stack to image files split into chunks."
          Aliases = [ "output"; "save"; "chunks"; "tiff"; "file" ]
          Inputs = [ makePort "IN" imageAny ]
          Outputs = [ makePort "ToSink" Sink ]
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String
                makeParameter "chunkX" "Chunk X" "12u" (BasicType.Numeric UInt32)
                makeParameter "chunkY" "Chunk Y" "13u" (BasicType.Numeric UInt32)
                makeParameter "chunkZ" "Chunk Z" "14u" (BasicType.Numeric UInt32) ] }

        { Id = "SqrtFloat64"
          DisplayName = "sqrt"
          Category = "Arithmetic"
          Description = "Apply square root to each pixel."
          Aliases = [ "sqrt"; "square root"; "arithmetic" ]
          Inputs = [ makePort "IN" imageFloat64 ]
          Outputs = [ makePort "OUT" imageFloat64 ]
          Parameters = [] }

        makeScalarImageFunction "ImageAddScalar" "imageAddScalar" "Arithmetic" UInt8
        makeScalarImageFunction "ImageAddScalar" "imageAddScalar" "Arithmetic" Float64
        makeScalarImageFunction "ImageMulScalar" "imageMulScalar" "Arithmetic" UInt8
        makeScalarImageFunction "ImageMulScalar" "imageMulScalar" "Arithmetic" Float64
        makeScalarImageFunction "ImageDivScalar" "imageDivScalar" "Arithmetic" UInt8
        makeScalarImageFunction "ImageDivScalar" "imageDivScalar" "Arithmetic" Float64
        makeScalarImageFunction "ScalarMulImage" "scalarMulImage" "Arithmetic" UInt8
        makeScalarImageFunction "ScalarMulImage" "scalarMulImage" "Arithmetic" Float64

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

        { Id = "ConvGauss"
          DisplayName = "convGauss"
          Category = "Filters"
          Description = "Apply Gaussian convolution."
          Aliases = [ "gaussian"; "smooth"; "blur"; "filter" ]
          Inputs = [ makePort "IN" imageFloat64 ]
          Outputs = [ makePort "OUT" imageFloat64 ]
          Parameters = [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "FiniteDiff"
          DisplayName = "finiteDiff"
          Category = "Filters"
          Description = "Apply finite difference derivative filters."
          Aliases = [ "derivative"; "difference"; "filter" ]
          Inputs = [ makePort "IN" imageFloat64 ]
          Outputs = [ makePort "OUT" imageFloat64 ]
          Parameters =
              [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "axis1" "Axis 1" "1u" (BasicType.Numeric UInt32)
                makeParameter "axis2" "Axis 2" "2u" (BasicType.Numeric UInt32) ] }

        { Id = "AddNormalNoiseUInt8"
          DisplayName = "addNormalNoiseUInt8"
          Category = "Filters"
          Description = "Add normally distributed noise to each image."
          Aliases = [ "noise"; "random"; "normal" ]
          Inputs = [ makePort "IN" imageUInt8 ]
          Outputs = [ makePort "OUT" imageUInt8 ]
          Parameters =
              [ makeParameter "mean" "Mean" "128.0" (BasicType.Numeric Float64)
                makeParameter "std" "Std" "50.0" (BasicType.Numeric Float64) ] }

        { Id = "AddNormalNoiseFloat64"
          DisplayName = "addNormalNoiseFloat64"
          Category = "Filters"
          Description = "Add normally distributed noise to each image."
          Aliases = [ "noise"; "random"; "normal" ]
          Inputs = [ makePort "IN" imageFloat64 ]
          Outputs = [ makePort "OUT" imageFloat64 ]
          Parameters =
              [ makeParameter "mean" "Mean" "128.0" (BasicType.Numeric Float64)
                makeParameter "std" "Std" "50.0" (BasicType.Numeric Float64) ] }

        { Id = "Threshold"
          DisplayName = "threshold"
          Category = "Segmentation"
          Description = "Threshold an image into a binary UInt8 image."
          Aliases = [ "binary"; "mask"; "segment" ]
          Inputs = [ makePort "IN" imageAny ]
          Outputs = [ makePort "OUT" imageUInt8 ]
          Parameters =
              [ makeParameter "lower" "Lower" "128.0" (BasicType.Numeric Float64)
                makeParameter "upper" "Upper" "infinity" (BasicType.Numeric Float64) ] }

        { Id = "Erode"
          DisplayName = "erode"
          Category = "Morphology"
          Description = "Erode a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "IN" imageUInt8 ]
          Outputs = [ makePort "OUT" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1u" (BasicType.Numeric UInt32) ] }

        { Id = "Dilate"
          DisplayName = "dilate"
          Category = "Morphology"
          Description = "Dilate a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "IN" imageUInt8 ]
          Outputs = [ makePort "OUT" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1u" (BasicType.Numeric UInt32) ] }

        { Id = "Opening"
          DisplayName = "opening"
          Category = "Morphology"
          Description = "Apply morphological opening to a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "IN" imageUInt8 ]
          Outputs = [ makePort "OUT" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1u" (BasicType.Numeric UInt32) ] }

        { Id = "Closing"
          DisplayName = "closing"
          Category = "Morphology"
          Description = "Apply morphological closing to a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "IN" imageUInt8 ]
          Outputs = [ makePort "OUT" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1u" (BasicType.Numeric UInt32) ] }

        { Id = "ConnectedComponents"
          DisplayName = "connectedComponents"
          Category = "Segmentation"
          Description = "Label connected binary components."
          Aliases = [ "components"; "labels"; "segmentation" ]
          Inputs = [ makePort "IN" imageUInt8 ]
          Outputs = [ makePort "OUT" imageUInt64 ]
          Parameters = [ makeParameter "windowSize" "Window size" "3u" (BasicType.Numeric UInt32) ] }

        { Id = "PermuteAxes"
          DisplayName = "permuteAxes"
          Category = "Geometry"
          Description = "Transpose stack axes."
          Aliases = [ "transpose"; "axes"; "permute" ]
          Inputs = [ makePort "IN" imageAny ]
          Outputs = [ makePort "OUT" imageAny ]
          Parameters =
              [ makeParameter "axes" "Axes" "(0u,1u,2u)" BasicType.String
                makeParameter "tileSize" "Tile size" "64u" (BasicType.Numeric UInt32) ] }

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
