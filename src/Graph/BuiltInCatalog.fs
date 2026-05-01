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

  let private standardNumericTypes =
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

  let private concreteNumericTypes =
      standardNumericTypes @ [ Complex ]

  let private scalarBasicTypes =
      (standardNumericTypes |> List.map BasicType.Numeric) @ [ Bool; String ]

  let private readRandomTypes =
      [ UInt8; Float64 ]

  let private readChunksTypes =
      [ UInt8 ]

  let private zeroTypes =
      [ UInt8; Float64 ]

  let private scalarImageFunctionTypes =
      [ UInt8; Float64 ]

  let private scalarImageFunctions =
      [ "ImageAddScalar", "imageAddScalar", "Arithmetic"
        "ImageMulScalar", "imageMulScalar", "Arithmetic"
        "ImageDivScalar", "imageDivScalar", "Arithmetic"
        "ScalarMulImage", "scalarMulImage", "Arithmetic" ]

  let private makePort name portType =
      { Name = name
        Type = portType }

  let private makeParameter key label defaultValue parameterType =
      { Key = key
        Label = label
        DefaultValue = defaultValue
        Type = parameterType }

  let private availableMemoryParameter =
      makeParameter "availableMemory" "Available memory" (string (pown 2 30)) BasicType.String

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
      Category = "Sources / Sinks"
      Description = "Read a stack from chunked image files."
      Aliases = [ "input"; "load"; "tiff"; "file" ]
      Inputs = []
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ availableMemoryParameter
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
            makeParameter "targetType" "Target type" (fsharpTypeName tp) BasicType.String ] }

  let makeZero (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "Zero" + str
      DisplayName = "zero" + str
      Category = "Sources / Sinks"
      Description = "Create a zero-valued synthetic stack."
      Aliases = [ "empty"; "synthetic"; "source" ]
      Inputs = []
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "width" "Width" "64u" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64u" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1u" (BasicType.Numeric UInt32) ] }

  let makeReadRandom (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "ReadRandom" + str
      DisplayName = "readRandom" + str
      Category = "Sources / Sinks"
      Description = "Read a randomized subset of stack files."
      Aliases = [ "random"; "input"; "tiff"; "file" ]
      Inputs = []
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "depth" "Depth" "1u" (BasicType.Numeric UInt32)
            makeParameter "input" "Input" "input" BasicType.String
            makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

  let makeReadChunks (tp: NumericType) =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = "ReadChunks" + str
      DisplayName = "readChunks" + str
      Category = "Sources / Sinks"
      Description = "Read stack chunks from image files."
      Aliases = [ "chunks"; "input"; "tiff"; "file" ]
      Inputs = []
      Outputs = [ makePort "OUT" pt ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "input" "Input" "input" BasicType.String
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

  let makeMulPair tp =
    let str = NumericType.toString tp
    let pt = PortType.numericToImage tp
    { Id = if tp = UInt8 then "MulPair" else "MulPair" + str
      DisplayName = "mulPair" + str
      Category = "Arithmetic"
      Description = "Multiply two image streams of the same numeric type pairwise. Code generation inserts zip or shared fan-out as needed."
      Aliases = [ "multiply"; "mask"; "pair"; "zip"; fsharpTypeName tp; str ]
      Inputs = [ makePort "A" pt; makePort "B" pt ]
      Outputs = [ makePort "OUT" pt ]
      Parameters = [] }

  let orderedFunctions =
      [ yield! (scalarBasicTypes |> List.map makeScalar)

        yield! (standardNumericTypes |> List.map makeRead)

        yield! (readRandomTypes |> List.map makeReadRandom)

        yield! (readChunksTypes |> List.map makeReadChunks)

        yield! (zeroTypes |> List.map makeZero)

        { Id = "Write"
          DisplayName = "write"
          Category = "Sources / Sinks"
          Description = "Write a processed stack to image files."
          Aliases = [ "output"; "save"; "tiff"; "file" ]
          Inputs = [ makePort "IN" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

        { Id = "WriteInChunks"
          DisplayName = "writeInChunks"
          Category = "Sources / Sinks"
          Description = "Write a stack to image files split into chunks."
          Aliases = [ "output"; "save"; "chunks"; "tiff"; "file" ]
          Inputs = [ makePort "IN" imageAny ]
          Outputs = []
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

        yield!
            (scalarImageFunctions
             |> List.collect (fun (operation, displayName, category) ->
                 scalarImageFunctionTypes
                 |> List.map (makeScalarImageFunction operation displayName category)))

        yield! (concreteNumericTypes |> List.map makeMulPair)

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

        yield! (standardNumericTypes |> List.map makeCast) ]

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
