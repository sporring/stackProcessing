namespace Studio.Graph

module BuiltInCatalog =
  let any = PortType.Any
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
  let translationTable = PortType.Custom "TranslationTable"

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

  let private numericDefaultValue tp =
      match tp with
      | UInt8
      | Int8
      | UInt16
      | Int16
      | UInt32
      | Int32 -> "1"
      | UInt64
      | Int64 -> "1"
      | Float32 -> "1.0"
      | Float64
      | Number -> "1.0"
      | Complex -> "1.0"

  let private scalarDefaultValue tp =
      match tp with
      | BasicType.Bool -> "true"
      | BasicType.String -> "value"
      | BasicType.Map -> "map"
      | BasicType.Unit -> "()"
      | BasicType.Numeric numericType -> numericDefaultValue numericType

  let makeGenericScalar () =
    { Id = "Scalar"
      DisplayName = "scalar"
      Category = "Sources / Sinks"
      Description = "Bind a scalar value for graph parameters."
      Aliases = [ "value"; "parameter"; "constant"; "let"; "UInt8"; "Float64"; "String"; "Bool" ]
      Inputs = []
      Outputs = [ makePort "Value" (Scalar(Numeric Float64)) ]
      Parameters =
          [ makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "value" "Value" (scalarDefaultValue (Numeric Float64)) (Numeric Float64) ] }

  let makeScalarOp () =
    { Id = "ScalarOp"
      DisplayName = "scalarOp"
      Category = "Arithmetic"
      Description = "Combine two scalar values with an arithmetic operation."
      Aliases = [ "scalar"; "arithmetic"; "add"; "subtract"; "multiply"; "divide"; "+"; "-"; "*"; "/" ]
      Inputs = []
      Outputs = [ makePort "Float64" (Scalar(BasicType.Numeric Float64)) ]
      Parameters =
          [ makeParameter "operation" "Operation" "*" BasicType.String
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "a" "A" (numericDefaultValue Float64) (BasicType.Numeric Float64)
            makeParameter "b" "B" (numericDefaultValue Float64) (BasicType.Numeric Float64) ] }

  let makeGenericRead () =
    { Id = "Read"
      DisplayName = "read"
      Category = "Sources / Sinks"
      Description = "Read a stack from chunked image files."
      Aliases = [ "input"; "load"; "tiff"; "file"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "input" "Input" "input" BasicType.String
            makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

  let makeGenericReadRandom () =
    { Id = "ReadRandom"
      DisplayName = "readRandom"
      Category = "Sources / Sinks"
      Description = "Read a randomized subset of stack files."
      Aliases = [ "random"; "input"; "tiff"; "file"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
            makeParameter "input" "Input" "input" BasicType.String
            makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

  let makeGenericReadChunks () =
    { Id = "ReadChunks"
      DisplayName = "readChunks"
      Category = "Sources / Sinks"
      Description = "Read stack chunks from image files."
      Aliases = [ "chunks"; "input"; "tiff"; "file"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "input" "Input" "input" BasicType.String
            makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

  let makeGenericCast () =
    { Id = "Cast"
      DisplayName = "cast"
      Category = "Type conversions"
      Description = "Convert stream element type."
      Aliases = [ "convert"; "uint8"; "float"; "type"; "UInt8"; "Float64" ]
      Inputs = [ makePort "Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ makeParameter "sourceType" "Source type" "Float64" BasicType.String
            makeParameter "targetType" "Target type" "Float64" BasicType.String ] }

  let makeGenericZero () =
    { Id = "Zero"
      DisplayName = "zero"
      Category = "Sources / Sinks"
      Description = "Create a zero-valued synthetic stack."
      Aliases = [ "empty"; "synthetic"; "source"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32) ] }

  let makeGenericCreateByEuler2DTransform () =
    { Id = "CreateByEuler2DTransform"
      DisplayName = "createByEuler2DTransform"
      Category = "Sources / Sinks"
      Description = "Create a synthetic stack by applying an Euler 2D transform to a seed image."
      Aliases = [ "synthetic"; "source"; "euler"; "transform"; "rotation"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "UInt8" BasicType.String
            makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "64" (BasicType.Numeric UInt32)
            makeParameter "boxSize" "Box size" "16" (BasicType.Numeric UInt32)
            makeParameter "transform" "Transform" "Diagonal" BasicType.String ] }

  let makeScalarImageOperation id displayName description aliases =
    { Id = id
      DisplayName = displayName
      Category = "Arithmetic"
      Description = description
      Aliases = aliases @ [ "scalar"; "arithmetic"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ makeParameter "operation" "Operation" "*" BasicType.String
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "value" "Value" (numericDefaultValue Float64) (BasicType.Numeric Float64) ] }

  let makeGenericAddNormalNoise () =
    { Id = "AddNormalNoise"
      DisplayName = "addNormalNoise"
      Category = "Statistics"
      Description = "Add normally distributed noise to each image."
      Aliases = [ "noise"; "random"; "normal"; "statistics"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "mean" "Mean" "128.0" (BasicType.Numeric Float64)
            makeParameter "std" "Std" "50.0" (BasicType.Numeric Float64) ] }

  let makePairOperation id displayName description aliases parameters =
    { Id = id
      DisplayName = displayName
      Category = "Arithmetic"
      Description = description
      Aliases = aliases @ [ "pair"; "zip"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "I: Float64" imageFloat64; makePort "J: Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters = parameters }

  let orderedFunctions =
      [ makeGenericScalar()

        makeScalarOp()

        makeGenericRead()

        makeGenericReadRandom()

        makeGenericReadChunks()

        makeGenericZero()

        makeGenericCreateByEuler2DTransform()

        { Id = "ComputeStats"
          DisplayName = "computeStats"
          Category = "Statistics"
          Description = "Reduce an image stream to aggregate image statistics."
          Aliases = [ "statistics"; "stats"; "mean"; "std"; "min"; "max"; "reducer" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs =
              [ makePort "NumPixels: UInt32" (Scalar(BasicType.Numeric UInt32))
                makePort "Mean: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "Std: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "Min: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "Max: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "Sum: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "Var: Float64" (Scalar(BasicType.Numeric Float64)) ]
          Parameters = [] }

        { Id = "Write"
          DisplayName = "write"
          Category = "Sources / Sinks"
          Description = "Write a processed stack to image files."
          Aliases = [ "output"; "save"; "tiff"; "file" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

        { Id = "WriteThrough"
          DisplayName = "writeThrough"
          Category = "Sources / Sinks"
          Description = "Write a processed stack to image files and pass it through unchanged."
          Aliases = [ "output"; "save"; "tiff"; "file"; "through"; "side effect" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

        { Id = "WriteInChunks"
          DisplayName = "writeInChunks"
          Category = "Sources / Sinks"
          Description = "Write a stack to image files split into chunks."
          Aliases = [ "output"; "save"; "chunks"; "tiff"; "file" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String
                makeParameter "chunkX" "Chunk X" "12" (BasicType.Numeric UInt32)
                makeParameter "chunkY" "Chunk Y" "13" (BasicType.Numeric UInt32)
                makeParameter "chunkZ" "Chunk Z" "14" (BasicType.Numeric UInt32) ] }

        { Id = "Tap"
          DisplayName = "tap"
          Category = "Debug"
          Description = "Print each streamed value and pass it through unchanged."
          Aliases = [ "debug"; "trace"; "log"; "inspect" ]
          Inputs = [ makePort "Any" any ]
          Outputs = [ makePort "Any" any ]
          Parameters = [ makeParameter "label" "Label" "tap" BasicType.String ] }

        { Id = "Print"
          DisplayName = "print"
          Category = "Visualization"
          Description = "Print one or more scalar values in the generated program."
          Aliases = [ "debug"; "trace"; "log"; "sink"; "inspect"; "printfn" ]
          Inputs = []
          Outputs = []
          Parameters =
              [ makeParameter "format" "Format" "{input1}" BasicType.String
                makeParameter "input1" "Input 1" "input1" BasicType.String
                makeParameter "input2" "Input 2" "input2" BasicType.String
                makeParameter "input3" "Input 3" "input3" BasicType.String
                makeParameter "input4" "Input 4" "input4" BasicType.String
                makeParameter "input5" "Input 5" "input5" BasicType.String
                makeParameter "input6" "Input 6" "input6" BasicType.String
                makeParameter "input7" "Input 7" "input7" BasicType.String
                makeParameter "input8" "Input 8" "input8" BasicType.String ] }

        { Id = "Histogram"
          DisplayName = "histogram"
          Category = "Visualization"
          Description = "Compute and show an image histogram."
          Aliases = [ "plot"; "chart"; "histogram"; "visualize"; "show" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters = [] }

        { Id = "HistogramData"
          DisplayName = "histogramData"
          Category = "Visualization"
          Description = "Reduce an image stream to histogram points that can be printed or plotted."
          Aliases = [ "plot"; "chart"; "histogram"; "points"; "reducer" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Map" (Scalar BasicType.Map) ]
          Parameters = [] }

        { Id = "Chart"
          DisplayName = "chart"
          Category = "Visualization"
          Description = "Render map-like x/y or key/value data as a Plotly.NET chart."
          Aliases = [ "plot"; "chart"; "histogram"; "visualize"; "show"; "scatter"; "line"; "bar"; "column"; "area"; "pie"; "doughnut" ]
          Inputs = []
          Outputs = []
          Parameters =
              [ makeParameter "kind" "Kind" "Column" BasicType.String
                makeParameter "input" "Input" "map" BasicType.Map ] }

        { Id = "ShowImage"
          DisplayName = "showImage"
          Category = "Visualization"
          Description = "Show each image as a heatmap."
          Aliases = [ "plot"; "image"; "heatmap"; "visualize"; "show" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters = [] }

        { Id = "SqrtFloat64"
          DisplayName = "sqrt"
          Category = "Arithmetic"
          Description = "Apply square root to each pixel."
          Aliases = [ "sqrt"; "square root"; "arithmetic" ]
          Inputs = [ makePort "IN" imageFloat64 ]
          Outputs = [ makePort "OUT" imageFloat64 ]
          Parameters = [] }

        makeScalarImageOperation
            "ImageOpScalar"
            "imageOpScalar"
            "Apply an arithmetic operation with the image on the left and a scalar on the right."
            [ "add"; "subtract"; "multiply"; "divide"; "image"; "value"; "scale"; "+"; "-"; "*"; "/" ]

        makeScalarImageOperation
            "ScalarOpImage"
            "scalarOpImage"
            "Apply an arithmetic operation with a scalar on the left and the image on the right."
            [ "add"; "subtract"; "multiply"; "divide"; "inverse"; "value"; "image"; "+"; "-"; "*"; "/" ]

        makeGenericAddNormalNoise()

        makePairOperation
            "ImageOpImage"
            "imageOpImage"
            "Combine two image streams of the same numeric type pairwise with an arithmetic operation. Code generation inserts zip or shared fan-out as needed."
            [ "add"; "sum"; "subtract"; "multiply"; "mask"; "divide"; "ratio"; "arithmetic"; "+"; "-"; "*"; "/" ]
            [ makeParameter "operation" "Operation" "*" BasicType.String
              makeParameter "type" "Type" "Float64" BasicType.String ]

        makePairOperation
            "MaxOfPair"
            "maxOfPair"
            "Take the pixelwise maximum of two image streams of the same numeric type pairwise. Code generation inserts zip or shared fan-out as needed."
            [ "maximum"; "max"; "pair"; "arithmetic" ]
            [ makeParameter "type" "Type" "Float64" BasicType.String ]

        makePairOperation
            "MinOfPair"
            "minOfPair"
            "Take the pixelwise minimum of two image streams of the same numeric type pairwise. Code generation inserts zip or shared fan-out as needed."
            [ "minimum"; "min"; "pair"; "arithmetic" ]
            [ makeParameter "type" "Type" "Float64" BasicType.String ]

        { Id = "DiscreteGaussian"
          DisplayName = "discreteGaussian"
          Category = "Filters"
          Description = "Apply a Gaussian smoothing filter."
          Aliases = [ "gaussian"; "smooth"; "blur"; "filter" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
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
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "FiniteDiff"
          DisplayName = "finiteDiff"
          Category = "Filters"
          Description = "Apply finite difference derivative filters."
          Aliases = [ "derivative"; "difference"; "filter" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "axis1" "Axis 1" "1" (BasicType.Numeric UInt32)
                makeParameter "axis2" "Axis 2" "2" (BasicType.Numeric UInt32) ] }

        { Id = "Threshold"
          DisplayName = "threshold"
          Category = "Segmentation"
          Description = "Threshold an image into a binary UInt8 image."
          Aliases = [ "binary"; "mask"; "segment"; "UInt8"; "Float64"; "type" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "lower" "Lower" "128.0" (BasicType.Numeric Float64)
                makeParameter "upper" "Upper" "infinity" (BasicType.Numeric Float64) ] }

        { Id = "Erode"
          DisplayName = "erode"
          Category = "Morphology"
          Description = "Erode a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Dilate"
          DisplayName = "dilate"
          Category = "Morphology"
          Description = "Dilate a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Opening"
          DisplayName = "opening"
          Category = "Morphology"
          Description = "Apply morphological opening to a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Closing"
          DisplayName = "closing"
          Category = "Morphology"
          Description = "Apply morphological closing to a binary UInt8 image."
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "ConnectedComponents"
          DisplayName = "connectedComponents"
          Category = "Segmentation"
          Description = "Label connected binary components."
          Aliases = [ "components"; "labels"; "segmentation" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt64" imageUInt64 ]
          Parameters = [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "ComponentTranslationTable"
          DisplayName = "componentTranslationTable"
          Category = "Segmentation"
          Description = "Reduce a UInt64 connected-component label stream to a chunk translation table."
          Aliases = [ "connected"; "components"; "translation"; "table"; "reducer"; "labels" ]
          Inputs = [ makePort "UInt64" imageUInt64 ]
          Outputs = [ makePort "TranslationTable" translationTable ]
          Parameters = [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "CollapseComponentLabels"
          DisplayName = "collapseComponentLabels"
          Category = "Segmentation"
          Description = "Collapse chunk-local component labels using a translation table."
          Aliases = [ "connected"; "components"; "translation"; "table"; "update"; "labels" ]
          Inputs = [ makePort "UInt64" imageUInt64 ]
          Outputs = [ makePort "UInt64" imageUInt64 ]
          Parameters =
              [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32)
                makeParameter "translationTable" "Translation table" "" BasicType.String ] }

        { Id = "PermuteAxes"
          DisplayName = "permuteAxes"
          Category = "Geometry"
          Description = "Transpose stack axes."
          Aliases = [ "transpose"; "axes"; "permute" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "axes" "Axes" "(0,1,2)" BasicType.String
                makeParameter "tileSize" "Tile size" "64" (BasicType.Numeric UInt32) ] }

        makeGenericCast() ]

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
