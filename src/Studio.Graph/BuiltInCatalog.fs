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
  let connectedComponentLabels = PortType.Tuple(imageUInt64, PortType.Scalar(BasicType.Numeric UInt64))

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
      DisplayName = "a"
      Category = "Sources / Sinks"
      Summary = "Bind a scalar value for graph parameters."
      Description = ""
      Aliases = [ "value"; "parameter"; "constant"; "let"; "UInt8"; "Float64"; "String"; "Bool" ]
      Inputs = []
      Outputs = [ makePort "Value" (Scalar(Numeric Float64)) ]
      Parameters =
          [ makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "value" "Value" (scalarDefaultValue (Numeric Float64)) (Numeric Float64) ] }

  let makeFileDirectory () =
    { Id = "FileDirectory"
      DisplayName = "file/directory"
      Category = "Sources / Sinks"
      Summary = "Ask for a file or directory when the graph is run and emit its path as a string."
      Description = "When Run is pressed, Studio highlights each File/Directory box and opens the corresponding picker. Run stops if the user cancels one of the prompts."
      Aliases = [ "file"; "directory"; "folder"; "path"; "input"; "output"; "String" ]
      Inputs = []
      Outputs = [ makePort "Value" (Scalar BasicType.String) ]
      Parameters =
          [ makeParameter "kind" "Kind" "Directory" BasicType.String
            makeParameter "value" "Value" "" BasicType.String ] }

  let makeScalarOp () =
    { Id = "ScalarOp"
      DisplayName = "a op b"
      Category = "Arithmetic"
      Summary = "Combine two scalar values with an arithmetic operation."
      Description = ""
      Aliases = [ "scalar"; "arithmetic"; "add"; "subtract"; "multiply"; "divide"; "+"; "-"; "*"; "/" ]
      Inputs = []
      Outputs = [ makePort "Float64" (Scalar(BasicType.Numeric Float64)) ]
      Parameters =
          [ makeParameter "operation" "Operation" "*" BasicType.String
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "a" "A" (numericDefaultValue Float64) (BasicType.Numeric Float64)
            makeParameter "b" "B" (numericDefaultValue Float64) (BasicType.Numeric Float64) ] }

  let makeScalarFunction () =
    { Id = "ScalarFunction"
      DisplayName = "f(a)"
      Category = "Arithmetic"
      Summary = "Apply a standard F# function to a scalar value."
      Description = ""
      Aliases = [ "scalar"; "function"; "unary"; "abs"; "acos"; "asin"; "atan"; "cos"; "sin"; "tan"; "exp"; "log10"; "log"; "round"; "sqrt"; "square"; "arithmetic" ]
      Inputs = []
      Outputs = [ makePort "Float64" (Scalar(BasicType.Numeric Float64)) ]
      Parameters =
          [ makeParameter "function" "Function" "sqrt" BasicType.String
            makeParameter "a" "A" (numericDefaultValue Float64) (BasicType.Numeric Float64) ] }

  let makeGenericRead () =
    { Id = "Read"
      DisplayName = "read"
      Category = "Sources / Sinks"
      Summary = "Read a stack from chunked image files."
      Description = ""
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
      Summary = "Read a randomized subset of stack files."
      Description = ""
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
      Summary = "Read stack chunks from image files."
      Description = ""
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
      Summary = "Convert stream element type."
      Description = ""
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
      Summary = "Create a zero-valued synthetic stack."
      Description = ""
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
      Summary = "Create a synthetic stack by applying an Euler 2D transform to a seed image."
      Description = ""
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
      Summary = description
      Description = ""
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
      Summary = "Add normally distributed noise to each image."
      Description = ""
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
      Summary = description
      Description = ""
      Aliases = aliases @ [ "pair"; "zip"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "I: Float64" imageFloat64; makePort "J: Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters = parameters }

  let orderedFunctions =
      [ makeGenericScalar()

        makeFileDirectory()
    
        makeScalarOp()

        makeScalarFunction()

        makeGenericRead()

        makeGenericReadRandom()

        makeGenericReadChunks()

        makeGenericZero()

        makeGenericCreateByEuler2DTransform()

        { Id = "ComputeStats"
          DisplayName = "computeStats"
          Category = "Statistics"
          Summary = "Reduce an image stream to aggregate image statistics."
          Description = ""
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
          Summary = "Write a processed stack to image files."
          Description = ""
          Aliases = [ "output"; "save"; "tiff"; "file" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

        { Id = "WriteThrough"
          DisplayName = "writeThrough"
          Category = "Sources / Sinks"
          Summary = "Write a processed stack to image files and pass it through unchanged."
          Description = ""
          Aliases = [ "output"; "save"; "tiff"; "file"; "through"; "side effect" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

        { Id = "WriteChunkSlices"
          DisplayName = "writeChunkSlices"
          Category = "Sources / Sinks"
          Summary = "Write connected-component label chunks slice-by-slice and pass labels plus object counts through unchanged."
          Description = ""
          Aliases = [ "output"; "save"; "chunks"; "labels"; "connected"; "components"; "side effect" ]
          Inputs = [ makePort "Labels + count" connectedComponentLabels ]
          Outputs = [ makePort "Labels + count" connectedComponentLabels ]
          Parameters =
              [ makeParameter "output" "Output" "tmp" BasicType.String
                makeParameter "suffix" "Suffix" ".mha" BasicType.String
                makeParameter "windowSize" "Window size" "8" (BasicType.Numeric UInt32) ] }

        { Id = "WriteInChunks"
          DisplayName = "writeInChunks"
          Category = "Sources / Sinks"
          Summary = "Write a stack to image files split into chunks."
          Description = ""
          Aliases = [ "output"; "save"; "chunks"; "tiff"; "file" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String
                makeParameter "chunkX" "Chunk X" "12" (BasicType.Numeric UInt32)
                makeParameter "chunkY" "Chunk Y" "13" (BasicType.Numeric UInt32)
                makeParameter "chunkZ" "Chunk Z" "14" (BasicType.Numeric UInt32) ] }

        { Id = "GetStackInfo"
          DisplayName = "getStackInfo"
          Category = "Sources / Sinks"
          Summary = "Inspect a stack directory and expose file information fields as scalar outputs."
          Description = ""
          Aliases = [ "info"; "metadata"; "file"; "stack"; "width"; "height"; "depth"; "size"; "dimensions"; "component" ]
          Inputs = []
          Outputs =
              [ makePort "Dimensions: UInt32" (Scalar(BasicType.Numeric UInt32))
                makePort "Size: UInt64 list" (Custom "UInt64List")
                makePort "ComponentType: String" (Scalar BasicType.String)
                makePort "NumberOfComponents: UInt32" (Scalar(BasicType.Numeric UInt32))
                makePort "Width: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Height: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Depth: UInt64" (Scalar(BasicType.Numeric UInt64)) ]
          Parameters =
              [ makeParameter "input" "Name" "input" BasicType.String
                makeParameter "suffix" "Suffix" ".tiff" BasicType.String ] }

        { Id = "Tap"
          DisplayName = "tap"
          Category = "Debug"
          Summary = "Print each streamed value and pass it through unchanged."
          Description = ""
          Aliases = [ "debug"; "trace"; "log"; "inspect" ]
          Inputs = [ makePort "Any" any ]
          Outputs = [ makePort "Any" any ]
          Parameters = [ makeParameter "label" "Label" "tap" BasicType.String ] }

        { Id = "Print"
          DisplayName = "print"
          Category = "Visualization"
          Summary = "Print one or more scalar values in the generated program."
          Description = ""
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
          Summary = "Compute and show an image histogram."
          Description = ""
          Aliases = [ "plot"; "chart"; "histogram"; "visualize"; "show" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters = [] }

        { Id = "HistogramData"
          DisplayName = "histogramData"
          Category = "Visualization"
          Summary = "Reduce an image stream to histogram points that can be printed or plotted."
          Description = ""
          Aliases = [ "plot"; "chart"; "histogram"; "points"; "reducer" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Map" (Scalar BasicType.Map) ]
          Parameters = [] }

        { Id = "Chart"
          DisplayName = "chart"
          Category = "Visualization"
          Summary = "Render map-like x/y or key/value data as a Plotly.NET chart."
          Description = ""
          Aliases = [ "plot"; "chart"; "histogram"; "visualize"; "show"; "scatter"; "line"; "bar"; "column"; "area"; "pie"; "doughnut" ]
          Inputs = []
          Outputs = []
          Parameters =
              [ makeParameter "kind" "Kind" "Column" BasicType.String
                makeParameter "input" "Input" "map" BasicType.Map ] }

        { Id = "ShowImage"
          DisplayName = "showImage"
          Category = "Visualization"
          Summary = "Show each image as a heatmap."
          Description = ""
          Aliases = [ "plot"; "image"; "heatmap"; "visualize"; "show" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters = [] }

        { Id = "UnaryImageFunction"
          DisplayName = "f(I)"
          Category = "Arithmetic"
          Summary = "Apply a standard unary function to each pixel."
          Description = ""
          Aliases = [ "function"; "unary"; "abs"; "acos"; "asin"; "atan"; "cos"; "sin"; "tan"; "exp"; "log10"; "log"; "round"; "sqrt"; "square"; "arithmetic" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters = [ makeParameter "function" "Function" "sqrt" BasicType.String ] }

        makeScalarImageOperation
            "ImageOpScalar"
            "I op a"
            "Apply an arithmetic operation with the image on the left and a scalar on the right."
            [ "add"; "subtract"; "multiply"; "divide"; "image"; "value"; "scale"; "+"; "-"; "*"; "/" ]

        makeScalarImageOperation
            "ScalarOpImage"
            "a op I"
            "Apply an arithmetic operation with a scalar on the left and the image on the right."
            [ "add"; "subtract"; "multiply"; "divide"; "inverse"; "value"; "image"; "+"; "-"; "*"; "/" ]

        makeGenericAddNormalNoise()

        makePairOperation
            "ImageOpImage"
            "I op J"
            "Combine two image streams of the same numeric type pairwise with an arithmetic, max, or min operation. Code generation inserts zip or shared fan-out as needed."
            [ "add"; "sum"; "subtract"; "multiply"; "mask"; "divide"; "ratio"; "maximum"; "max"; "minimum"; "min"; "arithmetic"; "+"; "-"; "*"; "/" ]
            [ makeParameter "operation" "Operation" "*" BasicType.String
              makeParameter "type" "Type" "Float64" BasicType.String ]

        { Id = "DiscreteGaussian"
          DisplayName = "discreteGaussian"
          Category = "Filters"
          Summary = "Apply a Gaussian smoothing filter."
          Description = ""
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
          Summary = "Apply Gaussian convolution."
          Description = ""
          Aliases = [ "gaussian"; "smooth"; "blur"; "filter" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "FiniteDiff"
          DisplayName = "finiteDiff"
          Category = "Filters"
          Summary = "Apply finite difference derivative filters."
          Description = ""
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
          Summary = "Threshold an image into a binary UInt8 image."
          Description = ""
          Aliases = [ "binary"; "mask"; "segment"; "UInt8"; "Float64"; "type" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "lower" "Lower" "128.0" (BasicType.Numeric Float64)
                makeParameter "upper" "Upper" "infinity" (BasicType.Numeric Float64) ] }

        { Id = "Erode"
          DisplayName = "erode"
          Category = "Binary Morphology"
          Summary = "Erode a binary UInt8 image."
          Description = ""
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Dilate"
          DisplayName = "dilate"
          Category = "Binary Morphology"
          Summary = "Dilate a binary UInt8 image."
          Description = ""
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Opening"
          DisplayName = "opening"
          Category = "Binary Morphology"
          Summary = "Apply morphological opening to a binary UInt8 image."
          Description = ""
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Closing"
          DisplayName = "closing"
          Category = "Binary Morphology"
          Summary = "Apply morphological closing to a binary UInt8 image."
          Description = ""
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "ConnectedComponents"
          DisplayName = "connectedComponents"
          Category = "Binary Morphology"
          Summary = "Label connected binary components."
          Description = ""
          Aliases = [ "components"; "labels"; "segmentation" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "Labels + count" connectedComponentLabels ]
          Parameters = [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "ComponentTranslationTable"
          DisplayName = "componentTranslationTable"
          Category = "Binary Morphology"
          Summary = "Reduce connected-component label chunks plus object counts to a chunk translation table."
          Description = ""
          Aliases = [ "connected"; "components"; "translation"; "table"; "reducer"; "labels" ]
          Inputs = [ makePort "Labels + count" connectedComponentLabels ]
          Outputs = [ makePort "TranslationTable" translationTable ]
          Parameters = [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "CollapseComponentLabels"
          DisplayName = "collapseComponentLabels"
          Category = "Binary Morphology"
          Summary = "Collapse chunk-local component labels using a translation table."
          Description = ""
          Aliases = [ "connected"; "components"; "translation"; "table"; "update"; "labels" ]
          Inputs = [ makePort "UInt64" imageUInt64 ]
          Outputs = [ makePort "UInt64" imageUInt64 ]
          Parameters =
              [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32)
                makeParameter "translationTable" "Translation table" "" BasicType.String ] }

        { Id = "PermuteAxes"
          DisplayName = "permuteAxes"
          Category = "Geometry"
          Summary = "Transpose stack axes."
          Description = ""
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
