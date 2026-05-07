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
  let intList = PortType.Custom "IntList"
  let uint64List = PortType.Custom "UInt64List"

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

  let private suffixParameter defaultValue =
      makeParameter "suffix" "Format" defaultValue BasicType.String

  let private readFormatDescription =
      "Reads one image per stack slice from a directory. TIFF input accepts both .tif and .tiff files when TIFF is selected; JPEG input accepts both .jpg and .jpeg files when JPEG is selected. Format choices restrict the image type menu to combinations expected to work with SimpleITK/ITK: TIFF supports common 8/16/32-bit integer and 32/64-bit floating-point scalar images; PNG supports UInt8 and UInt16; JPEG and BMP support UInt8; MetaImage, NRRD, and NIfTI support the broad scalar numeric set used by Studio."

  let private writeFormatDescription =
      "Writes one image per stack slice. The selected format controls which image types can be connected to the input pin: TIFF supports common 8/16/32-bit integer and 32/64-bit floating-point scalar images; PNG supports UInt8 and UInt16; JPEG and BMP support UInt8; MetaImage, NRRD, and NIfTI support the broad scalar numeric set used by Studio. Cast before write when a format cannot store the current image type."

  let private chunkWriteFormatDescription =
      "Writes stack slabs split into chunk files for later slab reading. The selected format controls which image types can be connected to the input pin, using the same constraints as write. TIFF output uses the exact selected suffix, either .tif or .tiff."

  let private zarrFormatDescription =
      "Reads or writes an OME-Zarr volume through ZarrNET. The current native .NET implementation is used here for UInt8 and UInt16 scalar images. readZarrSlab serves 2D slices from a selected timepoint/channel/resolution; writeZarr writes a single timepoint/channel volume and exposes chunk sizes and physical voxel spacing so Studio can be used as a stack-to-Zarr converter."

  let private nexusFormatDescription =
      "Reads a rank-3 NeXus/HDF5 detector stack through PureHDF using an explicit dataset path and axis mapping. This covers common MAX IV and ESRF detector-stack layouts while keeping streaming slice reads larger-than-memory friendly. Compressed detector files that use external HDF5 filters may require a later native/plugin fallback."

  let private intensityDescription =
      "Intensity filters change the numeric values of each pixel without changing the stack geometry. They are slice-local and therefore fit the streaming model naturally. Clamp limits values to a range. ShiftScale applies (input + shift) * scale. Intensity stretch maps a selected input range linearly to a selected output range."

  let private shiftScaleDescription =
      "Uses SimpleITK's ShiftScaleImageFilter. Each output pixel is computed as (input + shift) * scale, and the image geometry is unchanged. Use this with computeStats for two-pass stack normalization: first compute mean and standard deviation, then set shift = -mean and scale = 1/std. The operation itself is streaming-friendly because the shift and scale are already known when the image pass starts."

  let private intensityStretchDescription =
      "Linearly maps an input intensity range to an output intensity range using the same shift/scale semantics as shiftScale. Values are not clipped: pixels outside the input range continue linearly outside the output range. This is useful with computeStats or quantiles when the source min/max or robust quantile limits are estimated in an earlier reducer pass."

  let private quantilesDescription =
      "Estimates quantile values from a histogram map. q1 is always emitted. q2, q3, q4, and q5 are optional output slots controlled by the corresponding enabled parameters. Each q value must be between 0 and 1. The result is based on the cumulative histogram counts, so accuracy depends on the histogram key resolution."

  let private localDenoiseDescription =
      "These denoising filters are local-neighborhood operations rather than global iterative solvers. Median uses a radius in x, y, and z and is streamed through windows large enough to cover the z-neighborhood. Bilateral is edge-preserving and can be slower; use the window size to give the z-neighborhood enough context. No recursive Gaussian, curvature-flow, or anisotropic-diffusion filters are included here because their iteration/global-dependency structure is less friendly to LMIP streaming."

  let private edgeDescription =
      "These edge and derivative-like filters are local operators that can be evaluated on streaming z-windows. Gradient magnitude estimates local change strength. Sobel emphasizes edges using a small derivative stencil. Laplacian computes a second-derivative response. Recursive Gaussian and Canny variants are intentionally not included in this first pass because they are less obviously aligned with the 3D LMIP streaming model."

  let private comparisonDescription =
      "Compare two synchronized image streams pixel by pixel and emit a UInt8 mask. Pixels where the comparison is true become non-zero, and false pixels become zero. The two inputs must have the same selected numeric type and compatible geometry. This is the mask-building counterpart to I op J and compiles to the corresponding StackProcessing comparison stage."

  let private maskDescription =
      "Masking combines an image stream with a UInt8 mask stream. Non-zero mask pixels keep the input image value, and masked-out pixels are replaced by the outside value. This is useful after thresholding, connected-component cleanup, or comparison stages. The image type parameter controls the left input and output type; the right input is always a UInt8 mask."

  let private grayscaleMorphologyDescription =
      "Grayscale morphology applies min/max-style neighborhood operations to intensity images rather than binary masks. Erode darkens or shrinks bright structures; dilate brightens or expands them. Opening removes small bright structures, closing fills small dark gaps, white top-hat extracts bright details smaller than the structuring element, black top-hat extracts dark details, and morphological gradient emphasizes local contrast boundaries. These are local filters and are streamed through z-windows large enough to cover the selected radius."

  let private binaryMorphologyDescription =
      "Binary morphology operates on UInt8 masks where non-zero pixels are treated as foreground. The reconstruction variants preserve connected mask structure while removing or filling features selected by a marker or structuring element. Fully connected controls whether diagonal neighbors count as connected. Window size should be at least 2 * radius + 1 for radius-based operations so the z-neighborhood is available while streaming."

  let private labelAnalysisDescription =
      "Label analysis stages inspect labeled images rather than changing intensities. Label shape statistics measure object geometry, label intensity statistics measure intensity values inside labeled regions, overlap measures compare two label images, label contour extracts object boundaries, and changeLabel remaps one label value to another. Statistics stages emit scalar/map-like data and are usually terminal or diagnostic parts of a graph."

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
      Description = readFormatDescription
      Aliases = [ "input"; "load"; "tiff"; "file"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "input" "Input" "input" BasicType.String
            suffixParameter ".tiff" ] }

  let makeGenericReadRandom () =
    { Id = "ReadRandom"
      DisplayName = "readRandom"
      Category = "Sources / Sinks"
      Summary = "Read a randomized subset of stack files."
      Description = readFormatDescription
      Aliases = [ "random"; "input"; "tiff"; "file"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
            makeParameter "input" "Input" "input" BasicType.String
            suffixParameter ".tiff" ] }

  let makeGenericReadSlab () =
    { Id = "ReadSlab"
      DisplayName = "readSlab"
      Category = "Sources / Sinks"
      Summary = "Read chunked stack files as a normal 2D slice stream."
      Description = "Reads slab files assembled from chunks produced by writeInSlabs, then serves their 2D slices to the pipeline. Use the same format and suffix that were used when writing the chunk files; chunk filenames encode x/y/z chunk positions."
      Aliases = [ "slab"; "chunks"; "input"; "tiff"; "file"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "input" "Input" "input" BasicType.String
            suffixParameter ".tiff" ] }

  let makeGenericReadZarrSlab () =
    { Id = "ReadZarrSlab"
      DisplayName = "readZarrSlab"
      Category = "Sources / Sinks"
      Summary = "Read an OME-Zarr dataset as a normal 2D slice stream."
      Description = zarrFormatDescription
      Aliases = [ "zarr"; "ome-zarr"; "slab"; "input"; "UInt8"; "UInt16"; "type" ]
      Inputs = []
      Outputs = [ makePort "UInt8" imageUInt8 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "UInt8" BasicType.String
            makeParameter "input" "Input" "input.zarr" BasicType.String
            makeParameter "slabDepth" "Slab depth" "8" (BasicType.Numeric UInt32)
            makeParameter "multiscaleIndex" "Multiscale index" "0" (BasicType.Numeric Int32)
            makeParameter "datasetIndex" "Dataset index" "0" (BasicType.Numeric Int32)
            makeParameter "timepoint" "Timepoint" "0" (BasicType.Numeric Int32)
            makeParameter "channel" "Channel" "0" (BasicType.Numeric Int32)
            makeParameter "maxParallelChunks" "Max parallel chunks" "0" (BasicType.Numeric Int32) ] }

  let makeGenericReadNexusSlab () =
    { Id = "ReadNexusSlab"
      DisplayName = "readNexusSlab"
      Category = "Sources / Sinks"
      Summary = "Read a NeXus/HDF5 detector stack as a normal 2D slice stream."
      Description = nexusFormatDescription
      Aliases = [ "nexus"; "hdf5"; "h5"; "slab"; "input"; "MAX IV"; "ESRF"; "DanMAX"; "ForMAX"; "HOAHub"; "type" ]
      Inputs = []
      Outputs = [ makePort "UInt16" imageUInt16 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "UInt16" BasicType.String
            makeParameter "input" "Input" "scan.h5" BasicType.String
            makeParameter "datasetPath" "Dataset path" "/entry/data/data" BasicType.String
            makeParameter "slabDepth" "Slab depth" "8" (BasicType.Numeric UInt32)
            makeParameter "frameAxis" "Frame axis" "0" (BasicType.Numeric Int32)
            makeParameter "yAxis" "Y axis" "1" (BasicType.Numeric Int32)
            makeParameter "xAxis" "X axis" "2" (BasicType.Numeric Int32) ] }

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

        makeGenericReadSlab()

        makeGenericReadZarrSlab()

        makeGenericReadNexusSlab()

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
          Description = writeFormatDescription
          Aliases = [ "output"; "save"; "tiff"; "file" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                suffixParameter ".tiff" ] }

        { Id = "WriteThrough"
          DisplayName = "writeThrough"
          Category = "Sources / Sinks"
          Summary = "Write a processed stack to image files and pass it through unchanged."
          Description = writeFormatDescription
          Aliases = [ "output"; "save"; "tiff"; "file"; "through"; "side effect" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                suffixParameter ".tiff" ] }

        { Id = "WriteSlabSlices"
          DisplayName = "writeSlabSlices"
          Category = "Sources / Sinks"
          Summary = "Write connected-component label slabs slice-by-slice and pass labels plus object counts through unchanged."
          Description = "Writes connected-component label slabs slice-by-slice and passes the labels plus object counts onward. MetaImage (.mha/.mhd) is the safest default for label data because it supports large integer scalar images."
          Aliases = [ "output"; "save"; "slabs"; "labels"; "connected"; "components"; "side effect" ]
          Inputs = [ makePort "Labels + count" connectedComponentLabels ]
          Outputs = [ makePort "Labels + count" connectedComponentLabels ]
          Parameters =
              [ makeParameter "output" "Output" "tmp" BasicType.String
                suffixParameter ".mha"
                makeParameter "windowSize" "Window size" "8" (BasicType.Numeric UInt32) ] }

        { Id = "WriteInSlabs"
          DisplayName = "writeInSlabs"
          Category = "Sources / Sinks"
          Summary = "Write a stack as slabs split into chunk files."
          Description = chunkWriteFormatDescription
          Aliases = [ "output"; "save"; "slabs"; "chunks"; "tiff"; "file" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output" BasicType.String
                suffixParameter ".tiff"
                makeParameter "chunkX" "Chunk X" "12" (BasicType.Numeric UInt32)
                makeParameter "chunkY" "Chunk Y" "13" (BasicType.Numeric UInt32)
                makeParameter "chunkZ" "Chunk Z" "14" (BasicType.Numeric UInt32) ] }

        { Id = "WriteZarr"
          DisplayName = "writeZarr"
          Category = "Sources / Sinks"
          Summary = "Write a stream of 2D slices as an OME-Zarr volume."
          Description = zarrFormatDescription
          Aliases = [ "zarr"; "ome-zarr"; "output"; "save"; "convert"; "UInt8"; "UInt16" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output.zarr" BasicType.String
                makeParameter "name" "Name" "image" BasicType.String
                makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
                makeParameter "chunkX" "Chunk X" "64" (BasicType.Numeric UInt32)
                makeParameter "chunkY" "Chunk Y" "64" (BasicType.Numeric UInt32)
                makeParameter "chunkZ" "Chunk Z" "8" (BasicType.Numeric UInt32)
                makeParameter "physicalSizeX" "Physical size X" "1.0" (BasicType.Numeric Float64)
                makeParameter "physicalSizeY" "Physical size Y" "1.0" (BasicType.Numeric Float64)
                makeParameter "physicalSizeZ" "Physical size Z" "1.0" (BasicType.Numeric Float64)
                makeParameter "maxConcurrentWrites" "Max concurrent writes" "0" (BasicType.Numeric Int32) ] }

        { Id = "WriteNexus"
          DisplayName = "writeNexus"
          Category = "Sources / Sinks"
          Summary = "Write a stream of 2D slices as a rank-3 NeXus/HDF5 detector dataset."
          Description = "Writes a rank-3 HDF5 dataset through PureHDF using incremental hyperslab writes. This is intended as a simple stack-to-HDF5/NeXus converter; use dataset path and axis parameters to choose the detector-stack layout."
          Aliases = [ "nexus"; "hdf5"; "h5"; "output"; "save"; "convert" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "output.h5" BasicType.String
                makeParameter "datasetPath" "Dataset path" "/entry/data/data" BasicType.String
                makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
                makeParameter "chunkX" "Chunk X" "64" (BasicType.Numeric UInt32)
                makeParameter "chunkY" "Chunk Y" "64" (BasicType.Numeric UInt32)
                makeParameter "chunkZ" "Chunk Z" "8" (BasicType.Numeric UInt32)
                makeParameter "frameAxis" "Frame axis" "0" (BasicType.Numeric Int32)
                makeParameter "yAxis" "Y axis" "1" (BasicType.Numeric Int32)
                makeParameter "xAxis" "X axis" "2" (BasicType.Numeric Int32) ] }

        { Id = "GetStackInfo"
          DisplayName = "getStackInfo"
          Category = "Sources / Sinks"
          Summary = "Inspect a stack directory and expose file information fields as scalar outputs."
          Description = "Reads metadata from the first matching stack slice and combines it with the number of matching files. TIFF input accepts both .tif and .tiff files when TIFF is selected; JPEG input accepts both .jpg and .jpeg files when JPEG is selected."
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
                suffixParameter ".tiff" ] }

        { Id = "GetChunkInfo"
          DisplayName = "getChunkInfo"
          Category = "Sources / Sinks"
          Summary = "Inspect a chunked stack directory and expose chunk layout and top-left file metadata."
          Description = "Reads metadata for chunk files produced by writeInSlabs. The Chunks output is the chunk-grid dimensions, Size is the full stack size, and the component outputs come from the top-left chunk file. TIFF input accepts both .tif and .tiff files when TIFF is selected; JPEG input accepts both .jpg and .jpeg files when JPEG is selected."
          Aliases = [ "info"; "metadata"; "file"; "chunk"; "chunks"; "width"; "height"; "depth"; "size"; "component" ]
          Inputs = []
          Outputs =
              [ makePort "Chunks: int list" intList
                makePort "Size: UInt64 list" uint64List
                makePort "ComponentType: String" (Scalar BasicType.String)
                makePort "NumberOfComponents: UInt32" (Scalar(BasicType.Numeric UInt32))
                makePort "ChunkX: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "ChunkY: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "ChunkZ: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "Width: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Height: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Depth: UInt64" (Scalar(BasicType.Numeric UInt64)) ]
          Parameters =
              [ makeParameter "input" "Name" "input" BasicType.String
                suffixParameter ".tiff" ] }

        { Id = "GetZarrInfo"
          DisplayName = "getZarrInfo"
          Category = "Sources / Sinks"
          Summary = "Inspect an OME-Zarr dataset and expose chunk layout and image dimensions."
          Description = "Reads metadata from a selected OME-Zarr multiscale dataset. Chunks reports the storage chunk shape, Size is the x/y/z image size, and ComponentType maps to the Zarr dtype used by readZarrSlab/writeZarr."
          Aliases = [ "info"; "metadata"; "zarr"; "ome-zarr"; "chunk"; "chunks"; "width"; "height"; "depth"; "size"; "component" ]
          Inputs = []
          Outputs =
              [ makePort "Chunks: int list" intList
                makePort "Size: UInt64 list" uint64List
                makePort "ComponentType: String" (Scalar BasicType.String)
                makePort "NumberOfComponents: UInt32" (Scalar(BasicType.Numeric UInt32))
                makePort "ChunkX: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "ChunkY: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "ChunkZ: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "Width: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Height: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Depth: UInt64" (Scalar(BasicType.Numeric UInt64)) ]
          Parameters =
              [ makeParameter "input" "Name" "input.zarr" BasicType.String
                makeParameter "multiscaleIndex" "Multiscale index" "0" (BasicType.Numeric Int32)
                makeParameter "datasetIndex" "Dataset index" "0" (BasicType.Numeric Int32) ] }

        { Id = "GetNexusInfo"
          DisplayName = "getNexusInfo"
          Category = "Sources / Sinks"
          Summary = "Inspect a NeXus/HDF5 detector stack and expose chunk layout and image dimensions."
          Description = nexusFormatDescription
          Aliases = [ "info"; "metadata"; "nexus"; "hdf5"; "h5"; "chunk"; "chunks"; "width"; "height"; "depth"; "size"; "component" ]
          Inputs = []
          Outputs =
              [ makePort "Chunks: int list" intList
                makePort "Size: UInt64 list" uint64List
                makePort "ComponentType: String" (Scalar BasicType.String)
                makePort "NumberOfComponents: UInt32" (Scalar(BasicType.Numeric UInt32))
                makePort "ChunkX: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "ChunkY: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "ChunkZ: Int32" (Scalar(BasicType.Numeric Int32))
                makePort "Width: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Height: UInt64" (Scalar(BasicType.Numeric UInt64))
                makePort "Depth: UInt64" (Scalar(BasicType.Numeric UInt64)) ]
          Parameters =
              [ makeParameter "input" "Name" "scan.h5" BasicType.String
                makeParameter "datasetPath" "Dataset path" "/entry/data/data" BasicType.String
                makeParameter "frameAxis" "Frame axis" "0" (BasicType.Numeric Int32)
                makeParameter "yAxis" "Y axis" "1" (BasicType.Numeric Int32)
                makeParameter "xAxis" "X axis" "2" (BasicType.Numeric Int32) ] }

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

        { Id = "Quantiles"
          DisplayName = "quantiles"
          Category = "Statistics"
          Summary = "Estimate quantiles from a histogram."
          Description = quantilesDescription
          Aliases = [ "quantile"; "percentile"; "histogram"; "statistics"; "robust"; "range" ]
          Inputs = []
          Outputs =
              [ makePort "q1: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "q2: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "q3: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "q4: Float64" (Scalar(BasicType.Numeric Float64))
                makePort "q5: Float64" (Scalar(BasicType.Numeric Float64)) ]
          Parameters =
              [ makeParameter "histogram" "Histogram" "" BasicType.Map
                makeParameter "q1" "q1" "0.5" (BasicType.Numeric Float64)
                makeParameter "useQ2" "Enable q2" "false" BasicType.Bool
                makeParameter "q2" "q2" "0.01" (BasicType.Numeric Float64)
                makeParameter "useQ3" "Enable q3" "false" BasicType.Bool
                makeParameter "q3" "q3" "0.99" (BasicType.Numeric Float64)
                makeParameter "useQ4" "Enable q4" "false" BasicType.Bool
                makeParameter "q4" "q4" "0.25" (BasicType.Numeric Float64)
                makeParameter "useQ5" "Enable q5" "false" BasicType.Bool
                makeParameter "q5" "q5" "0.75" (BasicType.Numeric Float64) ] }

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

        { Id = "Convolve"
          DisplayName = "convolve"
          Category = "Filters"
          Summary = "Convolve an image stream with a kernel image."
          Description = "Applies the StackProcessing convolve stage. Kernel is a raw F# expression that evaluates to an Image of the same pixel type as the input stream. Output region and Boundary accept the ImageFunctions union case names exposed by the DSL."
          Aliases = [ "convolution"; "kernel"; "filter"; "same"; "valid"; "boundary" ]
          Inputs = [ makePort "Image" imageAny ]
          Outputs = [ makePort "Image" imageAny ]
          Parameters =
              [ makeParameter "kernel" "Kernel" "Image<float>([3u; 3u; 3u])" BasicType.String
                makeParameter "outputRegionMode" "Output region" "None" BasicType.String
                makeParameter "boundaryCondition" "Boundary" "None" BasicType.String
                makeParameter "windowSize" "Window size" "None" BasicType.String ] }

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

        { Id = "Clamp"
          DisplayName = "clamp"
          Category = "Intensity"
          Summary = "Limit image intensities to a lower and upper bound."
          Description = intensityDescription
          Aliases = [ "intensity"; "clip"; "limit"; "range"; "minimum"; "maximum" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "lower" "Lower" "0.0" (BasicType.Numeric Float64)
                makeParameter "upper" "Upper" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "ShiftScale"
          DisplayName = "shiftScale"
          Category = "Intensity"
          Summary = "Apply (input + shift) * scale to image intensities."
          Description = shiftScaleDescription
          Aliases = [ "intensity"; "shift"; "scale"; "offset"; "gain" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "shift" "Shift" "0.0" (BasicType.Numeric Float64)
                makeParameter "scale" "Scale" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "IntensityStretch"
          DisplayName = "intensityStretch"
          Category = "Intensity"
          Summary = "Linearly map one intensity range to another."
          Description = intensityStretchDescription
          Aliases = [ "intensity"; "stretch"; "contrast"; "linear"; "range"; "scale" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "inputMinimum" "Input minimum" "0.0" (BasicType.Numeric Float64)
                makeParameter "inputMaximum" "Input maximum" "1.0" (BasicType.Numeric Float64)
                makeParameter "outputMinimum" "Output minimum" "0.0" (BasicType.Numeric Float64)
                makeParameter "outputMaximum" "Output maximum" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "Median"
          DisplayName = "median"
          Category = "Filters"
          Summary = "Apply a local median denoising filter."
          Description = localDenoiseDescription
          Aliases = [ "denoise"; "median"; "salt"; "pepper"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "Bilateral"
          DisplayName = "bilateral"
          Category = "Filters"
          Summary = "Apply an edge-preserving bilateral denoising filter."
          Description = localDenoiseDescription
          Aliases = [ "denoise"; "bilateral"; "edge"; "preserve"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "domainSigma" "Domain sigma" "2.0" (BasicType.Numeric Float64)
                makeParameter "rangeSigma" "Range sigma" "50.0" (BasicType.Numeric Float64)
                makeParameter "windowSize" "Window size" "7" (BasicType.Numeric UInt32) ] }

        { Id = "GradientMagnitude"
          DisplayName = "gradientMagnitude"
          Category = "Filters"
          Summary = "Compute local gradient magnitude."
          Description = edgeDescription
          Aliases = [ "edge"; "gradient"; "magnitude"; "derivative"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "SobelEdge"
          DisplayName = "sobelEdge"
          Category = "Filters"
          Summary = "Compute a Sobel edge response."
          Description = edgeDescription
          Aliases = [ "edge"; "sobel"; "gradient"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "Laplacian"
          DisplayName = "laplacian"
          Category = "Filters"
          Summary = "Compute a local Laplacian response."
          Description = edgeDescription
          Aliases = [ "edge"; "laplacian"; "second"; "derivative"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "ImageComparison"
          DisplayName = "I cmp J"
          Category = "Segmentation"
          Summary = "Compare two image streams pixelwise and emit a UInt8 mask."
          Description = comparisonDescription
          Aliases = [ "compare"; "mask"; "greater"; "less"; "equal"; "threshold"; "<"; ">"; "=" ]
          Inputs = [ makePort "I" imageAny; makePort "J" imageAny ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "operation" "Operation" ">" BasicType.String
                makeParameter "type" "Type" "Float64" BasicType.String ] }

        { Id = "MaskLogic"
          DisplayName = "M op N"
          Category = "Segmentation"
          Summary = "Combine two UInt8 masks with logical operations."
          Description = "Mask logic operates on UInt8 mask streams and emits a UInt8 mask. Use it to combine threshold, comparison, and segmentation results before masking or writing. Non-zero values are treated as true by the underlying SimpleITK logical filters. The operation selector compiles to andMask, orMask, or xorMask. Use notMask when only a single mask should be inverted."
          Aliases = [ "mask"; "logic"; "and"; "or"; "xor"; "binary"; "boolean" ]
          Inputs = [ makePort "UInt8" imageUInt8; makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "operation" "Operation" "and" BasicType.String ] }

        { Id = "NotMask"
          DisplayName = "notMask"
          Category = "Segmentation"
          Summary = "Invert a UInt8 mask."
          Description = "Inverts a UInt8 mask stream using SimpleITK's logical Not filter. This is intended for masks, not for grayscale intensity inversion. For grayscale negatives, estimate the relevant maximum first and use shiftScale. The operation is slice-local and fits streaming naturally."
          Aliases = [ "mask"; "logic"; "not"; "invert"; "binary"; "boolean" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [] }

        { Id = "Mask"
          DisplayName = "mask"
          Category = "Segmentation"
          Summary = "Apply a UInt8 mask to an image stream."
          Description = maskDescription
          Aliases = [ "mask"; "apply"; "binary"; "select"; "outside"; "segment" ]
          Inputs = [ makePort "Image" imageAny; makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "Image" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "outsideValue" "Outside value" "0.0" (BasicType.Numeric Float64) ] }

        { Id = "GrayscaleErode"
          DisplayName = "grayscaleErode"
          Category = "Grayscale Morphology"
          Summary = "Erode a grayscale image with a local structuring element."
          Description = grayscaleMorphologyDescription
          Aliases = [ "grayscale"; "morphology"; "erode"; "minimum"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "GrayscaleDilate"
          DisplayName = "grayscaleDilate"
          Category = "Grayscale Morphology"
          Summary = "Dilate a grayscale image with a local structuring element."
          Description = grayscaleMorphologyDescription
          Aliases = [ "grayscale"; "morphology"; "dilate"; "maximum"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "GrayscaleOpening"
          DisplayName = "grayscaleOpening"
          Category = "Grayscale Morphology"
          Summary = "Remove small bright grayscale structures."
          Description = grayscaleMorphologyDescription
          Aliases = [ "grayscale"; "morphology"; "opening"; "bright"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "GrayscaleClosing"
          DisplayName = "grayscaleClosing"
          Category = "Grayscale Morphology"
          Summary = "Fill small dark grayscale gaps."
          Description = grayscaleMorphologyDescription
          Aliases = [ "grayscale"; "morphology"; "closing"; "dark"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "WhiteTopHat"
          DisplayName = "whiteTopHat"
          Category = "Grayscale Morphology"
          Summary = "Extract small bright grayscale details."
          Description = grayscaleMorphologyDescription
          Aliases = [ "grayscale"; "morphology"; "white"; "top"; "hat"; "bright"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "BlackTopHat"
          DisplayName = "blackTopHat"
          Category = "Grayscale Morphology"
          Summary = "Extract small dark grayscale details."
          Description = grayscaleMorphologyDescription
          Aliases = [ "grayscale"; "morphology"; "black"; "top"; "hat"; "dark"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "MorphologicalGradient"
          DisplayName = "morphologicalGradient"
          Category = "Grayscale Morphology"
          Summary = "Compute local grayscale morphological contrast."
          Description = grayscaleMorphologyDescription
          Aliases = [ "grayscale"; "morphology"; "gradient"; "edge"; "contrast"; "filter" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "BinaryFillHoles"
          DisplayName = "binaryFillHoles"
          Category = "Binary Morphology"
          Summary = "Fill holes in a binary UInt8 stack."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "holes"; "fill"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "BinaryContour"
          DisplayName = "binaryContour"
          Category = "Binary Morphology"
          Summary = "Extract the contour of a binary UInt8 mask."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "contour"; "edge"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "fullyConnected" "Fully connected" "false" BasicType.Bool
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "BinaryMedian"
          DisplayName = "binaryMedian"
          Category = "Binary Morphology"
          Summary = "Apply a median filter to a binary UInt8 mask."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "median"; "denoise"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "BinaryOpeningByReconstruction"
          DisplayName = "binaryOpeningByReconstruction"
          Category = "Binary Morphology"
          Summary = "Open a binary mask by reconstruction."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "opening"; "reconstruction"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "fullyConnected" "Fully connected" "false" BasicType.Bool
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "BinaryClosingByReconstruction"
          DisplayName = "binaryClosingByReconstruction"
          Category = "Binary Morphology"
          Summary = "Close a binary mask by reconstruction."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "closing"; "reconstruction"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "fullyConnected" "Fully connected" "false" BasicType.Bool
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "BinaryReconstructionByDilation"
          DisplayName = "binaryReconstructionByDilation"
          Category = "Binary Morphology"
          Summary = "Reconstruct a binary marker under a binary mask by dilation."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "reconstruction"; "dilation"; "marker"; "mask" ]
          Inputs = [ makePort "Marker" imageUInt8; makePort "Mask" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "fullyConnected" "Fully connected" "false" BasicType.Bool ] }

        { Id = "BinaryReconstructionByErosion"
          DisplayName = "binaryReconstructionByErosion"
          Category = "Binary Morphology"
          Summary = "Reconstruct a binary marker over a binary mask by erosion."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "reconstruction"; "erosion"; "marker"; "mask" ]
          Inputs = [ makePort "Marker" imageUInt8; makePort "Mask" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "fullyConnected" "Fully connected" "false" BasicType.Bool ] }

        { Id = "VotingBinaryHoleFilling"
          DisplayName = "votingBinaryHoleFilling"
          Category = "Binary Morphology"
          Summary = "Fill binary holes by neighborhood voting."
          Description = binaryMorphologyDescription
          Aliases = [ "morphology"; "binary"; "voting"; "holes"; "fill"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32)
                makeParameter "majorityThreshold" "Majority threshold" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

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

        { Id = "RelabelComponents"
          DisplayName = "relabelComponents"
          Category = "Binary Morphology"
          Summary = "Relabel connected-component labels and remove small components."
          Description = ""
          Aliases = [ "components"; "labels"; "relabel"; "size"; "filter" ]
          Inputs = [ makePort "UInt64" imageUInt64 ]
          Outputs = [ makePort "UInt64" imageUInt64 ]
          Parameters =
              [ makeParameter "minimumObjectSize" "Minimum object size" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "Watershed"
          DisplayName = "watershed"
          Category = "Segmentation"
          Summary = "Apply watershed segmentation."
          Description = ""
          Aliases = [ "segmentation"; "watershed"; "labels"; "basins" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "level" "Level" "0.0" (BasicType.Numeric Float64)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "SignedDistanceMap"
          DisplayName = "signedDistanceMap"
          Category = "Segmentation"
          Summary = "Compute a band-limited signed distance map from a binary UInt8 image."
          Description =
            "Computes signed distances in streaming z-windows. Each window has depth stride + 2 * bandRadius, and only the center stride slices are emitted. Input mask values different from 0 are treated as object pixels. Distances inside the object are negative and distances outside are positive. Values whose absolute distance is not less than bandRadius are set to NaN because they may depend on object boundaries outside the current window."
          Aliases = [ "distance"; "map"; "signed"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "bandRadius" "Band radius" "8" (BasicType.Numeric UInt32)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "OtsuThreshold"
          DisplayName = "otsuThreshold"
          Category = "Segmentation"
          Summary = "Threshold an image stack using Otsu's method."
          Description =
            "Estimates a single Otsu threshold from a random sample of input slices before the streaming pass starts.\n\nThe sampled pixel values are binned into the requested number of histogram bins. The threshold is chosen by brute-force maximization of Otsu's between-class variance over those bins, then applied to the full stream with the ordinary binary threshold stage.\n\nIncrease the sample count for stacks whose foreground/background balance changes strongly along z. Increase the bin count for broad continuous-valued images, but keep it modest enough that the sampled histogram remains stable. Output pixels are UInt8, with values at or above the estimated threshold set to 1 and lower values set to 0."
          Aliases = [ "threshold"; "otsu"; "binary"; "mask"; "segment" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sampleCount" "Sample slices" "16" (BasicType.Numeric UInt32)
                makeParameter "bins" "Bins" "256" (BasicType.Numeric UInt32) ] }

        { Id = "MomentsThreshold"
          DisplayName = "momentsThreshold"
          Category = "Segmentation"
          Summary = "Threshold an image stack using moment-preserving thresholding."
          Description =
            "Estimates a single moment-preserving threshold from a random sample of input slices before the streaming pass starts.\n\nThe sampled pixel values are binned into the requested number of histogram bins. The threshold is estimated from the first three histogram moments using Tsai's moment-preserving method, then applied to the full stream with the ordinary binary threshold stage.\n\nIncrease the sample count for stacks whose class balance changes strongly along z. Increase the bin count for broad continuous-valued images, but keep it modest enough that the sampled histogram remains stable. Output pixels are UInt8, with values at or above the estimated threshold set to 1 and lower values set to 0."
          Aliases = [ "threshold"; "moments"; "binary"; "mask"; "segment" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sampleCount" "Sample slices" "16" (BasicType.Numeric UInt32)
                makeParameter "bins" "Bins" "256" (BasicType.Numeric UInt32) ] }

        { Id = "ComponentTranslationTable"
          DisplayName = "componentTranslationTable"
          Category = "Binary Morphology"
          Summary = "Reduce connected-component label slabs plus object counts to a chunk translation table."
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

        { Id = "LabelShapeStatistics"
          DisplayName = "labelShapeStatistics"
          Category = "Segmentation"
          Summary = "Measure object geometry in a labeled image."
          Description = labelAnalysisDescription
          Aliases = [ "label"; "statistics"; "shape"; "objects"; "measure" ]
          Inputs = [ makePort "Labels" imageAny ]
          Outputs = [ makePort "Shape statistics" (Custom "LabelShapeStatistics") ]
          Parameters =
              [ makeParameter "type" "Type" "UInt64" BasicType.String
                makeParameter "windowSize" "Window size" "8" (BasicType.Numeric UInt32) ] }

        { Id = "LabelIntensityStatistics"
          DisplayName = "labelIntensityStatistics"
          Category = "Segmentation"
          Summary = "Measure intensities inside labeled regions."
          Description = labelAnalysisDescription
          Aliases = [ "label"; "statistics"; "intensity"; "objects"; "measure" ]
          Inputs = [ makePort "Labels" imageAny; makePort "Intensity" imageAny ]
          Outputs = [ makePort "Intensity statistics" (Custom "LabelIntensityStatistics") ]
          Parameters =
              [ makeParameter "labelType" "Label type" "UInt64" BasicType.String
                makeParameter "intensityType" "Intensity type" "Float64" BasicType.String ] }

        { Id = "LabelOverlapMeasures"
          DisplayName = "labelOverlapMeasures"
          Category = "Segmentation"
          Summary = "Compare overlap between two labeled images."
          Description = labelAnalysisDescription
          Aliases = [ "label"; "statistics"; "overlap"; "dice"; "jaccard"; "measure" ]
          Inputs = [ makePort "Source" imageAny; makePort "Target" imageAny ]
          Outputs = [ makePort "Overlap measures" (Custom "LabelOverlapMeasures") ]
          Parameters = [ makeParameter "type" "Type" "UInt64" BasicType.String ] }

        { Id = "LabelContour"
          DisplayName = "labelContour"
          Category = "Segmentation"
          Summary = "Extract contours from a labeled image."
          Description = labelAnalysisDescription
          Aliases = [ "label"; "contour"; "boundary"; "objects"; "edge" ]
          Inputs = [ makePort "Labels" imageAny ]
          Outputs = [ makePort "Labels" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "UInt64" BasicType.String
                makeParameter "fullyConnected" "Fully connected" "false" BasicType.Bool
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "ChangeLabel"
          DisplayName = "changeLabel"
          Category = "Segmentation"
          Summary = "Replace one label value with another."
          Description = labelAnalysisDescription
          Aliases = [ "label"; "change"; "map"; "replace"; "relabel" ]
          Inputs = [ makePort "Labels" imageAny ]
          Outputs = [ makePort "Labels" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "UInt64" BasicType.String
                makeParameter "fromLabel" "From label" "1.0" (BasicType.Numeric Float64)
                makeParameter "toLabel" "To label" "0.0" (BasicType.Numeric Float64) ] }

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

        { Id = "Resize"
          DisplayName = "resize"
          Category = "Geometry"
          Summary = "Axis-aligned streaming resize to an explicit x/y/z size."
          Description = "Resizes an image stack without rotation. Each x-y slice is resampled independently with SimpleITK using nearest-neighbor or linear interpolation, then z is interpolated from a streaming two-slice window. The first input coordinate (0,0,0) is always included in the output. No pre-filtering is applied, so downsampling can alias; this is intentional for speed and to keep x-y and z behavior comparable."
          Aliases = [ "resize"; "resample"; "scale"; "axis"; "aligned"; "nearest"; "linear"; "geometry" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float32" BasicType.String
                makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
                makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
                makeParameter "depth" "Depth" "64" (BasicType.Numeric UInt32)
                makeParameter "interpolation" "Interpolation" "Linear" BasicType.String ] }

        { Id = "Resample"
          DisplayName = "resample"
          Category = "Geometry"
          Summary = "Axis-aligned streaming resampling by x/y/z scale factors."
          Description = "Resamples an image stack by positive x/y/z factors. Factors larger than 1 make the stack larger and factors below 1 make it smaller. Each x-y slice is resampled independently with SimpleITK using nearest-neighbor or linear interpolation, then z is interpolated from a streaming two-slice window. The first input coordinate (0,0,0) is always included in the output. No pre-filtering is applied, so downsampling can alias; this is intentional for speed and to keep x-y and z behavior comparable."
          Aliases = [ "resample"; "resize"; "scale"; "axis"; "aligned"; "nearest"; "linear"; "geometry" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float32" BasicType.String
                makeParameter "factorX" "Factor X" "1.0" (BasicType.Numeric Float64)
                makeParameter "factorY" "Factor Y" "1.0" (BasicType.Numeric Float64)
                makeParameter "factorZ" "Factor Z" "1.0" (BasicType.Numeric Float64)
                makeParameter "interpolation" "Interpolation" "Linear" BasicType.String ] }

        { Id = "ResampleAffineTrilinearSlices"
          DisplayName = "resampleAffineTrilinearSlices"
          Category = "Geometry"
          Summary = "Run the chunked affine trilinear resampler over chunk files."
          Description = "Advanced chunk utility. This DSL function returns a sequence of output slices rather than a streaming Plan stage, so the Studio box is terminal and exposes the geometry, affine, interpolation, and background arguments as raw F# expressions."
          Aliases = [ "resample"; "affine"; "trilinear"; "chunks"; "geometry"; "transform" ]
          Inputs = []
          Outputs = []
          Parameters =
              [ makeParameter "type" "Type" "Float32" BasicType.String
                makeParameter "input" "Input" "input" BasicType.String
                suffixParameter ".tiff"
                makeParameter "lerp" "Lerp" "(fun a b t -> a + (b - a) * t)" BasicType.String
                makeParameter "windowSize" "Window size" "8" (BasicType.Numeric Int32)
                makeParameter "inputGeometry" "Input geometry" "{ W = 64; H = 64; D = 64; Origin = TinyLinAlg.v3 0.0 0.0 0.0; Spacing = TinyLinAlg.v3 1.0 1.0 1.0; Direction = { m00 = 1.0; m01 = 0.0; m02 = 0.0; m10 = 0.0; m11 = 1.0; m12 = 0.0; m20 = 0.0; m21 = 0.0; m22 = 1.0 } }" BasicType.String
                makeParameter "outputGeometry" "Output geometry" "{ W = 64; H = 64; D = 64; Origin = TinyLinAlg.v3 0.0 0.0 0.0; Spacing = TinyLinAlg.v3 1.0 1.0 1.0; Direction = { m00 = 1.0; m01 = 0.0; m02 = 0.0; m10 = 0.0; m11 = 1.0; m12 = 0.0; m20 = 0.0; m21 = 0.0; m22 = 1.0 } }" BasicType.String
                makeParameter "affine" "Affine" "{ A = { m00 = 1.0; m01 = 0.0; m02 = 0.0; m10 = 0.0; m11 = 1.0; m12 = 0.0; m20 = 0.0; m21 = 0.0; m22 = 1.0 }; T = TinyLinAlg.v3 0.0 0.0 0.0; C = TinyLinAlg.v3 0.0 0.0 0.0 }" BasicType.String
                makeParameter "background" "Background" "0.0f" (BasicType.Numeric Float32) ] }

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
