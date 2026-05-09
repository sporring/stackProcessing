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
  let vectorImageFloat64 = PortType.Custom "VectorImageFloat64"
  let translationTable = PortType.Custom "TranslationTable"
  let connectedComponentLabels = PortType.Tuple(imageUInt64, PortType.Scalar(BasicType.Numeric UInt64))
  let mesh = PortType.Custom "Mesh"
  let pointSet = PortType.Custom "PointSet"
  let float64Matrix = PortType.Custom "Float64Matrix"
  let biasModel = PortType.Custom "BiasModel"
  let serialSliceManifest = PortType.Custom "SerialSliceManifest"
  let serialTransPair = PortType.Tuple(imageAny, serialSliceManifest)
  let streamedObjects = PortType.Custom "StreamedObjects"
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

  let private readVolumeDescription =
      "Reads a single 2D or 3D volume file as a normal slice stream. Multipage TIFF/BigTIFF pages are read forward-only through the streaming TIFF reader; other volume formats use SimpleITK's extract-region reader so each z-slice is requested independently rather than loading the full volume first. The Type selector is the emitted stream type, so readable TIFF pixel types are cast to that type after page import."

  let private writeFormatDescription =
      "Writes one image per stack slice. The selected format controls which image types can be connected to the input pin: TIFF supports common 8/16/32-bit integer and 32/64-bit floating-point scalar images; PNG supports UInt8 and UInt16; JPEG and BMP support UInt8; MetaImage, NRRD, and NIfTI support the broad scalar numeric set used by Studio. Cast before write when a format cannot store the current image type."

  let private writeVolumeDescription =
      "Writes a slice stream as one volume file. The current streaming implementation writes multipage TIFF or BigTIFF pages incrementally, without materializing the full stack in memory. Use .btf or .bigtiff for BigTIFF output."

  let private chunkWriteFormatDescription =
      "Writes stack slabs split into chunk files for later slab reading. The selected format controls which image types can be connected to the input pin, using the same constraints as write. TIFF output uses the exact selected suffix, either .tif or .tiff."

  let private zarrFormatDescription =
      "Reads or writes an OME-Zarr volume through ZarrNET. The current native .NET implementation is used here for UInt8 and UInt16 scalar images. readZarrSlab serves 2D slices from a selected timepoint/channel/resolution; writeZarr writes a single timepoint/channel volume and exposes chunk sizes and physical voxel spacing so Studio can be used as a stack-to-Zarr converter."

  let private nexusFormatDescription =
      "Reads a rank-3 NeXus/HDF5 detector stack through PureHDF using an explicit dataset path and axis mapping. This covers common MAX IV and ESRF detector-stack layouts while keeping streaming slice reads larger-than-memory friendly. Compressed detector files that use external HDF5 filters may require a later native/plugin fallback."

  let private pointSetDescription =
      "Reads and writes coordinate point sets as clean CSV. The header is x,y,z,scale,response. x, y, and z are image coordinates; scale and response are optional when reading and are written as floating-point values. This format is intentionally simple so point detections can be exchanged with Python, R, spreadsheets, and visualization tools without custom parsers."

  let private csvWriteDescription =
      "Writes a non-image data stream to CSV. Use the data kind selector to choose whether the input is a PointSet, a Float64 matrix, or a histogram map. Histogram rows are written as key,count and are sorted by key."

  let private scalarDescription =
      "Creates a named scalar value that can be connected to parameter inputs on other boxes.\n\nUse it for thresholds, scale factors, file names, booleans, or other values that should be visible in the graph instead of typed directly into a parameter field.\n\nFor numeric types, the names pi and e are accepted and compile to the standard mathematical constants. For strings, pi and e remain ordinary text."

  let private fileDirectoryDescription =
      "Prompts for a file or directory when the graph is run.\n\nUse this when the path should be chosen interactively instead of saved as fixed text in the graph. The selected path is emitted as a string and can be linked to read, write, or metadata boxes.\n\nIf the picker is cancelled, the run is stopped before processing starts."

  let private scalarArithmeticDescription =
      "Combines two scalar values with a simple arithmetic operation.\n\nThe result is another scalar that can be used as a parameter elsewhere in the graph. This is useful for derived constants, such as converting a measured range into a scale factor or combining quantile outputs.\n\nBoth inputs are ordinary parameters unless linked from another scalar-producing box."

  let private scalarFunctionDescription =
      "Applies one standard mathematical function to a scalar value.\n\nExamples include sqrt, abs, log, exp, sin, cos, and square. Use this when a parameter should be derived from another scalar rather than typed by hand.\n\nThe output is a Float64 scalar and can be linked to numeric parameter fields."

  let private castDescription =
      "Converts every image slice from one numeric pixel type to another.\n\nUse cast before writing to formats that only support certain pixel types, or before filters that expect a particular type. Values are converted using the normal SimpleITK cast behavior, so converting from floating point to integer can round or truncate and may lose precision.\n\nThe image size and slice order are unchanged."

  let private zeroDescription =
      "Creates a synthetic stack filled with zero-valued pixels.\n\nThis is useful for testing, for generating blank masks, or for building small example graphs without reading files. Width and height define each 2D slice, and depth defines the number of slices in the stack.\n\nChoose the output type to match the filters or writer that follow."

  let private coordinateDescription axis =
      $"Creates a synthetic Float64 image stack where every pixel stores its {axis}-coordinate. Width and height define the slice shape, depth defines the number of emitted slices, and z is taken from the slice index. These sources are useful when building coordinate-aware models and diagnostics."

  let private biasModelDescription =
      "Fits a low-order 3D polynomial bias field to streamed image slices. The z-coordinate is the slice index, so this works naturally with readRandom: sample slices to estimate the model, then connect the resulting BiasModel to correctBias on the full stream. The masked variant only uses non-zero mask pixels for fitting."

  let private correctBiasDescription =
      "Subtracts a fitted polynomial BiasModel from each streamed slice and emits Float64 corrected images. The masked variant only subtracts inside non-zero mask pixels and leaves pixels outside the mask unchanged."

  let private serialSectionsDescription =
      "Serial-section tools operate slice-wise for stacks acquired by physical sectioning, shaving, cutting, or other per-slice interactions with the material. They keep z as an ordered section index and use serial* names to distinguish acquisition correction from ordinary 3D image filtering."

  let private euler2DDescription =
      "Creates a synthetic stack by transforming a simple seed image through a sequence of 2D Euler transforms.\n\nIt is mainly a demonstration and testing source for registration, resampling, and motion-like examples. The generated stack contains a box-like object whose position changes according to the selected transform pattern.\n\nUse read boxes for real experimental data."

  let private scalarImageOperationDescription =
      "Applies an arithmetic operation between every image pixel and one scalar value.\n\nI op a means the image value is on the left, for example I - a or I / a. a op I means the scalar is on the left, for example a - I or a / I.\n\nUse these boxes for offsets, gains, ratios, and simple intensity formulas where the same scalar is applied to the whole stack."

  let private pairImageOperationDescription =
      "Combines two synchronized image streams pixel by pixel.\n\nBoth inputs should have the same selected numeric type and compatible image geometry. The operation can add, subtract, multiply, divide, take a pixelwise maximum, or take a pixelwise minimum.\n\nUse this for image arithmetic such as ratios, differences, masks represented as 0/1 images, or combining two processed versions of the same stack."

  let private normalNoiseDescription =
      "Adds normally distributed random noise to each image slice.\n\nThe mean parameter controls the average added value, and std controls the noise spread. This is mainly useful for simulations, robustness tests, and example graphs.\n\nFor integer images, remember that the result is converted back to the selected pixel type, so values may be clipped or rounded by the image representation."

  let private tapDescription =
      "Prints each value that passes through the box and then forwards it unchanged.\n\nUse tap while debugging a graph to check that a branch is producing values at the expected point. The label parameter is printed together with the value, which makes several taps easier to tell apart.\n\nFor large image streams, prefer using tap sparingly because printing every element can slow a run."

  let private printDescription =
      "Prints scalar values from the generated program.\n\nUse the format text to decide how the values appear; placeholders such as {input1} are replaced by linked inputs. This is useful for reporting computed statistics, thresholds, quantiles, or file information at the end of a run.\n\nIt is intended for scalar summaries, not for printing image pixels."

  let private histogramDescription =
      "Computes an intensity histogram from the image stream and immediately shows it as a chart.\n\nThe x-axis is the pixel value or histogram bin key, and the y-axis is the number of pixels observed. Use this to inspect intensity distributions before choosing thresholds, stretches, or quantiles.\n\nFor very large or continuous-valued images, consider sampling first so the chart stays readable."

  let private histogramDataDescription =
      "Computes an intensity histogram and emits it as data instead of displaying it directly.\n\nThis is the preferred input to quantiles, otsuThresholdFromHistogram, and momentsThresholdFromHistogram. The histogram is reduced over the whole connected stream, so the output is available after the histogram branch has been drained.\n\nUse readRandom or readRange upstream when an estimated histogram is enough."

  let private estimateHistogramDescription =
      "Randomly samples whole slices from an image stack, downsamples pixels within those slices, and emits an estimated histogram plus diagnostics. This is a reducer-style source/sink: it does not stream sampled images onward.\n\nType selects the pixel type used to read the sampled slices. Slices is the number of randomly chosen stack slices. Input and suffix identify the image stack. Down is the in-slice pixel stride: 1 uses every pixel in the sampled slices, 2 uses every second x/y pixel, and so on. Estimator selects DKW, Holdout, or DKWAndHoldout. Confidence is used by DKW to report a distribution-free half-width for the empirical CDF.\n\nThe histogram output can feed quantiles, threshold estimators, chart, writeCSV, or histogramEqualization. Samples reports the number of pixels used. CDF half-width is the DKW epsilon at the selected confidence. Holdout max CDF delta splits sampled pixels into two alternating halves and reports the maximum CDF difference between them."

  let private chartDescription =
      "Displays map-like data as a Plotly chart.\n\nUse it with histogramData, object-size histograms, or other key/value summaries. The kind parameter selects the visual style, such as column, scatter, line, area, pie, or doughnut.\n\nChart is a visualization sink: it helps inspect results and does not produce image data for later boxes."

  let private showImageDescription =
      "Shows each incoming image slice as a heatmap.\n\nThis is useful for quick visual inspection of small or sampled stacks. For large stacks, display can be slower than the processing itself, so use it on short ranges or diagnostic branches.\n\nFor final output, use write instead."

  let private sumProjectionDescription =
      "Builds one 2D projection image by summing intensities through the stack direction.\n\nUse it as a quick volume visualizer before showImage or write. The function selector transforms each pixel before it is added: Identity sums raw intensity, Abs sums magnitude, Square emphasizes strong signals, SqrtAbs compresses bright values, and Log1pAbs gives a logarithmic-looking projection without failing on negative values.\n\nThe output is Float64 and contains one image slice."

  let private unaryImageFunctionDescription =
      "Applies one standard mathematical function independently to every pixel.\n\nExamples include sqrt, abs, log, exp, sin, cos, and square. The image geometry and slice order are unchanged, only the pixel values are transformed.\n\nUse this for simple intensity formulas that do not depend on neighboring pixels."

  let private gaussianDescription =
      "Smooths an image with a Gaussian-shaped neighborhood.\n\nSigma controls the blur width: larger sigma removes larger-scale noise and softens edges more strongly. Boundary and output-region settings control how pixels near the edge of the available volume are treated.\n\nUse Gaussian smoothing before derivatives, feature detection, or thresholding when small noise should be suppressed."

  let private convolveDescription =
      "Applies a user-specified convolution kernel to the image stack.\n\nThe kernel defines how neighboring pixels are weighted and summed, so this box can implement smoothing, sharpening, derivatives, or custom local filters. Output region controls whether edge pixels are preserved or trimmed, and boundary controls how missing neighborhood values are handled.\n\nFor most users, smoothWGauss or finiteDiff are easier starting points."

  let private finiteDiffDescription =
      "Computes the smallest centered finite-difference derivative estimator.\n\nUse it to emphasize changes along selected axes, such as edges, ridges, or directional gradients. Smooth the image explicitly before this box when derivative noise should be suppressed.\n\nThe result is an intensity image that often needs scaling, thresholding, or visualization before interpretation."

  let private dogKeypointsDescription =
      "Detects local Difference-of-Gaussian extrema in streaming z-windows. Each window builds a small 3D Gaussian scale space, subtracts adjacent scales, and emits points that are strict local maxima or minima in x, y, z, and scale. Only the center stride slices of each window are emitted, while the z padding covers the largest Gaussian support and the one-slice extrema neighborhood.\n\nThis is a detector only, not a complete SIFT descriptor implementation. Coordinates are reported in pixel units and z uses the source slice index. Increase scale levels or sigma for larger features; increase contrast threshold to suppress weak/noisy extrema."

  let private siftKeypointsDescription =
      "Detects SIFT-style keypoint locations as local Difference-of-Gaussian extrema in streaming z-windows. The output is a point set containing x, y, z, scale, and response. Descriptor vectors and canonical orientations are not emitted by the current point-set representation."

  let private streamingKeypointDescription =
      "Detects 3D keypoints with a bounded z-window local operator and emits x, y, z, scale, and response as a PointSet. These detectors are streaming-friendly: they never need the full stack in memory, but the selected scale controls the window padding."

  let private streamedObjectsDescription =
      "Streams completed connected objects from a binary mask. Each input slice is inspected for non-zero foreground pixels, object fronts touching the advancing z-boundary are carried forward, and objects are emitted once the next slice proves they cannot continue. Six-connectivity uses face contacts only; TwentySix-connectivity also allows diagonal contacts. paintObjects converts the emitted integer positions back into UInt8 mask slices with value 1 at object pixels and 0 elsewhere."

  let private thresholdDescription =
      "Turns a numeric image into a UInt8 binary mask.\n\nPixels between lower and upper, including the limits, become foreground; pixels outside the range become background. Use infinity as the upper limit when you want a simple lower-threshold operation.\n\nThresholds can be typed directly or linked from computeStats, quantiles, otsuThresholdFromHistogram, or momentsThresholdFromHistogram."

  let private binaryShapeDescription =
      "Changes the shape of a UInt8 binary mask using a local neighborhood.\n\nErode removes foreground pixels near object boundaries and can break thin connections. Dilate expands foreground regions and can close small gaps. Opening is erosion followed by dilation and tends to remove small bright objects. Closing is dilation followed by erosion and tends to fill small dark gaps.\n\nThe radius controls the neighborhood size."

  let private connectedComponentsDescription =
      "Labels connected foreground regions in a binary mask.\n\nThe output image stores an integer label for each component, with background left as zero. The count output reports how many local components were found before any later global relabeling.\n\nUse componentTranslationTable and collapseComponentLabels when labels need to be made consistent across streamed slabs."

  let private relabelComponentsDescription =
      "Renumbers connected-component labels and removes components below a chosen size.\n\nThis is used after connectedComponents when small labeled objects should be discarded and the remaining labels should be compacted. The minimum object size is measured in voxels.\n\nFor direct cleanup of binary masks, removeSmallObjects is usually the simpler box."

  let private collapseLabelsDescription =
      "Applies a connected-component translation table to a labeled image stream.\n\nUse it after componentTranslationTable to turn slab-local labels into consistent whole-stack labels. Background remains zero, while labels that belong to the same physical object are mapped to the same final value.\n\nThis is part of the connected-component workflow for larger-than-memory stacks."

  let private permuteAxesDescription =
      "Reorders the x, y, and z axes of a stack.\n\nUse this when detector data or intermediate results need a different orientation, for example turning z into x or swapping x and y. Axis permutation can require chunked access because changing the z-axis changes which pixels belong to each output slice.\n\nTile size controls the working block size used during the transpose."

  let private intensityDescription =
      "Intensity filters change the numeric values of each pixel without changing the stack geometry. They are slice-local and therefore fit the streaming model naturally. Clamp limits values to a range. ShiftScale applies (input + shift) * scale. Intensity stretch maps a selected input range linearly to a selected output range."

  let private shiftScaleDescription =
      "Uses SimpleITK's ShiftScaleImageFilter. Each output pixel is computed as (input + shift) * scale, and the image geometry is unchanged. Use this with computeStats for two-pass stack normalization: first compute mean and standard deviation, then set shift = -mean and scale = 1/std. The operation itself is streaming-friendly because the shift and scale are already known when the image pass starts."

  let private intensityStretchDescription =
      "Linearly maps an input intensity range to an output intensity range using the same shift/scale semantics as shiftScale. Values are not clipped: pixels outside the input range continue linearly outside the output range. This is useful with computeStats or quantiles when the source min/max or robust quantile limits are estimated in an earlier reducer pass."

  let private histogramEqualizationDescription =
      "Applies 3D histogram equalization from a histogram map estimated over the connected stack or a representative sample. The stage streams slices and emits Float64 values in the range 0..1. Histogram bins are treated as an empirical cumulative distribution, with interpolation between sampled bin keys so readRandom estimates can still be used on continuous-valued images."

  let private quantilesDescription =
      "Estimates quantile values from a histogram map. q1 is always emitted. q2, q3, q4, and q5 are optional output slots controlled by the corresponding enabled parameters. Each q value must be between 0 and 1. The result is based on the cumulative histogram counts, so accuracy depends on the histogram key resolution."

  let private computeStatsDescription =
      "Computes whole-stream summary statistics for an image stack.\n\nThe outputs include pixel count, mean, standard deviation, minimum, maximum, sum, and sum of squares. Use these values to choose thresholds, normalize intensities, or report basic measurements.\n\nBecause the result summarizes the input stream, it is usually used on a separate branch before a second processing pass that applies the chosen parameters."

  let private localDenoiseDescription =
      "These denoising filters are local-neighborhood smoothing operations rather than global iterative solvers. smoothWMedian uses a radius in x, y, and z and is streamed through windows large enough to cover the z-neighborhood. smoothWBilateral is edge-preserving and can be slower; use the window size to give the z-neighborhood enough context. No recursive Gaussian, curvature-flow, or anisotropic-diffusion filters are included here because their iteration/global-dependency structure is less friendly to LMIP streaming."

  let private edgeDescription =
      "These edge and derivative-like filters are local operators that can be evaluated on streaming z-windows. Gradient magnitude estimates local change strength. Sobel emphasizes edges using a small derivative stencil. Laplacian computes a second-derivative response. Recursive Gaussian and Canny variants are intentionally not included in this first pass because they are less obviously aligned with the 3D LMIP streaming model."

  let private comparisonDescription =
      "Compare two synchronized image streams pixel by pixel and emit a UInt8 mask. Pixels where the comparison is true become non-zero, and false pixels become zero. The two inputs must have the same selected numeric type and compatible geometry. This is the mask-building counterpart to I op J and compiles to the corresponding StackProcessing comparison stage."

  let private grayscaleMorphologyDescription =
      "Grayscale morphology applies min/max-style neighborhood operations to intensity images rather than binary masks. Erode darkens or shrinks bright structures; dilate brightens or expands them. Opening removes small bright structures, closing fills small dark gaps, white top-hat extracts bright details smaller than the structuring element, black top-hat extracts dark details, and morphological gradient emphasizes local contrast boundaries. These are local filters and are streamed through z-windows large enough to cover the selected radius."

  let private binaryMorphologyDescription =
      "Binary morphology operates on UInt8 masks where non-zero pixels are treated as foreground.\n\nUse these boxes to clean masks, adjust object boundaries, extract contours, and remove or fill small connected regions. Radius-based operations use a local neighborhood, while removeSmallObjects and fillSmallHoles make decisions from connected component size.\n\nChoose connectivity carefully: Six uses face contact, while TwentySix also treats diagonal contact as connected."

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
      Description = scalarDescription
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
      Description = fileDirectoryDescription
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
      Description = scalarArithmeticDescription
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
      Description = scalarFunctionDescription
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

  let makeGenericReadVolume () =
    { Id = "ReadVolume"
      DisplayName = "readVolume"
      Category = "Sources / Sinks"
      Summary = "Read one volume file as a slice stream."
      Description = readVolumeDescription
      Aliases = [ "volume"; "multipage"; "tiff"; "bigtiff"; "ome"; "nrrd"; "nifti"; "mha"; "source"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "input" "Input" "volume.tiff" BasicType.String ] }

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

  let makeGenericEstimateHistogram () =
    { Id = "EstimateHistogram"
      DisplayName = "estimateHistogram"
      Category = "Statistics"
      Summary = "Randomly sample slices and estimate a histogram with diagnostics."
      Description = estimateHistogramDescription
      Aliases = [ "histogram"; "estimate"; "random"; "confidence"; "dkw"; "holdout"; "sample"; "cdf"; "type" ]
      Inputs = []
      Outputs =
          [ makePort "Map" (Scalar BasicType.Map)
            makePort "Samples: UInt64" (Scalar(BasicType.Numeric UInt64))
            makePort "CDF half-width: Float64" (Scalar(BasicType.Numeric Float64))
            makePort "Holdout max CDF delta: Float64" (Scalar(BasicType.Numeric Float64)) ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "slices" "Slices" "16" (BasicType.Numeric UInt32)
            makeParameter "input" "Input" "input" BasicType.String
            suffixParameter ".tiff"
            makeParameter "down" "Down" "4" (BasicType.Numeric UInt32)
            makeParameter "estimator" "Estimator" "DKWAndHoldout" BasicType.String
            makeParameter "confidence" "Confidence" "0.95" (BasicType.Numeric Float64) ] }

  let makeGenericReadRange () =
    { Id = "ReadRange"
      DisplayName = "readRange"
      Category = "Sources / Sinks"
      Summary = "Read a clamped range of stack files."
      Description = "Reads a regular subset of a stack as first, first+step, first+2*step and so on, stopping at or before last. Indices are zero-based. First and last are clamped to the available stack range, and last accepts Matlab-like notation: end is the final image, end-1 is the second-to-last image, and so on. Step must be non-zero; use a negative step to read backwards."
      Aliases = [ "range"; "subset"; "input"; "end"; "matlab"; "tiff"; "file"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "first" "First" "0" BasicType.String
            makeParameter "step" "Step" "1" (BasicType.Numeric Int32)
            makeParameter "last" "Last" "end" BasicType.String
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
      Description = castDescription
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
      Description = zeroDescription
      Aliases = [ "empty"; "synthetic"; "source"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32) ] }

  let makeGenericNormalNoise () =
    { Id = "NormalNoise"
      DisplayName = "normalNoise"
      Category = "Sources / Sinks"
      Summary = "Create a synthetic stack of normally distributed noise."
      Description = "Creates a synthetic image stack with normally distributed pixel values. This is the source-form sibling of addNormalNoise: it behaves like zero followed by addNormalNoise, using the requested width, height, depth, mean, and standard deviation."
      Aliases = [ "noise"; "random"; "normal"; "synthetic"; "source"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
            makeParameter "mean" "Mean" "0.0" (BasicType.Numeric Float64)
            makeParameter "std" "Std" "1.0" (BasicType.Numeric Float64) ] }

  let makeGenericSaltAndPepperNoise () =
    { Id = "SaltAndPepperNoise"
      DisplayName = "saltAndPepperNoise"
      Category = "Sources / Sinks"
      Summary = "Create a synthetic salt-and-pepper noise stack."
      Description = "Creates a synthetic image stack by applying SimpleITK's salt-and-pepper noise filter to zero-valued slices. Probability controls how often pixels are replaced by salt or pepper values."
      Aliases = [ "noise"; "random"; "salt"; "pepper"; "impulse"; "synthetic"; "source"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
            makeParameter "probability" "Probability" "0.01" (BasicType.Numeric Float64) ] }

  let makeGenericShotNoise () =
    { Id = "ShotNoise"
      DisplayName = "shotNoise"
      Category = "Sources / Sinks"
      Summary = "Create a synthetic shot-noise stack."
      Description = "Creates a synthetic image stack by applying SimpleITK's shot-noise filter to zero-valued slices. Scale controls the shot-noise scale used by the native filter."
      Aliases = [ "noise"; "random"; "shot"; "poisson"; "synthetic"; "source"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
            makeParameter "scale" "Scale" "1.0" (BasicType.Numeric Float64) ] }

  let makeGenericSpeckleNoise () =
    { Id = "SpeckleNoise"
      DisplayName = "speckleNoise"
      Category = "Sources / Sinks"
      Summary = "Create a synthetic speckle-noise stack."
      Description = "Creates a synthetic image stack by applying SimpleITK's speckle-noise filter to zero-valued slices. Std controls the standard deviation used by the native filter."
      Aliases = [ "noise"; "random"; "speckle"; "multiplicative"; "synthetic"; "source"; "UInt8"; "Float64"; "type" ]
      Inputs = []
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ availableMemoryParameter
            makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
            makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
            makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32)
            makeParameter "std" "Std" "1.0" (BasicType.Numeric Float64) ] }

  let makeGenericCreateByEuler2DTransform () =
    { Id = "CreateByEuler2DTransform"
      DisplayName = "createByEuler2DTransform"
      Category = "Sources / Sinks"
      Summary = "Create a synthetic stack by applying an Euler 2D transform to a seed image."
      Description = euler2DDescription
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
      Description = scalarImageOperationDescription
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
      Description = normalNoiseDescription
      Aliases = [ "noise"; "random"; "normal"; "statistics"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "mean" "Mean" "128.0" (BasicType.Numeric Float64)
            makeParameter "std" "Std" "50.0" (BasicType.Numeric Float64) ] }

  let makeGenericAddSaltAndPepperNoise () =
    { Id = "AddSaltAndPepperNoise"
      DisplayName = "addSaltAndPepperNoise"
      Category = "Statistics"
      Summary = "Add salt-and-pepper noise to each image."
      Description = "Applies SimpleITK's salt-and-pepper noise filter to each image slice. Probability controls how often pixels are replaced by salt or pepper values."
      Aliases = [ "noise"; "random"; "salt"; "pepper"; "impulse"; "statistics"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "probability" "Probability" "0.01" (BasicType.Numeric Float64) ] }

  let makeGenericAddShotNoise () =
    { Id = "AddShotNoise"
      DisplayName = "addShotNoise"
      Category = "Statistics"
      Summary = "Add shot noise to each image."
      Description = "Applies SimpleITK's shot-noise filter to each image slice. Scale controls the shot-noise scale used by the native filter."
      Aliases = [ "noise"; "random"; "shot"; "poisson"; "statistics"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "scale" "Scale" "1.0" (BasicType.Numeric Float64) ] }

  let makeGenericAddSpeckleNoise () =
    { Id = "AddSpeckleNoise"
      DisplayName = "addSpeckleNoise"
      Category = "Statistics"
      Summary = "Add speckle noise to each image."
      Description = "Applies SimpleITK's speckle-noise filter to each image slice. Std controls the standard deviation used by the native filter."
      Aliases = [ "noise"; "random"; "speckle"; "multiplicative"; "statistics"; "UInt8"; "Float64"; "type" ]
      Inputs = [ makePort "Float64" imageFloat64 ]
      Outputs = [ makePort "Float64" imageFloat64 ]
      Parameters =
          [ makeParameter "type" "Type" "Float64" BasicType.String
            makeParameter "std" "Std" "1.0" (BasicType.Numeric Float64) ] }

  let makePairOperation id displayName description aliases parameters =
    { Id = id
      DisplayName = displayName
      Category = "Arithmetic"
      Summary = description
      Description = pairImageOperationDescription
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

        makeGenericReadVolume()

        makeGenericReadRandom()

        makeGenericEstimateHistogram()

        makeGenericReadRange()

        makeGenericReadSlab()

        makeGenericReadZarrSlab()

        makeGenericReadNexusSlab()

        { Id = "ReadPointSet"
          DisplayName = "readPointSet"
          Category = "Sources / Sinks"
          Summary = "Read coordinate points from a CSV point-set file."
          Description = pointSetDescription
          Aliases = [ "points"; "csv"; "keypoints"; "coordinates"; "features"; "read"; "source" ]
          Inputs = []
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ availableMemoryParameter
                makeParameter "input" "Input" "points.csv" BasicType.String ] }

        makeGenericZero()

        { Id = "CoordinateX"
          DisplayName = "coordinateX"
          Category = "Sources / Sinks"
          Summary = "Create a Float64 stack of x-coordinate values."
          Description = coordinateDescription "x"
          Aliases = [ "coordinate"; "x"; "source"; "synthetic"; "position" ]
          Inputs = []
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ availableMemoryParameter
                makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
                makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
                makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32) ] }

        { Id = "CoordinateY"
          DisplayName = "coordinateY"
          Category = "Sources / Sinks"
          Summary = "Create a Float64 stack of y-coordinate values."
          Description = coordinateDescription "y"
          Aliases = [ "coordinate"; "y"; "source"; "synthetic"; "position" ]
          Inputs = []
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ availableMemoryParameter
                makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
                makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
                makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32) ] }

        { Id = "CoordinateZ"
          DisplayName = "coordinateZ"
          Category = "Sources / Sinks"
          Summary = "Create a Float64 stack of z-coordinate values."
          Description = coordinateDescription "z"
          Aliases = [ "coordinate"; "z"; "source"; "synthetic"; "position"; "slice" ]
          Inputs = []
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ availableMemoryParameter
                makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
                makeParameter "height" "Height" "64" (BasicType.Numeric UInt32)
                makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32) ] }

        makeGenericNormalNoise()

        makeGenericSaltAndPepperNoise()

        makeGenericShotNoise()

        makeGenericSpeckleNoise()

        makeGenericCreateByEuler2DTransform()

        { Id = "ComputeStats"
          DisplayName = "computeStats"
          Category = "Statistics"
          Summary = "Reduce an image stream to aggregate image statistics."
          Description = computeStatsDescription
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

        { Id = "FitBiasModel"
          DisplayName = "fitBiasModel"
          Category = "Statistics"
          Summary = "Fit a 3D polynomial bias model from streamed image slices."
          Description = biasModelDescription
          Aliases = [ "bias"; "background"; "polynomial"; "flatfield"; "illumination"; "reducer"; "random" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "BiasModel" biasModel ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "order" "Order" "2" (BasicType.Numeric Int32)
                makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32) ] }

        { Id = "FitBiasModelMasked"
          DisplayName = "fitBiasModelMasked"
          Category = "Statistics"
          Summary = "Fit a 3D polynomial bias model inside a mask."
          Description = biasModelDescription
          Aliases = [ "bias"; "background"; "polynomial"; "flatfield"; "illumination"; "mask"; "reducer"; "random" ]
          Inputs = [ makePort "Number" imageAny; makePort "Mask" imageUInt8 ]
          Outputs = [ makePort "BiasModel" biasModel ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "order" "Order" "2" (BasicType.Numeric Int32)
                makeParameter "depth" "Depth" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Volume"
          DisplayName = "volume"
          Category = "Statistics"
          Summary = "Reduce a UInt8 0-1 mask stream to real-world object volume."
          Description = "Counts foreground voxels in a UInt8 mask stream where pixels must be 0 or 1, then multiplies by xUnit * yUnit * zUnit. Use the unit parameters to convert voxel counts into physical volume units."
          Aliases = [ "volume"; "mask"; "voxel"; "object"; "measure"; "reducer" ]
          Inputs = [ makePort "UInt8 mask" imageUInt8 ]
          Outputs = [ makePort "Volume: Float64" (Scalar(BasicType.Numeric Float64)) ]
          Parameters =
              [ makeParameter "xUnit" "X unit" "1.0" (BasicType.Numeric Float64)
                makeParameter "yUnit" "Y unit" "1.0" (BasicType.Numeric Float64)
                makeParameter "zUnit" "Z unit" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "CorrectBias"
          DisplayName = "correctBias"
          Category = "Intensity"
          Summary = "Subtract a fitted polynomial bias model."
          Description = correctBiasDescription
          Aliases = [ "bias"; "background"; "polynomial"; "flatfield"; "illumination"; "subtract"; "correct" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "model" "Model" "biasModel" BasicType.String ] }

        { Id = "CorrectBiasMasked"
          DisplayName = "correctBiasMasked"
          Category = "Intensity"
          Summary = "Subtract a fitted polynomial bias model inside a mask."
          Description = correctBiasDescription
          Aliases = [ "bias"; "background"; "polynomial"; "flatfield"; "illumination"; "mask"; "subtract"; "correct" ]
          Inputs = [ makePort "Number" imageAny; makePort "Mask" imageUInt8 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "model" "Model" "biasModel" BasicType.String ] }

        { Id = "SerialPolynomialBiasCorrect"
          DisplayName = "serialPolynomialBiasCorrect"
          Category = "Serial Sections"
          Summary = "Fit and subtract a 2D polynomial bias field independently per slice."
          Description = serialSectionsDescription
          Aliases = [ "serial"; "slice"; "slicewise"; "bias"; "polynomial"; "section"; "illumination" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "order" "Order" "2" (BasicType.Numeric Int32) ] }

        { Id = "SerialEstTrans"
          DisplayName = "serialEstTrans"
          Category = "Serial Sections"
          Summary = "Estimate pairwise slicewise affine transforms from 2D SIFT-style keypoints."
          Description = serialSectionsDescription + "\n\nserialEstTrans supports two methods. SiftAffine detects 2D Difference-of-Gaussian/SIFT-style keypoint locations in neighboring slices, uses displacement voting for a robust initial pairing region, then registers the keypoint sets with the affine point-set optimizer and accumulates the pairwise affine transforms from the first slice. SSDTranslation uses direct sum-of-squared-differences image matching and emits translation-only affine matrices. The current point representation uses keypoint positions, scale, and response; descriptor vectors and canonical orientations are not part of the Studio box yet."
          Aliases = [ "serial"; "slice"; "slicewise"; "manifest"; "registration"; "translation"; "alignment"; "sift"; "keypoint"; "DoG" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs =
              [ makePort "Number" imageAny
                makePort "SerialSliceManifest" serialSliceManifest ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "maxShift" "Max shift" "8" (BasicType.Numeric Int32)
                makeParameter "method" "Method" "SiftAffine" BasicType.String
                makeParameter "sigma0" "Sigma 0" "1.6" (BasicType.Numeric Float64)
                makeParameter "scaleFactor" "Scale factor" "1.41421356237" (BasicType.Numeric Float64)
                makeParameter "scaleLevels" "Scale levels" "4" (BasicType.Numeric UInt32)
                makeParameter "contrastThreshold" "Contrast threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "maxKeypoints" "Max keypoints" "50" (BasicType.Numeric UInt32)
                makeParameter "matchTolerance" "Match tolerance" "1.5" (BasicType.Numeric Float64)
                makeParameter "maxIterations" "Max iterations" "60" (BasicType.Numeric Int32)
                makeParameter "initialLinearStep" "Linear step" "0.05" (BasicType.Numeric Float64)
                makeParameter "initialTranslationStep" "Translation step" "1.0" (BasicType.Numeric Float64)
                makeParameter "minStep" "Min step" "0.0001" (BasicType.Numeric Float64)
                makeParameter "stepShrink" "Step shrink" "0.5" (BasicType.Numeric Float64) ] }

        { Id = "SerialApplyTrans"
          DisplayName = "serialApplyTrans"
          Category = "Serial Sections"
          Summary = "Apply slicewise serial-section transforms on the original slice canvas."
          Description = serialSectionsDescription
          Aliases = [ "serial"; "slice"; "slicewise"; "manifest"; "transform"; "registration"; "apply" ]
          Inputs =
              [ makePort "Number" imageAny
                makePort "SerialSliceManifest" serialSliceManifest ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "background" "Background" "0.0" (BasicType.Numeric Float64) ] }

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

        { Id = "WriteVolume"
          DisplayName = "writeVolume"
          Category = "Sources / Sinks"
          Summary = "Write a stream as one multipage TIFF/BigTIFF volume."
          Description = writeVolumeDescription
          Aliases = [ "volume"; "multipage"; "tiff"; "bigtiff"; "ome"; "write"; "save" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "volume.tiff" BasicType.String ] }

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

        { Id = "WriteMesh"
          DisplayName = "writeMesh"
          Category = "Sources / Sinks"
          Summary = "Write a streamed triangle mesh to OBJ, STL, or PLY."
          Description = "Writes triangle sets produced by marchingCubes. OBJ and ASCII STL are written in a streaming-friendly pass. ASCII PLY is also supported, but its header needs vertex and face counts, so triangle sets are counted before the file is finalized. Use OBJ for the broadest compatibility during exploratory streaming workflows."
          Aliases = [ "mesh"; "surface"; "triangles"; "obj"; "stl"; "ply"; "write"; "save" ]
          Inputs = [ makePort "Mesh" mesh ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "surface.obj" BasicType.String
                makeParameter "format" "Format" "auto" BasicType.String ] }

        { Id = "SurfaceArea"
          DisplayName = "surfaceArea"
          Category = "Geometry"
          Summary = "Reduce a triangle mesh stream to real-world surface area."
          Description = "Sums triangle areas from streamed triangle sets, scaling x, y, and z coordinates by the supplied unit sizes before area is computed. This lets marching-cubes surfaces be measured in physical units when voxel spacing is anisotropic."
          Aliases = [ "area"; "surface"; "mesh"; "triangles"; "measure"; "reducer" ]
          Inputs = [ makePort "Mesh" mesh ]
          Outputs = [ makePort "Area: Float64" (Scalar(BasicType.Numeric Float64)) ]
          Parameters =
              [ makeParameter "xUnit" "X unit" "1.0" (BasicType.Numeric Float64)
                makeParameter "yUnit" "Y unit" "1.0" (BasicType.Numeric Float64)
                makeParameter "zUnit" "Z unit" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "WritePointSet"
          DisplayName = "writePointSet"
          Category = "Sources / Sinks"
          Summary = "Write coordinate point sets to a CSV file."
          Description = pointSetDescription
          Aliases = [ "points"; "csv"; "keypoints"; "coordinates"; "features"; "write"; "save" ]
          Inputs = [ makePort "PointSet" pointSet ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "points" BasicType.String
                suffixParameter ".csv" ] }

        { Id = "PointPairDistances"
          DisplayName = "pointPairDistances"
          Category = "Geometry"
          Summary = "Reduce point sets to a vectorized Euclidean distance matrix."
          Description = "Collects streamed point sets, scales x, y, and z coordinates by the supplied unit sizes, and computes the Euclidean distance between every pair. The result is a row-major vectorized Float64 distance matrix that can be converted back to a matrix with unvectorizeMatrix."
          Aliases = [ "points"; "distance"; "matrix"; "pairwise"; "euclidean"; "dog"; "keypoints"; "reducer" ]
          Inputs = [ makePort "PointSet" pointSet ]
          Outputs = [ makePort "Distance matrix" float64Matrix ]
          Parameters =
              [ makeParameter "xUnit" "X unit" "1.0" (BasicType.Numeric Float64)
                makeParameter "yUnit" "Y unit" "1.0" (BasicType.Numeric Float64)
                makeParameter "zUnit" "Z unit" "1.0" (BasicType.Numeric Float64) ] }

        { Id = "AffineRegistration"
          DisplayName = "affineRegistration"
          Category = "Geometry"
          Summary = "Register two point sets and emit affine transform matrices."
          Description = "Registers moving points to fixed points using the point-set affine optimizer. The outputs are 4x4 homogeneous Float64 matrices: Transform maps moving coordinates to fixed coordinates, and Inverse transform maps fixed coordinates back to moving coordinates. The matrices can be written with writeMatrix."
          Aliases = [ "registration"; "affine"; "points"; "transform"; "matrix"; "alignment"; "reducer" ]
          Inputs =
              [ makePort "Fixed PointSet" pointSet
                makePort "Moving PointSet" pointSet ]
          Outputs =
              [ makePort "Transform" float64Matrix
                makePort "Inverse transform" float64Matrix ]
          Parameters =
              [ makeParameter "maxIterations" "Max iterations" "200" (BasicType.Numeric Int32)
                makeParameter "initialLinearStep" "Linear step" "0.05" (BasicType.Numeric Float64)
                makeParameter "initialTranslationStep" "Translation step" "1.0" (BasicType.Numeric Float64)
                makeParameter "minStep" "Min step" "0.0001" (BasicType.Numeric Float64)
                makeParameter "stepShrink" "Step shrink" "0.5" (BasicType.Numeric Float64) ] }

        { Id = "WriteMatrix"
          DisplayName = "writeMatrix"
          Category = "Sources / Sinks"
          Summary = "Write a vectorized Float64 matrix to a file."
          Description = "Writes a row-major vectorized Float64 matrix, such as the output of pointPairDistances. The current supported format is CSV, with one row per matrix row; the suffix parameter leaves the surface open for additional formats later."
          Aliases = [ "matrix"; "csv"; "write"; "save"; "distance" ]
          Inputs = [ makePort "Float64 matrix" float64Matrix ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "matrix" BasicType.String
                suffixParameter ".csv" ] }

        { Id = "WriteCSV"
          DisplayName = "writeCSV"
          Category = "Sources / Sinks"
          Summary = "Write point sets, matrices, or histograms to CSV."
          Description = csvWriteDescription
          Aliases = [ "csv"; "write"; "save"; "points"; "matrix"; "histogram"; "keypoints"; "distance" ]
          Inputs = [ makePort "Data" any ]
          Outputs = []
          Parameters =
              [ makeParameter "output" "Output" "data" BasicType.String
                makeParameter "dataKind" "Data" "PointSet" BasicType.String ] }

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
          Description = tapDescription
          Aliases = [ "debug"; "trace"; "log"; "inspect" ]
          Inputs = [ makePort "Any" any ]
          Outputs = [ makePort "Any" any ]
          Parameters = [ makeParameter "label" "Label" "tap" BasicType.String ] }

        { Id = "Print"
          DisplayName = "print"
          Category = "Visualization"
          Summary = "Print one or more scalar values in the generated program."
          Description = printDescription
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
          Description = histogramDescription
          Aliases = [ "plot"; "chart"; "histogram"; "visualize"; "show" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters = [] }

        { Id = "HistogramData"
          DisplayName = "histogramData"
          Category = "Visualization"
          Summary = "Reduce an image stream to histogram points that can be printed or plotted."
          Description = histogramDataDescription
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
          Description = chartDescription
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
          Description = showImageDescription
          Aliases = [ "plot"; "image"; "heatmap"; "visualize"; "show" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = []
          Parameters = [] }

        { Id = "SumProjection"
          DisplayName = "sumProjection"
          Category = "Visualization"
          Summary = "Reduce a stack to one summed projection image."
          Description = sumProjectionDescription
          Aliases = [ "projection"; "sum"; "volume"; "visualize"; "mip"; "z"; "intensity"; "show"; "write" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "function" "Function" "Identity" BasicType.String ] }

        { Id = "UnaryImageFunction"
          DisplayName = "f(I)"
          Category = "Arithmetic"
          Summary = "Apply a standard unary function to each pixel."
          Description = unaryImageFunctionDescription
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

        makeGenericAddSaltAndPepperNoise()

        makeGenericAddShotNoise()

        makeGenericAddSpeckleNoise()

        makePairOperation
            "ImageOpImage"
            "I op J"
            "Combine two image streams of the same numeric type pairwise with an arithmetic, max, or min operation. Code generation inserts zip or shared fan-out as needed."
            [ "add"; "sum"; "subtract"; "multiply"; "mask"; "divide"; "ratio"; "maximum"; "max"; "minimum"; "min"; "arithmetic"; "+"; "-"; "*"; "/" ]
            [ makeParameter "operation" "Operation" "*" BasicType.String
              makeParameter "type" "Type" "Float64" BasicType.String ]

        { Id = "ComplexFromReIm"
          DisplayName = "toComplex"
          Category = "Complex Images"
          Summary = "Compose real and imaginary Float64 image streams into a native complex image stream."
          Description = "Combines synchronized Float64 real and imaginary images into one native ComplexFloat64 image stream."
          Aliases = [ "complex"; "real"; "imaginary"; "compose"; "toComplex"; "fourier" ]
          Inputs = [ makePort "Re: Float64" imageFloat64; makePort "Im: Float64" imageFloat64 ]
          Outputs = [ makePort "Complex" imageComplex ]
          Parameters = [] }

        { Id = "ComplexPolar"
          DisplayName = "polarToComplex"
          Category = "Complex Images"
          Summary = "Compose modulus and argument Float64 image streams into a native complex image stream."
          Description = "Combines synchronized modulus and angle images into one native ComplexFloat64 image stream using polar coordinates."
          Aliases = [ "complex"; "polar"; "modulus"; "argument"; "phase"; "angle"; "fourier" ]
          Inputs = [ makePort "Modulus: Float64" imageFloat64; makePort "Arg: Float64" imageFloat64 ]
          Outputs = [ makePort "Complex" imageComplex ]
          Parameters = [] }

        { Id = "ComplexRe"
          DisplayName = "Re"
          Category = "Complex Images"
          Summary = "Extract the real part of a complex image stream."
          Description = "Extracts the real component from each native complex pixel and emits a Float64 image stream."
          Aliases = [ "complex"; "real"; "re"; "fourier" ]
          Inputs = [ makePort "Complex" imageComplex ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [] }

        { Id = "ComplexIm"
          DisplayName = "Im"
          Category = "Complex Images"
          Summary = "Extract the imaginary part of a complex image stream."
          Description = "Extracts the imaginary component from each native complex pixel and emits a Float64 image stream."
          Aliases = [ "complex"; "imaginary"; "im"; "fourier" ]
          Inputs = [ makePort "Complex" imageComplex ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [] }

        { Id = "ComplexModulus"
          DisplayName = "modulus"
          Category = "Complex Images"
          Summary = "Compute the modulus of a complex image stream."
          Description = "Computes the complex modulus at each pixel and emits a Float64 image stream."
          Aliases = [ "complex"; "abs"; "absolute"; "magnitude"; "modulus"; "fourier" ]
          Inputs = [ makePort "Complex" imageComplex ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [] }

        { Id = "ComplexArg"
          DisplayName = "arg"
          Category = "Complex Images"
          Summary = "Compute the argument angle of a complex image stream."
          Description = "Computes the complex phase angle at each pixel and emits a Float64 image stream."
          Aliases = [ "complex"; "arg"; "argument"; "phase"; "angle"; "fourier" ]
          Inputs = [ makePort "Complex" imageComplex ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [] }

        { Id = "ComplexConjugate"
          DisplayName = "conjugate"
          Category = "Complex Images"
          Summary = "Compute the complex conjugate of a complex image stream."
          Description = "Keeps the real part and negates the imaginary part of each native complex pixel."
          Aliases = [ "complex"; "conjugate"; "fourier" ]
          Inputs = [ makePort "Complex" imageComplex ]
          Outputs = [ makePort "Complex" imageComplex ]
          Parameters = [] }

        { Id = "FFT"
          DisplayName = "FFT"
          Category = "Fourier"
          Summary = "Compute a chunk-backed 3D FFT from a scalar image stack."
          Description = "Streams scalar slices into a temporary chunk workspace, computes slice-wise XY FFT followed by a z-direction FFT, and emits native complex frequency-domain slices."
          Aliases = [ "fft"; "fourier"; "frequency"; "3d"; "chunk"; "spectrum" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Complex" imageComplex ]
          Parameters =
            [ makeParameter "type" "Type" "Float64" BasicType.String
              makeParameter "chunkX" "Chunk X" "64" (BasicType.Numeric UInt32)
              makeParameter "chunkY" "Chunk Y" "64" (BasicType.Numeric UInt32)
              makeParameter "chunkZ" "Chunk Z" "16" (BasicType.Numeric UInt32) ] }

        { Id = "InvFFT"
          DisplayName = "invFFT"
          Category = "Fourier"
          Summary = "Compute the inverse chunk-backed 3D FFT."
          Description = "Streams complex frequency-domain slices through a temporary chunk workspace, applies the inverse z transform and inverse XY transforms, and emits real Float64 slices."
          Aliases = [ "ifft"; "inverse"; "fourier"; "frequency"; "3d"; "chunk" ]
          Inputs = [ makePort "Complex" imageComplex ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
            [ makeParameter "chunkX" "Chunk X" "64" (BasicType.Numeric UInt32)
              makeParameter "chunkY" "Chunk Y" "64" (BasicType.Numeric UInt32)
              makeParameter "chunkZ" "Chunk Z" "16" (BasicType.Numeric UInt32) ] }

        { Id = "ShiftFFT"
          DisplayName = "shiftFFT"
          Category = "Fourier"
          Summary = "Shift the zero-frequency component to the center of a complex spectrum."
          Description = "Streams complex frequency-domain slices through a temporary chunk workspace and circularly shifts each axis by half its size."
          Aliases = [ "fftshift"; "shift"; "fourier"; "frequency"; "center"; "spectrum" ]
          Inputs = [ makePort "Complex" imageComplex ]
          Outputs = [ makePort "Complex" imageComplex ]
          Parameters =
            [ makeParameter "chunkX" "Chunk X" "64" (BasicType.Numeric UInt32)
              makeParameter "chunkY" "Chunk Y" "64" (BasicType.Numeric UInt32)
              makeParameter "chunkZ" "Chunk Z" "16" (BasicType.Numeric UInt32) ] }

        { Id = "ToVectorImage"
          DisplayName = "toVectorImage"
          Category = "Vector Images"
          Summary = "Combine two Float64 image streams into one two-component vector-valued image stream."
          Description = "Combines synchronized scalar image streams into vector-valued pixels while preserving the image domain. This is the vector-valued counterpart to stack: stack adds a spatial axis, while toVectorImage adds pixel components."
          Aliases = [ "vector"; "components"; "compose"; "toVectorImage"; "field" ]
          Inputs = [ makePort "I: Float64" imageFloat64; makePort "J: Float64" imageFloat64 ]
          Outputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Parameters = [] }

        { Id = "VectorElement"
          DisplayName = "vectorElement"
          Category = "Vector Images"
          Summary = "Extract one scalar component from a vector-valued image stream."
          Description = "Extracts a component from each vector-valued pixel and emits an ordinary Float64 image stream. Component indices are zero-based."
          Aliases = [ "vector"; "component"; "extract"; "element"; "channel" ]
          Inputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [ makeParameter "component" "Component" "0" (BasicType.Numeric UInt32) ] }

        { Id = "AppendVectorElement"
          DisplayName = "appendVectorElement"
          Category = "Vector Images"
          Summary = "Append a scalar image stream as a new vector component."
          Description = "Combines a vector-valued image stream and a synchronized scalar image stream by appending the scalar pixel as the last vector component. Use it with toVectorImage to build three-component vector images for vectorCross3D."
          Aliases = [ "vector"; "append"; "component"; "compose"; "field" ]
          Inputs = [ makePort "Vector Float64" vectorImageFloat64; makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Parameters = [] }

        { Id = "VectorMapElements"
          DisplayName = "mapVectorElements"
          Category = "Vector Images"
          Summary = "Apply a scalar function independently to every vector component."
          Description = "Maps a simple Float64 function over every component of every vector-valued pixel while preserving the image domain and component count."
          Aliases = [ "vector"; "map"; "component"; "abs"; "sqrt"; "function" ]
          Inputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Outputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Parameters = [ makeParameter "function" "Function" "sqrt" BasicType.String ] }

        { Id = "VectorDot"
          DisplayName = "vectorDot"
          Category = "Vector Images"
          Summary = "Compute the pixelwise dot product of two vector-valued image streams."
          Description = "Combines synchronized vector-valued image streams by taking the dot product at each pixel, producing an ordinary Float64 image stream."
          Aliases = [ "vector"; "dot"; "inner"; "product" ]
          Inputs = [ makePort "U: Vector Float64" vectorImageFloat64; makePort "V: Vector Float64" vectorImageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters = [] }

        { Id = "VectorCross3D"
          DisplayName = "vectorCross3D"
          Category = "Vector Images"
          Summary = "Compute the pixelwise 3D cross product of two vector-valued image streams."
          Description = "Combines synchronized three-component vector-valued image streams by taking the 3D cross product at each pixel, producing another vector-valued image stream."
          Aliases = [ "vector"; "cross"; "3d"; "product" ]
          Inputs = [ makePort "U: Vector Float64" vectorImageFloat64; makePort "V: Vector Float64" vectorImageFloat64 ]
          Outputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Parameters = [] }

        { Id = "VectorAngleTo"
          DisplayName = "vectorAngleTo"
          Category = "Vector Images"
          Summary = "Compute the per-pixel angle to a fixed vector."
          Description = "Computes the angle in radians between each vector-valued pixel and a fixed reference vector."
          Aliases = [ "vector"; "angle"; "orientation"; "direction" ]
          Inputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
            [ makeParameter "x" "X" "1.0" (BasicType.Numeric Float64)
              makeParameter "y" "Y" "0.0" (BasicType.Numeric Float64)
              makeParameter "z" "Z" "0.0" (BasicType.Numeric Float64) ] }

        { Id = "Gradient"
          DisplayName = "gradient"
          Category = "Vector Images"
          Summary = "Compute a 3-component finite-difference gradient field."
          Description = "Computes finite-difference derivatives along x, y, and z and emits them as three-component vector-valued pixels ordered as dx, dy, dz."
          Aliases = [ "gradient"; "derivative"; "finite"; "difference"; "vector"; "field" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Parameters =
            [ makeParameter "order" "Order" "1" (BasicType.Numeric UInt32)
              makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "StructureTensor"
          DisplayName = "structureTensor"
          Category = "Vector Images"
          Summary = "Compute structure-tensor eigensystem fields."
          Description = "Pre-smooths the scalar image with sigma, computes the finite-difference gradient, forms the six unique components of the symmetric exterior product, smooths those tensor components with rho, and emits four 3-vector streams: eigenvalues followed by the three eigenvector fields."
          Aliases = [ "structure"; "tensor"; "eigen"; "orientation"; "gradient"; "matrix" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs =
            [ makePort "Eigenvalues" vectorImageFloat64
              makePort "Eigenvector 0" vectorImageFloat64
              makePort "Eigenvector 1" vectorImageFloat64
              makePort "Eigenvector 2" vectorImageFloat64 ]
          Parameters =
            [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
              makeParameter "rho" "Rho" "2.0" (BasicType.Numeric Float64) ] }

        { Id = "PCA"
          DisplayName = "PCA"
          Category = "Vector Images"
          Summary = "Reduce vector images to a principal-component eigensystem."
          Description = "Computes principal component analysis over all vector pixels in the input stream. Components selects the vector dimensionality, from 2 to 8 in Studio. The reducer emits singleton vector streams: eigenvalues followed by one eigenvector stream per component, sorted by descending eigenvalue."
          Aliases = [ "pca"; "principal"; "component"; "covariance"; "eigen"; "vector"; "reducer" ]
          Inputs = [ makePort "Vector Float64" vectorImageFloat64 ]
          Outputs =
            [ makePort "Eigenvalues" vectorImageFloat64
              makePort "Eigenvector 0" vectorImageFloat64
              makePort "Eigenvector 1" vectorImageFloat64
              makePort "Eigenvector 2" vectorImageFloat64
              makePort "Eigenvector 3" vectorImageFloat64
              makePort "Eigenvector 4" vectorImageFloat64
              makePort "Eigenvector 5" vectorImageFloat64
              makePort "Eigenvector 6" vectorImageFloat64
              makePort "Eigenvector 7" vectorImageFloat64 ]
          Parameters = [ makeParameter "components" "Components" "3" (BasicType.Numeric UInt32) ] }

        { Id = "SmoothWGauss"
          DisplayName = "smoothWGauss"
          Category = "Filters"
          Summary = "Apply a Gaussian smoothing filter."
          Description = gaussianDescription
          Aliases = [ "gaussian"; "smooth"; "blur"; "filter" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "outputRegionMode" "Output region" "None" BasicType.String
                makeParameter "boundaryCondition" "Boundary" "None" BasicType.String
                makeParameter "windowSize" "Window size" "None" BasicType.String ] }

        { Id = "Convolve"
          DisplayName = "convolve"
          Category = "Filters"
          Summary = "Convolve an image stream with a kernel image."
          Description = convolveDescription
          Aliases = [ "convolution"; "kernel"; "filter"; "same"; "valid"; "boundary" ]
          Inputs = [ makePort "Image" imageAny ]
          Outputs = [ makePort "Image" imageAny ]
          Parameters =
              [ makeParameter "kernel" "Kernel" "Image<float>([3u; 3u; 3u])" BasicType.String
                makeParameter "outputRegionMode" "Output region" "None" BasicType.String
                makeParameter "boundaryCondition" "Boundary" "None" BasicType.String
                makeParameter "windowSize" "Window size" "None" BasicType.String ] }

        { Id = "FiniteDiff"
          DisplayName = "finiteDiff"
          Category = "Filters"
          Summary = "Apply finite difference derivative filters."
          Description = finiteDiffDescription
          Aliases = [ "derivative"; "difference"; "filter" ]
          Inputs = [ makePort "Float64" imageFloat64 ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "axis1" "Axis 1" "1" (BasicType.Numeric UInt32)
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

        { Id = "HistogramEqualization"
          DisplayName = "histogramEqualization"
          Category = "Intensity"
          Summary = "Equalize intensities from a 3D histogram estimate."
          Description = histogramEqualizationDescription
          Aliases = [ "intensity"; "histogram"; "equalize"; "equalization"; "contrast"; "cdf"; "3d" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Float64" imageFloat64 ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "histogram" "Histogram" "" BasicType.Map ] }

        { Id = "CreatePadding"
          DisplayName = "createPadding"
          Category = "Geometry"
          Summary = "Pad an image volume on all six sides."
          Description = "Pads the x and y sides of each 2D slice with SimpleITK's constant padding filter, and pads the z direction by streaming generated constant slices before and after the input stream. The six side parameters are before/after x, before/after y, and before/after z. Padding in z increases the stream length, while x/y padding increases each slice size. The padding value is converted to the selected image type."
          Aliases = [ "padding"; "pad"; "constant"; "border"; "geometry"; "x"; "y"; "z" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "beforeX" "Before X" "0" (BasicType.Numeric UInt32)
                makeParameter "afterX" "After X" "0" (BasicType.Numeric UInt32)
                makeParameter "beforeY" "Before Y" "0" (BasicType.Numeric UInt32)
                makeParameter "afterY" "After Y" "0" (BasicType.Numeric UInt32)
                makeParameter "beforeZ" "Before Z" "0" (BasicType.Numeric UInt32)
                makeParameter "afterZ" "After Z" "0" (BasicType.Numeric UInt32)
                makeParameter "value" "Value" "0.0" (BasicType.Numeric Float64) ] }

        { Id = "Crop"
          DisplayName = "crop"
          Category = "Geometry"
          Summary = "Crop an image volume on all six sides."
          Description = "Removes pixels from the x and y sides of every slice with SimpleITK's crop filter, and removes slices from the beginning and end of the z stream. Cropping before z skips and releases those incoming slices immediately. Cropping after z buffers only the requested trailing count, so the pipeline can truncate once the last needed streaming element has passed."
          Aliases = [ "crop"; "trim"; "remove"; "border"; "geometry"; "x"; "y"; "z" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Number" imageAny ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "beforeX" "Before X" "0" (BasicType.Numeric UInt32)
                makeParameter "afterX" "After X" "0" (BasicType.Numeric UInt32)
                makeParameter "beforeY" "Before Y" "0" (BasicType.Numeric UInt32)
                makeParameter "afterY" "After Y" "0" (BasicType.Numeric UInt32)
                makeParameter "beforeZ" "Before Z" "0" (BasicType.Numeric UInt32)
                makeParameter "afterZ" "After Z" "0" (BasicType.Numeric UInt32) ] }

        { Id = "SmoothWMedian"
          DisplayName = "smoothWMedian"
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

        { Id = "SmoothWBilateral"
          DisplayName = "smoothWBilateral"
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
          Description = "Mask logic operates on UInt8 mask streams and emits a UInt8 mask. Use it to combine threshold, comparison, and segmentation results before masking or writing. Non-zero values are treated as true by the underlying SimpleITK logical filters. The operation selector compiles to maskAnd, maskOr, or maskXor. Use maskNot when only a single mask should be inverted."
          Aliases = [ "mask"; "logic"; "and"; "or"; "xor"; "binary"; "boolean" ]
          Inputs = [ makePort "UInt8" imageUInt8; makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "operation" "Operation" "and" BasicType.String ] }

        { Id = "MaskNot"
          DisplayName = "maskNot"
          Category = "Segmentation"
          Summary = "Invert a UInt8 mask."
          Description = "Inverts a UInt8 mask stream using SimpleITK's logical Not filter. This is intended for masks, not for grayscale intensity inversion. For grayscale negatives, estimate the relevant maximum first and use shiftScale. The operation is slice-local and fits streaming naturally."
          Aliases = [ "mask"; "logic"; "not"; "invert"; "binary"; "boolean" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [] }

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

        { Id = "RemoveSmallObjects"
          DisplayName = "removeSmallObjects"
          Category = "Binary Morphology"
          Summary = "Remove connected foreground objects up to a maximum voxel count."
          Description =
            "Streams a binary mask and removes completed foreground components whose voxel count is less than or equal to the requested maximum volume.\n\nThe stage carries only components that still touch the advancing z-frontier. Once an object cannot continue into the next slice, it is either painted to zero or left unchanged. Six-connectivity uses face contacts only; TwentySix-connectivity also allows diagonal contacts across and within slices.\n\nThis replaces reconstruction-style object cleanup in the LMIP DSL because its decision rule is finite, explicit, and local to completed connected components."
          Aliases = [ "morphology"; "binary"; "objects"; "remove"; "small"; "cleanup"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "maximumVolume" "Maximum volume" "64" (BasicType.Numeric UInt64)
                makeParameter "connectivity" "Connectivity" "Six" BasicType.String ] }

        { Id = "FillSmallHoles"
          DisplayName = "fillSmallHoles"
          Category = "Binary Morphology"
          Summary = "Fill enclosed background holes up to a maximum voxel count."
          Description =
            "Streams a binary mask and fills completed background components whose voxel count is less than or equal to the requested maximum volume.\n\nBackground components touching the x-y border, the first z-slice, or the final z-slice are treated as exterior and are preserved. Other completed background components are holes and are painted to one when small enough. Six-connectivity uses face contacts only; TwentySix-connectivity also allows diagonal contacts.\n\nThis is the LMIP-oriented replacement for neighborhood voting hole filling when the desired operation is connected-hole cleanup by size."
          Aliases = [ "morphology"; "binary"; "holes"; "fill"; "small"; "cleanup"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "maximumVolume" "Maximum volume" "64" (BasicType.Numeric UInt64)
                makeParameter "connectivity" "Connectivity" "Six" BasicType.String ] }

        { Id = "Threshold"
          DisplayName = "threshold"
          Category = "Segmentation"
          Summary = "Threshold an image into a binary UInt8 image."
          Description = thresholdDescription
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
          Description = binaryShapeDescription
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Dilate"
          DisplayName = "dilate"
          Category = "Binary Morphology"
          Summary = "Dilate a binary UInt8 image."
          Description = binaryShapeDescription
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Opening"
          DisplayName = "opening"
          Category = "Binary Morphology"
          Summary = "Apply morphological opening to a binary UInt8 image."
          Description = binaryShapeDescription
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "Closing"
          DisplayName = "closing"
          Category = "Binary Morphology"
          Summary = "Apply morphological closing to a binary UInt8 image."
          Description = binaryShapeDescription
          Aliases = [ "morphology"; "binary"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [ makeParameter "radius" "Radius" "1" (BasicType.Numeric UInt32) ] }

        { Id = "ConnectedComponents"
          DisplayName = "connectedComponents"
          Category = "Binary Morphology"
          Summary = "Label connected binary components."
          Description = connectedComponentsDescription
          Aliases = [ "components"; "labels"; "segmentation" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "Labels + count" connectedComponentLabels ]
          Parameters = [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "StreamConnectedObjects"
          DisplayName = "streamConnectedObjects"
          Category = "Segmentation"
          Summary = "Emit completed connected objects from a binary mask stream."
          Description = streamedObjectsDescription
          Aliases = [ "objects"; "components"; "connected"; "stream"; "coordinates"; "mask" ]
          Inputs = [ makePort "UInt8" imageUInt8 ]
          Outputs = [ makePort "Objects" streamedObjects ]
          Parameters =
              [ makeParameter "connectivity" "Connectivity" "Six" BasicType.String ] }

        { Id = "PaintObjects"
          DisplayName = "paintObjects"
          Category = "Segmentation"
          Summary = "Paint streamed object coordinates into UInt8 mask slices."
          Description = streamedObjectsDescription
          Aliases = [ "paint"; "objects"; "coordinates"; "mask"; "UInt8" ]
          Inputs = [ makePort "Objects" streamedObjects ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters =
              [ makeParameter "width" "Width" "64" (BasicType.Numeric UInt32)
                makeParameter "height" "Height" "64" (BasicType.Numeric UInt32) ] }

        { Id = "PaintObjectsCropped"
          DisplayName = "paintObjectsCropped"
          Category = "Segmentation"
          Summary = "Paint each streamed object into its minimal UInt8 mask slices."
          Description = streamedObjectsDescription
          Aliases = [ "paint"; "objects"; "coordinates"; "mask"; "cropped"; "minimal"; "bounding"; "box" ]
          Inputs = [ makePort "Objects" streamedObjects ]
          Outputs = [ makePort "UInt8" imageUInt8 ]
          Parameters = [] }

        { Id = "RelabelComponents"
          DisplayName = "relabelComponents"
          Category = "Binary Morphology"
          Summary = "Relabel connected-component labels and remove small components."
          Description = relabelComponentsDescription
          Aliases = [ "components"; "labels"; "relabel"; "size"; "filter" ]
          Inputs = [ makePort "UInt64" imageUInt64 ]
          Outputs = [ makePort "UInt64" imageUInt64 ]
          Parameters =
              [ makeParameter "minimumObjectSize" "Minimum object size" "1" (BasicType.Numeric UInt32)
                makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "MarchingCubes"
          DisplayName = "marchingCubes"
          Category = "Geometry"
          Summary = "Extract a streamed triangle surface from consecutive image slices."
          Description = "Builds a triangle mesh from a rolling two-slice window. Each pair of consecutive slices forms the cubes needed for a local isosurface, so the operation is larger-than-memory friendly in z. The surface value is compared directly to the selected image type values; using the desired label or threshold value avoids an extra subtraction or recentering stage. The implementation triangulates each cube locally and emits triangle sets for writeMesh."
          Aliases = [ "marching"; "cubes"; "mesh"; "surface"; "isosurface"; "triangles"; "geometry" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "Mesh" mesh ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "surfaceValue" "Surface value" "0.5" (BasicType.Numeric Float64) ] }

        { Id = "DogKeypoints"
          DisplayName = "dogKeypoints"
          Category = "Geometry"
          Summary = "Detect streamed Difference-of-Gaussian keypoints."
          Description = dogKeypointsDescription
          Aliases = [ "dog"; "sift"; "keypoints"; "features"; "points"; "scale"; "gaussian" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sigma0" "Sigma 0" "1.0" (BasicType.Numeric Float64)
                makeParameter "scaleFactor" "Scale factor" "1.6" (BasicType.Numeric Float64)
                makeParameter "scaleLevels" "Scale levels" "4" (BasicType.Numeric UInt32)
                makeParameter "contrastThreshold" "Contrast threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "SiftKeypoints"
          DisplayName = "siftKeypoints"
          Category = "Geometry"
          Summary = "Detect streamed SIFT-style keypoints."
          Description = siftKeypointsDescription
          Aliases = [ "sift"; "dog"; "keypoints"; "features"; "points"; "scale"; "gaussian" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sigma0" "Sigma 0" "1.0" (BasicType.Numeric Float64)
                makeParameter "scaleFactor" "Scale factor" "1.6" (BasicType.Numeric Float64)
                makeParameter "scaleLevels" "Scale levels" "4" (BasicType.Numeric UInt32)
                makeParameter "contrastThreshold" "Contrast threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "LogBlobKeypoints"
          DisplayName = "logBlobKeypoints"
          Category = "Geometry"
          Summary = "Detect 3D Laplacian-of-Gaussian blob keypoints."
          Description = streamingKeypointDescription
          Aliases = [ "log"; "blob"; "keypoints"; "features"; "points"; "gaussian" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "threshold" "Threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "HessianKeypoints"
          DisplayName = "hessianKeypoints"
          Category = "Geometry"
          Summary = "Detect 3D blob, tube, or sheet keypoints from Hessian eigenvalues."
          Description = streamingKeypointDescription
          Aliases = [ "hessian"; "blob"; "tube"; "sheet"; "vessel"; "keypoints"; "features"; "points" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "responseKind" "Response" "Blob" BasicType.String
                makeParameter "threshold" "Threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "Harris3DKeypoints"
          DisplayName = "harris3DKeypoints"
          Category = "Geometry"
          Summary = "Detect 3D corner or junction keypoints from the structure tensor."
          Description = streamingKeypointDescription
          Aliases = [ "harris"; "corner"; "junction"; "structure"; "tensor"; "keypoints"; "features"; "points" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "rho" "Rho" "1.5" (BasicType.Numeric Float64)
                makeParameter "k" "K" "0.04" (BasicType.Numeric Float64)
                makeParameter "threshold" "Threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "Forstner3DKeypoints"
          DisplayName = "forstner3DKeypoints"
          Category = "Geometry"
          Summary = "Detect 3D Förstner-style junction keypoints from the structure tensor."
          Description = streamingKeypointDescription
          Aliases = [ "forstner"; "foerstner"; "corner"; "junction"; "structure"; "tensor"; "keypoints"; "features"; "points" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "rho" "Rho" "1.5" (BasicType.Numeric Float64)
                makeParameter "threshold" "Threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "PhaseCongruencyKeypoints"
          DisplayName = "phaseCongruencyKeypoints"
          Category = "Geometry"
          Summary = "Detect contrast-normalized local phase keypoints with a bounded window."
          Description = streamingKeypointDescription
          Aliases = [ "phase"; "congruency"; "local"; "contrast"; "keypoints"; "features"; "points" ]
          Inputs = [ makePort "Number" imageAny ]
          Outputs = [ makePort "PointSet" pointSet ]
          Parameters =
              [ makeParameter "type" "Type" "Float64" BasicType.String
                makeParameter "sigma" "Sigma" "1.0" (BasicType.Numeric Float64)
                makeParameter "threshold" "Threshold" "0.03" (BasicType.Numeric Float64)
                makeParameter "stride" "Stride" "8" (BasicType.Numeric UInt32) ] }

        { Id = "SignedDistanceBand"
          DisplayName = "signedDistanceBand"
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

        { Id = "OtsuThresholdFromHistogram"
          DisplayName = "otsuThresholdFromHistogram"
          Category = "Statistics"
          Summary = "Estimate an Otsu threshold from a histogram."
          Description =
            "Takes a histogram, usually from histogramData on a sampled or random-read branch, and returns a scalar threshold by maximizing Otsu's between-class variance.\n\nThis box does not threshold images itself. Link its scalar output to the lower bound of the standard threshold box, and use infinity or another upper bound there. Keeping threshold estimation separate from threshold application makes the two-pass LMIP structure explicit."
          Aliases = [ "threshold"; "otsu"; "histogram"; "statistics"; "scalar"; "segment" ]
          Inputs = []
          Outputs = [ makePort "Threshold: Float64" (Scalar(BasicType.Numeric Float64)) ]
          Parameters =
              [ makeParameter "histogram" "Histogram" "" BasicType.Map ] }

        { Id = "MomentsThresholdFromHistogram"
          DisplayName = "momentsThresholdFromHistogram"
          Category = "Statistics"
          Summary = "Estimate a moment-preserving threshold from a histogram."
          Description =
            "Takes a histogram, usually from histogramData on a sampled or random-read branch, and returns a scalar threshold from the first three histogram moments using Tsai's moment-preserving method.\n\nThis box does not threshold images itself. Link its scalar output to the lower bound of the standard threshold box, and use infinity or another upper bound there. Keeping threshold estimation separate from threshold application makes the two-pass LMIP structure explicit."
          Aliases = [ "threshold"; "moments"; "histogram"; "statistics"; "scalar"; "segment" ]
          Inputs = []
          Outputs = [ makePort "Threshold: Float64" (Scalar(BasicType.Numeric Float64)) ]
          Parameters =
              [ makeParameter "histogram" "Histogram" "" BasicType.Map ] }

        { Id = "ComponentTranslationTable"
          DisplayName = "componentTranslationTable"
          Category = "Binary Morphology"
          Summary = "Reduce connected-component label slabs to a translation table with streaming component statistics."
          Description = "Consumes connected-component label slabs and their local object counts. It records label equivalences across slab boundaries, accumulates per-slab voxel counts and bounding boxes on the fly, then reduces both the label mapping and statistics to the final whole-stack component labels. The resulting table can be used by collapseComponentLabels and also contains global component statistics."
          Aliases = [ "connected"; "components"; "translation"; "table"; "statistics"; "stats"; "reducer"; "labels" ]
          Inputs = [ makePort "Labels + count" connectedComponentLabels ]
          Outputs = [ makePort "TranslationTable" translationTable ]
          Parameters = [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32) ] }

        { Id = "CollapseComponentLabels"
          DisplayName = "collapseComponentLabels"
          Category = "Binary Morphology"
          Summary = "Collapse chunk-local component labels using a translation table."
          Description = collapseLabelsDescription
          Aliases = [ "connected"; "components"; "translation"; "table"; "update"; "labels" ]
          Inputs = [ makePort "UInt64" imageUInt64 ]
          Outputs = [ makePort "UInt64" imageUInt64 ]
          Parameters =
              [ makeParameter "windowSize" "Window size" "3" (BasicType.Numeric UInt32)
                makeParameter "translationTable" "Translation table" "" BasicType.String ] }

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
          Description = permuteAxesDescription
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
