module Tests.StudioGraphTests

open System.IO
open System.Text
open Expecto
open Studio.Graph

let private savedParam key value useInput =
    { Key = key; Value = value; UseInput = useInput }

let private savedNode id functionId parameters =
    { Id = id
      FunctionId = functionId
      X = 12.5
      Y = -7.0
      Parameters = parameters |> List.toArray }

[<Tests>]
let domainSuite =
    testList "Studio.Graph domain" [
        testCase "numeric and basic type strings roundtrip" <| fun _ ->
            let numericTypes =
                [ Number; UInt8; Int8; UInt16; Int16; UInt32; Int32; UInt64; Int64; Float32; Float64; Complex64; Complex ]

            for numericType in numericTypes do
                let text = NumericType.toString numericType
                Expect.equal (NumericType.tryParse text) (Some numericType) $"NumericType should parse {text}."
                Expect.equal (BasicType.tryParse text) (Some(BasicType.Numeric numericType)) $"BasicType should parse numeric {text}."

            Expect.equal (BasicType.tryParse "String") (Some BasicType.String) "String should parse."
            Expect.equal (BasicType.tryParse "Map") (Some BasicType.Map) "Map should parse."
            Expect.equal (BasicType.tryParse "nope") None "Unknown type should not parse."

        testCase "port compatibility supports Any and Number wildcards" <| fun _ ->
            Expect.isTrue (PortType.canConnect Any (Scalar String)) "Any output should connect to concrete input."
            Expect.isTrue (PortType.canConnect (Image UInt8) (Image Number)) "Concrete image should connect to Number image input."
            Expect.isTrue (PortType.canConnect (Image Number) (Image Float64)) "Number image should connect to concrete image input."
            Expect.isTrue (PortType.canConnect (Custom "ObjectSizeStats") (Custom "Record")) "Object-size statistics should connect to the generic Expand record input."
            Expect.isFalse (PortType.canConnect (Scalar String) (Scalar(BasicType.Numeric Float64))) "Different concrete scalar types should not connect."

        testCase "image file format table captures important pixel type restrictions" <| fun _ ->
            Expect.isFalse (ImageFileFormat.supports ".tiff" Float64) "TIFF stack output should not offer Float64 because StackIO.write rejects it."
            Expect.isFalse (ImageFileFormat.supports ".tif" Float64) "TIF output should be treated like TIFF."
            Expect.isTrue (ImageFileFormat.readSupports ".tiff" Float64) "TIFF input should still allow Float64."
            Expect.isTrue (ImageFileFormat.supports ".tiff" Float32) "TIFF stack output should accept Float32."
            Expect.isFalse (ImageFileFormat.supports ".tiff" UInt64) "TIFF should not be offered for 64-bit integer stack output."
            Expect.equal ImageFileFormat.readFormats.Head.Label "TIFF (.tif or .tiff)" "Read format choices should present TIFF suffixes as aliases."
            Expect.isTrue (ImageFileFormat.readFormats |> List.exists (fun format -> format.Label = "JPEG (.jpg or .jpeg)" && format.Suffix = ".jpg")) "Read format choices should present JPEG suffixes as aliases."
            Expect.isFalse (ImageFileFormat.readFormats |> List.exists (fun format -> format.Suffix = ".jpeg")) "Read format choices should not show a duplicate JPEG entry."
            Expect.isTrue (ImageFileFormat.supports ".png" UInt16) "PNG should support 16-bit unsigned slices."
            Expect.isFalse (ImageFileFormat.supports ".png" Float32) "PNG should not accept floating-point output."
            Expect.isTrue (ImageFileFormat.supports ".jpg" UInt8) "JPEG should accept 8-bit unsigned slices."
            Expect.isTrue (ImageFileFormat.supports ".jpeg" UInt8) "JPEG alias should accept 8-bit unsigned slices."
            Expect.isFalse (ImageFileFormat.supports ".jpg" UInt16) "JPEG should be kept to 8-bit output."

        testCase "FunctionDefinition.matches searches display category summary description and aliases" <| fun _ ->
            let read = BuiltInCatalog.find "Read"
            Expect.isTrue (FunctionDefinition.matches "read" read) "Display name should match."
            Expect.isTrue (FunctionDefinition.matches "Sources" read) "Category should match."
            Expect.isTrue (FunctionDefinition.matches "slice" read) "Summary should match."

            let withDescription =
                { read with
                    Summary = ""
                    Description = "Detailed text for unusually specific matching." }

            Expect.isTrue (FunctionDefinition.matches "unusually specific" withDescription) "Description should match."
            Expect.isTrue (FunctionDefinition.matches "tiff" read) "Alias should match."
            Expect.isFalse (FunctionDefinition.matches "definitely-not-here" read) "Unrelated search should not match."
    ]

[<Tests>]
let catalogSuite =
    testList "Studio.Graph catalog" [
        testCase "catalog exposes expected generic functions" <| fun _ ->
            let ids = BuiltInCatalog.orderedFunctions |> List.map _.Id
            Expect.containsAll ids ["Scalar"; "FileDirectory"; "Read"; "ReadRandom"; "EstimateHistogram"; "ReadRange"; "ReadPointSet"; "Zero"; "CoordinateX"; "CoordinateY"; "CoordinateZ"; "NormalNoise"; "SaltAndPepperNoise"; "PoissonNoise"; "Write"; "WriteChunks"; "WriteMesh"; "WritePointSet"; "WriteMatrix"; "Expand"; "GetChunkInfo"; "GetZarrInfo"; "GetNexusInfo"; "Resize"; "Resample"; "CreatePadding"; "Crop"; "MarchingCubes"; "ObjectSurfaceArea"; "DogKeypoints"; "SiftKeypoints"; "LogBlobKeypoints"; "HessianKeypoints"; "Forstner3DKeypoints"; "PhaseCongruencyKeypoints"; "PointPairDistances"; "AffineRegistrationMatrix"; "AffineRegistrationInverseMatrix"; "StreamConnectedObjects"; "MeasureObjects"; "ObjectSizes"; "ObjectSizeStats"; "PaintObjects"; "PaintObjectsCropped"; "ImageOpImage"; "ComputeStats"; "FitBiasModel"; "FitBiasModelMasked"; "CorrectBias"; "CorrectBiasMasked"; "SerialPolynomialBiasCorrect"; "SerialEstTrans"; "SerialApplyTrans"; "SerialEstBoundingBox"; "ObjectVolume"; "Quantiles"; "Chart"; "SumProjection"; "ImHistogram"; "ImHistogramData"; "Histogram"] "Important Studio functions should be in the palette catalog."
            Expect.isFalse (ids |> List.contains "WriteVolume") "Volume-file writing should be selected from the combined Write box."
            Expect.isFalse (ids |> List.contains "WriteZarr") "Zarr writing should be selected from the combined Write box."
            Expect.isFalse (ids |> List.contains "WriteNexus") "NeXus writing should be selected from the combined Write box."
            Expect.isFalse (ids |> List.contains "GetStackInfo") "Stack info should come from read/write boxes in Studio."
            Expect.isFalse (ids |> List.contains "WriteThrough") "writeThrough is a DSL/internal primitive and should stay hidden from Studio."
            Expect.equal (BuiltInCatalog.find "WriteChunks").Outputs.[0].Type (PortType.Custom "ChunkInfo") "writeChunks should expose ChunkInfo for downstream print/expand branches."
            Expect.containsAll ids ["AddNormalNoise"; "AddSaltAndPepperNoise"; "AddPoissonNoise"; "AddSpeckleNoise"] "Noise add-stage boxes should be available in Studio."
            Expect.equal (BuiltInCatalog.find "CoordinateX").Outputs.[0].Type (PortType.Image NumericType.Float64) "coordinateX should expose its fixed Float64 image output."
            Expect.equal (BuiltInCatalog.find "CoordinateY").Outputs.[0].Type (PortType.Image NumericType.Float64) "coordinateY should expose its fixed Float64 image output."
            Expect.equal (BuiltInCatalog.find "CoordinateZ").Outputs.[0].Type (PortType.Image NumericType.Float64) "coordinateZ should expose its fixed Float64 image output."
            Expect.containsAll ids ["Convolve"; "SignedDistanceBand"; "OtsuThresholdFromHistogram"; "MomentsThresholdFromHistogram"; "ResampleAffine"] "The StackProcessing DSL algorithms requested for Studio should be in the palette catalog."
            Expect.isFalse (ids |> List.contains "ReadSlab") "ReadSlab is retired from the Chunk Studio surface."
            Expect.isFalse (ids |> List.contains "RelabelComponents") "Legacy connected-component relabeling is retired from the Chunk Studio surface."
            Expect.isFalse (ids |> List.contains "ComponentTranslationTable") "Legacy connected-component translation tables are retired from the Chunk Studio surface."
            Expect.isFalse (ids |> List.contains "CollapseComponentLabels") "Legacy connected-component label collapse is retired from the Chunk Studio surface."
            Expect.isFalse (ids |> List.contains "BinaryFillHoles") "binaryFillHoles is a whole-stack SimpleITK operation and should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "SignedDistanceMap") "signedDistanceMap is the lower-level whole-image name; Studio should expose signedDistanceBand."
            Expect.isFalse (ids |> List.contains "Watershed") "watershed is not exposed as an LMIP Studio box because basin labels are not local to independent z-windows."
            Expect.isFalse (ids |> List.contains "OtsuThreshold") "whole-image-style otsuThreshold should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "MomentsThreshold") "whole-image-style momentsThreshold should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "SampledOtsuThreshold") "sampledOtsuThreshold hides the histogram-to-threshold step and should not be exposed."
            Expect.isFalse (ids |> List.contains "SampledMomentsThreshold") "sampledMomentsThreshold hides the histogram-to-threshold step and should not be exposed."
            Expect.containsAll ids ["Clamp"; "ShiftScale"; "IntensityStretch"; "HistogramEqualization"; "SmoothWMedian"; "SmoothWBilateral"; "GradientMagnitudeSquared"; "SobelEdge"; "Laplacian"; "ImageComparison"; "MaskLogic"; "MaskNot"] "The high-value SimpleITK filter families should be available in Studio."
            Expect.isFalse (ids |> List.contains "Mask") "mask is intentionally not exposed; use binary arithmetic/logical stages directly."

        testCase "read source boxes default to Float32 image streams" <| fun _ ->
            for functionId in [ "Read"; "ReadRandom"; "ReadRange" ] do
                let definition = BuiltInCatalog.find functionId
                let typeParameter = definition.Parameters |> List.find (fun parameter -> parameter.Key = "type")

                Expect.equal typeParameter.DefaultValue "Float32" $"{functionId} should default its Type parameter to Float32."
                Expect.equal definition.Outputs.[0].Type (PortType.Image NumericType.Float32) $"{functionId} should expose a Float32 image output by default."

            let histogram = BuiltInCatalog.find "EstimateHistogram"
            let histogramType = histogram.Parameters |> List.find (fun parameter -> parameter.Key = "type")
            Expect.equal histogramType.DefaultValue "Float32" "EstimateHistogram should sample Float32 images by default."

        testCase "geometry measurement catalog uses mesh and scalar reducer ports" <| fun _ ->
            let ids = BuiltInCatalog.orderedFunctions |> List.map _.Id
            let surfaceArea = BuiltInCatalog.find "ObjectSurfaceArea"
            let volume = BuiltInCatalog.find "ObjectVolume"
            let pointPairDistances = BuiltInCatalog.find "PointPairDistances"
            let affineRegistration = BuiltInCatalog.find "AffineRegistrationMatrix"
            let affineRegistrationInverse = BuiltInCatalog.find "AffineRegistrationInverseMatrix"
            let writeMatrix = BuiltInCatalog.find "WriteMatrix"

            Expect.equal surfaceArea.Inputs.[0].Type (PortType.Custom "Mesh") "objectSurfaceArea should consume streamed triangle sets."
            Expect.equal surfaceArea.Outputs.[0].Type (PortType.Scalar(BasicType.Numeric Float64)) "objectSurfaceArea should emit a scalar Float64 reducer output."
            Expect.equal surfaceArea.Parameters.Length 3 "objectSurfaceArea should expose x/y/z unit parameters."
            Expect.equal volume.Inputs.[0].Type (PortType.Image UInt8) "objectVolume should consume UInt8 0-1 mask slices."
            Expect.equal volume.Outputs.[0].Type (PortType.Scalar(BasicType.Numeric Float64)) "objectVolume should emit a scalar Float64 reducer output."
            Expect.equal volume.Parameters.Length 3 "objectVolume should expose x/y/z unit parameters."
            Expect.equal pointPairDistances.Inputs.[0].Type (PortType.Custom "PointSet") "pointPairDistances should consume point sets."
            Expect.equal pointPairDistances.Outputs.[0].Type (PortType.Custom "Float64Matrix") "pointPairDistances should emit a vectorized Float64 matrix."
            Expect.equal (affineRegistration.Inputs |> List.map _.Type) [ PortType.Custom "PointSet"; PortType.Custom "PointSet" ] "affineRegistration should consume fixed and moving point sets."
            Expect.equal (affineRegistration.Outputs |> List.map _.Type) [ PortType.Custom "Float64Matrix" ] "affineRegistrationMatrix should emit one transform matrix."
            Expect.equal (affineRegistrationInverse.Outputs |> List.map _.Type) [ PortType.Custom "Float64Matrix" ] "affineRegistrationInverseMatrix should emit one inverse matrix."
            Expect.equal writeMatrix.Inputs.[0].Type (PortType.Custom "Float64Matrix") "writeMatrix should consume vectorized Float64 matrices."
            Expect.isFalse (ids |> List.contains "Normalize") "normalize is intentionally not exposed as a streaming Studio box; use computeStats plus shiftScale."
            Expect.isFalse (ids |> List.contains "RescaleIntensity") "rescaleIntensity is intentionally not exposed as a streaming Studio box; use sampled statistics or quantiles plus intensityStretch."
            Expect.isFalse (ids |> List.contains "InvertIntensity") "invertIntensity requires a known maximum; use estimated statistics plus shiftScale."
            Expect.containsAll ids ["GrayscaleErode"; "GrayscaleDilate"; "GrayscaleOpening"; "GrayscaleClosing"; "WhiteTopHat"; "BlackTopHat"; "MorphologicalGradient"] "Grayscale morphology filters should be available in Studio."
            Expect.containsAll ids ["DilateZonohedral"; "ErodeZonohedral"; "OpeningZonohedral"; "ClosingZonohedral"; "BinaryContour"; "BinaryMedian"; "RemoveSmallObjects"; "FillSmallHoles"; "LabelContour"; "ChangeLabel"] "Extra binary morphology and label analysis filters should be available in Studio."
            Expect.isFalse (ids |> List.contains "BinaryOpeningByReconstruction") "reconstruction filters are not exposed as LMIP Studio boxes because geodesic propagation is not bounded by a local z-window."
            Expect.isFalse (ids |> List.contains "BinaryClosingByReconstruction") "reconstruction filters are not exposed as LMIP Studio boxes because geodesic propagation is not bounded by a local z-window."
            Expect.isFalse (ids |> List.contains "BinaryReconstructionByDilation") "binary reconstruction by dilation is intentionally kept out of Studio's LMIP layer."
            Expect.isFalse (ids |> List.contains "BinaryReconstructionByErosion") "binary reconstruction by erosion is intentionally kept out of Studio's LMIP layer."
            Expect.isFalse (ids |> List.contains "VotingBinaryHoleFilling") "votingBinaryHoleFilling is replaced in Studio by connected-size fillSmallHoles."
            Expect.isFalse (ids |> List.contains "LabelShapeStatistics") "labelShapeStatistics is whole-image/block-local and should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "LabelIntensityStatistics") "labelIntensityStatistics is whole-image/block-local and should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "LabelOverlapMeasures") "labelOverlapMeasures is whole-image/block-local and should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "BinaryThinning") "binaryThinning requires iterative skeletonization and is out of scope for LMIP streaming."
            Expect.isFalse (ids |> List.contains "MaxOfPair") "maxOfPair should be available through ImageOpImage, not as a legacy palette box."
            Expect.isFalse (ids |> List.contains "MinOfPair") "minOfPair should be available through ImageOpImage, not as a legacy palette box."
            Expect.contains ids "SmoothWGauss" "Gaussian smoothing should be exposed with the user-facing smoothWGauss name."
            Expect.isFalse (ids |> List.contains "DiscreteGaussian") "discreteGaussian is an internal/Image-level name, not a Studio box."
            Expect.isFalse (ids |> List.contains "ConvGauss") "convGauss is not kept as a legacy Studio box."
            Expect.containsAll ids ["ToVectorImage"; "AppendVectorElement"; "VectorElement"; "VectorRange"; "VectorCast"; "ColorToVector3"; "VectorMapElements"; "VectorDot"; "VectorCross3D"; "VectorAngleTo"; "Gradient"; "StructureTensor"; "SymmetricMatrixEigensystem"; "SymmetricMatrixEigenvalues"; "SymmetricMatrixEigenvector"] "Vector-valued image composition and pixelwise vector operations should be available in Studio."
            Expect.containsAll ids ["ComplexFromReIm"; "ComplexPolar"; "ComplexRe"; "ComplexIm"; "ComplexModulus"; "ComplexArg"; "ComplexConjugate"] "Complex-valued image composition and unary operations should be available in Studio."
            Expect.containsAll ids ["FFT"; "InvFFT"; "ShiftFFT"] "Chunk-backed Fourier transforms should be available in Studio."

        testCase "vector image catalog uses vector-valued ports" <| fun _ ->
            let vectorType = PortType.Custom "VectorImageFloat64"
            let colorType = PortType.Custom "ColorImage"
            let toVectorImage = BuiltInCatalog.find "ToVectorImage"
            let appendVectorElement = BuiltInCatalog.find "AppendVectorElement"
            let vectorElement = BuiltInCatalog.find "VectorElement"
            let vectorRange = BuiltInCatalog.find "VectorRange"
            let vectorCast = BuiltInCatalog.find "VectorCast"
            let intensityStretch = BuiltInCatalog.find "IntensityStretch"
            let colorToVector3 = BuiltInCatalog.find "ColorToVector3"
            let vectorDot = BuiltInCatalog.find "VectorDot"
            let vectorCross3D = BuiltInCatalog.find "VectorCross3D"
            let vectorAngleTo = BuiltInCatalog.find "VectorAngleTo"
            let gradient = BuiltInCatalog.find "Gradient"
            let structureTensor = BuiltInCatalog.find "StructureTensor"
            let symmetricMatrixEigensystem = BuiltInCatalog.find "SymmetricMatrixEigensystem"
            let symmetricMatrixEigenvalues = BuiltInCatalog.find "SymmetricMatrixEigenvalues"
            let symmetricMatrixEigenvector = BuiltInCatalog.find "SymmetricMatrixEigenvector"
            let covarianceMatrix = BuiltInCatalog.find "CovarianceMatrix"
            let pca = BuiltInCatalog.find "PCA"

            Expect.equal toVectorImage.Outputs.[0].Type vectorType "toVectorImage should emit a vector-valued image stream."
            Expect.equal appendVectorElement.Inputs.[0].Type vectorType "appendVectorElement should consume an existing vector image."
            Expect.equal appendVectorElement.Inputs.[1].Type (PortType.Image Float64) "appendVectorElement should append a scalar Float64 image."
            Expect.equal appendVectorElement.Outputs.[0].Type vectorType "appendVectorElement should keep the stream vector-valued."
            Expect.equal vectorElement.Outputs.[0].Type (PortType.Image Float64) "vectorElement should extract a scalar image."
            Expect.equal vectorRange.Inputs.[0].Type vectorType "vectorRange should consume a vector-valued image."
            Expect.equal vectorRange.Outputs.[0].Type vectorType "vectorRange should emit a vector-valued image."
            Expect.equal vectorCast.Inputs.[0].Type vectorType "vectorCast should consume a vector-valued image."
            Expect.equal vectorCast.Outputs.[0].Type colorType "vectorCast to UInt8 should emit a color image."
            Expect.isTrue (PortType.canConnect vectorType intensityStretch.Inputs.[0].Type) "intensityStretch should consume a vector-valued image."
            Expect.isTrue (PortType.canConnect intensityStretch.Outputs.[0].Type vectorType) "intensityStretch should emit a vector-valued image when connected to vector stages."
            Expect.equal colorToVector3.Inputs.[0].Type colorType "colorToVector3 should consume a color image."
            Expect.equal colorToVector3.Outputs.[0].Type vectorType "colorToVector3 should emit a vector-valued image."
            Expect.equal vectorDot.Outputs.[0].Type (PortType.Image Float64) "vectorDot should reduce vectors to scalar pixels."
            Expect.equal vectorCross3D.Outputs.[0].Type vectorType "vectorCross3D should preserve vector-valued pixels."
            Expect.equal vectorAngleTo.Outputs.[0].Type (PortType.Image Float64) "vectorAngleTo should reduce vectors to scalar angles."
            Expect.equal gradient.Inputs.[0].Type (PortType.Image Float32) "gradient should consume scalar Float32 chunk slices."
            Expect.equal gradient.Outputs.[0].Type (PortType.Custom "VectorImageFloat32") "gradient should emit Float32 vector-valued chunks."
            Expect.equal structureTensor.Inputs.[0].Type (PortType.Image Float32) "structureTensor should consume scalar Float32 chunk slices."
            Expect.equal (structureTensor.Outputs |> List.map _.Type) [ PortType.Custom "VectorImageFloat32" ] "structureTensor should expose one Float32 six-component tensor output."
            Expect.equal symmetricMatrixEigensystem.Inputs.[0].Type (PortType.Custom "VectorImageFloat32") "symmetricMatrixEigensystem should consume Float32 vector-valued tensor chunks."
            Expect.equal symmetricMatrixEigensystem.Outputs.[0].Type (PortType.Custom "VectorImageFloat32") "symmetricMatrixEigensystem should expose one Float32 vectorized 3x4 eigensystem output."
            Expect.equal symmetricMatrixEigenvalues.Inputs.[0].Type (PortType.Custom "VectorImageFloat32") "symmetricMatrixEigenvalues should consume Float32 vector-valued tensor chunks."
            Expect.equal symmetricMatrixEigenvalues.Outputs.[0].Type (PortType.Custom "VectorImageFloat32") "symmetricMatrixEigenvalues should expose one Float32 three-component eigenvalue vector output."
            Expect.equal symmetricMatrixEigenvector.Inputs.[0].Type (PortType.Custom "VectorImageFloat32") "symmetricMatrixEigenvector should consume Float32 vector-valued tensor chunks."
            Expect.equal symmetricMatrixEigenvector.Outputs.[0].Type (PortType.Custom "VectorImageFloat32") "symmetricMatrixEigenvector should expose one Float32 three-component vector output."
            Expect.equal covarianceMatrix.Inputs.[0].Type (PortType.Custom "VectorImageFloat32") "covarianceMatrix should consume Float32 vector-valued chunks."
            Expect.equal covarianceMatrix.Outputs.[0].Type (PortType.Custom "Float64Matrix") "covarianceMatrix should expose one global Float64 matrix."
            Expect.equal pca.Inputs.[0].Type (PortType.Custom "VectorImageFloat32") "PCA should consume Float32 vector-valued chunks."
            Expect.equal pca.Outputs.Length 1 "PCA should expose one global covariance matrix output."
            Expect.equal pca.Outputs.[0].Type (PortType.Custom "Float64Matrix") "PCA output should be a Float64 matrix."

        testCase "complex image catalog uses complex and Float64 ports" <| fun _ ->
            let fromReIm = BuiltInCatalog.find "ComplexFromReIm"
            let polar = BuiltInCatalog.find "ComplexPolar"
            let re = BuiltInCatalog.find "ComplexRe"
            let im = BuiltInCatalog.find "ComplexIm"
            let modulus = BuiltInCatalog.find "ComplexModulus"
            let arg = BuiltInCatalog.find "ComplexArg"
            let conjugate = BuiltInCatalog.find "ComplexConjugate"

            Expect.equal fromReIm.Inputs.[0].Type (PortType.Image Float64) "toComplex should consume a Float64 real image."
            Expect.equal fromReIm.Inputs.[1].Type (PortType.Image Float64) "toComplex should consume a Float64 imaginary image."
            Expect.equal fromReIm.Outputs.[0].Type (PortType.Image Complex64) "toComplex should emit a compact Complex64 image."
            Expect.equal polar.Outputs.[0].Type (PortType.Image Complex64) "polarToComplex should emit a compact Complex64 image."
            Expect.equal re.Outputs.[0].Type (PortType.Image Float64) "Re should emit Float64."
            Expect.equal im.Outputs.[0].Type (PortType.Image Float64) "Im should emit Float64."
            Expect.equal modulus.Outputs.[0].Type (PortType.Image Float64) "modulus should emit Float64."
            Expect.equal arg.Outputs.[0].Type (PortType.Image Float64) "arg should emit Float64."
            Expect.equal conjugate.Inputs.[0].Type (PortType.Image Complex64) "conjugate should consume Complex64."
            Expect.equal conjugate.Outputs.[0].Type (PortType.Image Complex64) "conjugate should emit Complex64."

        testCase "fourier catalog uses scalar complex and chunk parameters" <| fun _ ->
            let fft = BuiltInCatalog.find "FFT"
            let invFFT = BuiltInCatalog.find "InvFFT"
            let shiftFFT = BuiltInCatalog.find "ShiftFFT"

            Expect.equal fft.Inputs.[0].Type (PortType.Image Float32) "FFT should consume Float32 chunk slices."
            Expect.equal fft.Outputs.[0].Type (PortType.Image Complex64) "FFT should emit compact Complex64 slices."
            Expect.equal invFFT.Inputs.[0].Type (PortType.Image Complex64) "invFFT should consume Complex64 slices."
            Expect.equal invFFT.Outputs.[0].Type (PortType.Image Float32) "invFFT should emit Float32 slices."
            Expect.equal shiftFFT.Inputs.[0].Type (PortType.Image Complex64) "shiftFFT should consume Complex64 slices."
            Expect.equal shiftFFT.Outputs.[0].Type (PortType.Image Complex64) "shiftFFT should emit Complex64 slices."
            Expect.containsAll (fft.Parameters |> List.map _.Key) [ "type"; "chunkX"; "chunkY"; "chunkZ" ] "FFT should expose type and chunk controls."

        testCase "file directory source emits a string scalar" <| fun _ ->
            let fileDirectory = BuiltInCatalog.find "FileDirectory"

            Expect.equal fileDirectory.DisplayName "file/directory" "File/directory should have a compact palette name."
            Expect.equal fileDirectory.Inputs [] "File/directory is a source."
            Expect.equal fileDirectory.Outputs.[0].Type (Scalar BasicType.String) "File/directory should emit a string path."
            Expect.equal fileDirectory.Parameters.[0].Key "kind" "The picker kind should be the first parameter."
            Expect.equal fileDirectory.Parameters.[1].Key "value" "The resolved path should be stored in value."

        testCase "catalog entries have palette summaries and detailed descriptions" <| fun _ ->
            let functions = BuiltInCatalog.orderedFunctions
            Expect.isNonEmpty functions "The built-in catalog should not be empty."
            Expect.isTrue (functions |> List.forall (fun f -> not (System.String.IsNullOrWhiteSpace f.Summary))) "Every palette entry should have hover summary text."
            Expect.isTrue (functions |> List.forall (fun f -> not (System.String.IsNullOrWhiteSpace f.Description))) "Every palette entry should have non-programmer-oriented description text."

        testCase "debug and visualization functions expose parameter input ports" <| fun _ ->
            let print = BuiltInCatalog.find "Print"
            let chart = BuiltInCatalog.find "Chart"

            Expect.equal (print.Parameters |> List.filter (fun p -> p.Key.StartsWith("input")) |> List.length) 8 "Print should expose eight activatable inputs."
            Expect.equal print.Outputs [] "Print is a side-effecting scalar sink."
            Expect.equal chart.Inputs [] "Chart receives map data through an always-on parameter port."
            Expect.equal chart.Parameters.[0].Key "kind" "Chart should expose the chart kind first."
            Expect.equal chart.Parameters.[1].Type BasicType.Map "Chart input should be map-like histogram data."

        testCase "connected component catalog exposes direct Chunk label images" <| fun _ ->
            let connected = BuiltInCatalog.find "ConnectedComponents"

            Expect.equal connected.Outputs.[0].Type (Image UInt32) "Connected components should stream compact UInt32 label slices directly."
            Expect.throws (fun () -> BuiltInCatalog.find "ComponentTranslationTable" |> ignore) "The legacy translation-table reducer should not be exposed."

        testCase "find fails clearly for missing function" <| fun _ ->
            Expect.throws (fun () -> BuiltInCatalog.find "MissingFunction" |> ignore) "Missing function lookup should fail."
    ]

[<Tests>]
let persistenceSuite =
    testList "Studio.Graph persistence" [
        testCase "serialize deserialize roundtrips saved graph" <| fun _ ->
            let graph =
                { Version = 1
                  Nodes =
                    [| savedNode "n1" "Scalar" [ savedParam "type" "String" false; savedParam "value" "input" false ]
                       savedNode "n2" "Read" [ savedParam "input" "input" true ] |]
                  Edges =
                    [| { FromNode = "n1"; FromKind = "scalarOutput"; FromPort = 0; ToNode = "n2"; ToKind = "parameterInput"; ToPort = 0 } |] }

            let json = PipelineGraphStorage.serialize graph
            let restored = PipelineGraphStorage.deserialize json
            Expect.equal restored graph "Saved graph should survive JSON roundtrip."
            Expect.stringContains json "\"functionId\": \"Scalar\"" "JSON should use camelCase field names."

        testCase "writeJsonAsync truncates and writes stream" <| fun _ ->
            let graph =
                { Version = 2
                  Nodes = [| savedNode "scalar" "Scalar" [ savedParam "type" "Float64" false ] |]
                  Edges = [||] }

            use stream = new MemoryStream()
            let oldBytes = Encoding.UTF8.GetBytes("old text that should be removed")
            stream.Write(oldBytes, 0, oldBytes.Length)
            stream.Position <- 0L
            PipelineGraphStorage.writeJsonAsync stream graph
            |> Async.AwaitTask
            |> Async.RunSynchronously

            stream.Position <- 0L
            let restored =
                PipelineGraphStorage.readJsonAsync stream
                |> Async.AwaitTask
                |> Async.RunSynchronously

            Expect.equal restored graph "Stream roundtrip should preserve graph."
            Expect.isFalse ((Encoding.UTF8.GetString(stream.ToArray())).Contains("old text")) "Old stream contents should be truncated before writing."
    ]
