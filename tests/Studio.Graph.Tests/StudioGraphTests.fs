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
                [ Number; UInt8; Int8; UInt16; Int16; UInt32; Int32; UInt64; Int64; Float32; Float64; Complex ]

            for numericType in numericTypes do
                let text = NumericType.toString numericType
                Expect.equal (NumericType.tryParse text) (Some numericType) $"NumericType should parse {text}."
                Expect.equal (BasicType.tryParse text) (Some(BasicType.Numeric numericType)) $"BasicType should parse numeric {text}."

            Expect.equal (BasicType.tryParse "String") (Some BasicType.String) "String should parse."
            Expect.equal (BasicType.tryParse "Histogram") (Some BasicType.Map) "Histogram is currently stored as Map."
            Expect.equal (BasicType.tryParse "nope") None "Unknown type should not parse."

        testCase "port compatibility supports Any and Number wildcards" <| fun _ ->
            Expect.isTrue (PortType.canConnect Any (Scalar String)) "Any output should connect to concrete input."
            Expect.isTrue (PortType.canConnect (Image UInt8) (Image Number)) "Concrete image should connect to Number image input."
            Expect.isTrue (PortType.canConnect (Image Number) (Image Float64)) "Number image should connect to concrete image input."
            Expect.isFalse (PortType.canConnect (Scalar String) (Scalar(BasicType.Numeric Float64))) "Different concrete scalar types should not connect."

        testCase "image file format table captures important pixel type restrictions" <| fun _ ->
            Expect.isTrue (ImageFileFormat.supports ".tiff" Float64) "TIFF should support floating-point image stacks."
            Expect.isTrue (ImageFileFormat.supports ".tif" Float64) "TIF should be treated like TIFF."
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
            Expect.isTrue (FunctionDefinition.matches "chunked" read) "Summary should match."

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
            Expect.containsAll ids ["Scalar"; "FileDirectory"; "Read"; "ReadRandom"; "ReadRange"; "ReadSlab"; "ReadZarrSlab"; "ReadNexusSlab"; "ReadPointSet"; "Zero"; "CoordinateX"; "CoordinateY"; "CoordinateZ"; "NormalNoise"; "SaltAndPepperNoise"; "ShotNoise"; "SpeckleNoise"; "Write"; "WriteInSlabs"; "WriteZarr"; "WriteNexus"; "WriteMesh"; "WritePointSet"; "WriteMatrix"; "GetStackInfo"; "GetChunkInfo"; "GetZarrInfo"; "GetNexusInfo"; "Resize"; "Resample"; "CreatePadding"; "Crop"; "MarchingCubes"; "SurfaceArea"; "DogKeypoints"; "SiftKeypoints"; "LogBlobKeypoints"; "HessianKeypoints"; "Harris3DKeypoints"; "Forstner3DKeypoints"; "PhaseCongruencyKeypoints"; "PointPairDistances"; "AffineRegistration"; "StreamConnectedObjects"; "PaintObjects"; "PaintObjectsCropped"; "ImageOpImage"; "ComputeStats"; "FitBiasModel"; "FitBiasModelMasked"; "CorrectBias"; "CorrectBiasMasked"; "SerialPolynomialBiasCorrect"; "SerialKeypoints2D"; "SerialKeypointTranslationManifest"; "SerialImageTranslationManifest"; "SerialApplyManifest"; "SerialApplyManifestInBoundingBox"; "Volume"; "Quantiles"; "Chart"; "SumProjection"] "Important Studio functions should be in the palette catalog."
            Expect.containsAll ids ["AddNormalNoise"; "AddSaltAndPepperNoise"; "AddShotNoise"; "AddSpeckleNoise"] "Noise add-stage boxes should be available in Studio."
            Expect.containsAll ids ["Convolve"; "RelabelComponents"; "SignedDistanceBand"; "OtsuThresholdFromHistogram"; "MomentsThresholdFromHistogram"; "ResampleAffineTrilinearSlices"] "The StackProcessing DSL algorithms requested for Studio should be in the palette catalog."
            Expect.isFalse (ids |> List.contains "BinaryFillHoles") "binaryFillHoles is a whole-stack SimpleITK operation and should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "SignedDistanceMap") "signedDistanceMap is the lower-level whole-image name; Studio should expose signedDistanceBand."
            Expect.isFalse (ids |> List.contains "Watershed") "watershed is not exposed as an LMIP Studio box because basin labels are not local to independent z-windows."
            Expect.isFalse (ids |> List.contains "OtsuThreshold") "whole-image-style otsuThreshold should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "MomentsThreshold") "whole-image-style momentsThreshold should not be exposed as an LMIP Studio box."
            Expect.isFalse (ids |> List.contains "SampledOtsuThreshold") "sampledOtsuThreshold hides the histogram-to-threshold step and should not be exposed."
            Expect.isFalse (ids |> List.contains "SampledMomentsThreshold") "sampledMomentsThreshold hides the histogram-to-threshold step and should not be exposed."
            Expect.containsAll ids ["Clamp"; "ShiftScale"; "IntensityStretch"; "SmoothWMedian"; "SmoothWBilateral"; "GradientMagnitude"; "SobelEdge"; "Laplacian"; "ImageComparison"; "MaskLogic"; "MaskNot"] "The high-value SimpleITK filter families should be available in Studio."
            Expect.isFalse (ids |> List.contains "Mask") "mask is intentionally not exposed; use binary arithmetic/logical stages directly."

        testCase "geometry measurement catalog uses mesh and scalar reducer ports" <| fun _ ->
            let ids = BuiltInCatalog.orderedFunctions |> List.map _.Id
            let surfaceArea = BuiltInCatalog.find "SurfaceArea"
            let volume = BuiltInCatalog.find "Volume"
            let pointPairDistances = BuiltInCatalog.find "PointPairDistances"
            let affineRegistration = BuiltInCatalog.find "AffineRegistration"
            let writeMatrix = BuiltInCatalog.find "WriteMatrix"

            Expect.equal surfaceArea.Inputs.[0].Type (PortType.Custom "Mesh") "surfaceArea should consume streamed triangle sets."
            Expect.equal surfaceArea.Outputs.[0].Type (PortType.Scalar(BasicType.Numeric Float64)) "surfaceArea should emit a scalar Float64 reducer output."
            Expect.equal surfaceArea.Parameters.Length 3 "surfaceArea should expose x/y/z unit parameters."
            Expect.equal volume.Inputs.[0].Type (PortType.Image UInt8) "volume should consume UInt8 0-1 mask slices."
            Expect.equal volume.Outputs.[0].Type (PortType.Scalar(BasicType.Numeric Float64)) "volume should emit a scalar Float64 reducer output."
            Expect.equal volume.Parameters.Length 3 "volume should expose x/y/z unit parameters."
            Expect.equal pointPairDistances.Inputs.[0].Type (PortType.Custom "PointSet") "pointPairDistances should consume point sets."
            Expect.equal pointPairDistances.Outputs.[0].Type (PortType.Custom "Float64Matrix") "pointPairDistances should emit a vectorized Float64 matrix."
            Expect.equal (affineRegistration.Inputs |> List.map _.Type) [ PortType.Custom "PointSet"; PortType.Custom "PointSet" ] "affineRegistration should consume fixed and moving point sets."
            Expect.equal (affineRegistration.Outputs |> List.map _.Type) [ PortType.Custom "Float64Matrix"; PortType.Custom "Float64Matrix" ] "affineRegistration should emit transform and inverse matrices."
            Expect.equal writeMatrix.Inputs.[0].Type (PortType.Custom "Float64Matrix") "writeMatrix should consume vectorized Float64 matrices."
            Expect.isFalse (ids |> List.contains "Normalize") "normalize is intentionally not exposed as a streaming Studio box; use computeStats plus shiftScale."
            Expect.isFalse (ids |> List.contains "RescaleIntensity") "rescaleIntensity is intentionally not exposed as a streaming Studio box; use sampled statistics or quantiles plus intensityStretch."
            Expect.isFalse (ids |> List.contains "IntensityWindow") "intensityWindow overlaps with intensityStretch and should not be exposed as a separate Studio box."
            Expect.isFalse (ids |> List.contains "InvertIntensity") "invertIntensity requires a known maximum; use estimated statistics plus shiftScale."
            Expect.containsAll ids ["GrayscaleErode"; "GrayscaleDilate"; "GrayscaleOpening"; "GrayscaleClosing"; "WhiteTopHat"; "BlackTopHat"; "MorphologicalGradient"] "Grayscale morphology filters should be available in Studio."
            Expect.containsAll ids ["BinaryContour"; "BinaryMedian"; "RemoveSmallObjects"; "FillSmallHoles"; "LabelContour"; "ChangeLabel"] "Extra binary morphology and label analysis filters should be available in Studio."
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
            Expect.containsAll ids ["ToVectorImage"; "AppendVectorElement"; "VectorElement"; "VectorMapElements"; "VectorDot"; "VectorCross3D"; "VectorAngleTo"; "Gradient"; "StructureTensor"] "Vector-valued image composition and pixelwise vector operations should be available in Studio."
            Expect.containsAll ids ["ComplexFromReIm"; "ComplexPolar"; "ComplexRe"; "ComplexIm"; "ComplexModulus"; "ComplexArg"; "ComplexConjugate"] "Complex-valued image composition and unary operations should be available in Studio."
            Expect.containsAll ids ["FFT"; "InvFFT"; "ShiftFFT"] "Chunk-backed Fourier transforms should be available in Studio."

        testCase "vector image catalog uses vector-valued ports" <| fun _ ->
            let vectorType = PortType.Custom "VectorImageFloat64"
            let toVectorImage = BuiltInCatalog.find "ToVectorImage"
            let appendVectorElement = BuiltInCatalog.find "AppendVectorElement"
            let vectorElement = BuiltInCatalog.find "VectorElement"
            let vectorDot = BuiltInCatalog.find "VectorDot"
            let vectorCross3D = BuiltInCatalog.find "VectorCross3D"
            let vectorAngleTo = BuiltInCatalog.find "VectorAngleTo"
            let gradient = BuiltInCatalog.find "Gradient"
            let structureTensor = BuiltInCatalog.find "StructureTensor"
            let pca = BuiltInCatalog.find "PCA"

            Expect.equal toVectorImage.Outputs.[0].Type vectorType "toVectorImage should emit a vector-valued image stream."
            Expect.equal appendVectorElement.Inputs.[0].Type vectorType "appendVectorElement should consume an existing vector image."
            Expect.equal appendVectorElement.Inputs.[1].Type (PortType.Image Float64) "appendVectorElement should append a scalar Float64 image."
            Expect.equal appendVectorElement.Outputs.[0].Type vectorType "appendVectorElement should keep the stream vector-valued."
            Expect.equal vectorElement.Outputs.[0].Type (PortType.Image Float64) "vectorElement should extract a scalar image."
            Expect.equal vectorDot.Outputs.[0].Type (PortType.Image Float64) "vectorDot should reduce vectors to scalar pixels."
            Expect.equal vectorCross3D.Outputs.[0].Type vectorType "vectorCross3D should preserve vector-valued pixels."
            Expect.equal vectorAngleTo.Outputs.[0].Type (PortType.Image Float64) "vectorAngleTo should reduce vectors to scalar angles."
            Expect.equal gradient.Inputs.[0].Type (PortType.Image Float64) "gradient should consume scalar Float64 slices."
            Expect.equal gradient.Outputs.[0].Type vectorType "gradient should emit vector-valued pixels."
            Expect.equal structureTensor.Inputs.[0].Type (PortType.Image Float64) "structureTensor should consume scalar Float64 slices."
            Expect.equal (structureTensor.Outputs |> List.map _.Type) [ vectorType; vectorType; vectorType; vectorType ] "structureTensor should expose four vector-valued outputs."
            Expect.equal pca.Inputs.[0].Type vectorType "PCA should consume vector-valued images."
            Expect.equal pca.Outputs.Length 9 "PCA should expose eigenvalues plus up to eight eigenvector outputs."
            Expect.equal (pca.Outputs |> List.map _.Type) (List.replicate 9 vectorType) "PCA outputs should be vector-valued."

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
            Expect.equal fromReIm.Outputs.[0].Type (PortType.Image Complex) "toComplex should emit a native complex image."
            Expect.equal polar.Outputs.[0].Type (PortType.Image Complex) "polarToComplex should emit a native complex image."
            Expect.equal re.Outputs.[0].Type (PortType.Image Float64) "Re should emit Float64."
            Expect.equal im.Outputs.[0].Type (PortType.Image Float64) "Im should emit Float64."
            Expect.equal modulus.Outputs.[0].Type (PortType.Image Float64) "modulus should emit Float64."
            Expect.equal arg.Outputs.[0].Type (PortType.Image Float64) "arg should emit Float64."
            Expect.equal conjugate.Inputs.[0].Type (PortType.Image Complex) "conjugate should consume Complex."
            Expect.equal conjugate.Outputs.[0].Type (PortType.Image Complex) "conjugate should emit Complex."

        testCase "fourier catalog uses scalar complex and chunk parameters" <| fun _ ->
            let fft = BuiltInCatalog.find "FFT"
            let invFFT = BuiltInCatalog.find "InvFFT"
            let shiftFFT = BuiltInCatalog.find "ShiftFFT"

            Expect.equal fft.Inputs.[0].Type (PortType.Image Number) "FFT should consume scalar numeric images."
            Expect.equal fft.Outputs.[0].Type (PortType.Image Complex) "FFT should emit complex slices."
            Expect.equal invFFT.Inputs.[0].Type (PortType.Image Complex) "invFFT should consume complex slices."
            Expect.equal invFFT.Outputs.[0].Type (PortType.Image Float64) "invFFT should emit Float64 slices."
            Expect.equal shiftFFT.Inputs.[0].Type (PortType.Image Complex) "shiftFFT should consume complex slices."
            Expect.equal shiftFFT.Outputs.[0].Type (PortType.Image Complex) "shiftFFT should emit complex slices."
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

        testCase "connected component catalog uses pair and reducer types" <| fun _ ->
            let connected = BuiltInCatalog.find "ConnectedComponents"
            let table = BuiltInCatalog.find "ComponentTranslationTable"

            Expect.equal connected.Outputs.[0].Type (Tuple(Image UInt64, Scalar(BasicType.Numeric UInt64))) "Connected components should stream labels with object counts."
            Expect.equal table.Inputs.[0].Type (Tuple(Image UInt64, Scalar(BasicType.Numeric UInt64))) "Translation-table reducer should consume connected-component pairs."
            Expect.equal table.Outputs.[0].Type (Custom "TranslationTable") "Translation-table output should be a custom parameter value."

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
