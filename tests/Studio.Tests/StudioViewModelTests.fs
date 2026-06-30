module Tests.StudioViewModelTests

open System
open System.IO
open Expecto
open NodeEditor.Model
open NodeEditor.Mvvm
open Studio.Graph
open Studio.Models
open Studio.ViewModels

let private p key value useInput =
    { Key = key
      Value = value
      UseInput = useInput }

let private node id functionId parameters =
    { Id = id
      FunctionId = functionId
      X = 0.0
      Y = 0.0
      Parameters = parameters |> List.toArray }

let private edge fromNode fromKind fromPort toNode toKind toPort =
    { FromNode = fromNode
      FromKind = fromKind
      FromPort = fromPort
      ToNode = toNode
      ToKind = toKind
      ToPort = toPort }

let private graph nodes edges =
    { Version = 1
      Nodes = nodes |> List.toArray
      Edges = edges |> List.toArray }

let private pipelineNodes (vm: MainWindowViewModel) =
    let drawing = vm.Editor.Drawing :?> DrawingNodeViewModel

    drawing.Nodes
    |> Seq.choose (function
        | :? PipelineNodeViewModel as node -> Some node
        | _ -> None)

let private typeParameter (node: PipelineNodeViewModel) =
    node.State.Parameters
    |> Seq.find (fun parameter -> parameter.Key = "type")

let private parameter key (node: PipelineNodeViewModel) =
    node.State.Parameters
    |> Seq.find (fun parameter -> parameter.Key = key)

let private optionStates (parameter: PipelineParameterViewModel) =
    parameter.Options
    |> Seq.map (fun option -> option.Value, option.IsEnabled)
    |> Map.ofSeq

let private visibleParameterKeys (node: PipelineNodeViewModel) =
    node.State.Parameters
    |> Seq.filter _.IsVisible
    |> Seq.map _.Key
    |> Seq.toList

let private imagePins kind (node: PipelineNodeViewModel) =
    node.Pins
    |> Seq.choose (function
        | :? PipelinePinViewModel as pin when pin.Kind = kind ->
            match pin.Port.Type with
            | PortType.Image _ -> Some pin
            | _ -> None
        | _ ->
            None)
    |> Seq.toList

let private findRepoFile relativePath =
    let rec search (directory: DirectoryInfo) =
        let candidate = Path.Combine(directory.FullName, relativePath)

        if File.Exists candidate then
            candidate
        elif isNull directory.Parent then
            failtestf "Could not find %s from %s" relativePath AppContext.BaseDirectory
        else
            search directory.Parent

    search (DirectoryInfo(AppContext.BaseDirectory))

[<Tests>]
let viewModelSuite =
    testList "Studio ViewModels" [
        testCase "serial transform type dropdown is constrained by connected image type" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 2000.0)

            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float32" false
                      p "format" "Volume file" false
                      p "input" "volumedata.tif" false ]

            let estimator =
                node "estimator" "SerialEstTrans"
                    [ p "type" "Float32" false
                      p "searchRadius" "8" false ]

            let apply =
                node "apply" "SerialApplyTrans"
                    [ p "type" "Float32" false
                      p "background" "0.0" false ]

            vm.ImportGraph(
                graph
                    [ read; estimator; apply ]
                    [ edge "read" "dataOutput" 0 "estimator" "dataInput" 0
                      edge "estimator" "dataOutput" 0 "apply" "dataInput" 0 ])

            let nodesById =
                pipelineNodes vm
                |> Seq.map (fun node -> node.State.Definition.Id, node)
                |> Map.ofSeq

            let estimatorOptions = nodesById["SerialEstTrans"] |> typeParameter |> optionStates
            let applyOptions = nodesById["SerialApplyTrans"] |> typeParameter |> optionStates

            Expect.isTrue estimatorOptions["Float32"] "The connected Float32 estimator type should remain selectable."
            Expect.isFalse estimatorOptions["Float64"] "The connected estimator should gray out Float64."
            Expect.isTrue applyOptions["Float32"] "The connected Float32 apply type should remain selectable."
            Expect.isFalse applyOptions["Float64"] "The connected apply box should gray out Float64."

        testCase "synthetic sources can choose every image type even when connected" <| fun _ ->
            let sourceParameters functionId =
                [ p "availableMemory" "2147483648UL" false
                  p "type" "UInt8" false
                  p "width" "64" false
                  p "height" "64" false
                  p "depth" "64" false ]
                @ match functionId with
                  | "NormalNoise" -> [ p "mean" "0.0" false; p "std" "1.0" false ]
                  | "SaltAndPepperNoise" -> [ p "probability" "0.01" false ]
                  | "PoissonNoise" -> [ p "lambda" "1.0" false ]
                  | "SpeckleNoise" -> [ p "std" "1.0" false ]
                  | "CreateByEuler2DTransform" ->
                      [ p "polygon" "[ { X = 16.0; Y = 16.0 }; { X = 48.0; Y = 16.0 }; { X = 48.0; Y = 48.0 }; { X = 16.0; Y = 48.0 } ]" false
                        p "transform" "Diagonal" false ]
                  | _ -> []

            for functionId in [ "Zero"; "NormalNoise"; "SaltAndPepperNoise"; "PoissonNoise"; "SpeckleNoise"; "CreateByEuler2DTransform" ] do
                let vm = MainWindowViewModel()
                vm.SetDrawingSize(2000.0, 2000.0)

                let source = node "source" functionId (sourceParameters functionId)

                let threshold =
                    node "threshold" "Threshold"
                        [ p "type" "UInt8" false
                          p "lower" "128.0" false
                          p "upper" "infinity" false ]

                vm.ImportGraph(graph [ source; threshold ] [ edge "source" "dataOutput" 0 "threshold" "dataInput" 0 ])

                let sourceNode =
                    pipelineNodes vm
                    |> Seq.find (fun node -> node.State.Definition.Id = functionId)

                let options = sourceNode |> typeParameter |> optionStates

                for typeName in [ "UInt8"; "Int8"; "UInt16"; "Int16"; "UInt32"; "Int32"; "UInt64"; "Int64"; "Float32"; "Float64"; "Complex" ] do
                    Expect.isTrue options[typeName] $"{functionId} should allow {typeName}."

        testCase "intensity stretch stream pins use selected image type" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 2000.0)

            let stretch =
                node "stretch" "IntensityStretch"
                    [ p "type" "Float32" false
                      p "inputMinimum" "0.0" false
                      p "inputMaximum" "1.0" false
                      p "outputMinimum" "0.0" false
                      p "outputMaximum" "1.0" false ]

            vm.ImportGraph(graph [ stretch ] [])

            let stretchNode =
                pipelineNodes vm
                |> Seq.exactlyOne

            let inputPin =
                stretchNode.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            let outputPin =
                stretchNode.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataOutput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            Expect.equal inputPin.Port.Name "Float32" "IntensityStretch's input hover text should show the selected concrete type."
            Expect.equal outputPin.Port.Name "Float32" "IntensityStretch's output hover text should show the selected concrete type."
            Expect.equal inputPin.Port.Type (PortType.Image Float32) "IntensityStretch's input should guard on the selected type."
            Expect.equal outputPin.Port.Type (PortType.Image Float32) "IntensityStretch's output should guard on the selected type."

        testCase "typed same-image boxes expose concrete stream pins" <| fun _ ->
            let sameTypeBoxes =
                [ "Clamp"
                  "ShiftScale"
                  "IntensityStretch"
                  "CreatePadding"
                  "Crop"
                  "SmoothWMedian"
                  "SmoothWBilateral"
                  "GradientMagnitudeSquared"
                  "SobelEdge"
                  "Laplacian"
                  "GrayscaleErode"
                  "GrayscaleDilate"
                  "GrayscaleOpening"
                  "GrayscaleClosing"
                  "LabelContour"
                  "ChangeLabel"
                  "Resize"
                  "Resample"
                  "ImageOpScalar"
                  "ScalarOpImage"
                  "AddNormalNoise"
                  "AddSaltAndPepperNoise"
                  "AddPoissonNoise"
                  "AddSpeckleNoise" ]

            for functionId in sameTypeBoxes do
                let vm = MainWindowViewModel()
                vm.SetDrawingSize(2000.0, 2000.0)
                vm.AddElement(functionId)

                let node =
                    pipelineNodes vm
                    |> Seq.exactlyOne

                (node |> parameter "type").Value <- "Float32"

                let inputs = imagePins DataInput node
                let outputs = imagePins DataOutput node

                Expect.isNonEmpty inputs $"{functionId} should have a typed image input."
                Expect.isNonEmpty outputs $"{functionId} should have a typed image output."

                for pin in inputs @ outputs do
                    Expect.equal pin.Port.Name "Float32" $"{functionId} image pin hover text should show the selected type."
                    Expect.equal pin.Port.Type (PortType.Image Float32) $"{functionId} image pin should guard on the selected type."

        testCase "typed image-to-other boxes expose concrete image inputs" <| fun _ ->
            let inputTypedBoxes =
                [ "HistogramEqualization", "Float32", [ PortType.Image Float32 ]
                  "ImageComparison", "Float32", [ PortType.Image UInt8 ]
                  "FitBiasModel", "Float32", []
                  "FitBiasModelMasked", "Float32", []
                  "CorrectBias", "Float32", [ PortType.Image Float64 ]
                  "CorrectBiasMasked", "Float32", [ PortType.Image Float64 ]
                  "Threshold", "Float32", [ PortType.Image UInt8 ]
                  "MarchingCubes", "Float32", []
                  "DogKeypoints", "Float32", []
                  "SiftKeypoints", "Float32", []
                  "LogBlobKeypoints", "Float32", []
                  "HessianKeypoints", "Float32", []
                  "Harris3DKeypoints", "Float32", []
                  "Forstner3DKeypoints", "Float32", []
                  "PhaseCongruencyKeypoints", "Float32", []
                  "SumProjection", "Float32", []
                  "FFT", "Float32", [] ]

            for functionId, typeName, expectedImageOutputs in inputTypedBoxes do
                let vm = MainWindowViewModel()
                vm.SetDrawingSize(2000.0, 2000.0)
                vm.AddElement(functionId)

                let node =
                    pipelineNodes vm
                    |> Seq.exactlyOne

                (node |> parameter "type").Value <- typeName
                let expectedInputType =
                    typeName
                    |> NumericType.tryParse
                    |> Option.map PortType.Image
                    |> Option.defaultWith (fun () -> failtestf "Bad test type %s" typeName)

                let typedInputs =
                    imagePins DataInput node
                    |> List.filter (fun pin -> pin.Port.Type = expectedInputType)

                Expect.isNonEmpty typedInputs $"{functionId} should guard its selected image input type."
                typedInputs
                |> List.iter (fun pin -> Expect.equal pin.Port.Name typeName $"{functionId} image input hover text should show the selected type.")

                let actualImageOutputTypes =
                    imagePins DataOutput node
                    |> List.map _.Port.Type

                for expectedOutput in expectedImageOutputs do
                    Expect.contains actualImageOutputTypes expectedOutput $"{functionId} should keep its declared image output type."

        testCase "typed source boxes expose concrete stream outputs" <| fun _ ->
            let sourceBoxes =
                [ "Read"
                  "ReadRandom"
                  "ReadRange"
                  "Zero"
                  "NormalNoise"
                  "SaltAndPepperNoise"
                  "PoissonNoise"
                  "SpeckleNoise"
                  "CreateByEuler2DTransform" ]

            for functionId in sourceBoxes do
                let vm = MainWindowViewModel()
                vm.SetDrawingSize(2000.0, 2000.0)
                vm.AddElement(functionId)

                let node =
                    pipelineNodes vm
                    |> Seq.exactlyOne

                (node |> parameter "type").Value <- "Float32"

                let outputs = imagePins DataOutput node

                Expect.isNonEmpty outputs $"{functionId} should have a typed image output."

                for pin in outputs do
                    Expect.equal pin.Port.Name "Float32" $"{functionId} image output hover text should show the selected type."
                    Expect.equal pin.Port.Type (PortType.Image Float32) $"{functionId} image output should guard on the selected type."

        testCase "serial estimator method is a dropdown" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("SerialEstTrans")

            let estimator =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "SerialEstTrans")

            let methodParameter = estimator |> parameter "method"
            let options =
                methodParameter.Options
                |> Seq.map _.Value
                |> Seq.toList

            Expect.sequenceEqual options [ "dogAffine"; "siftAffine"; "SSDAffine" ] "The serial estimator method should use fixed dropdown options."
            Expect.equal methodParameter.Value "dogAffine" "The catalog default should remain selected unless changed."

        testCase "serial estimator parameters are enabled by selected method" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("SerialEstTrans")

            let estimator =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "SerialEstTrans")

            let methodParameter = estimator |> parameter "method"
            let scaleParameter = estimator |> parameter "scale"
            let pixelFractionParameter = estimator |> parameter "pixelFraction"

            Expect.isTrue scaleParameter.IsValueEnabled "Scale should be active for dog/sift keypoint methods."
            Expect.isFalse pixelFractionParameter.IsValueEnabled "Pixel fraction should be inactive for dog/sift keypoint methods."

            methodParameter.Value <- "SSDAffine"

            Expect.isFalse scaleParameter.IsValueEnabled "Scale should be inactive for SSD affine."
            Expect.isTrue pixelFractionParameter.IsValueEnabled "Pixel fraction should be active for SSD affine."

        testCase "expand accepts records on top edge" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("Expand")

            let info =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "Expand")

            let inputPin =
                info.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            Expect.equal inputPin.Alignment PinAlignment.Top "Expand should place the record input on the top edge."
            Expect.floatClose Accuracy.high inputPin.X (info.Width / 2.0) "The top input should be horizontally centered."
            Expect.floatClose Accuracy.high inputPin.Y 0.0 "The top input should sit on the top edge."

        testCase "read stream output is centered despite metadata output" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("Read")

            let readNode =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "Read")

            let streamPin =
                readNode.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataOutput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            let infoPin =
                readNode.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = ReducerOutput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            Expect.equal streamPin.Alignment PinAlignment.Right "The image stream should leave on the right edge."
            Expect.floatClose Accuracy.high streamPin.X readNode.Width "The stream output should sit on the right edge."
            Expect.floatClose Accuracy.high streamPin.Y (readNode.Height / 2.0) "The stream output should ignore bottom metadata pins when placed vertically."
            Expect.equal infoPin.Alignment PinAlignment.Bottom "The metadata record should leave on the bottom edge."
            Expect.floatClose Accuracy.high infoPin.X (readNode.Width / 2.0) "The single metadata output should be horizontally centered."
            Expect.floatClose Accuracy.high infoPin.Y readNode.Height "The metadata output should sit on the bottom edge."

        testCase "read shows only parameters for selected source format and keeps all type options enabled" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("Read")

            let readNode =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "Read")

            let typeOptions = readNode |> parameter "type" |> optionStates

            Expect.isTrue typeOptions["UInt32"] "Read output type should be a requested cast target, not limited by file format."
            Expect.isTrue typeOptions["Float64"] "Read should be able to emit Float64 even when reading TIFF stacks."

            (readNode |> parameter "type").Value <- "Float64"
            let stackSuffixOptions = readNode |> parameter "suffix" |> optionStates

            Expect.isTrue stackSuffixOptions[".jpg"] "JPEG stack input should remain selectable when the read output type is Float64."
            Expect.isTrue stackSuffixOptions[".png"] "PNG stack input should remain selectable when the read output type is Float64."
            Expect.isTrue stackSuffixOptions[".bmp"] "BMP stack input should remain selectable when the read output type is Float64."

            Expect.sequenceEqual
                (visibleParameterKeys readNode)
                [ "availableMemory"; "type"; "format"; "input"; "suffix" ]
                "Image stack read should show only stack file parameters."

            (readNode |> parameter "format").Value <- "Volume file"

            Expect.sequenceEqual
                (visibleParameterKeys readNode)
                [ "availableMemory"; "type"; "format"; "input"; "suffix" ]
                "Volume file read should still expose its file type selector."

            let volumeSuffixOptions = readNode |> parameter "suffix" |> optionStates
            let volumeSuffixParameter = readNode |> parameter "suffix"

            Expect.isTrue volumeSuffixOptions[".tiff"] "TIFF should be the selectable volume-file type."
            Expect.isFalse (volumeSuffixOptions.ContainsKey ".jpg") "Image-stack formats should not appear in the volume-file suffix menu."
            Expect.equal volumeSuffixParameter.Value ".tiff" "Volume-file reads should default to TIFF when the previous file type no longer applies."
            Expect.isNotNull volumeSuffixParameter.SelectedOption "Rebuilding the file-type menu should keep the ComboBox selection visible."

            (readNode |> parameter "format").Value <- "OME-Zarr"

            Expect.sequenceEqual
                (visibleParameterKeys readNode)
                [ "availableMemory"; "type"; "format"; "input"; "thickDepth"; "multiscaleIndex"; "datasetIndex"; "timepoint"; "channel"; "maxParallelChunks" ]
                "OME-Zarr read should show only zarr-specific parameters."

            (readNode |> parameter "format").Value <- "NeXus/HDF5"

            Expect.sequenceEqual
                (visibleParameterKeys readNode)
                [ "availableMemory"; "type"; "format"; "input"; "datasetPath"; "frameAxis"; "yAxis"; "xAxis" ]
                "NeXus read should show only HDF5-specific parameters."

        testCase "read inspects source type and only auto-defaults until user chooses a type" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("Read")

            let readNode =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "Read")

            let sampleFile = findRepoFile (Path.Combine("samples", "data", "gaussianField", "image_000.tiff"))
            let sampleDir = Path.GetDirectoryName sampleFile
            let sourceType = (StackIO.getImageInfo sampleDir ".tiff").componentType

            (readNode |> parameter "input").Value <- sampleDir

            Expect.isTrue readNode.State.HasSourceInfo "Read should show source information once the path exists."
            Expect.isFalse readNode.State.SourceInfoIsError "Existing TIFF stack should inspect without an error."
            Expect.stringContains readNode.State.SourceInfoText $"Source type: {sourceType}" "Read should show the true source component type."
            Expect.equal (readNode |> parameter "type").Value sourceType "Unconnected read output should default to the detected source type."

            let manualType = if sourceType = "Float64" then "Float32" else "Float64"
            (readNode |> parameter "type").Value <- manualType
            (readNode |> parameter "input").Value <- sampleDir

            Expect.equal (readNode |> parameter "type").Value manualType "Manual read output type selection should be respected."
            Expect.stringContains readNode.State.SourceInfoText $"Source type: {sourceType}" "Source type note should remain the true underlying type."

            let missingDir = Path.Combine(Path.GetTempPath(), $"studio-missing-source-{Guid.NewGuid():N}")
            (readNode |> parameter "input").Value <- missingDir

            Expect.isTrue readNode.State.SourceInfoIsError "Missing read source should be shown as an error."
            Expect.stringContains readNode.State.SourceInfoText "does not exist" "Missing read source should name the filesystem error."

        testCase "read input parameters expose a browse affordance" <| fun _ ->
            for functionId in [ "Read"; "ReadRandom"; "ReadRange" ] do
                let vm = MainWindowViewModel()
                vm.SetDrawingSize(1200.0, 800.0)
                vm.AddElement(functionId)

                let readNode =
                    pipelineNodes vm
                    |> Seq.exactlyOne

                Expect.isTrue ((readNode |> parameter "input").IsBrowseVisible) $"{functionId} input should show a browse button."
                Expect.isFalse ((readNode |> parameter "type").IsBrowseVisible) $"{functionId} type should not show a browse button."

        testCase "imported expand adapts output ports from connected record type" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)

            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let expand =
                node "expand" "Expand" []

            vm.ImportGraph(
                graph
                    [ read; expand ]
                    [ edge "read" "reducerOutput" 0 "expand" "dataInput" 0 ])

            let expandNode =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "Expand")

            let outputPins =
                expandNode.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = ReducerOutput -> Some pin
                    | _ -> None)
                |> Seq.toList

            Expect.equal expandNode.State.RecordType (Some (PortType.Custom "ImageInfo")) "Expand should remember the connected record type."
            Expect.equal outputPins.Length 12 "Expand should expose the ImageInfo fields once it is connected to ImageInfo."
            Expect.sequenceEqual
                (outputPins |> List.map _.Port.Name)
                [ "Format: String"; "Dimensions: UInt32"; "Size: UInt64 list"; "ComponentType: String"; "NumberOfComponents: UInt32"; "Chunks: int list"; "ChunkX: Int32"; "ChunkY: Int32"; "ChunkZ: Int32"; "Width: UInt64"; "Height: UInt64"; "Depth: UInt64" ]
                "ImageInfo fields should appear as reducer outputs."

        testCase "resample sample imports without changing connectors during collection notification" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 1400.0)

            let samplePath = findRepoFile (Path.Combine("samples", "resample", "resample.json"))
            let graph = PipelineGraphStorage.load samplePath

            Expect.isTrue (File.Exists samplePath) "The resample sample should exist."
            try
                vm.ImportGraph graph
            with ex ->
                failtestf "Importing resample/resample.json should not throw, but got: %s" ex.Message

            let exported = vm.ExportGraph()
            let exportedFunctionIdByNode =
                exported.Nodes
                |> Array.map (fun node -> node.Id, node.FunctionId)
                |> Map.ofArray

            Expect.equal exported.Edges.Length graph.Edges.Length "Importing resample should preserve every saved edge."
            Expect.isTrue
                (exported.Edges
                 |> Array.exists (fun edge ->
                     exportedFunctionIdByNode[edge.FromNode] = "NormalNoise"
                     && exportedFunctionIdByNode[edge.ToNode] = "Resample"))
                "The imported source-to-resample stream edge should survive."

        testCase "normalize sample imports with current read random parameter ports" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 1400.0)

            let samplePath = findRepoFile (Path.Combine("samples", "normalize", "normalize.json"))
            let graph = PipelineGraphStorage.load samplePath

            Expect.isTrue (File.Exists samplePath) "The normalize sample should exist."
            try
                vm.ImportGraph graph
            with ex ->
                failtestf "Importing normalize/normalize.json should not throw, but got: %s" ex.Message

            let exported = vm.ExportGraph()

            Expect.equal exported.Edges.Length graph.Edges.Length "Importing normalize should preserve every saved edge."

        testCase "changing read type adapts downstream image boxes and keeps metadata links" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 1400.0)

            let samplePath = findRepoFile (Path.Combine("samples", "resize", "resize.json"))
            vm.ImportGraph(PipelineGraphStorage.load samplePath)

            let nodes =
                pipelineNodes vm
                |> Seq.toArray

            let readNode = nodes |> Array.find (fun node -> node.State.Definition.Id = "Read")
            let resampleNode = nodes |> Array.find (fun node -> node.State.Definition.Id = "Resize")
            let writeNode = nodes |> Array.find (fun node -> node.State.Definition.Id = "Write")
            let drawing = vm.Editor.Drawing :?> DrawingNodeViewModel

            let streamConnectorCountBefore =
                drawing.Connectors
                |> Seq.filter (fun connector ->
                    match connector.Start, connector.End with
                    | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                        Object.ReferenceEquals(startPin.Parent, readNode)
                        && startPin.Kind = DataOutput
                        && Object.ReferenceEquals(endPin.Parent, resampleNode)
                        && endPin.Kind = DataInput
                    | _ ->
                        false)
                |> Seq.length

            Expect.equal streamConnectorCountBefore 1 "Read should be connected to Resize before the type change."

            (readNode |> parameter "type").Value <- "UInt16"

            Expect.equal (readNode |> parameter "type").Value "UInt16" "The read type parameter should change."
            Expect.equal (resampleNode |> parameter "type").Value "UInt16" "Resize should adapt to the changed source image type."

            let resampleInput =
                resampleNode.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            Expect.equal resampleInput.Port.Name "UInt16" "Resize's image input hover text should show the concrete type."

            (readNode |> parameter "type").Value <- "Float64"

            Expect.equal (resampleNode |> parameter "type").Value "UInt16" "Resize should not adapt when doing so would break the downstream TIFF writer."

            let readInputStillConnected =
                drawing.Connectors
                |> Seq.exists (fun connector ->
                    match connector.Start, connector.End with
                    | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                        Object.ReferenceEquals(startPin.Parent, readNode)
                        && startPin.Kind = DataOutput
                        && Object.ReferenceEquals(endPin.Parent, resampleNode)
                        && endPin.Kind = DataInput
                    | _ ->
                        false)

            Expect.isFalse readInputStillConnected "Changing to Float64 should cut the changed Read-to-Resize edge before mutating the downstream chain."

            let writeInputStillConnected =
                drawing.Connectors
                |> Seq.exists (fun connector ->
                    match connector.Start, connector.End with
                    | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                        Object.ReferenceEquals(startPin.Parent, resampleNode)
                        && startPin.Kind = DataOutput
                        && Object.ReferenceEquals(endPin.Parent, writeNode)
                        && endPin.Kind = DataInput
                    | _ ->
                        false)

            Expect.isTrue writeInputStillConnected "The still-compatible Resize-to-Write edge should stay connected."

            ignore writeNode
    ]
