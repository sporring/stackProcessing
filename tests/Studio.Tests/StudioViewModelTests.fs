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
                      edge "estimator" "dataOutput" 0 "apply" "dataInput" 0
                      edge "estimator" "dataOutput" 1 "apply" "dataInput" 1 ])

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
                [ "availableMemory"; "type"; "format"; "input"; "slabDepth"; "multiscaleIndex"; "datasetIndex"; "timepoint"; "channel"; "maxParallelChunks" ]
                "OME-Zarr read should show only zarr-specific parameters."

            (readNode |> parameter "format").Value <- "NeXus/HDF5"

            Expect.sequenceEqual
                (visibleParameterKeys readNode)
                [ "availableMemory"; "type"; "format"; "input"; "slabDepth"; "datasetPath"; "frameAxis"; "yAxis"; "xAxis" ]
                "NeXus read should show only HDF5-specific parameters."

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

            Expect.equal expandNode.State.RecordType (Some (PortType.Custom "StackInfo")) "Expand should remember the connected record type."
            Expect.equal outputPins.Length 7 "Expand should expose the StackInfo fields once it is connected to StackInfo."
            Expect.sequenceEqual
                (outputPins |> List.map _.Port.Name)
                [ "Dimensions: UInt32"; "Size: UInt64 list"; "ComponentType: String"; "NumberOfComponents: UInt32"; "Width: UInt64"; "Height: UInt64"; "Depth: UInt64" ]
                "StackInfo fields should appear as reducer outputs."

        testCase "resample sample imports without changing connectors during collection notification" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 1400.0)

            let samplePath = findRepoFile (Path.Combine("samples", "resample", "pipeline.json"))
            let graph = PipelineGraphStorage.load samplePath

            Expect.isTrue (File.Exists samplePath) "The resample sample should exist."
            try
                vm.ImportGraph graph
            with ex ->
                failtestf "Importing resample/pipeline.json should not throw, but got: %s" ex.Message

            let exported = vm.ExportGraph()

            Expect.equal exported.Edges.Length graph.Edges.Length "Importing resample should preserve every saved edge."
            Expect.isTrue
                (exported.Edges |> Array.exists (fun edge -> edge.FromNode = "node-4" && edge.ToNode = "node-5"))
                "The imported Expand-to-Print field edges should survive."

        testCase "changing read type adapts downstream image boxes and keeps metadata links" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 1400.0)

            let samplePath = findRepoFile (Path.Combine("samples", "resample", "pipeline.json"))
            vm.ImportGraph(PipelineGraphStorage.load samplePath)

            let nodes =
                pipelineNodes vm
                |> Seq.toArray

            let readNode = nodes |> Array.find (fun node -> node.State.Definition.Id = "Read")
            let resampleNode = nodes |> Array.find (fun node -> node.State.Definition.Id = "Resample")
            let writeNode = nodes |> Array.find (fun node -> node.State.Definition.Id = "Write")
            let expandNodes = nodes |> Array.filter (fun node -> node.State.Definition.Id = "Expand")
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

            Expect.equal streamConnectorCountBefore 1 "Read should be connected to Resample before the type change."

            (readNode |> parameter "type").Value <- "Float32"

            Expect.equal (readNode |> parameter "type").Value "Float32" "The read type parameter should change."
            Expect.equal (resampleNode |> parameter "type").Value "Float32" "Resample should adapt to the changed source image type."

            let resampleInput =
                resampleNode.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            Expect.equal resampleInput.Port.Name "Float32" "Resample's image input hover text should show the concrete type."

            let readMetadataStillConnected =
                drawing.Connectors
                |> Seq.exists (fun connector ->
                    match connector.Start, connector.End with
                    | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                        Object.ReferenceEquals(startPin.Parent, readNode)
                        && startPin.Kind = ReducerOutput
                        && expandNodes |> Array.exists (fun expandNode -> Object.ReferenceEquals(endPin.Parent, expandNode))
                        && endPin.Kind = DataInput
                    | _ ->
                        false)

            Expect.isTrue readMetadataStillConnected "Changing the stream type should not remove the unchanged StackInfo connection."

            (readNode |> parameter "type").Value <- "Float64"

            Expect.equal (resampleNode |> parameter "type").Value "Float32" "Resample should not adapt when doing so would break the downstream TIFF writer."

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

            Expect.isFalse readInputStillConnected "Changing to Float64 should cut the changed Read-to-Resample edge before mutating the downstream chain."

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

            Expect.isTrue writeInputStillConnected "The still-compatible Resample-to-Write edge should stay connected."

            let writeMetadataStillConnected =
                drawing.Connectors
                |> Seq.exists (fun connector ->
                    match connector.Start, connector.End with
                    | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                        Object.ReferenceEquals(startPin.Parent, writeNode)
                        && startPin.Kind = ReducerOutput
                        && endPin.Kind = ParameterInput
                    | _ ->
                        false)

            Expect.isTrue writeMetadataStillConnected "Changing read stream type should not remove the unchanged Write StackInfo connection."
    ]
