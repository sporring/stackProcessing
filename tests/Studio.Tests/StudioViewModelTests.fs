module Tests.StudioViewModelTests

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

[<Tests>]
let viewModelSuite =
    testList "Studio ViewModels" [
        testCase "serial transform type dropdown is constrained by connected image type" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(2000.0, 2000.0)

            let read =
                node "read" "ReadVolume"
                    [ p "availableMemory" "1024" false
                      p "type" "Float32" false
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

        testCase "stack info expand accepts stack info on top edge" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("StackInfoExpand")

            let info =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "StackInfoExpand")

            let inputPin =
                info.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            Expect.equal inputPin.Alignment PinAlignment.Top "StackInfoExpand should place the StackInfo input on the top edge."
            Expect.floatClose Accuracy.high inputPin.X (info.Width / 2.0) "The top input should be horizontally centered."
            Expect.floatClose Accuracy.high inputPin.Y 0.0 "The top input should sit on the top edge."

        testCase "chunk info expand accepts chunk info on top edge" <| fun _ ->
            let vm = MainWindowViewModel()
            vm.SetDrawingSize(1200.0, 800.0)
            vm.AddElement("ChunkInfoExpand")

            let info =
                pipelineNodes vm
                |> Seq.find (fun node -> node.State.Definition.Id = "ChunkInfoExpand")

            let inputPin =
                info.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                    | _ -> None)
                |> Seq.exactlyOne

            Expect.equal inputPin.Alignment PinAlignment.Top "ChunkInfoExpand should place the ChunkInfo input on the top edge."
            Expect.floatClose Accuracy.high inputPin.X (info.Width / 2.0) "The top input should be horizontally centered."
            Expect.floatClose Accuracy.high inputPin.Y 0.0 "The top input should sit on the top edge."
    ]
