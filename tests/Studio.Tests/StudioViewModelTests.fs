module Tests.StudioViewModelTests

open Expecto
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
                      p "maxShift" "8" false ]

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
    ]
