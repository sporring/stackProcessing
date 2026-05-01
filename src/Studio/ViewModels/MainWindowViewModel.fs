namespace Studio.ViewModels

open System
open System.Collections.Generic
open System.Collections.ObjectModel
open System.Collections.Specialized
open System.ComponentModel
open System.Windows.Input
open Avalonia.Threading
open Graph
open NodeEditor.Mvvm
open NodeEditor.Model
open Studio.Models
open Studio.Services

type private SimpleCommand(execute: obj -> unit, canExecute: obj -> bool) =
    let canExecuteChanged = Event<EventHandler, EventArgs>()

    interface ICommand with
        member _.CanExecute(parameter) = canExecute parameter
        member _.Execute(parameter) = execute parameter

        [<CLIEvent>]
        member _.CanExecuteChanged = canExecuteChanged.Publish

type PipelinePinKind =
    | DataInput
    | DataOutput
    | ParameterInput
    | ScalarOutput

module PipelinePinKind =
    let toString kind =
        match kind with
        | DataInput -> "dataInput"
        | DataOutput -> "dataOutput"
        | ParameterInput -> "parameterInput"
        | ScalarOutput -> "scalarOutput"

    let ofString value =
        match value with
        | "parameterInput" -> ParameterInput
        | "scalarOutput" -> ScalarOutput
        | "dataOutput" -> DataOutput
        | _ -> DataInput

    let isInput kind =
        match kind with
        | DataInput
        | ParameterInput -> true
        | DataOutput
        | ScalarOutput -> false

    let isOutput kind = not (isInput kind)

type PipelinePinViewModel(alignment: PinAlignment, port: Port, kind: PipelinePinKind, ?parameterKey: string) =
    inherit PinViewModel()

    let mutable isActive = kind <> ParameterInput

    member _.Port = port
    member _.Kind = kind
    member _.ParameterKey = defaultArg parameterKey ""
    member _.IsActive = isActive

    member _.PinOpacity =
        if kind = ParameterInput && not isActive then 0.0 else 1.0

    member this.SetActive(value: bool) =
        if isActive <> value then
            isActive <- value
            this.OnPropertyChanged(nameof this.IsActive)
            this.OnPropertyChanged(nameof this.PinOpacity)

    member _.PinBrush =
        match kind with
        | ParameterInput
        | ScalarOutput -> "#8A5A22"
        | DataInput
        | DataOutput -> "#FFFFFF"

    member _.TrianglePoints =
        match kind with
        | ParameterInput
        | ScalarOutput -> "0,0 14,0 7,14"
        | DataInput -> "14,0 0,7 14,14"
        | DataOutput -> "0,0 14,7 0,14"

    member _.IsInput = PipelinePinKind.isInput kind
    member _.IsOutput = PipelinePinKind.isOutput kind

module private PortMapping =
    let parameterPort (parameter: PipelineParameterViewModel) =
        { Name = parameter.Label
          Type = Scalar parameter.ParameterType }

[<AllowNullLiteral>]
type PipelineNodeViewModel(
    state: PipelineNodeState,
    selectNode: PipelineNodeViewModel -> unit,
    getDrawingSize: unit -> float * float,
    markGraphDirty: unit -> unit,
    removePinConnections: IPin seq -> unit,
    refreshNodePins: PipelineNodeViewModel -> unit) as this =
    inherit NodeViewModel()

    let addPipelinePin x y alignment kind parameterKey (port: Port) =
        let pin = PipelinePinViewModel(alignment, port, kind, ?parameterKey = parameterKey)
        pin.Name <-
            match kind with
            | ParameterInput -> "__ParameterInput"
            | ScalarOutput -> "__ScalarOutput"
            | DataInput
            | DataOutput -> port.Name

        pin.Parent <- this
        pin.X <- x
        pin.Y <- y
        pin.Width <- 14.
        pin.Height <- 14.
        pin.Alignment <- alignment
        this.Pins.Add(pin :> IPin)
        pin :> IPin

    let nodeHeight =
        let portCount = max state.Definition.Inputs.Length state.Definition.Outputs.Length
        max 48. (20. + 22. * float (max 1 portCount))

    let verticalPinPosition index count =
        if count <= 1 then
            nodeHeight / 2.
        else
            let spacing = 22.
            let totalHeight = spacing * float (count - 1)
            (nodeHeight - totalHeight) / 2. + spacing * float index

    do
        this.Name <- state.Title
        this.Content <- PipelineNodeContent(state.Title, state, fun () -> selectNode this)
        this.Width <- 110.
        this.Height <- nodeHeight
        this.Pins <- ObservableCollection<IPin>()

        state.Parameters
        |> Seq.iter (fun parameter ->
            parameter.PropertyChanged.Add(fun args ->
                if args.PropertyName = nameof parameter.UseInput then
                    this.SyncParameterPinVisibility()
                    refreshNodePins this
                    markGraphDirty()))

        this.InitializePins()

    member private this.RemoveConnectionsForPin(pin: IPin) =
        removePinConnections [ pin ]

    member private this.TryFindParameterPin(parameterKey: string) =
        this.Pins
        |> Seq.tryPick (function
            | :? PipelinePinViewModel as pin when pin.Kind = ParameterInput && pin.ParameterKey = parameterKey ->
                Some(pin :> IPin)
            | _ -> None)

    member private this.SetParameterPinVisibility(parameter: PipelineParameterViewModel, pin: IPin) =
        if parameter.UseInput then
            pin.Width <- 14.
            pin.Height <- 14.

            match pin with
            | :? PipelinePinViewModel as parameterPin -> parameterPin.SetActive(true)
            | _ -> ()
        else
            this.RemoveConnectionsForPin(pin)
            pin.X <- -10000.
            pin.Y <- -10000.
            pin.Width <- 0.
            pin.Height <- 0.

            match pin with
            | :? PipelinePinViewModel as parameterPin -> parameterPin.SetActive(false)
            | _ -> ()

    member private this.AddParameterPin(index: int, count: int, parameter: PipelineParameterViewModel) =
        let spacing = this.Width / float (count + 1)
        let x = spacing * float (index + 1) - 7.
        let pin = addPipelinePin x 0. PinAlignment.Top ParameterInput (Some parameter.Key) (PortMapping.parameterPort parameter)
        this.SetParameterPinVisibility(parameter, pin)

        this.TryFindParameterPin(parameter.Key)
        |> Option.iter (fun pin -> this.SetParameterPinVisibility(parameter, pin))

    member private this.SyncParameterPinVisibility() =
        let parameters = state.Parameters |> Seq.toList

        parameters
        |> List.iteri (fun index parameter ->
            match this.TryFindParameterPin(parameter.Key) with
            | Some pin ->
                let spacing = this.Width / float (parameters.Length + 1)
                pin.X <- spacing * float (index + 1) - 7.
                pin.Y <- 0.
                this.SetParameterPinVisibility(parameter, pin)
            | None ->
                this.AddParameterPin(index, parameters.Length, parameter))

    member private this.InitializePins() =
        this.Pins.Clear()

        state.Definition.Inputs
        |> List.iteri (fun portIndex port ->
            addPipelinePin 0. (verticalPinPosition portIndex state.Definition.Inputs.Length) PinAlignment.Left DataInput None port |> ignore)

        state.Definition.Outputs
        |> List.iteri (fun portIndex port ->
            let kind =
                if state.Definition.Id.StartsWith("Scalar", StringComparison.Ordinal) then ScalarOutput else DataOutput

            let alignment =
                if kind = ScalarOutput then PinAlignment.Bottom else PinAlignment.Right

            let x =
                if kind = ScalarOutput then this.Width / 2. - 7. else 110.

            let y =
                if kind = ScalarOutput then nodeHeight else verticalPinPosition portIndex state.Definition.Outputs.Length

            addPipelinePin x y alignment kind None port |> ignore)

        let parameters = state.Parameters |> Seq.toList

        parameters
        |> List.iteri (fun index parameter -> this.AddParameterPin(index, parameters.Length, parameter))

        this.SyncParameterPinVisibility()

    member _.State = state

    member this.ClampToDrawing() =
        let drawingWidth, drawingHeight = getDrawingSize()
        let maxX = max 0. (drawingWidth - this.Width)
        let maxY = max 0. (drawingHeight - this.Height)

        this.X <- min maxX (max 0. this.X)
        this.Y <- min maxY (max 0. this.Y)

    override this.OnSelected() =
        base.OnSelected()
        selectNode this

    override this.OnMoved() =
        base.OnMoved()
        this.ClampToDrawing()
        markGraphDirty()

type MainWindowViewModel() as this =
    inherit ViewModelBase()

    let paletteGroups = ObservableCollection<PaletteGroupViewModel>()
    let mutable selectedNode: PipelineNodeViewModel = null
    let mutable generatedProgram = ""
    let mutable paletteSearch = ""
    let mutable graphDirty = false

    let editor =
        let editor = EditorViewModel()
        editor.Factory <- MyNodeFactory()
        editor.Templates <- editor.Factory.CreateTemplates()
        editor.Drawing <- editor.Factory.CreateDrawing("StackProcessing Pipeline")
        editor

    let drawing =
        editor.Drawing :?> DrawingNodeViewModel

    let updatePaletteGroups () =
        paletteGroups.Clear()

        let matchingFunctions =
            BuiltInCatalog.orderedFunctions
            |> List.filter (FunctionDefinition.matches paletteSearch)

        let expandedByDefault =
            not (String.IsNullOrWhiteSpace paletteSearch)

        matchingFunctions
        |> Seq.groupBy _.Category
        |> Seq.iter (fun (category, functions) ->
            paletteGroups.Add(PaletteGroupViewModel(category, functions, expandedByDefault)))

    let createState functionId =
        let definition = BuiltInCatalog.find functionId

        let parameters =
            definition.Parameters
            |> List.map (fun parameter ->
                PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type))

        PipelineNodeState(definition, parameters)

    let watchState (state: PipelineNodeState) =
        state.Parameters
        |> Seq.iter (fun parameter ->
            parameter.PropertyChanged.Add(fun _ -> this.MarkGraphDirty()))

    let pipelineNodes () =
        drawing.Nodes
        |> Seq.choose (function
            | :? PipelineNodeViewModel as node -> Some node
            | _ -> None)

    let pipelineStates () =
        pipelineNodes ()
        |> Seq.map _.State

    let parameterValues (state: PipelineNodeState) =
        state.Parameters
        |> Seq.map (fun parameter ->
            { Key = parameter.Key
              Value = parameter.Value
              UseInput = parameter.UseInput })
        |> Seq.toArray

    let setParameterValues (state: PipelineNodeState) (parameters: SavedParameter array) =
        let values =
            parameters
            |> Seq.map (fun parameter -> parameter.Key, parameter)
            |> Map.ofSeq

        for parameter in state.Parameters do
            match values |> Map.tryFind parameter.Key with
            | Some savedParameter ->
                parameter.Value <- savedParameter.Value
                parameter.UseInput <- savedParameter.UseInput
            | None -> ()

    let tryPin alignment (node: INode) =
        node.Pins
        |> Seq.tryFind (fun pin -> pin.Alignment = alignment)

    let pinByKindIndex kind index (node: PipelineNodeViewModel) =
        if kind = ParameterInput then
            node.State.Parameters
            |> Seq.tryItem index
            |> Option.bind (fun parameter ->
                node.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = ParameterInput && pin.ParameterKey = parameter.Key -> Some pin
                    | _ -> None)
                |> Seq.tryHead)
        else
            node.Pins
            |> Seq.choose (function
                | :? PipelinePinViewModel as pin when pin.Kind = kind -> Some pin
                | _ -> None)
            |> Seq.tryItem index

    let pinIndexByKind kind (pin: IPin) (node: PipelineNodeViewModel) =
        match pin with
        | :? PipelinePinViewModel as parameterPin when kind = ParameterInput ->
            node.State.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = parameterPin.ParameterKey)
        | _ ->
            node.Pins
            |> Seq.choose (function
                | :? PipelinePinViewModel as candidate when candidate.Kind = kind -> Some candidate
                | _ -> None)
            |> Seq.mapi (fun index candidate -> index, candidate)
            |> Seq.tryFind (fun (_, candidate) -> Object.ReferenceEquals(candidate, pin))
            |> Option.map fst

    let canConnectPins (startPin: IPin) (endPin: IPin) =
        match startPin, endPin with
        | :? PipelinePinViewModel as outputPin, (:? PipelinePinViewModel as inputPin)
            when outputPin.IsOutput && outputPin.IsActive && inputPin.IsInput && inputPin.IsActive ->
            PortType.canConnect outputPin.Port.Type inputPin.Port.Type
        | _ -> false

    let connectorOrientation (startPin: IPin) (endPin: IPin) =
        match startPin, endPin with
        | (:? PipelinePinViewModel as outputPin), (:? PipelinePinViewModel as inputPin)
            when outputPin.Kind = ScalarOutput || inputPin.Kind = ParameterInput ->
            ConnectorOrientation.Vertical
        | _ ->
            ConnectorOrientation.Horizontal

    let removePinConnections (pins: IPin seq) =
        let pins = pins |> Seq.toArray

        let connectors =
            drawing.Connectors
            |> Seq.filter (fun connector ->
                pins
                |> Array.exists (fun pin -> Object.ReferenceEquals(pin, connector.Start) || Object.ReferenceEquals(pin, connector.End)))
            |> Seq.toArray

        for connector in connectors do
            drawing.Connectors.Remove(connector) |> ignore

    let refreshNodePins (node: PipelineNodeViewModel) =
        if drawing.Nodes.Contains(node :> INode) then
            let connectors =
                drawing.Connectors
                |> Seq.filter (fun connector -> Object.ReferenceEquals(connector.Start.Parent, node) || Object.ReferenceEquals(connector.End.Parent, node))
                |> Seq.toArray

            for connector in connectors do
                drawing.Connectors.Remove(connector) |> ignore

            let index =
                drawing.Nodes
                |> Seq.tryFindIndex (fun candidate -> Object.ReferenceEquals(candidate, node))
                |> Option.defaultValue (drawing.Nodes.Count - 1)

            drawing.Nodes.Remove(node :> INode) |> ignore
            drawing.Nodes.Insert(index, node :> INode)

            Dispatcher.UIThread.Post(
                (fun () ->
                    for connector in connectors do
                        drawing.Connectors.Add(connector) |> ignore),
                DispatcherPriority.Background)

    let createNode index functionId =
        let node =
            PipelineNodeViewModel(
                createState functionId,
                (fun node -> this.SelectedNode <- node),
                (fun () -> drawing.Width, drawing.Height),
                (fun () -> this.MarkGraphDirty()),
                removePinConnections,
                refreshNodePins)

        watchState node.State

        node.X <- float (24 + index * 118)
        node.Y <- 66.
        node.ClampToDrawing()
        node

    let addConnector startPin endPin =
        let connector = ConnectorViewModel()
        connector.Start <- startPin
        connector.End <- endPin
        connector.Orientation <- connectorOrientation startPin endPin
        drawing.Connectors.Add(connector :> IConnector)
        this.MarkGraphDirty()

    (*
    let addSeedNodes () =
        let nodes =
            [ "Source"; "ReadFloat64"; "DiscreteGaussian"; "CastUInt8"; "Write"; "Sink" ]
            |> List.mapi createNode

        for node in nodes do
            drawing.Nodes.Add(node :> INode)
    *)
    do
        updatePaletteGroups()

        match drawing.Nodes with
        | :? INotifyCollectionChanged as nodes -> nodes.CollectionChanged.Add(fun _ -> this.MarkGraphDirty())
        | _ -> ()

        match drawing.Connectors with
        | :? INotifyCollectionChanged as connectors -> connectors.CollectionChanged.Add(fun _ -> this.MarkGraphDirty())
        | _ -> ()

        //addSeedNodes()

    member _.Editor = editor
    member _.PaletteGroups = paletteGroups

    member this.PaletteSearch
        with get () = paletteSearch
        and set value =
            if this.SetProperty(&paletteSearch, value) then
                updatePaletteGroups()

    member this.SelectedNode
        with get () = selectedNode
        and set value =
            if this.SetProperty(&selectedNode, value) then
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))

    member this.SelectedElement
        with get () = selectedNode
        and set value = this.SelectedNode <- value

    member _.HasSelectedElement = not (isNull selectedNode)

    member _.GeneratedProgram = generatedProgram

    member _.ConnectSeedPipeline() =
        if drawing.Connectors.Count = 0 then
            pipelineNodes ()
            |> Seq.pairwise
            |> Seq.iter (fun (left, right) ->
                match tryPin PinAlignment.Right left, tryPin PinAlignment.Left right with
                | Some startPin, Some endPin when canConnectPins startPin endPin ->
                    addConnector startPin endPin
                | _ -> ())

    member this.AddSourceCommand =
        SimpleCommand((fun _ -> this.AddElement("Source")), (fun _ -> true)) :> ICommand

    member this.AddReadCommand =
        SimpleCommand((fun _ -> this.AddElement("ReadFloat64")), (fun _ -> true)) :> ICommand

    member this.AddGaussianCommand =
        SimpleCommand((fun _ -> this.AddElement("DiscreteGaussian")), (fun _ -> true)) :> ICommand

    member this.AddCastCommand =
        SimpleCommand((fun _ -> this.AddElement("CastUInt8")), (fun _ -> true)) :> ICommand

    member this.AddWriteCommand =
        SimpleCommand((fun _ -> this.AddElement("Write")), (fun _ -> true)) :> ICommand

    member this.AddSinkCommand =
        SimpleCommand((fun _ -> this.AddElement("Sink")), (fun _ -> true)) :> ICommand

    member this.AddPaletteElementCommand =
        SimpleCommand(
            (fun parameter ->
                match parameter with
                | :? Function as definition -> this.AddElement(definition.Id)
                | :? string as functionId -> this.AddElement(functionId)
                | _ -> ()),
            (fun _ -> true))
        :> ICommand

    member this.DeleteSelectedCommand =
        SimpleCommand((fun _ -> this.DeleteSelectedElement()), (fun _ -> not (isNull selectedNode)))
        :> ICommand

    member this.RunCommand =
        SimpleCommand(
            (fun _ ->
                match this.ValidateGraph() with
                | Ok () -> generatedProgram <- PipelineCodeGenerator.generateSavedGraph (this.ExportGraph())
                | Error message -> generatedProgram <- message

                this.RaiseGeneratedProgramChanged()),
            (fun _ -> true))
        :> ICommand

    member _.ExportGraph() =
        let nodes = pipelineNodes () |> Seq.toArray

        let nodeIds =
            let ids = Dictionary<PipelineNodeViewModel, string>()

            nodes
            |> Array.iteri (fun index node -> ids.Add(node, $"node-{index + 1}"))

            ids

        let savedNodes =
            nodes
            |> Array.map (fun node ->
                { Id = nodeIds[node]
                  FunctionId = node.State.Definition.Id
                  X = node.X
                  Y = node.Y
                  Parameters = parameterValues node.State })

        let savedEdges =
            drawing.Connectors
            |> Seq.choose (fun connector ->
                match connector.Start, connector.End with
                | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                    match startPin.Parent, endPin.Parent with
                    | (:? PipelineNodeViewModel as startNode), (:? PipelineNodeViewModel as endNode) ->
                        match pinIndexByKind startPin.Kind startPin startNode, pinIndexByKind endPin.Kind endPin endNode with
                        | Some fromPort, Some toPort ->
                            Some
                                { FromNode = nodeIds[startNode]
                                  FromKind = PipelinePinKind.toString startPin.Kind
                                  FromPort = fromPort
                                  ToNode = nodeIds[endNode]
                                  ToKind = PipelinePinKind.toString endPin.Kind
                                  ToPort = toPort }
                        | _ -> None
                    | _ -> None
                | _ -> None)
            |> Seq.toArray

        let savedGraph =
            { Version = 1
              Nodes = savedNodes
              Edges = savedEdges }

        savedGraph

    member this.ExportGraphJson() =
        this.ExportGraph() |> PipelineGraphStorage.serialize

    member this.SaveGraph(path: string) =
        this.ExportGraph() |> PipelineGraphStorage.save path
        this.MarkGraphSaved()

    member this.ImportGraph(savedGraph: SavedGraph) =
        drawing.Connectors.Clear()
        drawing.Nodes.Clear()
        this.SelectedNode <- null

        let loadedNodes =
            savedGraph.Nodes
            |> Array.map (fun savedNode ->
                match BuiltInCatalog.tryFind savedNode.FunctionId with
                | None -> invalidOp $"Unknown function id in saved graph: {savedNode.FunctionId}"
                | Some _ ->
                    let node =
                        PipelineNodeViewModel(
                            createState savedNode.FunctionId,
                            (fun node -> this.SelectedNode <- node),
                            (fun () -> drawing.Width, drawing.Height),
                            (fun () -> this.MarkGraphDirty()),
                            removePinConnections,
                            refreshNodePins)

                    watchState node.State
                    node.X <- savedNode.X
                    node.Y <- savedNode.Y
                    node.ClampToDrawing()
                    setParameterValues node.State savedNode.Parameters
                    drawing.Nodes.Add(node :> INode)
                    savedNode.Id, node)
            |> Map.ofArray

        for edge in savedGraph.Edges do
            match loadedNodes |> Map.tryFind edge.FromNode, loadedNodes |> Map.tryFind edge.ToNode with
            | Some fromNode, Some toNode ->
                let fromKind =
                    if String.IsNullOrWhiteSpace edge.FromKind then DataOutput else PipelinePinKind.ofString edge.FromKind

                let toKind =
                    if String.IsNullOrWhiteSpace edge.ToKind then DataInput else PipelinePinKind.ofString edge.ToKind

                match pinByKindIndex fromKind edge.FromPort fromNode, pinByKindIndex toKind edge.ToPort toNode with
                | Some startPin, Some endPin when canConnectPins startPin endPin ->
                    addConnector startPin endPin
                | Some _, Some _ ->
                    invalidOp $"Saved edge has incompatible port types: {edge.FromNode}[{edge.FromPort}] -> {edge.ToNode}[{edge.ToPort}]"
                | _ ->
                    invalidOp $"Saved edge refers to a missing port: {edge.FromNode}[{edge.FromPort}] -> {edge.ToNode}[{edge.ToPort}]"
            | _ ->
                invalidOp $"Saved edge refers to a missing node: {edge.FromNode} -> {edge.ToNode}"

        this.MarkGraphSaved()

    member this.ImportGraphJson(json: string) =
        json |> PipelineGraphStorage.deserialize |> this.ImportGraph

    member this.LoadGraph(path: string) =
        path |> PipelineGraphStorage.load |> this.ImportGraph

    member _.HasGraph =
        drawing.Nodes.Count > 0 || drawing.Connectors.Count > 0

    member _.IsGraphDirty = graphDirty

    member this.MarkGraphDirty() =
        if not graphDirty then
            graphDirty <- true
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.IsGraphDirty))

    member this.MarkGraphSaved() =
        if graphDirty then
            graphDirty <- false
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.IsGraphDirty))

    member this.ClearGraph() =
        let shouldMarkDirty = this.HasGraph || graphDirty
        drawing.Connectors.Clear()
        drawing.Nodes.Clear()
        this.SelectedNode <- null
        generatedProgram <- ""
        this.RaiseGeneratedProgramChanged()

        if shouldMarkDirty then
            this.MarkGraphDirty()

    member this.AddElement(functionId: string) =
        let node = createNode drawing.Nodes.Count functionId
        node.X <- min (max 0. (drawing.Width - node.Width)) (24. + float (drawing.Nodes.Count % 6) * 118.)
        node.Y <- min (max 0. (drawing.Height - node.Height)) (24. + float (drawing.Nodes.Count / 6) * 72.)

        drawing.Nodes.Add(node :> INode)
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.AddElementAt(functionId: string, x: float, y: float) =
        let node = createNode drawing.Nodes.Count functionId
        node.X <- x - node.Width / 2.
        node.Y <- y - node.Height / 2.
        node.ClampToDrawing()

        drawing.Nodes.Add(node :> INode)
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.AddPaletteDragElementAt(functionId: string, x: float, y: float, isOutsideGraph: bool) =
        let node = createNode drawing.Nodes.Count functionId
        node.State.IsPaletteDragOutside <- isOutsideGraph
        node.X <- x - node.Width / 2.
        node.Y <- y - node.Height / 2.

        drawing.Nodes.Add(node :> INode)
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.MoveSelectedElementTo(x: float, y: float, shouldClamp: bool, isPaletteDragOutside: bool) =
        if not (isNull selectedNode) then
            selectedNode.State.IsPaletteDragOutside <- isPaletteDragOutside
            selectedNode.X <- x - selectedNode.Width / 2.
            selectedNode.Y <- y - selectedNode.Height / 2.

            if shouldClamp then
                selectedNode.ClampToDrawing()

            this.MarkGraphDirty()

    member this.DeleteSelectedElement() =
        if not (isNull selectedNode) then
            let nodes = pipelineNodes () |> Seq.toArray
            let currentIndex =
                nodes
                |> Array.tryFindIndex (fun node -> Object.ReferenceEquals(node, selectedNode))
                |> Option.defaultValue 0

            let pinsToRemove = selectedNode.Pins |> Seq.toArray

            let connectorsToRemove =
                drawing.Connectors
                |> Seq.filter (fun connector ->
                    pinsToRemove
                    |> Array.exists (fun pin -> Object.ReferenceEquals(pin, connector.Start) || Object.ReferenceEquals(pin, connector.End)))
                |> Seq.toArray

            for connector in connectorsToRemove do
                drawing.Connectors.Remove(connector) |> ignore

            drawing.Nodes.Remove(selectedNode) |> ignore

            let remaining = pipelineNodes () |> Seq.toArray
            if remaining.Length > 0 then
                this.SelectedNode <- remaining[min currentIndex (remaining.Length - 1)]
            else
                this.SelectedNode <- null

            this.MarkGraphDirty()

    member _.ValidateGraph() =
        let shouldRequirePin (pin: IPin) =
            match pin with
            | :? PipelinePinViewModel as pipelinePin ->
                match pipelinePin.Kind with
                | ParameterInput -> pipelinePin.IsActive
                | ScalarOutput -> drawing.IsPinConnected(pin)
                | DataInput
                | DataOutput -> true
            | _ -> true

        let missingPins =
            drawing.Nodes
            |> Seq.collect (fun node ->
                node.Pins
                |> Seq.filter (fun pin -> shouldRequirePin pin && not (drawing.IsPinConnected(pin)))
                |> Seq.map (fun pin -> $"{node.Name}.{pin.Name}"))
            |> Seq.toArray

        if missingPins.Length = 0 then
            Ok ()
        else
            let message =
                missingPins
                |> Seq.map (fun pin -> $"// - {pin}")
                |> String.concat Environment.NewLine

            Error($"// Cannot generate F# yet. Connect every input and output pin first.{Environment.NewLine}{message}")

    member _.SetDrawingSize(width: float, height: float) =
        if width > 0. && height > 0. then
            drawing.Width <- width
            drawing.Height <- height

            pipelineNodes ()
            |> Seq.iter _.ClampToDrawing()

    member this.DeleteSelectedElementIfInTrashZone(trashWidth: float, trashHeight: float, margin: float) =
        if not (isNull selectedNode) then
            let trashLeft = max 0. (drawing.Width - trashWidth - margin)
            let trashTop = max 0. (drawing.Height - trashHeight - margin)

            if selectedNode.X + selectedNode.Width >= trashLeft && selectedNode.Y + selectedNode.Height >= trashTop then
                this.DeleteSelectedElement()

    interface IGraphWindowController with
        member this.SetDrawingSize width height =
            this.SetDrawingSize(width, height)

        member this.DeleteSelectedElementIfInTrashZone trashWidth trashHeight margin =
            this.DeleteSelectedElementIfInTrashZone(trashWidth, trashHeight, margin)

    member private this.RaiseGeneratedProgramChanged() =
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.GeneratedProgram))
