namespace Studio.ViewModels

open System
open System.Collections.ObjectModel
open System.ComponentModel
open System.Windows.Input
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

type PipelinePinViewModel(alignment: PinAlignment) =
    inherit PinViewModel()

    member _.TrianglePoints =
        match alignment with
        | PinAlignment.Left -> "0,0 14,7 0,14"
        | PinAlignment.Right -> "0,0 14,7 0,14"
        | _ -> "0,0 14,7 0,14"

[<AllowNullLiteral>]
type PipelineNodeViewModel(
    state: PipelineNodeState,
    selectNode: PipelineNodeViewModel -> unit,
    getDrawingSize: unit -> float * float) as this =
    inherit NodeViewModel()

    let addPipelinePin x y alignment name =
        let pin = PipelinePinViewModel(alignment)
        pin.Name <- name
        pin.Parent <- this
        pin.X <- x
        pin.Y <- y
        pin.Width <- 14.
        pin.Height <- 14.
        pin.Alignment <- alignment
        this.Pins.Add(pin :> IPin)

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

        state.Definition.Inputs
        |> List.iteri (fun portIndex port ->
            addPipelinePin 0. (verticalPinPosition portIndex state.Definition.Inputs.Length) PinAlignment.Left port.Name)

        state.Definition.Outputs
        |> List.iteri (fun portIndex port ->
            addPipelinePin 110. (verticalPinPosition portIndex state.Definition.Outputs.Length) PinAlignment.Right port.Name)

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

type MainWindowViewModel() as this =
    inherit ViewModelBase()

    let paletteGroups = ObservableCollection<PaletteGroupViewModel>()
    let mutable selectedNode: PipelineNodeViewModel = null
    let mutable generatedProgram = ""
    let mutable paletteSearch = ""

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
                PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue))

        PipelineNodeState(definition, parameters)

    let pipelineNodes () =
        drawing.Nodes
        |> Seq.choose (function
            | :? PipelineNodeViewModel as node -> Some node
            | _ -> None)

    let pipelineStates () =
        pipelineNodes ()
        |> Seq.map _.State

    let tryPin alignment (node: INode) =
        node.Pins
        |> Seq.tryFind (fun pin -> pin.Alignment = alignment)

    let createNode index functionId =
        let node =
            PipelineNodeViewModel(
                createState functionId,
                (fun node -> this.SelectedNode <- node),
                (fun () -> drawing.Width, drawing.Height))

        node.X <- float (24 + index * 118)
        node.Y <- 66.
        node.ClampToDrawing()
        node

    let addConnector startPin endPin =
        let connector = ConnectorViewModel()
        connector.Start <- startPin
        connector.End <- endPin
        connector.Orientation <- ConnectorOrientation.Horizontal
        drawing.Connectors.Add(connector :> IConnector)

    let addSeedPipeline () =
        let nodes =
            [ "Source"; "Read"; "DiscreteGaussian"; "Cast"; "Write"; "Sink" ]
            |> List.mapi createNode

        for node in nodes do
            drawing.Nodes.Add(node :> INode)

        nodes
        |> Seq.pairwise
        |> Seq.iter (fun (left, right) ->
            match tryPin PinAlignment.Right left, tryPin PinAlignment.Left right with
            | Some startPin, Some endPin -> addConnector startPin endPin
            | _ -> ())

    do
        updatePaletteGroups()
        addSeedPipeline()

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

    member this.AddSourceCommand =
        SimpleCommand((fun _ -> this.AddElement("Source")), (fun _ -> true)) :> ICommand

    member this.AddReadCommand =
        SimpleCommand((fun _ -> this.AddElement("Read")), (fun _ -> true)) :> ICommand

    member this.AddGaussianCommand =
        SimpleCommand((fun _ -> this.AddElement("DiscreteGaussian")), (fun _ -> true)) :> ICommand

    member this.AddCastCommand =
        SimpleCommand((fun _ -> this.AddElement("Cast")), (fun _ -> true)) :> ICommand

    member this.AddWriteCommand =
        SimpleCommand((fun _ -> this.AddElement("Write")), (fun _ -> true)) :> ICommand

    member this.AddSinkCommand =
        SimpleCommand((fun _ -> this.AddElement("Sink")), (fun _ -> true)) :> ICommand

    member this.AddPaletteElementCommand =
        SimpleCommand(
            (fun parameter ->
                match parameter with
                | :? FunctionDefinition as definition -> this.AddElement(definition.Id)
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
                | Ok () -> generatedProgram <- PipelineCodeGenerator.generate (pipelineStates ())
                | Error message -> generatedProgram <- message

                this.RaiseGeneratedProgramChanged()),
            (fun _ -> true))
        :> ICommand

    member this.AddElement(functionId: string) =
        let node = createNode drawing.Nodes.Count functionId
        node.X <- min (max 0. (drawing.Width - node.Width)) (24. + float (drawing.Nodes.Count % 6) * 118.)
        node.Y <- min (max 0. (drawing.Height - node.Height)) (24. + float (drawing.Nodes.Count / 6) * 72.)

        drawing.Nodes.Add(node :> INode)
        this.SelectedNode <- node

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

    member _.ValidateGraph() =
        let missingPins =
            drawing.Nodes
            |> Seq.collect (fun node ->
                node.Pins
                |> Seq.filter (fun pin -> not (drawing.IsPinConnected(pin)))
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
