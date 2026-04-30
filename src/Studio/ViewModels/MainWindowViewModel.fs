namespace Studio.ViewModels

open System
open System.Collections.ObjectModel
open System.ComponentModel
open System.Windows.Input
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

type private PipelineNodeViewModel(
    element: PipelineElementViewModel,
    selectElement: PipelineElementViewModel -> unit,
    getDrawingSize: unit -> float * float) =
    inherit NodeViewModel()

    member _.Element = element

    member this.ClampToDrawing() =
        let drawingWidth, drawingHeight = getDrawingSize()
        let maxX = max 0. (drawingWidth - this.Width)
        let maxY = max 0. (drawingHeight - this.Height)

        this.X <- min maxX (max 0. this.X)
        this.Y <- min maxY (max 0. this.Y)

    override this.OnSelected() =
        base.OnSelected()
        selectElement element

    override this.OnMoved() =
        base.OnMoved()
        this.ClampToDrawing()

type PipelinePinViewModel(alignment: PinAlignment) =
    inherit PinViewModel()

    member _.TrianglePoints =
        match alignment with
        | PinAlignment.Left -> "0,0 14,7 0,14"
        | PinAlignment.Right -> "0,0 14,7 0,14"
        | _ -> "0,0 14,7 0,14"

type MainWindowViewModel() as this =
    inherit ViewModelBase()

    let elements = ObservableCollection<PipelineElementViewModel>()
    let mutable selectedElement: PipelineElementViewModel = null
    let mutable generatedProgram = ""

    let createElement kind =
        match kind with
        | PipelineElementKind.Source ->
            PipelineElementViewModel(
                kind,
                "source",
                [ PipelineParameterViewModel("Available memory", "availableMemory", "availableMemory") ])
        | PipelineElementKind.Read ->
            PipelineElementViewModel(
                kind,
                "read",
                [ PipelineParameterViewModel("Pixel type", "pixelType", "float")
                  PipelineParameterViewModel("Input", "input", "input")
                  PipelineParameterViewModel("Suffix", "suffix", ".tiff") ])
        | PipelineElementKind.DiscreteGaussian ->
            PipelineElementViewModel(
                kind,
                "discreteGaussian",
                [ PipelineParameterViewModel("Sigma", "sigma", "1.0")
                  PipelineParameterViewModel("Output region", "outputRegionMode", "None")
                  PipelineParameterViewModel("Boundary", "boundaryCondition", "None")
                  PipelineParameterViewModel("Window size", "windowSize", "15") ])
        | PipelineElementKind.Cast ->
            PipelineElementViewModel(
                kind,
                "cast",
                [ PipelineParameterViewModel("Source type", "sourceType", "float")
                  PipelineParameterViewModel("Target type", "targetType", "uint8") ])
        | PipelineElementKind.Write ->
            PipelineElementViewModel(
                kind,
                "write",
                [ PipelineParameterViewModel("Output", "output", "output")
                  PipelineParameterViewModel("Suffix", "suffix", ".tiff") ])
        | PipelineElementKind.Sink ->
            PipelineElementViewModel(kind, "sink", [])
        | _ ->
            PipelineElementViewModel(kind, "unknown", [])

    let editor =
        let editor = EditorViewModel()
        editor.Factory <- MyNodeFactory()
        editor.Templates <- editor.Factory.CreateTemplates()
        editor.Drawing <- editor.Factory.CreateDrawing("StackProcessing Pipeline")
        editor

    let drawing =
        editor.Drawing :?> DrawingNodeViewModel

    let parameterSubscriptions = ResizeArray<IDisposable>()

    let subscribeElement (this: MainWindowViewModel) (element: PipelineElementViewModel) =
        let handler =
            PropertyChangedEventHandler(fun _ _ -> ())

        for parameter in element.Parameters do
            parameter.PropertyChanged.AddHandler(handler)

        { new IDisposable with
            member _.Dispose() =
                for parameter in element.Parameters do
                    parameter.PropertyChanged.RemoveHandler(handler) }

    let makeNode (index: int) (element: PipelineElementViewModel) =
        let addPipelinePin (node: NodeViewModel) x y alignment name =
            let pin = PipelinePinViewModel(alignment)
            pin.Name <- name
            pin.Parent <- node
            pin.X <- x
            pin.Y <- y
            pin.Width <- 14.
            pin.Height <- 14.
            pin.Alignment <- alignment
            node.Pins.Add(pin :> IPin)

        let node =
            PipelineNodeViewModel(
                element,
                (fun selected -> this.SelectedElement <- selected),
                (fun () -> drawing.Width, drawing.Height))
        node.Name <- element.Title
        node.Content <- PipelineNodeContent(element.Title, element, fun () -> this.SelectedElement <- element)
        node.X <- float (24 + index * 118)
        node.Y <- 66.
        node.Width <- 110.
        node.Height <- 48.
        node.Pins <- ObservableCollection<IPin>()

        if element.Kind <> PipelineElementKind.Source then
            addPipelinePin node 0. 24. PinAlignment.Left "IN"

        if element.Kind <> PipelineElementKind.Sink then
            addPipelinePin node 110. 24. PinAlignment.Right "OUT"

        node.ClampToDrawing()
        node

    let tryPin alignment (node: NodeViewModel) =
        node.Pins
        |> Seq.tryFind (fun pin -> pin.Alignment = alignment)

    let refreshDrawing () =
        drawing.Nodes.Clear()
        drawing.Connectors.Clear()

        let nodes =
            elements
            |> Seq.mapi makeNode
            |> Seq.toArray

        for node in nodes do
            drawing.Nodes.Add(node :> INode)

        nodes
        |> Seq.pairwise
        |> Seq.iter (fun (left, right) ->
            match tryPin PinAlignment.Right left, tryPin PinAlignment.Left right with
            | Some startPin, Some endPin ->
                let connector = ConnectorViewModel()
                connector.Start <- startPin
                connector.End <- endPin
                connector.Orientation <- ConnectorOrientation.Horizontal
                drawing.Connectors.Add(connector :> IConnector)
            | _ -> ())

    do
        let seed =
            [ PipelineElementKind.Source
              PipelineElementKind.Read
              PipelineElementKind.DiscreteGaussian
              PipelineElementKind.Cast
              PipelineElementKind.Write
              PipelineElementKind.Sink ]
            |> List.map createElement

        for element in seed do
            elements.Add element

        refreshDrawing ()

        elements.CollectionChanged.Add(fun _ ->
            for subscription in parameterSubscriptions do
                subscription.Dispose()

            parameterSubscriptions.Clear()

            for element in elements do
                parameterSubscriptions.Add(subscribeElement this element)
            ())

        for element in elements do
            parameterSubscriptions.Add(subscribeElement this element)

    member this.Editor = editor
    member this.Elements = elements

    member this.SelectedElement
        with get () = selectedElement
        and set value =
            if this.SetProperty(&selectedElement, value) then
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))

    member _.HasSelectedElement = not (isNull selectedElement)

    member _.GeneratedProgram = generatedProgram

    member this.SelectElement(element: PipelineElementViewModel) =
        this.SelectedElement <- element

    member this.AddSourceCommand =
        SimpleCommand((fun _ -> this.AddElement(PipelineElementKind.Source)), (fun _ -> true)) :> ICommand

    member this.AddReadCommand =
        SimpleCommand((fun _ -> this.AddElement(PipelineElementKind.Read)), (fun _ -> true)) :> ICommand

    member this.AddGaussianCommand =
        SimpleCommand((fun _ -> this.AddElement(PipelineElementKind.DiscreteGaussian)), (fun _ -> true)) :> ICommand

    member this.AddCastCommand =
        SimpleCommand((fun _ -> this.AddElement(PipelineElementKind.Cast)), (fun _ -> true)) :> ICommand

    member this.AddWriteCommand =
        SimpleCommand((fun _ -> this.AddElement(PipelineElementKind.Write)), (fun _ -> true)) :> ICommand

    member this.AddSinkCommand =
        SimpleCommand((fun _ -> this.AddElement(PipelineElementKind.Sink)), (fun _ -> true)) :> ICommand

    member this.DeleteSelectedCommand =
        SimpleCommand((fun _ -> this.DeleteSelectedElement()), (fun _ -> not (isNull selectedElement)))
        :> ICommand

    member this.RunCommand =
        SimpleCommand(
            (fun _ ->
                match this.ValidateGraph() with
                | Ok () -> generatedProgram <- PipelineCodeGenerator.generate elements
                | Error message -> generatedProgram <- message

                this.RaiseGeneratedProgramChanged()),
            (fun _ -> true))
        :> ICommand

    member this.AddElement(kind: PipelineElementKind) =
        let element = createElement kind
        let insertIndex =
            elements
            |> Seq.tryFindIndex (fun item -> item.Kind = PipelineElementKind.Sink)
            |> Option.defaultValue elements.Count

        elements.Insert(insertIndex, element)

        let node = makeNode insertIndex element
        node.X <- min (max 0. (drawing.Width - node.Width)) (24. + float (drawing.Nodes.Count % 6) * 118.)
        node.Y <- min (max 0. (drawing.Height - node.Height)) (24. + float (drawing.Nodes.Count / 6) * 72.)
        drawing.Nodes.Add(node :> INode)

        this.SelectedElement <- element

    member this.DeleteSelectedElement() =
        if not (isNull selectedElement) then
            let nextIndex = max 0 (elements.IndexOf(selectedElement) - 1)
            let nodesToRemove =
                drawing.Nodes
                |> Seq.filter (fun node ->
                    match node with
                    | :? PipelineNodeViewModel as pipelineNode -> Object.ReferenceEquals(pipelineNode.Element, selectedElement)
                    | _ -> false)
                |> Seq.toArray

            let pinsToRemove =
                nodesToRemove
                |> Seq.collect _.Pins
                |> Seq.toArray

            let connectorsToRemove =
                drawing.Connectors
                |> Seq.filter (fun connector ->
                    pinsToRemove |> Array.exists (fun pin -> Object.ReferenceEquals(pin, connector.Start) || Object.ReferenceEquals(pin, connector.End)))
                |> Seq.toArray

            for connector in connectorsToRemove do
                drawing.Connectors.Remove(connector) |> ignore

            for node in nodesToRemove do
                drawing.Nodes.Remove(node) |> ignore

            elements.Remove(selectedElement) |> ignore

            if elements.Count > 0 then
                this.SelectedElement <- elements[min nextIndex (elements.Count - 1)]
            else
                this.SelectedElement <- null

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

            drawing.Nodes
            |> Seq.iter (fun node ->
                match node with
                | :? PipelineNodeViewModel as pipelineNode -> pipelineNode.ClampToDrawing()
                | _ -> ())

    member this.DeleteSelectedElementIfInTrashZone(trashWidth: float, trashHeight: float, margin: float) =
        if not (isNull selectedElement) then
            drawing.Nodes
            |> Seq.tryFind (fun node ->
                match node with
                | :? PipelineNodeViewModel as pipelineNode -> Object.ReferenceEquals(pipelineNode.Element, selectedElement)
                | _ -> false)
            |> Option.iter (fun node ->
                let trashLeft = max 0. (drawing.Width - trashWidth - margin)
                let trashTop = max 0. (drawing.Height - trashHeight - margin)
                let nodeRight = node.X + node.Width
                let nodeBottom = node.Y + node.Height

                if nodeRight >= trashLeft && nodeBottom >= trashTop then
                    this.DeleteSelectedElement())

    interface IGraphWindowController with
        member this.SetDrawingSize width height =
            this.SetDrawingSize(width, height)

        member this.DeleteSelectedElementIfInTrashZone trashWidth trashHeight margin =
            this.DeleteSelectedElementIfInTrashZone(trashWidth, trashHeight, margin)

    member private this.RaiseGeneratedProgramChanged() =
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.GeneratedProgram))
