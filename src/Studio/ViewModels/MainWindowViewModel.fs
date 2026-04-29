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

type private PipelineNodeViewModel(element: PipelineElementViewModel, selectElement: PipelineElementViewModel -> unit) =
    inherit NodeViewModel()

    member _.Element = element

    override this.OnSelected() =
        base.OnSelected()
        selectElement element

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
            PropertyChangedEventHandler(fun _ _ ->
                generatedProgram <- PipelineCodeGenerator.generate elements
                this.RaiseGeneratedProgramChanged())

        for parameter in element.Parameters do
            parameter.PropertyChanged.AddHandler(handler)

        { new IDisposable with
            member _.Dispose() =
                for parameter in element.Parameters do
                    parameter.PropertyChanged.RemoveHandler(handler) }

    let makeNode (index: int) (element: PipelineElementViewModel) =
        let node = PipelineNodeViewModel(element, fun selected -> this.SelectedElement <- selected)
        node.Name <- element.Title
        node.Content <- PipelineNodeContent(element.Title, fun () -> this.SelectedElement <- element)
        node.X <- float (24 + index * 118)
        node.Y <- 66.
        node.Width <- 110.
        node.Height <- 48.
        node.Pins <- ObservableCollection<IPin>()

        if element.Kind <> PipelineElementKind.Source then
            node.AddPin(0., 24., 10., 10., PinAlignment.Left, "IN") |> ignore

        if element.Kind <> PipelineElementKind.Sink then
            node.AddPin(110., 24., 10., 10., PinAlignment.Right, "OUT") |> ignore

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

        generatedProgram <- PipelineCodeGenerator.generate elements

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

        elements.CollectionChanged.Add(fun args ->
            for subscription in parameterSubscriptions do
                subscription.Dispose()

            parameterSubscriptions.Clear()

            for element in elements do
                parameterSubscriptions.Add(subscribeElement this element)

            refreshDrawing ()
            this.RaiseGeneratedProgramChanged())

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
        SimpleCommand(
            (fun _ ->
                if not (isNull selectedElement) then
                    let nextIndex = max 0 (elements.IndexOf(selectedElement) - 1)
                    elements.Remove(selectedElement) |> ignore
                    if elements.Count > 0 then
                        this.SelectedElement <- elements[min nextIndex (elements.Count - 1)]
                    else
                        this.SelectedElement <- null),
            (fun _ -> not (isNull selectedElement)))
        :> ICommand

    member this.RunCommand =
        SimpleCommand(
            (fun _ ->
                generatedProgram <- PipelineCodeGenerator.generate elements
                refreshDrawing ()
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
        this.SelectedElement <- element

    member private this.RaiseGeneratedProgramChanged() =
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.GeneratedProgram))
