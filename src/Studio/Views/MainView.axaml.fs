namespace Studio.Views

open System
open System.Threading.Tasks
open Avalonia
open Avalonia.Controls
open Avalonia.Controls.Primitives
open Avalonia.Input
open Avalonia.Interactivity
open Avalonia.Threading
open Avalonia.Markup.Xaml
open Avalonia.Media
open Avalonia.Controls.Shapes
open Avalonia.VisualTree
open NodeEditor.Controls
open NodeEditor.Model
open NodeEditor.Mvvm
open Studio.Models
open Studio.ViewModels

type MainView() as this =
    inherit UserControl()

    let pipelineKindFormat = "application/x-stackprocessing-pipeline-kind"
    let mutable paletteDragInProgress = false
    let mutable pendingPin: IPin option = None
    let mutable draggingPin: IPin option = None

    let pinCenter (pin: IPin) =
        if isNull pin.Parent then
            Point(pin.X + pin.Width / 2., pin.Y + pin.Height / 2.)
        else
            Point(pin.Parent.X + pin.X + pin.Width / 2., pin.Parent.Y + pin.Y + pin.Height / 2.)

    let showConnectionPreview (pin: IPin) (pointer: Point) =
        let preview = this.FindControl<Line>("ConnectionPreview")

        if not (isNull preview) then
            let start = pinCenter pin
            preview.StartPoint <- start
            preview.EndPoint <- pointer
            preview.IsVisible <- true

    let updateConnectionPreview (pointer: Point) =
        let preview = this.FindControl<Line>("ConnectionPreview")

        if not (isNull preview) && preview.IsVisible then
            preview.EndPoint <- pointer

    let hideConnectionPreview () =
        let preview = this.FindControl<Line>("ConnectionPreview")

        if not (isNull preview) then
            preview.IsVisible <- false

    let canConnectPins (first: IPin) (second: IPin) =
        let oppositeSides =
            match first.Alignment, second.Alignment with
            | PinAlignment.Right, PinAlignment.Left
            | PinAlignment.Left, PinAlignment.Right -> true
            | _ -> false

        oppositeSides && not (Object.ReferenceEquals(first.Parent, second.Parent))

    let setCompatiblePinHighlight (candidate: IPin option) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if not (isNull editor) then
            for visual in editor.GetVisualDescendants() do
                match visual with
                | :? Control as control ->
                    match control.DataContext with
                    | :? IPin as pin ->
                        control.Opacity <-
                            match draggingPin, candidate with
                            | Some firstPin, Some candidatePin
                                when Object.ReferenceEquals(pin, candidatePin) && canConnectPins firstPin candidatePin ->
                                1.0
                            | Some _, _ -> 0.55
                            | _ -> 1.0
                    | _ -> ()
                | _ -> ()

    let tryFindPinFromSource (source: obj) =
        match source with
        | :? Control as sourceControl ->
            match sourceControl.DataContext with
            | :? IPin as pin -> Some pin
            | _ ->
                sourceControl.GetVisualAncestors()
                |> Seq.choose (fun visual ->
                    match visual with
                    | :? Control as control ->
                        match control.DataContext with
                        | :? IPin as pin -> Some pin
                        | _ -> None
                    | _ -> None)
                |> Seq.tryHead
        | _ -> None

    let tryConnectPins (first: IPin) (second: IPin) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if isNull editor then
            false
        else
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing ->
                let outputPin, inputPin =
                    match first.Alignment, second.Alignment with
                    | PinAlignment.Right, PinAlignment.Left -> Some first, Some second
                    | PinAlignment.Left, PinAlignment.Right -> Some second, Some first
                    | _ -> None, None

                match outputPin, inputPin with
                | Some outputPin, Some inputPin when not (Object.ReferenceEquals(outputPin.Parent, inputPin.Parent)) ->
                    let alreadyConnected =
                        drawing.Connectors
                        |> Seq.exists (fun connector ->
                            Object.ReferenceEquals(connector.Start, outputPin)
                            && Object.ReferenceEquals(connector.End, inputPin))

                    if not alreadyConnected then
                        let connector = ConnectorViewModel()
                        connector.Start <- outputPin
                        connector.End <- inputPin
                        connector.Orientation <- ConnectorOrientation.Horizontal
                        drawing.Connectors.Add(connector :> IConnector)

                    true
                | _ -> false
            | _ -> false

    let syncGraphWindowSize () =
        let graphHost = this.FindControl<Grid>("GraphHost")
        let editor = this.FindControl<Editor>("PipelineEditor")

        if not (isNull graphHost) && not (isNull editor) && not (isNull editor.DrawingSource) then
            let width = graphHost.Bounds.Width
            let height = graphHost.Bounds.Height

            if width > 0. && height > 0. then
                editor.DrawingSource.Width <- width
                editor.DrawingSource.Height <- height

                editor.DrawingSource.Nodes
                |> Seq.iter (fun node ->
                    node.X <- min (max 0. (width - node.Width)) (max 0. node.X)
                    node.Y <- min (max 0. (height - node.Height)) (max 0. node.Y))

    let deleteSelectedNodeIfOverTrash () =
        let editor = this.FindControl<Editor>("PipelineEditor")

        let deleteSelectedElement () =
            if not (isNull this.DataContext) then
                let methodInfo = this.DataContext.GetType().GetMethod("DeleteSelectedElement")
                if not (isNull methodInfo) then
                    methodInfo.Invoke(this.DataContext, Array.empty) |> ignore

        if not (isNull editor) then
            match editor.DrawingSource, this.DataContext with
            | :? DrawingNodeViewModel as drawing, _ ->
                if not (isNull this.DataContext) then
                    let selectedElementProperty = this.DataContext.GetType().GetProperty("SelectedElement")
                    let selectedElement =
                        if isNull selectedElementProperty then
                            null
                        else
                            selectedElementProperty.GetValue(this.DataContext)

                    if not (isNull selectedElement) then
                        drawing.Nodes
                        |> Seq.tryFind (fun node ->
                            match node.Content with
                            | :? PipelineNodeContent as nodeContent ->
                                let nodeContentType = nodeContent.GetType()
                                let elementProperty = nodeContentType.GetProperty("Element")
                                not (isNull elementProperty)
                                && Object.ReferenceEquals(elementProperty.GetValue(nodeContent), selectedElement)
                            | _ -> false)
                        |> Option.iter (fun node ->
                            let trashLeft = max 0. (drawing.Width - 86. - 12.)
                            let trashTop = max 0. (drawing.Height - 42. - 12.)

                            if node.X + node.Width >= trashLeft && node.Y + node.Height >= trashTop then
                                deleteSelectedElement())
            | _ -> ()

    let clearNativeNodeSelection () =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if not (isNull editor) then
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing -> drawing.DeselectAllNodes()
            | _ -> ()

    do
        this.InitializeComponent()
        this.Loaded.Add(fun _ ->
            Dispatcher.UIThread.Post(fun () ->
                let graphHost = this.FindControl<Grid>("GraphHost")
                let editor = this.FindControl<Editor>("PipelineEditor")

                if not (isNull graphHost) then
                    graphHost.SizeChanged.Add(fun _ ->
                        Dispatcher.UIThread.Post(syncGraphWindowSize))

                    syncGraphWindowSize()

                if not (isNull editor) && not (isNull editor.ZoomControl) then
                    editor.ZoomControl.ResetZoomCommand()

                if not (isNull editor) then
                    let selectFromPointerSource (args: PointerEventArgs) =
                        match args.Source with
                        | :? Avalonia.Controls.Control as sourceControl ->
                            match sourceControl.DataContext with
                            | :? PipelineNodeContent as nodeContent ->
                                nodeContent.Select()
                            | _ ->
                                sourceControl.GetVisualAncestors()
                                |> Seq.choose (fun visual ->
                                    match visual with
                                    | :? Avalonia.Controls.Control as control ->
                                        match control.DataContext with
                                        | :? PipelineNodeContent as nodeContent -> Some nodeContent
                                        | _ -> None
                                    | _ -> None)
                                |> Seq.tryHead
                                |> Option.iter (fun nodeContent -> nodeContent.Select())
                        | _ -> ()

                    editor.AddHandler(
                        InputElement.PointerPressedEvent,
                        EventHandler<PointerPressedEventArgs>(fun _ args ->
                            match tryFindPinFromSource args.Source with
                            | Some pin ->
                                match pendingPin with
                                | Some firstPin when tryConnectPins firstPin pin ->
                                    pendingPin <- None
                                    draggingPin <- None
                                    hideConnectionPreview()
                                    setCompatiblePinHighlight None
                                    args.Handled <- true
                                | _ ->
                                    draggingPin <- Some pin
                                    pendingPin <- Some pin

                                    if not (isNull graphHost) then
                                        showConnectionPreview pin (args.GetPosition(graphHost))

                                    args.Handled <- false
                            | None ->
                                selectFromPointerSource args),
                        RoutingStrategies.Tunnel,
                        true)

                    editor.AddHandler(
                        InputElement.PointerMovedEvent,
                        EventHandler<PointerEventArgs>(fun _ args ->
                            match draggingPin with
                            | Some _ when not (isNull graphHost) ->
                                updateConnectionPreview (args.GetPosition(graphHost))
                                setCompatiblePinHighlight (tryFindPinFromSource args.Source)
                            | _ -> ()),
                        RoutingStrategies.Bubble,
                        true)

                    editor.AddHandler(
                        InputElement.PointerReleasedEvent,
                        EventHandler<PointerReleasedEventArgs>(fun _ args ->
                            match draggingPin, tryFindPinFromSource args.Source with
                            | Some firstPin, Some secondPin when tryConnectPins firstPin secondPin ->
                                pendingPin <- None
                                draggingPin <- None
                            | Some _, _ ->
                                draggingPin <- None
                            | None, _ -> ()

                            hideConnectionPreview()
                            setCompatiblePinHighlight None

                            Dispatcher.UIThread.Post(fun () ->
                                deleteSelectedNodeIfOverTrash()
                                clearNativeNodeSelection())),
                        RoutingStrategies.Bubble,
                        true)

                    editor.GetVisualDescendants()
                    |> Seq.iter (fun visual ->
                        match visual with
                        | :? ScrollViewer as scrollViewer ->
                            scrollViewer.HorizontalScrollBarVisibility <- ScrollBarVisibility.Disabled
                            scrollViewer.VerticalScrollBarVisibility <- ScrollBarVisibility.Disabled
                        | :? ScrollBar as scrollBar ->
                            scrollBar.IsVisible <- false
                        | _ -> ())))

    member private this.InitializeComponent() =
        AvaloniaXamlLoader.Load(this)

    member _.PipelineNodeClicked(sender: obj, args: RoutedEventArgs) =
        match sender with
        | :? Control as control ->
            match control.DataContext with
            | :? PipelineNodeContent as nodeContent ->
                nodeContent.Select()
                args.Handled <- true
            | _ -> ()
        | _ -> ()

    member _.PaletteElementPointerMoved(sender: obj, args: PointerEventArgs) =
        if not paletteDragInProgress then
            match sender with
            | :? Control as control when args.GetCurrentPoint(control).Properties.IsLeftButtonPressed ->
                match control.Tag with
                | :? string as kind ->
                    paletteDragInProgress <- true
                    let data = DataObject()
                    data.Set(pipelineKindFormat, kind)

                    DragDrop.DoDragDrop(args, data, DragDropEffects.Copy)
                    |> _.ContinueWith(Action<Task<DragDropEffects>>(fun _ -> paletteDragInProgress <- false))
                    |> ignore
                | _ -> ()
            | _ -> ()

    member _.PipelineEditorDragOver(_sender: obj, args: DragEventArgs) =
        if args.Data.Contains(pipelineKindFormat) then
            args.DragEffects <- DragDropEffects.Copy
            args.Handled <- true

    member _.PipelineEditorDrop(_sender: obj, args: DragEventArgs) =
        if args.Data.Contains(pipelineKindFormat) then
            match args.Data.Get(pipelineKindFormat) with
            | :? string as kind ->
                match Enum.TryParse<PipelineElementKind>(kind) with
                | true, elementKind ->
                    match this.DataContext with
                    | :? MainWindowViewModel as viewModel ->
                        viewModel.AddElement(elementKind)
                        args.Handled <- true
                    | _ -> ()
                | _ -> ()
            | _ -> ()

    member _.TrashDragOver(_sender: obj, args: DragEventArgs) =
        args.DragEffects <- DragDropEffects.Move
        args.Handled <- true

    member _.TrashDrop(_sender: obj, args: DragEventArgs) =
        if not (isNull this.DataContext) then
            let methodInfo = this.DataContext.GetType().GetMethod("DeleteSelectedElement")
            if not (isNull methodInfo) then
                methodInfo.Invoke(this.DataContext, Array.empty) |> ignore
            args.Handled <- true
