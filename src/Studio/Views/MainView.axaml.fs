namespace Studio.Views

open System
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

    let pipelineKindFormat = DataFormat.CreateStringApplicationFormat("stackprocessing-pipeline-kind")
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

    let tryFindConnectorFromSource (source: obj) =
        let connectorFromControl (control: Control) =
            match control with
            | :? Connector as connectorControl when not (isNull connectorControl.ConnectorSource) ->
                Some connectorControl.ConnectorSource
            | _ ->
                match control.DataContext with
                | :? IConnector as connector -> Some connector
                | _ -> None

        match source with
        | :? Control as sourceControl ->
            match connectorFromControl sourceControl with
            | Some connector -> Some connector
            | None ->
                sourceControl.GetVisualAncestors()
                |> Seq.choose (fun visual ->
                    match visual with
                    | :? Control as control -> connectorFromControl control
                    | _ -> None)
                |> Seq.tryHead
        | _ -> None

    let deleteConnector (connector: IConnector) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if isNull editor then
            false
        else
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing -> drawing.Connectors.Remove(connector)
            | _ -> false

    let distanceToSegment (point: Point) (startPoint: Point) (endPoint: Point) =
        let dx = endPoint.X - startPoint.X
        let dy = endPoint.Y - startPoint.Y
        let lengthSquared = dx * dx + dy * dy

        if lengthSquared = 0. then
            let px = point.X - startPoint.X
            let py = point.Y - startPoint.Y
            Math.Sqrt(px * px + py * py)
        else
            let t =
                ((point.X - startPoint.X) * dx + (point.Y - startPoint.Y) * dy) / lengthSquared
                |> max 0.
                |> min 1.

            let projection = Point(startPoint.X + t * dx, startPoint.Y + t * dy)
            let px = point.X - projection.X
            let py = point.Y - projection.Y
            Math.Sqrt(px * px + py * py)

    let tryFindConnectorAtPoint (point: Point) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if isNull editor then
            None
        else
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing ->
                drawing.Connectors
                |> Seq.choose (fun connector ->
                    if isNull connector.Start || isNull connector.End then
                        None
                    else
                        let distance = distanceToSegment point (pinCenter connector.Start) (pinCenter connector.End)
                        if distance <= 10. then Some(connector, distance) else None)
                |> Seq.sortBy snd
                |> Seq.tryHead
                |> Option.map fst
            | _ -> None

    let deleteConnectionsForPin (pin: IPin) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if isNull editor then
            false
        else
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing ->
                let connectors =
                    drawing.Connectors
                    |> Seq.filter (fun connector ->
                        Object.ReferenceEquals(connector.Start, pin)
                        || Object.ReferenceEquals(connector.End, pin))
                    |> Seq.toArray

                for connector in connectors do
                    drawing.Connectors.Remove(connector) |> ignore

                connectors.Length > 0
            | _ -> false

    let resetConnectionGesture () =
        pendingPin <- None
        draggingPin <- None
        hideConnectionPreview()
        setCompatiblePinHighlight None

    let tryFindPinAtPoint (point: Point) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if isNull editor then
            None
        else
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing ->
                drawing.Nodes
                |> Seq.collect _.Pins
                |> Seq.choose (fun pin ->
                    let center = pinCenter pin
                    let dx = center.X - point.X
                    let dy = center.Y - point.Y
                    let distance = Math.Sqrt(dx * dx + dy * dy)

                    if distance <= 18. then Some(pin, distance) else None)
                |> Seq.sortBy snd
                |> Seq.tryHead
                |> Option.map fst
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

                    let existingInputConnectors =
                        drawing.Connectors
                        |> Seq.filter (fun connector -> Object.ReferenceEquals(connector.End, inputPin))
                        |> Seq.toArray

                    if not alreadyConnected then
                        for connector in existingInputConnectors do
                            drawing.Connectors.Remove(connector) |> ignore

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
                            let point = args.GetCurrentPoint(editor)
                            let graphPoint =
                                if isNull graphHost then
                                    Point()
                                else
                                    args.GetPosition(graphHost)

                            if point.Properties.IsRightButtonPressed then
                                match tryFindPinFromSource args.Source, tryFindConnectorFromSource args.Source, tryFindConnectorAtPoint graphPoint with
                                | Some pin, _, _ -> deleteConnectionsForPin pin |> ignore
                                | None, Some connector, _ -> deleteConnector connector |> ignore
                                | None, None, Some connector -> deleteConnector connector |> ignore
                                | None, None, None -> ()

                                resetConnectionGesture()
                                args.PreventGestureRecognition()
                                args.Handled <- true
                            else
                                match tryFindPinFromSource args.Source, tryFindConnectorFromSource args.Source with
                                | Some pin, _ ->
                                    match pendingPin with
                                    | Some firstPin when tryConnectPins firstPin pin ->
                                        resetConnectionGesture()
                                        args.Handled <- true
                                    | _ ->
                                        draggingPin <- Some pin
                                        pendingPin <- Some pin

                                        if not (isNull graphHost) then
                                            showConnectionPreview pin (args.GetPosition(graphHost))

                                        args.Handled <- false
                                | None, Some connector when point.Properties.IsRightButtonPressed ->
                                    resetConnectionGesture()
                                    deleteConnector connector |> ignore
                                    args.Handled <- true
                                | None, _ ->
                                    selectFromPointerSource args),
                        RoutingStrategies.Tunnel,
                        true)

                    editor.AddHandler(
                        Control.ContextRequestedEvent,
                        EventHandler<ContextRequestedEventArgs>(fun _ args ->
                            args.Handled <- true),
                        RoutingStrategies.Tunnel,
                        true)

                    editor.AddHandler(
                        InputElement.PointerMovedEvent,
                        EventHandler<PointerEventArgs>(fun _ args ->
                            match draggingPin with
                            | Some _ when not (isNull graphHost) ->
                                let pointer = args.GetPosition(graphHost)
                                updateConnectionPreview pointer
                                setCompatiblePinHighlight (tryFindPinAtPoint pointer)
                            | _ -> ()),
                        RoutingStrategies.Bubble,
                        true)

                    editor.AddHandler(
                        InputElement.PointerReleasedEvent,
                        EventHandler<PointerReleasedEventArgs>(fun _ args ->
                            let point = args.GetCurrentPoint(editor)

                            if point.Properties.PointerUpdateKind = PointerUpdateKind.RightButtonReleased then
                                resetConnectionGesture()
                                args.PreventGestureRecognition()
                                args.Handled <- true
                            else
                                let targetPin =
                                    if isNull graphHost then
                                        tryFindPinFromSource args.Source
                                    else
                                        tryFindPinAtPoint (args.GetPosition(graphHost))

                                match draggingPin, targetPin with
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
                    let data = new DataTransfer()
                    data.Add(DataTransferItem.Create(pipelineKindFormat, kind))

                    DragDrop.DoDragDropAsync(args, data, DragDropEffects.Copy)
                    |> _.ContinueWith(Action<Threading.Tasks.Task<DragDropEffects>>(fun _ -> paletteDragInProgress <- false))
                    |> ignore
                | _ -> ()
            | _ -> ()

    member _.PipelineEditorDragOver(_sender: obj, args: DragEventArgs) =
        if not (isNull args.DataTransfer) && args.DataTransfer.Contains(pipelineKindFormat) then
            args.DragEffects <- DragDropEffects.Copy
            args.Handled <- true

    member _.PipelineEditorDrop(_sender: obj, args: DragEventArgs) =
        if not (isNull args.DataTransfer) && args.DataTransfer.Contains(pipelineKindFormat) then
            match args.DataTransfer.TryGetValue(pipelineKindFormat) with
            | kind when not (String.IsNullOrWhiteSpace kind) ->
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
