namespace Studio.Views

open System
open System.Collections.Generic
open System.Runtime.CompilerServices
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
open Avalonia.Layout
open Avalonia.Platform.Storage
open Avalonia.VisualTree
open Graph
open NodeEditor.Controls
open NodeEditor.Model
open NodeEditor.Mvvm
open Studio.Models
open Studio.ViewModels

type MainView() as this =
    inherit UserControl()

    let pipelineKindFormat = DataFormat.CreateStringApplicationFormat("stackprocessing-pipeline-kind")
    let mutable paletteDragInProgress = false
    let mutable paletteDragFunctionId: string option = None
    let mutable pendingPin: IPin option = None
    let mutable draggingPin: IPin option = None

    let pinCenter (pin: IPin) =
        if isNull pin.Parent then
            Point(pin.X + pin.Width / 2., pin.Y + pin.Height / 2.)
        else
            Point(pin.Parent.X + pin.X + pin.Width / 2., pin.Parent.Y + pin.Y + pin.Height / 2.)

    let jsonFileType () =
        let fileType = FilePickerFileType("Pipeline JSON")
        fileType.Patterns <- [ "*.json"; "*.JSON" ]
        fileType.MimeTypes <- [ "application/json"; "text/json"; "text/plain" ]
        fileType.AppleUniformTypeIdentifiers <- [ "public.json"; "public.text" ]
        fileType

    let localPath (file: IStorageFile) =
        if file.Path.IsFile then
            Some file.Path.LocalPath
        else
            None

    let parentWindow () =
        match TopLevel.GetTopLevel(this) with
        | :? Window as window -> window
        | _ -> null

    let showLoadErrorAsync (message: string) =
        task {
            let dialog = Window()
            dialog.Title <- "Could not load graph"
            dialog.Width <- 460.
            dialog.Height <- 180.
            dialog.WindowStartupLocation <- WindowStartupLocation.CenterOwner
            dialog.CanResize <- false

            let text =
                TextBlock(
                    Text = message,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = Thickness(16.))

            let ok =
                Button(
                    Content = "OK",
                    Width = 88.,
                    HorizontalAlignment = HorizontalAlignment.Right,
                    Margin = Thickness(16., 0., 16., 16.))

            ok.Click.Add(fun _ -> dialog.Close())

            let panel = DockPanel(LastChildFill = true)
            DockPanel.SetDock(ok, Dock.Bottom)
            panel.Children.Add(ok)
            panel.Children.Add(text)
            dialog.Content <- panel

            match parentWindow () with
            | null ->
                dialog.Show()
            | owner ->
                do! dialog.ShowDialog(owner)
        }

    let confirmIfGraphIsNonEmptyAsync (viewModel: MainWindowViewModel) title message =
        task {
            if viewModel.HasGraph then
                return! ConfirmationDialogs.confirmAsync (parentWindow()) title message
            else
                return true
        }

    let confirmIfGraphIsDirtyAsync (viewModel: MainWindowViewModel) title message =
        task {
            if viewModel.HasGraph && viewModel.IsGraphDirty then
                return! ConfirmationDialogs.confirmAsync (parentWindow()) title message
            else
                return true
        }

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

    let currentDrawing () =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if isNull editor then
            None
        else
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing -> Some drawing
            | _ -> None

    let isDataConnector (connector: IConnector) =
        match connector.Start, connector.End with
        | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
            startPin.Kind = DataOutput && endPin.Kind = DataInput
        | _ ->
            false

    let isReducerNode (node: INode) =
        match node with
        | :? PipelineNodeViewModel as pipelineNode -> pipelineNode.State.Definition.Id = "ComputeStats"
        | _ -> false

    let isDataNode (node: INode) =
        match node with
        | :? PipelineNodeViewModel as pipelineNode -> pipelineNode.State.Definition.Inputs.Length > 0
        | _ -> false

    let candidateIsDataConnector (outputPin: PipelinePinViewModel) (inputPin: PipelinePinViewModel) =
        outputPin.Kind = DataOutput && inputPin.Kind = DataInput

    let candidateIsParameterConnector (outputPin: PipelinePinViewModel) (inputPin: PipelinePinViewModel) =
        (outputPin.Kind = ScalarOutput || outputPin.Kind = ReducerOutput) && inputPin.Kind = ParameterInput

    let isCandidateDataConnector (candidate: (PipelinePinViewModel * PipelinePinViewModel) option) connectorEndParent =
        match candidate with
        | Some(outputPin: PipelinePinViewModel, inputPin: PipelinePinViewModel) ->
            candidateIsDataConnector outputPin inputPin
            && Object.ReferenceEquals(inputPin.Parent, connectorEndParent)
        | None ->
            false

    let dataAncestors (drawing: DrawingNodeViewModel) (candidate: (PipelinePinViewModel * PipelinePinViewModel) option) (node: INode) =
        let visited = HashSet<INode>(HashIdentity.Reference)

        let rec visit (node: INode) =
            if visited.Add node then
                drawing.Connectors
                |> Seq.filter (fun connector ->
                    isDataConnector connector
                    && Object.ReferenceEquals(connector.End.Parent, node))
                |> Seq.iter (fun connector -> visit connector.Start.Parent)

                match candidate with
                | Some(outputPin: PipelinePinViewModel, inputPin: PipelinePinViewModel) when isCandidateDataConnector candidate node ->
                    visit outputPin.Parent
                | _ ->
                    ()

        visit node
        visited

    let shareDataAncestor (drawing: DrawingNodeViewModel) (candidate: (PipelinePinViewModel * PipelinePinViewModel) option) left right =
        let leftAncestors = dataAncestors drawing candidate left
        let rightAncestors = dataAncestors drawing candidate right
        leftAncestors |> Seq.exists rightAncestors.Contains

    let sharedDataSources drawing candidate left right =
        let leftAncestors = dataAncestors drawing candidate left
        let rightAncestors = dataAncestors drawing candidate right

        let commonAncestors =
            leftAncestors
            |> Seq.filter rightAncestors.Contains
            |> Seq.toArray

        let hasIncomingData (node: INode) =
            drawing.Connectors
            |> Seq.exists (fun connector ->
                isDataConnector connector
                && Object.ReferenceEquals(connector.End.Parent, node))

        let sourceAncestors =
            commonAncestors
            |> Array.filter (hasIncomingData >> not)

        if sourceAncestors.Length > 0 then sourceAncestors else commonAncestors

    let parameterTargetsFrom (drawing: DrawingNodeViewModel) (candidate: (PipelinePinViewModel * PipelinePinViewModel) option) (node: INode) =
        seq {
            yield!
                drawing.Connectors
                |> Seq.choose (fun connector ->
                    match connector.Start, connector.End with
                    | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin)
                        when (startPin.Kind = ScalarOutput || startPin.Kind = ReducerOutput)
                             && endPin.Kind = ParameterInput
                             && Object.ReferenceEquals(startPin.Parent, node) ->
                        Some endPin.Parent
                    | _ ->
                        None)

            match candidate with
            | Some(outputPin: PipelinePinViewModel, inputPin: PipelinePinViewModel)
                when candidateIsParameterConnector outputPin inputPin
                     && Object.ReferenceEquals(outputPin.Parent, node) ->
                yield inputPin.Parent
            | _ ->
                ()
        }

    let nodesDependingOnReducerOutput (drawing: DrawingNodeViewModel) (candidate: (PipelinePinViewModel * PipelinePinViewModel) option) reducerNode =
        let visited = HashSet<INode>(HashIdentity.Reference)
        let dependents = ResizeArray<INode>()

        let rec visit node =
            parameterTargetsFrom drawing candidate node
            |> Seq.iter (fun target ->
                if visited.Add target then
                    dependents.Add target
                    visit target)

        visit reducerNode
        dependents |> Seq.toArray

    let reducerMapDependencySources (drawing: DrawingNodeViewModel) (outputPin: PipelinePinViewModel) (inputPin: PipelinePinViewModel) : INode array =
        let candidate = Some(outputPin, inputPin)

        drawing.Nodes
        |> Seq.filter isReducerNode
        |> Seq.collect (fun reducerNode ->
            nodesDependingOnReducerOutput drawing candidate reducerNode
            |> Seq.collect (fun dependentNode ->
                if isDataNode dependentNode && not (isReducerNode dependentNode) then
                    sharedDataSources drawing candidate reducerNode dependentNode
                else
                    Array.empty))
        |> Seq.distinctBy RuntimeHelpers.GetHashCode
        |> Seq.toArray

    let wouldCreateReducerMapDependency (drawing: DrawingNodeViewModel) (outputPin: PipelinePinViewModel) (inputPin: PipelinePinViewModel) =
        reducerMapDependencySources drawing outputPin inputPin |> Array.isEmpty |> not

    let canConnectPins (first: IPin) (second: IPin) =
        let outputPin, inputPin =
            match first, second with
            | (:? PipelinePinViewModel as firstPin), (:? PipelinePinViewModel as secondPin) when firstPin.IsOutput && secondPin.IsInput ->
                Some first, Some second
            | (:? PipelinePinViewModel as firstPin), (:? PipelinePinViewModel as secondPin) when firstPin.IsInput && secondPin.IsOutput ->
                Some second, Some first
            | _ -> None, None

        let compatiblePortTypes (outputPin: IPin) (inputPin: IPin) =
            match outputPin, inputPin with
            | :? PipelinePinViewModel as outputPin, (:? PipelinePinViewModel as inputPin) ->
                PortType.canConnect outputPin.Port.Type inputPin.Port.Type
            | _ -> false

        match outputPin, inputPin with
        | Some (:? PipelinePinViewModel as outputPin), Some (:? PipelinePinViewModel as inputPin) ->
            not (Object.ReferenceEquals(outputPin.Parent, inputPin.Parent))
            && outputPin.IsActive
            && inputPin.IsActive
            && compatiblePortTypes outputPin inputPin
            && (currentDrawing ()
                |> Option.map (fun drawing -> not (wouldCreateReducerMapDependency drawing outputPin inputPin))
                |> Option.defaultValue true)
        | _ -> false

    let connectorOrientation (startPin: IPin) (endPin: IPin) =
        match startPin, endPin with
        | (:? PipelinePinViewModel as outputPin), (:? PipelinePinViewModel as inputPin)
            when outputPin.Kind = ScalarOutput || inputPin.Kind = ParameterInput ->
            ConnectorOrientation.Vertical
        | _ ->
            ConnectorOrientation.Horizontal

    let inputPinIsFree (pin: IPin) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if isNull editor then
            false
        else
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing ->
                drawing.Connectors
                |> Seq.exists (fun connector -> Object.ReferenceEquals(connector.End, pin))
                |> not
            | _ -> false

    let setCompatiblePinHighlight (candidate: IPin option) =
        let editor = this.FindControl<Editor>("PipelineEditor")

        let clearProblemHighlights () =
            currentDrawing ()
            |> Option.iter (fun drawing ->
                drawing.Nodes
                |> Seq.choose (function
                    | :? PipelineNodeViewModel as node -> Some node
                    | _ -> None)
                |> Seq.iter (fun node -> node.State.IsProblemHighlighted <- false))

        let candidatePins (first: IPin) (second: IPin) =
            match first, second with
            | (:? PipelinePinViewModel as firstPin), (:? PipelinePinViewModel as secondPin) when firstPin.IsOutput && secondPin.IsInput ->
                Some(firstPin, secondPin)
            | (:? PipelinePinViewModel as firstPin), (:? PipelinePinViewModel as secondPin) when firstPin.IsInput && secondPin.IsOutput ->
                Some(secondPin, firstPin)
            | _ ->
                None

        let isBaseCompatible (outputPin: PipelinePinViewModel) (inputPin: PipelinePinViewModel) =
            not (Object.ReferenceEquals(outputPin.Parent, inputPin.Parent))
            && outputPin.IsActive
            && inputPin.IsActive
            && PortType.canConnect outputPin.Port.Type inputPin.Port.Type
            && (if inputPin.IsInput then inputPinIsFree inputPin else true)

        let highlightSemanticProblemSources () =
            clearProblemHighlights()

            match draggingPin, candidate, currentDrawing () with
            | Some firstPin, Some candidatePin, Some drawing ->
                match candidatePins firstPin candidatePin with
                | Some(outputPin, inputPin) when isBaseCompatible outputPin inputPin ->
                    reducerMapDependencySources drawing outputPin inputPin
                    |> Array.choose (function
                        | :? PipelineNodeViewModel as node -> Some(node: PipelineNodeViewModel)
                        | _ -> None)
                    |> Array.iter (fun node -> node.State.IsProblemHighlighted <- true)
                | _ -> ()
            | _ -> ()

        let isCompatibleHighlight (firstPin: IPin) (candidatePin: IPin) =
            match firstPin, candidatePin with
            | (:? PipelinePinViewModel as firstPin), (:? PipelinePinViewModel as candidatePin) ->
                candidatePin.IsActive
                && canConnectPins firstPin candidatePin
                && (if candidatePin.IsInput then inputPinIsFree candidatePin else true)
            | _ -> false

        highlightSemanticProblemSources()

        if not (isNull editor) then
            for visual in editor.GetVisualDescendants() do
                match visual with
                | :? Control as control ->
                    match control.DataContext with
                    | :? IPin as pin ->
                        control.Opacity <-
                            match draggingPin, candidate with
                            | Some firstPin, _ ->
                                match pin with
                                | :? PipelinePinViewModel when isCompatibleHighlight firstPin pin ->
                                    1.0
                                | _ -> 0.55
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
                    match first, second with
                    | (:? PipelinePinViewModel as firstPin), (:? PipelinePinViewModel as secondPin) when firstPin.IsOutput && secondPin.IsInput ->
                        Some first, Some second
                    | (:? PipelinePinViewModel as firstPin), (:? PipelinePinViewModel as secondPin) when firstPin.IsInput && secondPin.IsOutput ->
                        Some second, Some first
                    | _ -> None, None

                match outputPin, inputPin with
                | Some outputPin, Some inputPin when canConnectPins outputPin inputPin ->
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
                        connector.Orientation <- connectorOrientation outputPin inputPin

                        Dispatcher.UIThread.Post(
                            (fun () -> drawing.Connectors.Add(connector :> IConnector)),
                            DispatcherPriority.Background)

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
        match this.DataContext with
        | :? IGraphWindowController as controller ->
            controller.DeleteSelectedElementIfInTrashZone 86. 42. 12.
        | _ -> ()

    let clearNativeNodeSelection () =
        let editor = this.FindControl<Editor>("PipelineEditor")

        if not (isNull editor) then
            match editor.DrawingSource with
            | :? DrawingNodeViewModel as drawing -> drawing.DeselectAllNodes()
            | _ -> ()

    let isInsideGraphHost (point: Point) (graphHost: Grid) =
        point.X >= 0.
        && point.Y >= 0.
        && point.X <= graphHost.Bounds.Width
        && point.Y <= graphHost.Bounds.Height

    let showPaletteDragPreview functionId (rootPoint: Point) =
        let preview = this.FindControl<Border>("PaletteDragPreview")
        let label = this.FindControl<TextBlock>("PaletteDragPreviewLabel")

        if not (isNull preview) then
            Canvas.SetLeft(preview, rootPoint.X - preview.Width / 2.)
            Canvas.SetTop(preview, rootPoint.Y - preview.Height / 2.)
            preview.IsVisible <- true

        if not (isNull label) then
            let text =
                match BuiltInCatalog.tryFind functionId with
                | Some definition -> definition.DisplayName
                | None -> functionId

            label.Text <- text

    let hidePaletteDragPreview () =
        let preview = this.FindControl<Border>("PaletteDragPreview")

        if not (isNull preview) then
            preview.IsVisible <- false

    let movePaletteDragNode (args: PointerEventArgs) =
        match paletteDragFunctionId with
        | Some functionId ->
            let graphHost = this.FindControl<Grid>("GraphHost")

            if not (isNull graphHost) then
                let point = args.GetPosition(graphHost)
                let isOutsideGraph = not (isInsideGraphHost point graphHost)

                match this.DataContext with
                | :? MainWindowViewModel as viewModel ->
                    viewModel.MoveSelectedElementTo(point.X, point.Y, false, isOutsideGraph)
                | _ -> ()

                if isOutsideGraph then
                    showPaletteDragPreview functionId (args.GetPosition(this))
                else
                    hidePaletteDragPreview()

            args.Handled <- true
        | None -> ()

    let finishPaletteDrag (args: PointerReleasedEventArgs) =
        match paletteDragFunctionId with
        | Some _ ->
            paletteDragFunctionId <- None
            paletteDragInProgress <- false
            hidePaletteDragPreview()
            args.Pointer.Capture(null) |> ignore

            let graphHost = this.FindControl<Grid>("GraphHost")

            if not (isNull graphHost) then
                let dropPoint = args.GetPosition(graphHost)

                match this.DataContext with
                | :? MainWindowViewModel as viewModel ->
                    if isInsideGraphHost dropPoint graphHost then
                        viewModel.MoveSelectedElementTo(dropPoint.X, dropPoint.Y, true, false)
                    else
                        viewModel.DeleteSelectedElement()
                | _ -> ()

            args.Handled <- true
        | None -> ()

    do
        this.InitializeComponent()
        this.AddHandler(
            InputElement.PointerMovedEvent,
            EventHandler<PointerEventArgs>(fun _ args -> movePaletteDragNode args),
            RoutingStrategies.Tunnel,
            true)

        this.AddHandler(
            InputElement.PointerReleasedEvent,
            EventHandler<PointerReleasedEventArgs>(fun _ args -> finishPaletteDrag args),
            RoutingStrategies.Tunnel,
            true)

        this.Loaded.Add(fun _ ->
            Dispatcher.UIThread.Post(fun () ->
                let graphHost = this.FindControl<Grid>("GraphHost")
                let editor = this.FindControl<Editor>("PipelineEditor")

                if not (isNull graphHost) then
                    graphHost.SizeChanged.Add(fun _ ->
                        Dispatcher.UIThread.Post(syncGraphWindowSize))

                    syncGraphWindowSize()

                    graphHost.AddHandler(
                        DragDrop.DragOverEvent,
                        EventHandler<DragEventArgs>(fun _ args -> this.PipelineEditorDragOver(graphHost, args)),
                        RoutingStrategies.Tunnel,
                        true)

                    graphHost.AddHandler(
                        DragDrop.DropEvent,
                        EventHandler<DragEventArgs>(fun _ args -> this.PipelineEditorDrop(graphHost, args)),
                        RoutingStrategies.Tunnel,
                        true)

                if not (isNull editor) && not (isNull editor.ZoomControl) then
                    editor.ZoomControl.ResetZoomCommand()

                if not (isNull editor) then
                    editor.AddHandler(
                        DragDrop.DragOverEvent,
                        EventHandler<DragEventArgs>(fun _ args -> this.PipelineEditorDragOver(editor, args)),
                        RoutingStrategies.Tunnel,
                        true)

                    editor.AddHandler(
                        DragDrop.DropEvent,
                        EventHandler<DragEventArgs>(fun _ args -> this.PipelineEditorDrop(editor, args)),
                        RoutingStrategies.Tunnel,
                        true)

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

                                        args.PreventGestureRecognition()
                                        args.Handled <- true
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
                        | _ -> ())

                    match this.DataContext with
                    | :? MainWindowViewModel as viewModel ->
                        Dispatcher.UIThread.Post(fun () -> viewModel.ConnectSeedPipeline())
                    | _ -> ()))

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

    member _.SaveGraphClicked(_sender: obj, args: RoutedEventArgs) =
        task {
            args.Handled <- true

            match this.DataContext with
            | :? MainWindowViewModel as viewModel ->
                match viewModel.CurrentGraphPath with
                | Some path when viewModel.HasGraph ->
                    viewModel.SaveGraph(path)
                | _ ->
                    do! this.SaveGraphAsAsync()
            | _ -> ()
        }
        |> ignore

    member this.SaveGraphAsAsync() =
        task {
            match TopLevel.GetTopLevel(this), this.DataContext with
            | null, _ -> ()
            | _, (:? MainWindowViewModel as viewModel) ->
                let topLevel = TopLevel.GetTopLevel(this)
                let jsonType = jsonFileType()

                let options =
                    FilePickerSaveOptions(
                        Title = "Save pipeline graph",
                        SuggestedFileName = viewModel.SuggestedGraphFileName,
                        DefaultExtension = "json",
                        ShowOverwritePrompt = true,
                        FileTypeChoices = [ jsonType ],
                        SuggestedFileType = jsonType)

                let! file = topLevel.StorageProvider.SaveFilePickerAsync(options)

                if not (isNull file) then
                    let! stream = file.OpenWriteAsync()
                    use stream = stream
                    do! PipelineGraphStorage.writeJsonAsync stream (viewModel.ExportGraph())
                    file |> localPath |> Option.iter viewModel.SetCurrentGraphPath
                    viewModel.MarkGraphSaved()
            | _ -> ()
        }

    member this.SaveGraphAsClicked(_sender: obj, args: RoutedEventArgs) =
        task {
            args.Handled <- true
            do! this.SaveGraphAsAsync()
        }
        |> ignore

    member _.LoadGraphClicked(_sender: obj, args: RoutedEventArgs) =
        task {
            args.Handled <- true

            match TopLevel.GetTopLevel(this), this.DataContext with
            | null, _ -> ()
            | _, (:? MainWindowViewModel as viewModel) ->
                let! confirmed =
                    confirmIfGraphIsDirtyAsync
                        viewModel
                        "Load graph?"
                        "Loading a graph will replace the unsaved graph currently in memory. Continue?"

                if confirmed then
                    let topLevel = TopLevel.GetTopLevel(this)
                    let jsonType = jsonFileType()

                    let options =
                        FilePickerOpenOptions(
                            Title = "Load pipeline graph",
                            AllowMultiple = false,
                            FileTypeFilter = [ jsonType; FilePickerFileTypes.All ])

                    let! files = topLevel.StorageProvider.OpenFilePickerAsync(options)

                    match files |> Seq.tryHead with
                    | Some file ->
                        try
                            let! stream = file.OpenReadAsync()
                            use stream = stream
                            let! graph = PipelineGraphStorage.readJsonAsync stream
                            viewModel.ImportGraph(graph)
                            file |> localPath |> Option.iter viewModel.SetCurrentGraphPath
                        with ex ->
                            do! showLoadErrorAsync ex.Message
                    | None -> ()
            | _ -> ()
        }
        |> ignore

    member _.ClearGraphClicked(_sender: obj, args: RoutedEventArgs) =
        task {
            args.Handled <- true

            match this.DataContext with
            | :? MainWindowViewModel as viewModel ->
                let! confirmed =
                    confirmIfGraphIsDirtyAsync
                        viewModel
                        "Clear graph?"
                        "Clearing will delete the unsaved graph currently in memory. Continue?"

                if confirmed then
                    viewModel.ClearGraph()
            | _ -> ()
        }
        |> ignore

    member _.PaletteElementPointerPressed(sender: obj, args: PointerPressedEventArgs) =
        match sender with
        | :? Control as control when not paletteDragInProgress && args.GetCurrentPoint(control).Properties.IsLeftButtonPressed ->
            match control.Tag with
            | :? string as functionId ->
                paletteDragInProgress <- true
                paletteDragFunctionId <- Some functionId
                args.PreventGestureRecognition()
                args.Handled <- true
                args.Pointer.Capture(this) |> ignore

                let graphHost = this.FindControl<Grid>("GraphHost")

                if not (isNull graphHost) then
                    let point = args.GetPosition(graphHost)
                    let isOutsideGraph = not (isInsideGraphHost point graphHost)

                    match this.DataContext with
                    | :? MainWindowViewModel as viewModel ->
                        viewModel.AddPaletteDragElementAt(functionId, point.X, point.Y, isOutsideGraph)
                    | _ -> ()

                    if isOutsideGraph then
                        showPaletteDragPreview functionId (args.GetPosition(this))
                    else
                        hidePaletteDragPreview()
            | _ -> ()
        | _ -> ()

    member _.PipelineEditorDragOver(_sender: obj, args: DragEventArgs) =
        if not (isNull args.DataTransfer) && args.DataTransfer.Contains(pipelineKindFormat) then
            args.DragEffects <- DragDropEffects.Copy
            args.Handled <- true

    member _.PipelineEditorDrop(_sender: obj, args: DragEventArgs) =
        if not (isNull args.DataTransfer) && args.DataTransfer.Contains(pipelineKindFormat) then
            match args.DataTransfer.TryGetValue(pipelineKindFormat) with
            | functionId when not (String.IsNullOrWhiteSpace functionId) ->
                match this.DataContext with
                | :? MainWindowViewModel as viewModel ->
                    let graphHost = this.FindControl<Grid>("GraphHost")
                    let dropPoint =
                        if isNull graphHost then
                            Point()
                        else
                            args.GetPosition(graphHost)

                    viewModel.AddElementAt(functionId, dropPoint.X, dropPoint.Y)
                    args.Handled <- true
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
